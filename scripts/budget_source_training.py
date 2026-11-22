"""Fine-tune candidate scores against the expected reward of a twelve-person list."""
import argparse
import hashlib

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from distilled_source_pairs import OUT, Packed, PairNet, samples, write_json

PLAN=dict(budget=12,temperature=.45,learning_rate=.0004,nll_weight=.03,checkpoints=[0,2,5,9],
          selection="Most validation double hits, then lower validation pair NLL, then earlier epoch; checkpoint zero is the frozen additive model",
          loss="Negative expected product of the two inclusion probabilities plus 0.03 pair NLL. Shared soft cutoff chosen to allocate exactly 12 slots per event.")


def soft_membership(score,mask):
    t=PLAN["temperature"]
    with torch.no_grad():
        lo=score.masked_fill(~mask,torch.inf).amin(1,keepdim=True)-20*t
        hi=score.masked_fill(~mask,-torch.inf).amax(1,keepdim=True)+20*t
        for _ in range(25):
            mid=(lo+hi)/2
            count=((torch.sigmoid((score-mid)/t))*mask).sum(1,keepdim=True)
            lo=torch.where(count>12,mid,lo)
            hi=torch.where(count>12,hi,mid)
        cutoff=(lo+hi)/2
        g=torch.sigmoid((score-cutoff)/t)
        slope=(g*(1-g))*mask
        influence=slope/slope.sum(1,keepdim=True).clamp_min(1e-12)
    # Implicit derivative of sum(sigmoid((score-cutoff)/temperature)) == 12.
    cutoff=cutoff+((score-score.detach())*influence).sum(1,keepdim=True)
    return torch.sigmoid((score-cutoff)/t)*mask


def step(model,data,idx,soft):
    mask=data.mask[idx]
    p=model.node(model.encoder(data.x[idx])).squeeze(-1)
    a=data.anchors[idx]
    b=data.partners[idx]
    row=torch.arange(len(idx))
    g=soft_membership(p,mask)
    expected_partner=(data.soft[idx]*g).sum(1) if soft else g[row,b]
    reward=g[row,a]*expected_partner
    partner_score=(p*data.soft[idx]).sum(1) if soft else p[row,b]
    logits=p[:,:,None]+p[:,None,:]
    valid=mask[:,:,None] & mask[:,None,:] & ~data.diag
    logz=torch.logsumexp(logits.masked_fill(~valid,-torch.inf).flatten(1),1)-np.log(2.)
    nll=logz-p[row,a]-partner_score
    return (PLAN["nll_weight"]*nll-reward).mean()


def validate(model,data):
    hits=0
    nll=0.
    with torch.no_grad():
        for idx in np.array_split(np.arange(len(data.x)),15):
            scores=model.node(model.encoder(data.x[idx])).squeeze(-1).masked_fill(~data.mask[idx],-torch.inf)
            # Stable order on the saved sorted candidate roster for exact score ties.
            top=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:12]
            hits+=int(((top==data.anchors[idx,None]).any(1) & (top==data.partners[idx,None]).any(1)).sum())
            nll+=len(idx)*data.loss(model,idx,False).item()
    return hits,nll/len(data.x)


def train(name):
    folder=OUT/name
    write_json(folder/"budget_protocol.json",PLAN)
    params=np.load(folder/"scale.npz")
    mean,std=params["mean"],params["std"]
    trainset=Packed(samples(name,"fit"),mean,std)
    validation=Packed(samples(name,"cal"),mean,std)
    summary={}
    for tag in ("hard","soft"):
        model=PairNet(len(mean),False)
        model.load_state_dict(torch.load(folder/("node_"+tag+".pt"),weights_only=True))
        optimizer=torch.optim.AdamW(model.parameters(),lr=PLAN["learning_rate"],weight_decay=.005)
        rng=np.random.default_rng(22739)
        best=None;curve={}
        for epoch in range(10):
            if epoch:
                model.train()
                for idx in np.array_split(rng.permutation(len(trainset.x)),42):
                    optimizer.zero_grad()
                    loss=step(model,trainset,idx,tag=="soft")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                    optimizer.step()
            if epoch in PLAN["checkpoints"]:
                hits,nll=validate(model.eval(),validation)
                curve[epoch]=dict(hits=hits,nll=nll)
                if best is None or (hits,-nll)>best:
                    best=(hits,-nll)
                    selected=epoch
                    torch.save(model.state_dict(),folder/("budget_"+tag+".pt"))
        summary[tag]=dict(epoch=selected,curve=curve)
        print(name,tag,summary[tag],flush=True)
    write_json(folder/"budget_fit.json",summary)
    files=("budget_fit.json","budget_hard.pt","budget_soft.pt","budget_protocol.json","weights_lock.json")
    write_json(folder/"budget_lock.json",{f:hashlib.sha256((folder/f).read_bytes()).hexdigest() for f in files})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("network",choices=["company","Dept2"])
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        train(parser.parse_args().network)

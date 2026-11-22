"""Five cumulative public suffix snapshots for the authors' FGSSI architecture."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from threadpoolctl import threadpool_limits
from torch import nn

from matched_source_baselines import MAIN, OUT as MATCHED, RUNS, cohort, digest, evaluation_windows, load, make_windows, raw_data, read_rows, save_rows, sorted_nodes, tie_order, write_json
from distilled_source_pairs import pair_rank

OUT = MAIN.parent/"temporal_transfer"
UP = OUT/"fgssi_upstream"
assert digest(UP/"model.py") == "9b5e83f92a8586bd710a667338b7ff641092e38c63096bdbbdccfff5bf71c418"
assert digest(UP/"args.py") == "fe23434cd23dd362117feec3be2170a06b9a9647959e49af1b29ac74ffc2a4f0"
sys.path.insert(0, str(UP))
import model as upstream

SEED = 111
EPOCHS = (5, 10, 20)
upstream.args.device = "cpu"


class Geometry:
    def __init__(self, graph):
        self.nodes = graph["nodes"]
        self.index = {v:i for i,v in enumerate(self.nodes)}
        n = len(self.nodes)
        adj = np.zeros((n,n), np.float32)
        for u,v,_ in graph["edges"]:
            i,j = self.index[u],self.index[v]
            adj[i,j] = adj[j,i] = 1
        degree = adj.sum(1)
        assert degree.min() > 0
        inv = degree**-.5
        self.adj = csr_matrix(adj)
        self.degree = degree
        self.central = np.abs(eigsh(self.adj.astype(np.float64), k=1, which="LA", v0=np.ones(n))[1][:,0]).astype(np.float32)
        common = (self.adj@self.adj.T).toarray()
        dice = 2*common/(degree[:,None]+degree[None,:])
        identity = np.eye(n,dtype=np.float32)
        self.norm = csr_matrix((adj+identity)*inv[:,None]*inv[None,:])
        self.dice_norm = (adj+identity+dice)*inv[:,None]*inv[None,:]
        self.dice = dice

    def inputs(self, events):
        n = len(self.nodes)
        snap = np.zeros((len(events),5,n), np.float32)
        for j,e in enumerate(events):
            batches = e["observed"]
            # Endpoints include the first and last public batch. Short suffixes repeat frames.
            cuts = np.linspace(0,len(batches)-1,5).round().astype(int)
            for t,k in enumerate(cuts):
                snap[j,t,[self.index[v] for group in batches[:k+1] for v in group]] = 1
        flat = snap.reshape(-1,n).T
        infected = self.adj @ flat
        degree = self.degree[:,None]
        relative = infected/degree
        abs_degree = degree/degree.max()
        abs_neighbors = infected/degree.max()
        features = np.stack(np.broadcast_arrays(1-flat,flat,relative,1-relative,abs_degree,
            self.central[:,None],abs_neighbors,abs_degree-abs_neighbors),axis=-1).astype(np.float32)
        pooled = (self.norm @ features.reshape(n,-1)).reshape(n,-1,8).transpose(1,0,2)
        active = self.dice_norm @ flat
        conv = np.stack((self.dice_norm.sum(1)[:,None]-active, active),axis=-1).transpose(1,0,2)
        def tensor(a):
            return torch.from_numpy(np.ascontiguousarray(a.reshape(len(events),5,n,-1)))
        return tensor(conv),tensor(pooled),torch.from_numpy(snap)


class FGSSI(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleList(upstream.GCN(2,1,filters=[16],dropout=.2,behavior_identity=True) for _ in range(5))
        self.gru = nn.GRU(9,2,3,batch_first=True,bidirectional=True)
        self.att = upstream.GAT(4,-1,2,0,.01,4)

    def forward(self, conv, pooled, snap):
        B,T,N = snap.shape
        local = []
        for t,enc in enumerate(self.enc):
            layer = enc.gconv[0]
            local.append(enc.fc(torch.relu(conv[:,t]@layer.weight+layer.bias)))
        feature = (torch.relu(torch.stack(local,1))+snap[:,0,None,:,None])/2
        x = torch.cat((feature,pooled),dim=-1).transpose(1,2).reshape(B*N,T,9)
        h = self.gru(x)[0]
        heads = []
        # Upstream GAT returns only time row zero. The three non-neighbor
        # entries in that row retain zero logits before the five-way softmax.
        for att in self.att.attentions:
            wh = h @ att.W
            focal = wh[:,0] @ att.a[:2]
            two = focal + (wh[:,:2] @ att.a[2:]).squeeze(-1)
            alpha = torch.softmax(torch.nn.functional.pad(two,(0,3)),dim=1)
            heads.append(torch.nn.functional.elu((alpha[:,:,None]*wh).sum(1)))
        logits = torch.stack(heads).mean(0).reshape(B,N,2)
        return logits,feature[:,0,:,0]


def fidelity_check():
    torch.manual_seed(101)
    graph = dict(nodes=list(range(7)),edges=[(i,(i+1)%7,1) for i in range(7)]+[(1,4,2)])
    g = Geometry(graph)
    rows = [dict(observed=[[0],[2,1],[3],[4],[5]]),dict(observed=[[1],[4],[2]])]
    conv,pooled,snap = g.inputs(rows)
    model = FGSSI().eval()
    logits,_ = model(conv,pooled,snap)
    upstream.args.network_verNum = 7
    adj = torch.tensor(g.adj.toarray())
    state = torch.stack((1-snap,snap),dim=-1)
    for enc in model.enc:
        enc.gconv[0].Dice_M2 = torch.tensor(g.dice)
    enc_out = torch.stack([enc([adj,None,None,state],t) for t,enc in enumerate(model.enc)],1)
    x = torch.cat(((enc_out.relu()+snap[:,0,None,:,None])/2,pooled),dim=-1)
    h = torch.stack([model.gru(x[:,:,i])[0] for i in range(7)],dim=2)
    ref = torch.stack([model.att(item,None) for item in h])
    ours = logits.softmax(-1)
    params = list(model.parameters())
    probe = torch.randn_like(ours)
    grad = lambda out: torch.autograd.grad((out*probe).sum(),params)
    err = max(float((ref-ours).abs().max().detach()),*[float((a-b).abs().max()) for a,b in zip(grad(ref),grad(ours))])
    assert err < 1e-5 and not torch.equal(snap[:,0],snap[:,-1]),err
    return err


def data(name,phase):
    result=[]
    for graph,events,truth in cohort(name,phase):
        geo=Geometry(graph)
        conv,pooled,snap=geo.inputs(events)
        y=torch.zeros(len(events),len(geo.nodes))
        for i,r in enumerate(truth):
            y[i,[geo.index[v] for v in r["sources"]]]=1
        assert not (y*snap[:,-1]).any()
        result.append((conv,pooled,snap,y))
    return result


def validation(model,sets):
    total=n=0
    with torch.no_grad():
        for conv,pool,snap,y in sets:
            for idx in torch.arange(len(y)).split(16):
                logits,_=model(conv[idx],pool[idx],snap[idx])
                odds=(logits[:,:,1]-logits[:,:,0]).double()
                legal=snap[idx,-1]==0
                pair = odds[:,:,None]+odds[:,None,:]
                valid=legal[:,:,None]&legal[:,None,:]&torch.ones(pair.shape[1:],dtype=torch.bool).triu(1)
                total+=float((pair.masked_fill(~valid,-torch.inf).flatten(1).logsumexp(1)-(odds*y[idx]).sum(1)).sum())
                n+=len(idx)
    return total/n


def fit(name):
    folder=OUT/name
    folder.mkdir(parents=True,exist_ok=True)
    error=fidelity_check()
    train,cal=data(name,"fit"),data(name,"cal")
    assert sum(len(x[-1]) for x in train)==1000 and sum(len(x[-1]) for x in cal)==300
    torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED)
    model=FGSSI()
    opt=torch.optim.Adam([dict(params=model.enc.parameters(),lr=.003),dict(params=model.gru.parameters(),lr=.005),
                          dict(params=model.att.parameters(),lr=.001)])
    curve={};best=float("inf")
    for epoch in range(1,max(EPOCHS)+1):
        model.train()
        for wi in rng.permutation(len(train)):
            conv,pool,snap,y=train[wi]
            for idx in torch.from_numpy(rng.permutation(len(y))).split(8):
                logits,aux=model(conv[idx],pool[idx],snap[idx])
                legal=1-snap[idx,-1]
                count=legal.sum(1,keepdims=True)
                gold=y[idx]
                weight=torch.where(gold>0,1-2/count,2/count)*legal
                ce=-logits.log_softmax(-1).gather(-1,gold.long().unsqueeze(-1)).squeeze(-1)
                loss=((weight*(ce+2*(aux-gold).square())).sum(1)/count[:,0]).mean()
                opt.zero_grad();loss.backward();opt.step()
        if epoch in EPOCHS:
            value=validation(model.eval(),cal)
            curve[epoch]=value
            if value<best:
                best=value;torch.save(model.state_dict(),folder/"fgssi.pt")
            print(name,epoch,value,flush=True)
    write_json(folder/"selection.json",dict(checkpoints=curve,selected=min(curve,key=curve.get),upstream_error=error,
        weights=digest(folder/"fgssi.pt"),fit=1000,cal=300,seed=SEED))


def trained(name):
    folder=OUT/name
    assert digest(folder/"fgssi.pt")==load(folder/"selection.json")["weights"]
    net=FGSSI().eval()
    net.load_state_dict(torch.load(folder/"fgssi.pt",weights_only=True))
    return net


def predictions(net,graph,events):
    geo=Geometry(graph)
    for start in range(0,len(events),16):
        batch=events[start:start+16]
        with torch.no_grad():
            logits,_=net(*geo.inputs(batch))
        strength=(logits[:,:,1]-logits[:,:,0]).numpy().astype(np.float64)
        for e,all_v in zip(batch,strength,strict=True):
            seen={v for b in e["observed"] for v in b}
            nodes=sorted(set(geo.nodes)-seen)
            v=all_v[[geo.index[u] for u in nodes]]
            tie=tie_order(e["id"],len(nodes))
            yield e,nodes,v,[nodes[j] for j in sorted_nodes(v,tie)[:12]]


def evaluate(run):
    folder=MAIN/run
    plan=load(folder/"test_plan.json")
    net=trained(plan["source"])
    raw=raw_data(plan["network"])
    graphs,_=(make_windows(raw,plan["network"],.15,1.) if run.endswith("_new_windows") else evaluation_windows(plan["network"],raw))
    events=read_rows(folder/"public.jsonl")
    truth={r["id"]:r for r in read_rows(folder/"truth.jsonl")}
    result=[]
    for g in (g for g in graphs if g["window"] in plan["windows"]):
        items=[e for e in events if e["window"]==g["window"]]
        for e,nodes,v,chosen in predictions(net,g,items):
            a,b=[nodes.index(u) for u in truth[e["id"]]["sources"]]
            rank=pair_rank(np.add.outer(v,v),a,b,tie_order(e["id"],len(nodes)))
            result.append(dict(id=e["id"],window=g["window"],chosen=chosen,rank=rank))
    assert len(result)==len(events)
    save_rows(OUT/run/"fgssi_predictions.jsonl",result)
    print(run,len(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("name",choices=RUNS)
    p.add_argument("step",choices=["fit","evaluate"])
    args=p.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        if args.step=="fit":
            assert args.name in ("company","Dept2")
            fit(args.name)
        else:
            evaluate(args.name)

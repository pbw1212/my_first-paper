"""Retrain matched additive models with one historical feature family changed."""
import argparse
from collections import Counter

import numpy as np
import torch
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
from distilled_source_pairs import CHECKPOINTS, Packed, PairNet, PublicWindow
from evaluate_factorial_history import dataset, fitted
from factorial_fgssi import OUT as PRIOR, SEEDS
from five_frame_pairs import frames
from run_dept3_candidates import Features

OUT = PRIOR.parent / "history_components"
VARIANTS = ("undirected_long", "no_recent", "no_replay")
# PublicWindow emits 29 absolute columns, 29 within-roster ranks and six context columns.
LONG = [0, 1, 2, 3, *range(6, 12)]
RECENT = [4, 5, 12, *range(19, 26)]
REPLAY = [13, 14, 15, 16, 26, 27, 28]


class Designs:
    def __init__(self, graph, raw):
        self.temporal = PublicWindow(graph, raw)
        counts = Counter()
        for u, v, count in graph["edges"]:
            counts[min(u,v), max(u,v)] += count
        edges = [(a,b,count/2) for (u,v),count in sorted(counts.items()) for a,b in ((u,v),(v,u))]
        self.undirected = Features(dict(graph, edges=edges), [])

    def event(self, event):
        o = frames(event)
        (nodes, _, _), (full, _, _) = self.temporal.design(o)
        _, roster, neutral = self.undirected.event(o)
        assert nodes == roster and full.shape[1] == 64
        undirected = full.copy()
        changes = neutral[:, LONG]
        undirected[:, LONG] = changes
        undirected[:, np.array(LONG)+29] = (rankdata(changes, axis=0)-.5)/len(nodes)
        variants = dict(full=full, undirected_long=undirected)
        for name, cols in (("no_recent", RECENT), ("no_replay", REPLAY)):
            x = full.copy()
            x[:, cols] = 0
            x[:, np.array(cols)+29] = 0
            if name == "no_replay":
                x[:, 61:63] = 0  # event mean and maximum support, both from replay
            variants[name] = x
        return nodes, variants


def collect(network, phase, raw):
    groups = list(fg.cohort(network, phase)) if phase in ("fit","cal") else []
    if phase == "test":
        _, folder, _, graphs = dataset(network)
        public = fg.read_rows(folder/"public.jsonl")
        truth = {r["id"]:r for r in fg.read_rows(folder/"truth.jsonl")}
        for g in graphs:
            events = [o for o in public if o["window"] == g["window"]]
            groups.append((g, events, [truth[o["id"]] for o in events]))
    examples = []
    for g, events, labels in groups:
        window = Designs(g, raw)
        for o, y in zip(events, labels, strict=True):
            assert o["id"] == y["id"]
            nodes, variants = window.event(o)
            examples.append((o,nodes,variants,[nodes.index(s) for s in y["sources"]]))
    return examples


def packed(examples, name, mean, std):
    rows = []
    for _,nodes,variants,(a,b) in examples:
        zeros = np.broadcast_to(np.float16(0), (len(nodes),len(nodes)))
        rows.append((variants[name],zeros,zeros,a,b,np.zeros(len(nodes),np.float32)))
    return Packed(rows, mean, std)


def fit(train, val, name, folder):
    x = np.vstack([v[name] for _,_,v,_ in train])
    mean, std = x.mean(0), np.maximum(x.std(0),.025)
    np.savez(folder/(name+".npz"), mean=mean, std=std)
    training, validation = packed(train,name,mean,std), packed(val,name,mean,std)
    for seed in SEEDS:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = PairNet(len(mean),False)
        optimizer = torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.005)
        path = folder/f"{name}-{seed}.pt"
        curve = {}
        for epoch in range(1,max(CHECKPOINTS)+1):
            net.train()
            for batch in np.array_split(rng.permutation(len(train)), int(np.ceil(len(train)/24))):
                optimizer.zero_grad()
                training.loss(net,batch,False).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),5)
                optimizer.step()
            if epoch in CHECKPOINTS:
                net.eval()
                with torch.no_grad():
                    loss = sum(len(i)*validation.loss(net,i,False).item() for i in np.array_split(np.arange(len(val)),15))/len(val)
                if not curve or loss < min(curve.values()):
                    torch.save(net.state_dict(),path)
                curve[epoch] = loss
        fg.write_json(path.with_suffix(".json"),dict(curve=curve,epoch=min(curve,key=curve.get),sha256=fg.digest(path)))
    print(folder.name, name, flush=True)


def shortlist(model, x, nodes, tie):
    net, mean, std = model
    with torch.no_grad():
        z = torch.from_numpy(np.clip((x-mean)/std,-6,6))[None]
        logits = net.node(net.encoder(z))[0,:,0].numpy()
    return [nodes[i] for i in fg.sorted_nodes(logits,tie)[:12]]


def development(network):
    folder = OUT/network
    folder.mkdir(parents=True, exist_ok=True)
    raw = fg.raw_data(network)
    train, val = [collect(network,p,raw) for p in ("fit","cal")]
    assert (len(train),len(val)) == (1000,300)
    for name in VARIANTS:
        fit(train,val,name,folder)
    del train,val
    models = {("full",seed):fitted(network,seed,"C") for seed in SEEDS}
    for name in VARIANTS:
        with np.load(folder/(name+".npz")) as scale:
            for seed in SEEDS:
                net = PairNet(64,False).eval()
                net.load_state_dict(torch.load(folder/f"{name}-{seed}.pt",weights_only=True))
                models[name,seed] = net, scale["mean"], scale["std"]
    records = []
    for event,nodes,designs,indices in collect(network,"test",raw):
        tie = fg.tie_order(event["id"],len(nodes))
        sources = [nodes[i] for i in indices]
        lists = {f"{name}/{seed}":shortlist(model,designs[name],nodes,tie) for (name,seed),model in models.items()}
        records.append(dict(id=event["id"],window=event["window"],sources=sources,lists=lists))
    assert len(records)==1200
    fg.save_rows(folder/"predictions.jsonl",records)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("network",choices=("Dept2","company"))
    torch.set_num_threads(1)
    with threadpool_limits(1):
        development(p.parse_args().network)

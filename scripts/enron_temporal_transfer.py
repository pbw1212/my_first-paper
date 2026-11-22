"""Zero-shot evaluation on calendar-locked, consecutive Enron email windows."""
import argparse
import gzip
from collections import Counter
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from build_dept3_pilot import mask, simulate
from distilled_source_pairs import PairNet, PublicWindow, draw, load_events, shortlist
from fgssi_suffix import MAIN, MATCHED, OUT, digest, load, pair_rank, predictions, read_rows, save_rows, tie_order, trained, write_json
from matched_source_baselines import cover_scores, distances
from matched_suffix_ablation import snapshot
import five_frame_pairs

ROOT=OUT.parents[1]
FOLDER=OUT/"Enron"
RAW=ROOT/"data/raw/enron-igraph.rda"
CONTACTS=FOLDER/"contacts.txt.gz"
RAW_SHA="68df616583a009c323131574c833f615443756087dda9a97e27a0ad8d658fd5a"
START=978307200
WIDTH=14*86400
PLAN=dict(source="Dept2", target="Enron (igraphdata, Priebe et al.), no parameter estimated using target labels",
    dataset="https://raw.githubusercontent.com/igraph/igraphdata/d61dabac88607cf726323dcc68109fb5753aebc7/data/enron.rda",
    clean="125409 directed records. Parse UTC strings, remove pre-1998 / self contacts / duplicate (second,sender,recipient).",
    windows="All 26 consecutive complete 14-day windows in calendar year 2001 UTC, starting January 1; no activity or outcome filter.",
    sampling="First 60 mask-eligible ordinary events per window, existing draw cap of 60000 attempts. Record even if short.",
    start=START,width=WIDTH,root_seed=202611190,source_count=2,shortlist=12,primary="both true sources in 12",
    secondary=["exact pair Top1/5/32","MRR","paired differences per window","number of direct emitters"],
    comparisons=["pair_soft","fgssi","jordan","snapshot","pair_hard","five_hard"],
    fgssi="Author demo graph encoders, three-layer bidirectional GRU and temporal attention; five public cumulative frames spanning the first to last visible batch; no absolute time or hidden-prefix membership.")


def prepare():
    import rdata
    assert digest(RAW)==RAW_SHA
    graph=rdata.read_rda(RAW)["enron"]
    assert int(graph[0][0])==184 and bool(graph[1][0])
    u,v=graph[2].astype(int),graph[3].astype(int)
    t=np.asarray(graph[8][3]["Time"]).astype("datetime64[s]").astype(np.int64)
    assert len(t)==len(u)==len(v)==125409 and max(u.max(),v.max())==183
    kept=(t>=883612800)&(u!=v)
    triples=sorted(set(zip(t[kept].tolist(),u[kept].tolist(),v[kept].tolist())))
    content="".join(f"{a} {b} {s}\n" for s,a,b in triples).encode("ascii")
    FOLDER.mkdir(parents=True,exist_ok=True)
    CONTACTS.write_bytes(gzip.compress(content,mtime=0))
    write_json(FOLDER/"provenance.json",dict(url=PLAN["dataset"],raw_sha256=RAW_SHA,contacts_sha256=digest(CONTACTS),
        original=len(t),pre1998=int((t<883612800).sum()),self_after_cut=int(((t>=883612800)&(u==v)).sum()),
        duplicates_after_filter=int(kept.sum())-len(triples),distinct=len(triples),vertices=len({w for _,a,b in triples for w in (a,b)})))


def frozen_inputs():
    return [MAIN/"Dept2"/f for f in ("pair_soft.pt","scale.npz","pair_hard.pt")]+[
        MATCHED/"Dept2"/f for f in ("jordan_selection.json","snapshot_pair.pt","snapshot_scale.npz")]+[
        OUT/"Dept2"/"fgssi.pt",OUT/"Dept2"/"selection.json",OUT/"Dept2"/"five_hard.pt",OUT/"Dept2"/"five_scale.npz",
        OUT/"Dept2"/"five_selection.json",ROOT/"scripts/five_frame_pairs.py",CONTACTS,RAW,
        ROOT/"scripts/fgssi_suffix.py",Path(__file__).resolve(),ROOT/"scripts/distilled_source_pairs.py",
        ROOT/"scripts/matched_source_baselines.py", ROOT/"scripts/build_dept3_pilot.py",ROOT/"scripts/temporal_pair_evidence.py"]


def freeze():
    config=dict(plan=PLAN,hashes={p.relative_to(ROOT).as_posix():digest(p) for p in frozen_inputs()})
    path=FOLDER/"protocol_lock.json"
    if path.exists():
        assert load(path)==config,"External protocol already frozen with different inputs"
    else:
        write_json(path,config)


def verify():
    config=load(FOLDER/"protocol_lock.json")
    assert config["plan"]==PLAN and all(digest(ROOT/f)==v for f,v in config["hashes"].items())


def windows(raw):
    for k in range(26):
        start=START+k*WIDTH
        key=f"Enron-{k+1:02}"
        history=Counter((u,v) for t,u,v in raw if t<start)
        roster={v for edge in history for v in edge}
        contacts=[(t,u,v) for t,u,v in raw if start<=t<start+WIDTH and u in roster and v in roster]
        stream=[(t,[(u,v) for _,u,v in rows]) for t,rows in groupby(contacts,key=lambda r:r[0])]
        graph=dict(window=key,start=start,end=start+WIDTH,nodes=sorted(roster),edges=[(u,v,c) for (u,v),c in sorted(history.items())])
        yield graph,stream


def generate():
    verify()
    obs=[];truth=[];flow={}
    for k,(g,stream) in enumerate(windows(load_events(CONTACTS))):
        items,labels,counts=draw(g,stream,PLAN["root_seed"]+k,0,60)
        obs.extend(items);truth.extend(labels)
        flow[g["window"]]=dict(counts,start=g["start"],end=g["end"],n_nodes=len(g["nodes"]),n_contacts=sum(len(e) for _,e in stream))
    assert len({r["id"] for r in obs})==len(obs) and len({r["seed"] for r in truth})==len(truth)
    save_rows(FOLDER/"public.jsonl",obs);save_rows(FOLDER/"truth.jsonl",truth)
    write_json(FOLDER/"generation.json",flow)


def pair_models():
    result={}
    for name in ("pair_soft","pair_hard","snapshot"):
        snap=name=="snapshot"
        folder=(MATCHED if snap else MAIN)/"Dept2"
        scale=np.load(folder/("snapshot_scale.npz" if snap else "scale.npz"))
        mean,std=scale["mean"],scale["std"]
        m=PairNet(len(mean),True).eval()
        m.load_state_dict(torch.load(folder/("snapshot_pair.pt" if snap else name+".pt"),weights_only=True))
        result[name]=m,mean,std
    return result


def evaluate():
    verify()
    raw=load_events(CONTACTS)
    public=read_rows(FOLDER/"public.jsonl")
    truth={r["id"]:r for r in read_rows(FOLDER/"truth.jsonl")}
    model=trained("Dept2")
    five=five_frame_pairs.trained("Dept2")
    nets=pair_models()
    jcfg=load(MATCHED/"Dept2"/"jordan_selection.json")
    result=[]
    for g,stream in windows(raw):
        events=[e for e in public if e["window"]==g["window"]]
        if not events:
            continue
        hist=PublicWindow(g,raw)
        index,d,penalty=distances(g,jcfg["direction"])
        for e,nodes,v,chosen in predictions(model,g,events):
            tie=tie_order(e["id"],len(nodes))
            sets={"fgssi":chosen};scores={"fgssi":np.add.outer(v,v)}
            visible=[v for batch in e["observed"] for v in batch]
            dist=d[np.ix_([index[v] for v in nodes],[index[v] for v in visible])]
            scores["jordan"]=cover_scores(dist,penalty)/jcfg["temperature"]
            five_nodes,scores["five_hard"]=five_frame_pairs.score(hist,e,five)
            assert five_nodes==nodes
            design=hist.design(e)
            flat_design=hist.design(snapshot(e))
            for name,(net,mean,std) in nets.items():
                base,(features,edge,overlap)=flat_design if name=="snapshot" else design
                assert base[0]==nodes
                with torch.no_grad():
                    scores[name]=net(torch.from_numpy(np.clip((features-mean)/std,-6,6))[None],
                        torch.tensor(edge.astype(np.float16),dtype=torch.float32)[None],torch.tensor(overlap.astype(np.float16),dtype=torch.float32)[None],
                        torch.ones(1,len(nodes),dtype=torch.bool))[0].numpy()
            for name in ("jordan","five_hard",*nets):
                sets[name]=[nodes[i] for i in shortlist(scores[name],tie)[0]]
            assert all(len(set(s))==len(s)==12 and set(s)<=set(nodes) for s in sets.values())
            y=truth[e["id"]]
            trajectory=simulate(stream,y["sources"],y["seed"])
            assert y["trajectory"]==trajectory and mask(trajectory)==(y["cut_batches"],e["observed"])
            a,b=[nodes.index(v) for v in y["sources"]]
            ranks={k:pair_rank(s,a,b,tie) for k,s in scores.items()}
            result.append(dict(id=e["id"],window=g["window"],n=len(nodes),sets=sets,ranks=ranks))
        print(g["window"],len(events),flush=True)
    assert len(result)==len(public)
    save_rows(FOLDER/"predictions.jsonl",result)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("step",choices=["prepare","freeze","generate","evaluate"])
    step=p.parse_args().step
    torch.set_num_threads(2)
    with threadpool_limits(2):
        {"prepare":prepare,"freeze":freeze,"generate":generate,"evaluate":evaluate}[step]()

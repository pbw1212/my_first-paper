"""Test the frozen Dept2 models on documented early speakers using historical reply contacts."""
from collections import Counter
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
from calendar_thread_calibration import DEST,TEST
from historical_thread_roles import OUT as ROLES,TARGET_MONTHS,read_headers
from public_thread_archive import LISTS, OUT as PYTHON
from public_thread_transfer import DAY,MONTHS,marked_thread,thread_forest
from train_source_mixtures import (ARMS,OUT as SIM,SEEDS,PublicWindow,activity,excluding_self,
                                   fitted_models,frames,law_arrays,rule_for)

OUT=SIM/"early_speakers"


def messages(name,late):
    if not late:
        return read_headers(name)
    msg={}
    conflicts=set()
    for p in (PYTHON/(name+"-headers.jsonl.gz"),DEST/(name+".gz")):
        with gzip.open(p,"rt",encoding="utf8") as f:
            for key,*rest in map(json.loads,f):
                v=tuple(rest)
                if key in msg and v!=msg[key]:
                    conflicts.add(key)
                msg[key]=v
    return {k:v for k,v in msg.items() if k not in conflicts}


def reference(name,late):
    if late or name not in LISTS:
        path=DEST/(name+"-predictions.jsonl") if late else ROLES/name/"predictions.jsonl"
        return {r["id"]:(r["sources"],r["ready"],r["lists"].get("volume",[])) for r in fg.read_rows(path)}
    gold={r["id"]:r for r in fg.read_rows(PYTHON/name/"truth.jsonl")}
    return {r["id"]:(gold[r["id"]]["sources"],gold[r["id"]]["scored"],r["lists"].get("volume",[]))
            for r in fg.read_rows(PYTHON/name/"predictions.jsonl")}


def rank(hist,e,models,w,index,means,volume):
    (nodes,_,_),(x,_,_)=hist.design(frames(e))
    tie=fg.tie_order(e["id"],len(nodes))
    def top(values):
        return [nodes[q] for q in fg.sorted_nodes(np.asarray(values),tie)[:12]]
    local=w[[index[v] for v in nodes]]
    lists={"volume":top([volume[n] for n in nodes])}
    for i,(arm,model,mu,sd) in enumerate(models):
        with torch.no_grad():
            logits=model.node(model.encoder(torch.from_numpy(np.clip((x-mu)/sd,-6,6))))[:,0].double().numpy()
        h=np.exp(logits-logits.max())
        u=h*local
        m=h*excluding_self(h)
        if rule_for(arm)=="equal":
            m=m/means[0]+(h*excluding_self(u)+u*excluding_self(h))/means[1]+u*excluding_self(u)/means[2]
        lists[f"{arm}/{SEEDS[i%len(SEEDS)]}"]=top(m)
    return lists


def run(name,late,models):
    msg=messages(name,late)
    forest,contacts=thread_forest(msg)
    old=reference(name,late)
    months=TEST if late else MONTHS if name in LISTS else TARGET_MONTHS
    rows=[]
    for y,m in months:
        key=f"{y}-{m:02d}"
        start=int(datetime(y,m,1,tzinfo=timezone.utc).timestamp())
        end=int(datetime(y+(m==12),m%12+1,1,tzinfo=timezone.utc).timestamp())
        history=[(t,u,v) for t,u,v in contacts if start-98*DAY<=t<start]
        cnt=Counter((u,v) for _,u,v in history)
        roster={v for edge in cnt for v in edge}
        volume=Counter(a for a,t,_,_ in msg.values() if start-28*DAY<=t<start)
        g=dict(start=start,window=key,nodes=sorted(roster),edges=[(*edge,k) for edge,k in sorted(cnt.items())])
        ix={v:i for i,v in enumerate(g["nodes"])}
        hist=None
        for root,thread in sorted(forest.items()):
            if not start<=msg[root][1]<end:
                continue
            _,payload=marked_thread(root,thread,msg)
            if payload is None:
                continue
            gold,batches=payload
            obs=[[v for v in grp if v in roster] for grp in batches]
            obs=[grp for grp in obs if grp]
            vis={v for grp in obs for v in grp}
            ready=len(vis)>=3 and len(roster-vis)>=12
            lists={}
            if ready:
                if hist is None:
                    hist=PublicWindow(g,history)
                    w=activity(g,history)
                    means=law_arrays(w)[3]
                lists=rank(hist,dict(id=root,window=key,observed=obs),models,w,ix,means,volume)
            assert old[root]==(gold,ready,lists.get("volume",[]))
            hit={k:int(set(gold)<=set(v)) for k,v in lists.items()}
            rows.append(dict(id=root,window=key,ready=ready,hits=hit))
    assert set(old)=={r["id"] for r in rows}
    tag=name+("-2019" if late else "")
    fg.save_rows(OUT/(tag+".jsonl"),rows)
    results=dict(eligible=len(rows),scored=sum(r["ready"] for r in rows),months=len(months),methods={})
    baseline=np.array([r["hits"].get("volume",0) for r in rows])
    results["volume_hits"]=int(baseline.sum())
    for arm in ARMS:
        hits=np.array([[r["hits"].get(f"{arm}/{s}",0) for s in SEEDS] for r in rows])
        results["methods"][arm]=dict(hits_by_seed=hits.sum(0).tolist(),both_pct=100*hits.mean(),
                                     delta_vs_volume_pp=100*(hits-baseline[:,None]).mean())
    return results


if __name__=="__main__":
    torch.set_num_threads(1)
    with threadpool_limits(1):
        OUT.mkdir(exist_ok=True)
        paths=[Path(__file__).resolve(),SIM/"protocol.json",*(SIM/"Dept2/fit").glob("*.pt"),*(SIM/"Dept2/fit").glob("*.npz")]
        spec=dict(transfer_source="Dept2",real_targets="2017-18 and 2019 Python lists; 2021-22 GCC and FFmpeg",
                  prior_input="one plus 28-day outgoing deduplicated replies, exactly as simulation contact counts",
                  baseline="28-day post volume",denominator="all mask-eligible discussions; same abstention and legal roster",
                  note="follow-up on previously evaluated real cohorts, target labels used only for scoring",
                  hashes={p.relative_to(SIM.parents[1]).as_posix():fg.digest(p) for p in paths})
        if not (OUT/"protocol.json").exists():
            fg.write_json(OUT/"protocol.json",spec)
        assert fg.load(OUT/"protocol.json")==spec
        models=fitted_models("Dept2")
        result={}
        for name,late in [(n,False) for n in (*LISTS,"gcc","ffmpeg-devel")]+[(n,True) for n in LISTS]:
            tag=name+("-2019" if late else "")
            result[tag]=run(name,late,models)
            print(tag,result[tag]["eligible"],flush=True)
        fg.write_json(OUT/"summary.json",result)

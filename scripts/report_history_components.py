"""Verify the public lists, replay masked events and aggregate paired endpoint hits."""
import gzip
import json
import random

import numpy as np

import fgssi_suffix as fg
from build_dept3_pilot import mask, simulate
from factorial_fgssi import OUT as OLD, SEEDS
from history_components import OUT, VARIANTS
from locked_dept4 import FOLDER, PLAN, RAW, config, load_events, windows


def summarize(rows, methods, contrasts, windows_list):
    totals = np.zeros(len(windows_list),int)
    counts = {name:np.zeros((len(SEEDS),len(totals)),int) for name in methods}
    one = {name:np.zeros(len(SEEDS),int) for name in methods}
    paired = {m:np.zeros((len(SEEDS),2),int) for m in contrasts}
    for r in rows:
        w = windows_list.index(r["window"])
        totals[w] += 1
        for s,seed in enumerate(SEEDS):
            hits = {}
            for name in methods:
                names = r["lists"][f"{name}/{seed}"]
                assert len(set(names))==len(names)==12
                h = len(set(r["sources"]).intersection(names))
                hits[name] = h==2
                counts[name][s,w] += h==2
                one[name][s] += h>=1
            for m in contrasts:
                a,b = m
                paired[m][s] += (hits[a] and not hits[b], hits[b] and not hits[a])
    result = dict(n=len(rows), windows=windows_list, totals=totals.tolist(), methods={}, differences={})
    for name,arr in counts.items():
        result["methods"][name] = dict(both_pct=100*arr.sum(1).mean()/len(rows),
            any_pct=100*one[name].mean()/len(rows), by_seed=arr.sum(1).tolist(), by_seed_window=arr.tolist())
    rng = np.random.default_rng(20261120)
    nw = len(windows_list)
    ids = rng.integers(nw,size=(10000,(nw+2)//3,1))
    blocks = ((ids+np.arange(3))%nw).reshape(len(ids),-1)[:,:nw]
    seeds = rng.integers(len(SEEDS),size=(len(ids),len(SEEDS),1))
    for a,b in contrasts:
        diff = counts[a]-counts[b]
        observed = 100*diff.sum(1)/len(rows)
        result["differences"][a+"-"+b] = dict(pp=observed.mean(),by_seed_pp=observed.tolist(),
            gains_losses=paired[a,b].tolist())
        if nw>=10:
            denominators = totals[blocks].sum(1)
            valid = denominators>0
            numerators = diff[seeds,blocks[:,None,:]].mean(1).sum(1)
            sims = 100*numerators[valid]/denominators[valid]
            result["differences"][a+"-"+b]["ci95_pp"]=np.quantile(sims,[.025,.975]).tolist()
            result["differences"][a+"-"+b]["bootstrap_nonempty"]=int(valid.sum())
    return result


def main():
    output = {}
    for name in ("Dept2","company"):
        rows = fg.read_rows(OUT/name/"predictions.jsonl")
        assert len(rows)==len({r["id"] for r in rows})==1200
        with gzip.open(OLD/name/"cells.jsonl.gz","rt",encoding="utf8") as archived:
            old = {r["id"]:r for r in map(json.loads,archived)}
        assert len(old)==len(rows)
        for r in rows:
            for seed in SEEDS:
                assert r["lists"][f"full/{seed}"]==old[r["id"]]["models"][f"C{seed}"]["marg12"]
        output[name] = summarize(rows,("full",*VARIANTS),[("full",v) for v in VARIANTS],sorted({r["window"] for r in rows}))
    assert config()==fg.load(FOLDER/"protocol_lock.json")
    truth = {r["id"]:r for r in fg.read_rows(FOLDER/"truth.jsonl")}
    public = {r["id"]:r for r in fg.read_rows(FOLDER/"public.jsonl")}
    rows = fg.read_rows(FOLDER/"predictions.jsonl")
    assert len(rows)==len(truth)==len(public) and {r["id"] for r in rows}==set(truth)==set(public)
    flow = fg.load(FOLDER/"flow.json")
    for i,(g,contacts) in enumerate(windows(load_events(RAW))):
        rng = random.Random(PLAN["first_seed"]+i)
        eligible = 0
        for j in range(PLAN["attempts"]):
            sid = f"{g['window']}-{j}"
            sources = sorted(rng.sample(g["nodes"],2))
            seed = rng.getrandbits(63)
            trajectory = simulate(contacts,sources,seed)
            masked = mask(trajectory)
            assert (sid in public)==(masked is not None)
            if masked is not None:
                eligible += 1
                y = truth[sid]
                assert (sources,seed,trajectory)==(y["sources"],y["seed"],y["trajectory"])
                assert masked==(y["cut_batches"],public[sid]["observed"])
        assert flow[i]["window"]==g["window"] and flow[i]["eligible"]==eligible
    for r in rows:
        r["sources"] = truth[r["id"]]["sources"]
    output["Dept4"] = summarize(rows,("A","C","F"),[("C","F"),("C","A")],[w["window"] for w in flow])
    output["Dept4"]["uniform_random_pct"] = float(np.mean([100*12*11/(r["candidates"]*(r["candidates"]-1)) for r in rows]))
    output["Dept4"]["candidates_range"] = [min(r["candidates"] for r in rows),max(r["candidates"] for r in rows)]
    output["audit"] = dict(development_archived_lists=2*1200*len(SEEDS),dept4_uniform_draws=sum(w["attempts"] for w in flow),
        files={f.relative_to(OUT).as_posix():fg.digest(f) for f in [
            OUT/"Dept2/predictions.jsonl",OUT/"company/predictions.jsonl",
            *(FOLDER/k for k in ("predictions.jsonl","public.jsonl","truth.jsonl","flow.json","protocol_lock.json"))]})
    fg.write_json(OUT/"summary.json",output)
    for name in ("Dept2","company","Dept4"):
        v=output[name]
        print(name,v["n"],{k:round(s["both_pct"],3) for k,s in v["methods"].items()},
              {k:[round(s["pp"],3),s.get("ci95_pp"),s["by_seed_pp"]] for k,s in v["differences"].items()})


if __name__=="__main__":
    main()

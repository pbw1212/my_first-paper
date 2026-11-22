"""Replay paired evaluation cohorts, validate decisions and tabulate window-level results."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_dept3_pilot import mask, simulate
from matched_source_baselines import (MAIN, ORDINARY, OUT, RUNS, cover_scores, digest, distances, evaluation_windows,
    load, make_windows, raw_data, read_rows, write_json)
from score_dept3_candidates import roles


def indexed(path):
    rows = read_rows(path)
    index = {r["id"]: r for r in rows}
    assert len(index) == len(rows)
    return index


def check_radius():
    g = dict(nodes=list(range(6)), edges=[(0,1,1), (1,2,1), (3,2,1), (4,5,1)])
    for direction in ("forward", "undirected"):
        _, dist, penalty = distances(g, direction)
        assert dist[0,5] == penalty and dist[0,2] == 2
        d = dist[np.ix_([0,1,3,4], [2,5])]
        scores = cover_scores(d, penalty)
        pairs = [(i,j) for i in range(4) for j in range(i+1,4)]
        def key(p):
            near = [min(d[p[0],k], d[p[1],k]) for k in range(2)]
            return max(near), sum(near)
        for p in pairs:
            for q in pairs:
                assert (key(p) < key(q)) == (scores[p] > scores[q])
                assert (key(p) == key(q)) == (scores[p] == scores[q])


def summarize(records):
    models = list(records[0]["hits"])
    result = {}
    for m in models:
        hits = np.array([r["hits"][m] for r in records])
        stats = dict(both=int(sum(hits==2)), any_hit=int(sum(hits>0)))
        if m in records[0]["ranks"]:
            ranks = np.array([r["ranks"][m] for r in records])
            stats.update(top1=int(sum(ranks==1)), top5=int(sum(ranks<=5)), top32=int(sum(ranks<=32)),
                         mrr=float(np.mean(1/ranks)))
        result[m] = stats
    contrasts = {}
    for m in models:
        if m == "pair_soft":
            continue
        wins = losses = 0
        for r in records:
            a, b = r["hits"]["pair_soft"]==2, r["hits"][m]==2
            wins += int(a and not b)
            losses += int(b and not a)
        contrast = dict(gained=wins, lost=losses)
        assert wins-losses == result["pair_soft"]["both"]-result[m]["both"]
        if m in records[0]["ranks"]:
            pairs = [(r["ranks"]["pair_soft"]<=32, r["ranks"][m]<=32) for r in records]
            contrast.update(top32_gained=sum(a and not b for a,b in pairs), top32_lost=sum(b and not a for a,b in pairs))
        contrasts[m] = contrast
    sizes = [r["n"] for r in records]
    return dict(n=len(records), legal_min=min(sizes), legal_max=max(sizes),
                uniform_expected_both=sum(12*11/(n*(n-1)) for n in sizes), methods=result, paired=contrasts)


def audit(run, used):
    main = MAIN/run
    out = OUT/run
    inputs = load(out/"inputs.json")
    assert all(digest(main/f)==h for f,h in inputs["main"].items())
    source = OUT/inputs["source"]
    assert all(digest(source/f)==h for f,h in inputs["config"].items())
    config = load(source/"snapshot_selection.json")
    assert all(digest(source/f)==h for f,h in config["hashes"].items())
    longer = load(source/"extended_gcnsi"/"gcnsi_selection.json")
    assert digest(source/"extended_gcnsi"/"gcnsi.pt")==longer["weights_sha256"]
    plan = load(main/"test_plan.json")
    assert all(digest(MAIN/inputs["source"]/f)==h for f,h in plan["lock"].items())
    name = plan["network"]
    raw = raw_data(name)
    graphs, streams = (make_windows(raw, name, .15, 1.) if plan.get("extra_window_check", False) else evaluation_windows(name,raw))
    graphs = {g["window"]: g for g in graphs}
    if name in ("Dept2", "company"):
        timeline = {g["window"]:g for g in make_windows(raw,name,.15,1.)[0]}
        periods = {}
        for phase in ("fit","cal"):
            keys = {r["window"] for r in read_rows(ORDINARY/name/(phase+"_public.jsonl"))}
            periods[phase] = (min(timeline[k]["start"] for k in keys), max(timeline[k]["end"] for k in keys))
        assert periods["fit"][1] <= periods["cal"][0]
        assert periods["cal"][1] <= min(graphs[k]["start"] for k in plan["windows"])
    else:
        assert inputs["source"]=="Dept2"
    public, truth, earlier = [indexed(main/f) for f in ("public.jsonl","truth.jsonl","predictions.jsonl")]
    previous_ranks = indexed(main/"likelihoods.jsonl")
    pred = indexed(out/"predictions.jsonl")
    snap = indexed(out/"snapshot_predictions.jsonl")
    gc_long = indexed(out/"extended_gcnsi_predictions.jsonl")
    assert public.keys()==truth.keys()==earlier.keys()==pred.keys()==snap.keys()==gc_long.keys()==previous_ranks.keys()
    collected = defaultdict(list)
    for k,p in pred.items():
        o,y,old = public[k],truth[k],earlier[k]
        key = p["window"]
        assert key==o["window"]==y["window"]==old["window"] and y["ordinary"]
        assert y["seed"] not in used
        used.add(y["seed"])
        trajectory = simulate(streams[key],y["sources"],y["seed"])
        assert trajectory==y["trajectory"] and mask(trajectory)==(y["cut_batches"],o["observed"])
        gold = set(y["sources"])
        legal = set(graphs[key]["nodes"])-{v for b in o["observed"] for v in b}
        assert len(gold)==2 and gold<=legal and p["n"]==len(legal)
        assert set(p["sets"]["pair_soft"])==set(old["sets"]["pair_soft"])
        choices = {**p["sets"], "snapshot":snap[k]["chosen"], "gcnsi_extended":gc_long[k]["chosen"]}
        choices.update(old["sets"])
        for members in choices.values():
            assert len(members)==len(set(members))==12 and set(members)<=legal
        ranks = {**p["ranks"], "snapshot":snap[k]["rank"], "gcnsi_extended":gc_long[k]["rank"]}
        for m,value in previous_ranks[k].get("ranks",{}).items():
            if m in ranks:
                assert ranks[m] == value
            if m in choices:
                ranks[m] = value
        assert all(1<=rank<=len(legal)*(len(legal)-1)//2 for rank in ranks.values())
        hits = {m: len(gold.intersection(v)) for m,v in choices.items()}
        for tag in ("gcnsi","gcnsi_extended"):
            if ranks[tag]==1:
                assert hits[tag]==2
        rec = dict(n=len(legal),ranks=ranks,hits=hits)
        direct,_ = roles(y)
        for tag in ("all", "window/"+key, "batch/"+str(p["batch"]), "emitters/"+str(len(direct))):
            collected[tag].append(rec)
    print(run, len(pred), flush=True)
    hashes = {f:digest(out/f) for f in ("predictions.jsonl","snapshot_predictions.jsonl","extended_gcnsi_predictions.jsonl","inputs.json")}
    return dict(groups={k:summarize(v) for k,v in collected.items()}, hashes=hashes,
                source=inputs["source"], snapshot_selection_sha256=digest(source/"snapshot_selection.json"),
                extended_selection_sha256=digest(source/"extended_gcnsi"/"gcnsi_selection.json"))


if __name__ == "__main__":
    check_radius()
    seeds = defaultdict(set)
    result = {}
    for run in RUNS:
        name = run.removesuffix("_new_windows")
        if not seeds[name]:
            source = name if name in ("company","Dept2") else "Dept2"
            seeds[name].update(r["seed"] for phase in ("fit","cal") for r in read_rows(ORDINARY/source/(phase+"_truth.jsonl")))
        result[run] = audit(run, seeds[name])
    for name in ("company","Dept2"):
        base = result[name]["groups"]
        extra = result[name+"_new_windows"]["groups"]
        assert not {k for k in base if k.startswith("window/")}.intersection(extra)
    assert [r["groups"]["all"]["n"] for r in result.values()] == [1200,1200,480,480,300,300,1050]
    write_json(OUT/"summary.json",result)
    write_json(OUT/"audit.json",dict(replayed=5010, main=3660, supplemental=1350, checks=[
        "full time-stamped trajectory and public suffix regenerated from raw contacts and archived seeds",
        "same event IDs and labels, unique twelve eligible names, ranks within full unordered candidate space",
        "comparison input and weight hashes verified, original ordered-model shortlists reproduced",
        "disjoint main and supplemental windows within network; no training or validation seed reused",
        "two-center radius and tie-breaking verified against a direct calculation on a disconnected graph"],
        upstream_errors={n:load(OUT/n/"gcnsi_selection.json")["upstream_max_errors"] for n in ("company","Dept2")},
        raw_contact_files={p.name:digest(p) for p in (OUT.parents[1]/"data/raw").iterdir()
                           if p.is_file() and "Dept3" not in p.name},
        development_inputs={n:{f:digest(ORDINARY/n/f) for phase in ("fit","cal")
                               for f in (phase+"_public.jsonl",phase+"_truth.jsonl")} for n in ("company","Dept2")},
        scripts={f:digest(Path(__file__).with_name(f)) for f in ("audit_matched_source_baselines.py","matched_source_baselines.py",
                                                                  "matched_suffix_ablation.py","matched_gcnsi_duration.py")},
        summary_sha256=digest(OUT/"summary.json")))

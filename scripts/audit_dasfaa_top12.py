"""Recover Table 3's coverage from paired archived ranks/lists and the common WP1 cohort."""
from math import comb
from statistics import mean

import numpy as np
import closure_common as cc

def index(records):
    by_id = {r["id"]: r for r in records}
    assert len(by_id) == len(records)
    return by_id


def snapshot_audit(folder, wp1, net, seeds):
    path = folder / "recent_member_evidence/single_snapshot" / net
    ids = cc.load(path / "protocol.json")["events"]
    assert len(ids) == len(set(ids)) == 100 and set(ids) <= wp1.keys()
    snapshots = index([r for file in sorted(path.glob("*.jsonl")) for r in cc.fg.read_rows(file)])
    assert snapshots.keys() == set(ids)
    truth = index(cc.fg.read_rows(folder / "distilled_source_pairs" / net / "truth.jsonl"))
    snapshot_hits = {
        method: sum(r[method] == sorted(truth[key]["sources"]) for key, r in snapshots.items())
        for method in ("snapshot", "bosoul")
    }
    archive = cc.load(path / "summary.json")
    assert all(archive[m]["n"] == len(ids) and archive[m]["top1"] == h for m, h in snapshot_hits.items())
    mlp = [sum(wp1[key]["pair_rank"][i] == 1 for key in ids) for i in range(len(seeds))]
    return dict(events=len(ids), seed_order=seeds, mlp_hits=mlp,
                mlp_seed_mean_pct=100 * mean(mlp) / len(ids), **snapshot_hits)


def main():
    results = {}
    snapshot = {}
    supplement = cc.REPO / "runs/dasfaa_baseline_completion"
    supplemental = cc.load(supplement / "summary.json")
    assert supplemental["seed_order"] == list(cc.SEEDS)
    for net in cc.NAMES:
        wp_path = cc.BASE / "wp1_budgets" / (net+".jsonl.gz")
        wp = index(cc.read_rows(wp_path))
        keys = list(wp)
        sizes = [wp[k]["n"] for k in keys]
        scores = {}
        def collect(name, records, func):
            assert records.keys() == wp.keys()
            values = np.array([func(records[k]) for k in keys])
            assert np.all((values >= 0) & (values <= 2))
            scores[name] = dict(both=float(100*np.mean(values == 2)), any=float(100*np.mean(values > 0)),
                                recall=float(50*np.mean(values)))
            return values
        def hits(ranks):
            return (np.asarray(ranks) <= 12).sum(1)
        mlp = collect("MLP", wp, lambda r: hits(r["ranks"]))
        random_both = mean(comb(12,2)/comb(n,2) for n in sizes)*100
        random_any = mean(1-comb(n-12,2)/comb(n,2) for n in sizes)*100
        scores["Random"] = dict(both=random_both, any=random_any, recall=(random_any+random_both)/2)
        folder = cc.ARCH / "runs"
        factor = folder / "factorial_history" / net
        if (factor / "cells.jsonl.gz").exists():
            collect("Static", index(cc.read_rows(factor / "cells.jsonl.gz")),
                    lambda r:[v["m"][1] for k,v in r["models"].items() if k.startswith("A")])
        if net == "Dept4":
            hist = folder/"history_components/Dept4"
            truth = index(cc.fg.read_rows(hist/"truth.jsonl"))
            collect("Static", index(cc.fg.read_rows(hist/"predictions.jsonl")),
                    lambda r:[len(set(v).intersection(truth[r["id"]]["sources"])) for k,v in r["lists"].items() if k.startswith("A/")])
        tree_file = cc.BASE / "wp3_robustness" / ("tree-"+net+".jsonl.gz")
        if tree_file.exists():
            data = index([r for r in cc.read_rows(tree_file) if r["budget"]=="12"])
            assert all(data[k]["n"] == wp[k]["n"] and data[k]["ranks"]["MLP"] == wp[k]["ranks"] for k in keys)
            for key,col in [("FGSSI","FGSSI"),("full","Tree")]:
                collect(col,data,lambda r:hits(r["ranks"][key]))
        else:
            collect("FGSSI",index(cc.read_rows(factor / "fgssi.jsonl.gz")),
                    lambda r:[v["m"][1] for v in r["models"].values()])
        if net in supplemental["results"]:
            row = supplemental["results"][net]
            source = supplement / wp_path.name
            assert row["n"] == len(wp) and row["cohort_sha256"] == cc.sha(wp_path)
            assert row["ranks_sha256"] == cc.sha(source)
            data = index(cc.read_rows(source))
            assert all(data[k]["n"] == wp[k]["n"] and data[k]["window"] == wp[k]["window"] for k in keys)
            method = row["method"]
            assert method not in scores
            collect(method, data, lambda r: hits(r["ranks"]))
            assert all(np.isclose(value, row[k]) for k, value in scores[method].items())
        if net in ("Dept2", "company"):
            snapshot[net] = snapshot_audit(folder, wp, net, list(cc.SEEDS))
        fresh = cc.REPO / "runs/dasfaa_gcnsi_transfer" / (net+".jsonl.gz")
        if fresh.exists():
            gr = index(cc.read_rows(fresh))
            if net == "GNU libc":
                truth = index([r for r in cc.fg.read_rows(folder / "heldout_libc_alpha/events.jsonl") if r["profile"]=="flat"])
            else:
                gold = (folder/"history_components/Dept4" if net=="Dept4" else folder/"temporal_transfer"/net)
                truth = index(cc.fg.read_rows(gold/"truth.jsonl"))
        else:
            gr = index(cc.fg.read_rows(folder/"matched_source_baselines"/net/"extended_gcnsi_predictions.jsonl"))
            truth = index(cc.fg.read_rows(folder/"distilled_source_pairs"/net/"truth.jsonl"))
        assert truth.keys()==wp.keys()
        gc = collect("GCNSI", gr, lambda r:[len(set(r["chosen"]).intersection(truth[r["id"]]["sources"]))])
        rec = dict(n=len(keys), pool=mean(sizes), coverage_pct=scores)
        if net=="GNU libc":
            windows=[wp[k]["window"] for k in keys]
            win_keys=cc.load(cc.BASE/"wp1_budgets/summary.json")[net]["keys"]
            delta=cc.interval((mlp==2).astype(float)-(gc==2),windows,win_keys)
            rec["MLP_minus_GCNSI_pp"]=[delta["mean"]*100,*[v*100 for v in delta["ci95"]]]
        results[net]=rec
    cc.dump(cc.REPO/"paper_dasfaa/evidence/top12.json",results)
    cc.dump(cc.REPO/"paper_dasfaa/evidence/snapshot_top1.json",snapshot)

if __name__ == "__main__":
    main()

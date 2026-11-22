"""Compare frozen shortlist predictions to historical activity on exactly the same events."""
import gzip
import json

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
import locked_dept4 as target
from evaluate_factorial_history import dataset
from factorial_fgssi import OUT as FACTORIAL, SEEDS
from ordinary_source_hybrid import OUT as ACTIVITY, Window, activity
from report_history_components import summarize

OUT = FACTORIAL.parent/"activity_control"


def run(name):
    if name=="Dept4":
        source,folder,raw = "Dept2",target.FOLDER,target.load_events(target.RAW)
        graphs = [g for g,_ in target.windows(raw)]
        assert target.config()==fg.load(folder/"protocol_lock.json")
        old = {r["id"]:{s:r["lists"][f"C/{s}"] for s in SEEDS} for r in fg.read_rows(folder/"predictions.jsonl")}
    else:
        source,folder,raw,graphs = dataset(name)
        with gzip.open(FACTORIAL/name/"cells.jsonl.gz","rt",encoding="utf8") as stream:
            old = {r["id"]:{s:r["models"][f"C{s}"]["marg12"] for s in SEEDS} for r in map(json.loads,stream)}
    public = fg.read_rows(folder/"public.jsonl")
    truth = {r["id"]:r["sources"] for r in fg.read_rows(folder/"truth.jsonl")}
    assert len(public)==len(old)==len(truth)
    control = ACTIVITY/source/"activity.joblib"
    assert fg.digest(control)==fg.load(control.parent/"hybrid_selection.json")["hashes"][control.name]
    model = joblib.load(control)
    rows = []
    for g in graphs:
        events = [e for e in public if e["window"]==g["window"]]
        if not events:
            continue
        hist = Window(g,raw)
        ix = hist.base.f.history.index
        x = np.column_stack((hist.base.f.static,hist.quiet))
        check = hist.features(events[0])
        assert np.array_equal(x[[ix[n] for n in check[0][0]]],activity(check))
        score = model.decision_function(x)
        for e in events:
            hidden = set(g["nodes"]).difference(n for batch in e["observed"] for n in batch)
            nodes = sorted(hidden)
            indices = [ix[n] for n in nodes]
            tie = fg.tie_order(e["id"],len(nodes))
            top = {key:[nodes[i] for i in fg.sorted_nodes(value[indices],tie)[:12]] for key,value in
                   (("activity",score),("sent28",x[:,4]))}
            lists = {f"{key}/{s}":lst for key,lst in top.items() for s in SEEDS}
            lists.update({f"full/{s}":lst for s,lst in old[e["id"]].items()})
            assert all(set(lst)<=hidden for lst in lists.values())
            rows.append(dict(id=e["id"],window=g["window"],sources=truth[e["id"]],lists=lists))
    assert len(rows)==len(public)==len({r["id"] for r in rows})
    result = summarize(rows,("full","activity","sent28"),[("full","activity"),("full","sent28")],[g["window"] for g in graphs])
    result["input_hashes"] = {p.name:fg.digest(p) for p in (control,folder/"public.jsonl",folder/"truth.jsonl")}
    fg.save_rows(OUT/(name+".jsonl"),[dict(id=r["id"],activity=r["lists"][f"activity/{SEEDS[0]}"],
                                          sent28=r["lists"][f"sent28/{SEEDS[0]}"]) for r in rows])
    return result


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(1):
        results = {name:run(name) for name in ("Dept2","company","Enron","Dept4")}
    fg.write_json(OUT/"summary.json",results)
    for name,r in results.items():
        print(name,r["n"],{k:round(v["both_pct"],2) for k,v in r["methods"].items()},
              r["differences"]["full-activity"].get("ci95_pp"))

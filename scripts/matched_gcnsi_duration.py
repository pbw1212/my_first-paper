"""Validation-only extension of the GCNSI training horizon; keep the first comparison intact."""
import argparse

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from matched_source_baselines import (MAIN, OUT, RUNS, adjacency, architecture, digest, encode, event_nodes,
    evaluation_windows, forward, gcn_fit, load, make_windows, oriented, pair_rank, raw_data, read_rows,
    save_rows, sorted_nodes, tie_order)


def evaluate(run):
    main = MAIN/run
    plan = load(main/"test_plan.json")
    trained = OUT/plan["source"]/"extended_gcnsi"
    selected = load(trained/"gcnsi_selection.json")
    original = load(OUT/plan["source"]/"gcnsi_selection.json")
    assert all(abs(rec["nll"]-selected["validation"][tag]["nll"]) < 1e-6
               for tag,rec in original["validation"].items())
    assert digest(trained/"gcnsi.pt")==selected["weights_sha256"]
    model = architecture().eval()
    model.load_state_dict(torch.load(trained/"gcnsi.pt",weights_only=True))
    raw = raw_data(plan["network"])
    graphs,_ = (make_windows(raw,plan["network"],.15,1.) if plan.get("extra_window_check",False)
                else evaluation_windows(plan["network"],raw))
    public = read_rows(main/"public.jsonl")
    truth = {r["id"]:r for r in read_rows(main/"truth.jsonl")}
    output = []
    for graph in graphs:
        if graph["window"] not in plan["windows"]:
            continue
        events = [r for r in public if r["window"]==graph["window"]]
        x,e,ix = encode(oriented(graph,selected["selected"]["direction"]),events)
        adj = adjacency(e,len(ix))
        with torch.no_grad():
            logits = torch.cat([forward(model,x[i:i+8],adj) for i in range(0,len(x),8)])
        strengths = (logits[:,:,1]-logits[:,:,0]).numpy().astype(np.float64)
        for event,v in zip(events,strengths,strict=True):
            nodes,_ = event_nodes(graph,event)
            gc = v[[ix[u] for u in nodes]]
            tie = tie_order(event["id"],len(nodes))
            chosen = [nodes[i] for i in sorted_nodes(gc,tie)[:12]]
            a,b = [nodes.index(u) for u in truth[event["id"]]["sources"]]
            output.append(dict(id=event["id"],chosen=chosen,rank=pair_rank(np.add.outer(gc,gc),a,b,tie)))
    assert len(output)==len(public)
    save_rows(OUT/run/"extended_gcnsi_predictions.jsonl",output)
    print(run,len(output),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("name",choices=RUNS)
    parser.add_argument("phase",choices=("fit","evaluate"))
    args=parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.phase=="fit":
            assert args.name in ("company","Dept2")
            gcn_fit(args.name,folder=OUT/args.name/"extended_gcnsi",epochs=(10,25,50,100,150))
        else:
            evaluate(args.name)

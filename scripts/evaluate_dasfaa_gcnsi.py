"""Score frozen GCNSI on the unchanged WP1 cohorts, including the three later transfers."""

import torch
from threadpoolctl import threadpool_limits

import closure_common as cc
import matched_source_baselines as gc

OUT = cc.REPO / "runs" / "dasfaa_gcnsi_transfer"


@torch.inference_mode()
def score(name, rows, windows, trained):
    selected = cc.load(trained / "gcnsi_selection.json")
    assert cc.sha(trained / "gcnsi.pt") == selected["weights_sha256"]
    model = gc.architecture().eval()
    model.load_state_dict(torch.load(trained / "gcnsi.pt", weights_only=True))
    found = {}
    for graph, _, items, _ in windows:
        if not items:
            continue
        events = [event for event, _ in items]
        x, edge, index = gc.encode(gc.oriented(graph, selected["selected"]["direction"]), events)
        adj = gc.adjacency(edge, len(index))
        for i in range(0, len(x), 8):
            logits = gc.forward(model, x[i:i+8], adj)
            values = (logits[:, :, 1] - logits[:, :, 0]).double().numpy()
            for (event, gold), scores in zip(items[i:i+8], values, strict=True):
                nodes, _ = gc.event_nodes(graph, event)
                key = event["id"]
                assert len(nodes) == rows[key]["n"] and key not in found
                tie = gc.tie_order(key, len(nodes))
                order = gc.sorted_nodes(scores[[index[n] for n in nodes]], tie)
                chosen = [nodes[j] for j in order[:12]]
                assert len(set(chosen)) == 12 and set(gold) <= set(nodes)
                found[key] = dict(id=key, window=graph["window"], n=len(nodes), chosen=chosen,
                                  hits=len(set(gold).intersection(chosen)))
    assert found.keys() == rows.keys()
    return found, selected


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    summary = {}
    for name in cc.NAMES:
        rows = {r["id"]: r for r in cc.read_rows(cc.BASE / "wp1_budgets" / (name + ".jsonl.gz"))}
        folder = cc.ARCH / "runs/matched_source_baselines"
        trained = folder / cc.source(name) / "extended_gcnsi"
        original = folder / name / "extended_gcnsi_predictions.jsonl"
        if original.exists():
            old = {r["id"]: r["chosen"] for r in gc.read_rows(original)}
            if name == "Dept2":
                check, _ = score(name, rows, cc.cohort(name), trained)
                assert {k: v["chosen"] for k, v in check.items()} == old
            truth = {r["id"]: r["sources"] for r in gc.read_rows(cc.ARCH / "runs/distilled_source_pairs" / name / "truth.jsonl")}
            assert old.keys() == truth.keys() == rows.keys()
            hits = [len(set(v).intersection(truth[k])) for k,v in old.items()]
            provenance = "archived"
        else:
            found, _ = score(name, rows, cc.cohort(name), trained)
            cc.save_rows(OUT / (name + ".jsonl.gz"), list(found.values()))
            hits = [r["hits"] for r in found.values()]
            provenance = "frozen transfer"
        selected = cc.load(trained / "gcnsi_selection.json")
        summary[name] = dict(n=len(rows), both=sum(h == 2 for h in hits), any=sum(h >= 1 for h in hits),
                             model_sha256=selected["weights_sha256"], source=cc.source(name),
                             direction=selected["selected"]["direction"], epoch=selected["selected"]["epoch"],
                             seed=cc.load(trained / "protocol.json")["seed"], provenance=provenance)
        print(name, summary[name], flush=True)
    cc.dump(OUT / "summary.json", summary)


if __name__ == "__main__":
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()

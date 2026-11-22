"""Produce the independently specified CollegeMsg cohort and fit its forward model."""

import hashlib
import json
import random
from collections import Counter

from build_dept3_pilot import DAY, ROOT, make_windows, mask, read_rows, save_rows, simulate
from dept3_forward_model import fit_forward, prefix_graph, transition_rows
from probe_cascade_yield import load_events
from run_dept3_candidates import write_json
from threadpoolctl import threadpool_limits


OUT = ROOT / "runs" / "collegemsg_replication"
SOURCE = ROOT / "data" / "raw" / "CollegeMsg.txt.gz"
CFG = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))


def prepare():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != CFG["raw_sha256"]:
        raise ValueError("Unexpected CollegeMsg archive")
    events = load_events(SOURCE)
    all_graphs, all_counts, stats = [], {}, {}
    for split in ("train", "validation"):
        graphs, streams = make_windows(events, split, *CFG["time_splits"][split], width=CFG["window_days"]*DAY)
        all_graphs.extend(graphs)
        rng = random.Random(CFG[split+"_seed"])
        inputs, labels, attempts = [], [], []
        counts = {g["window"]: Counter() for g in graphs}
        for _ in range(CFG["attempt_limit_per_window"]):
            order = graphs.copy()
            rng.shuffle(order)
            for g in order:
                key = g["window"]
                sources = sorted(rng.sample(g["nodes"], 2))
                seed = rng.getrandbits(63)
                tr = simulate(streams[key], sources, seed)
                masked = mask(tr)
                c = counts[key]
                c["attempts"] += 1
                c["zero_growth"] += not tr
                sid = None
                if masked:
                    cut, visible = masked
                    sid = f"college-{split}-{len(inputs)+1:04}"
                    inputs.append({"id": sid, "window": key, "k": 2, "observed": visible})
                    labels.append({"id": sid, "sources": sources, "cut_batches": cut, "trajectory": tr})
                    c["accepted"] += 1
                attempts.append({"window": key, "sources": sources, "draw_seed": seed, "id": sid,
                                 "new_members": sum(len(b["joins"]) for b in tr)})
                if len(inputs) == CFG[split+"_target"]:
                    break
            if len(inputs) == CFG[split+"_target"]:
                break
        save_rows(OUT / "public" / f"{split}.jsonl", inputs)
        save_rows(OUT / "truth" / f"{split}.jsonl", labels)
        save_rows(OUT / "truth" / f"{split}_attempts.jsonl", attempts)
        for g in graphs:
            key = g["window"]
            counts[key]["known_nodes"] = len(g["nodes"])
            counts[key]["contacts"] = sum(len(b) for _, b in streams[key])
        all_counts.update(counts)
        print(split, len(inputs), "accepted out of", len(attempts), "attempts", flush=True)
        if split == "train":
            saved = {r["id"]:r for r in labels}
            for g in graphs:
                key = g["window"]
                hist = prefix_graph(g, events)
                s = {"num": [], "den": [], "lengths": [], "attempts": counts[key]["attempts"], "zero": counts[key]["zero_growth"]}
                for a in attempts:
                    if a["window"] != key or a["new_members"] == 0:
                        continue
                    if a["id"] is None:
                        tr = simulate(streams[key], a["sources"], a["draw_seed"])
                    else:
                        lab = saved[a["id"]]
                        tr, cut = lab["trajectory"], lab["cut_batches"]
                        h = sum(len(b["joins"]) for b in tr[:cut])
                        s["lengths"].append((a["new_members"]-h, h))
                    for num, den, _ in transition_rows(hist, a["sources"], tr):
                        s["num"].extend(num)
                        s["den"].extend([den]*len(num))
                stats[key] = s
    save_rows(OUT / "public/graphs.jsonl", all_graphs)
    write_json(OUT / "generation.json", all_counts)
    fits = {str(c): fit_forward(stats,c) for c in (6,8,10,12)}
    write_json(OUT / "forward_models.json", fits)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        prepare()

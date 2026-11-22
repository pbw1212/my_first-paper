"""Generate CollegeMsg validation events conditioned on both simulated lineages reaching the suffix."""

import random
from collections import Counter

from build_dept3_pilot import DAY, ROOT, make_windows, mask, save_rows, simulate
from collegemsg_data import CFG, SOURCE
from probe_cascade_yield import load_events
from run_dept3_candidates import write_json
from score_dept3_candidates import roles


OUT = ROOT / "runs" / "visible_dual_source"
TARGET = 40
LIMIT = 250000


def generate():
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "protocol.json", {
        "definition": "Each of the two sampled sources has at least one descendant in the observed suffix; shared descendants count.",
        "sampling": "Existing five CollegeMsg validation weeks; uniform historical roster; original growth and masking; fresh draws.",
        "target_per_week": TARGET, "attempt_cap_per_week": LIMIT, "seed_per_week": "31117 + 1-based window number",
        "models": "Frozen CollegeMsg replication fits through training window 12; no source injection; full historical candidate roster.",
        "primary": "Fraction of all retained events with the true pair in the first 32 pairs, prefix128 versus direct, among a fixed 32-node proposal.",
        "secondary": ["node coverage", "top1", "per-window paired gains and losses", "32-particle sensitivity"]
    })
    raw = load_events(SOURCE)
    graphs, streams = make_windows(raw, "validation", *CFG["time_splits"]["validation"], width=7*DAY)
    inputs, labels, flow = [], [], {}
    for week, g in enumerate(graphs, 1):
        key = g["window"]
        rng = random.Random(31117 + week)
        stats = Counter()
        for attempt in range(1, LIMIT+1):
            sources = sorted(rng.sample(g["nodes"], 2))
            seed = rng.getrandbits(63)
            tr = simulate(streams[key], sources, seed)
            stats["attempts"] += 1
            stats["zero_growth"] += not tr
            masked = mask(tr)
            if not masked:
                continue
            stats["mask_eligible"] += 1
            cut, obs = masked
            lab = {"sources": sources, "cut_batches": cut, "trajectory": tr}
            direct, lineage = roles(lab)
            stats["both_direct"] += len(direct)==2
            if len(lineage) != 2:
                continue
            stats["both_visible"] += 1
            sid = f"evidence-{week:02}-{stats['both_visible']:03}"
            inputs.append({"id":sid,"window":key,"k":2,"observed":obs})
            labels.append(dict(lab,id=sid,window=key,draw_seed=seed,attempt=attempt))
            if stats["both_visible"] == TARGET:
                break
        flow[key] = dict(stats)
        print(key,dict(stats),flush=True)
    save_rows(OUT / "observations.jsonl", inputs)
    save_rows(OUT / "truth.jsonl", labels)
    write_json(OUT / "generation.json", flow)


if __name__ == "__main__":
    generate()

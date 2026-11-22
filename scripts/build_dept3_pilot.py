"""Build the Dept3 pilot. Only the public directory is needed for inference."""

import hashlib
import json
import random
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

from probe_cascade_yield import DAY, RAW, load_events


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "dept3_pilot"
SOURCE = RAW / "email-Eu-core-temporal-Dept3.txt.gz"
RAW_SHA = "b1f92288e855044bbc74bacac39d42b12e59ec2a0130ad8df419786eb46a7bff"
WIDTH = 14 * DAY
P = 0.3
TARGET = 100
MAX_ROUNDS = 1000
SPLITS = {"train": (0.15, 0.60, 11701), "validation": (0.60, 0.80, 11702)}


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def make_windows(events, split, left, right, width=WIDTH):
    first, last = events[0][0], events[-1][0]
    start = first + int((last - first) * left)
    limit = first + int((last - first) * right)
    graphs, streams = [], {}
    for t0 in range(start, limit - width + 1, width):
        counts = Counter((u, v) for t, u, v in events if t < t0)
        roster = {v for edge in counts for v in edge}
        key = f"{split}-{len(graphs) + 1:02}"
        graphs.append({"window": key, "start": t0, "end": t0 + width,
                       "nodes": sorted(roster),
                       "edges": [[u, v, w] for (u, v), w in sorted(counts.items())]})
        contacts = [(t, u, v) for t, u, v in events
                    if t0 <= t < t0 + width and u in roster and v in roster]
        streams[key] = [(t, [(u, v) for _, u, v in batch])
                        for t, batch in groupby(contacts, key=lambda event: event[0])]
    return graphs, streams


def simulate(stream, sources, draw_seed):
    rng = random.Random(draw_seed)
    active = set(sources)
    trajectory = []
    for t, contacts in stream:
        parents = defaultdict(list)
        for u, v in contacts:
            if u in active and v not in active and rng.random() < P:
                parents[v].append(u)
        if parents:
            trajectory.append({"t": t, "joins": [[v, sorted(parents[v])] for v in sorted(parents)]})
            active.update(parents)
    return trajectory


def mask(trajectory):
    groups = [[v for v, _ in batch["joins"]] for batch in trajectory]
    n = sum(map(len, groups))
    hidden_count = cut = 0
    for batch in groups:
        if hidden_count + len(batch) > n // 2:
            break
        hidden_count += len(batch)
        cut += 1
    if cut == 0 or n - hidden_count < 3:
        return None
    return cut, groups[cut:]


def build_split(events, name, left, right, seed, target=TARGET):
    graphs, streams = make_windows(events, name, left, right)
    rng = random.Random(seed)
    attempts, inputs, labels = [], [], []
    for _ in range(MAX_ROUNDS):
        order = graphs.copy()
        rng.shuffle(order)
        for graph in order:
            key = graph["window"]
            sources = sorted(rng.sample(graph["nodes"], 2))
            draw_seed = rng.getrandbits(63)
            trajectory = simulate(streams[key], sources, draw_seed)
            masked = mask(trajectory)
            sample_id = None
            if masked is not None:
                cut, observed = masked
                sample_id = f"{name}-{len(inputs) + 1:03}"
                inputs.append({"id": sample_id, "window": key, "k": 2, "observed": observed})
                labels.append({"id": sample_id, "sources": sources, "cut_batches": cut,
                               "draw_seed": draw_seed, "trajectory": trajectory})
            attempts.append({"window": key, "sources": sources, "draw_seed": draw_seed,
                             "new_members": sum(len(b["joins"]) for b in trajectory),
                             "accepted_id": sample_id})
            if len(inputs) == target:
                return graphs, inputs, labels, attempts
    raise RuntimeError(f"Only {len(inputs)} usable samples in {name}")


def main():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != RAW_SHA:
        raise ValueError("Raw input digest differs from the historical SNAP archive.")
    events = load_events(SOURCE)
    all_graphs, all_attempts = [], []
    for name, bounds in SPLITS.items():
        graphs, inputs, labels, attempts = build_split(events, name, *bounds)
        save_rows(OUT / "public" / f"{name}.jsonl", inputs)
        save_rows(OUT / "truth" / f"{name}.jsonl", labels)
        all_graphs.extend(graphs)
        all_attempts.extend(attempts)
        print(name, "windows", len(graphs), "attempts", len(attempts), "accepted", len(inputs))
    save_rows(OUT / "public" / "graphs.jsonl", all_graphs)
    save_rows(OUT / "truth" / "attempts.jsonl", all_attempts)


if __name__ == "__main__":
    main()

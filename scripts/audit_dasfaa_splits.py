"""Recompute the manuscript's split, cleaning, and baseline checks from the original archive."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (111, 314159, 431, 2027, 9001)


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf8") as stream:
        return [json.loads(line) for line in stream]


def index(records):
    result = {row["id"]: row for row in records}
    assert len(result) == len(records)
    return result


def digest(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf8")).hexdigest()


def paired(cells, keys, first, second):
    totals = Counter(row["window"] for row in cells)
    assert set(totals) == set(keys)
    scores = np.zeros((len(keys), len(SEEDS)))
    for row in cells:
        models = row["models"]
        for j, seed in enumerate(SEEDS):
            hit_a = models[first + str(seed)]["m"][1] == 2
            hit_b = models[second + str(seed)]["m"][1] == 2
            scores[keys.index(row["window"]), j] += int(hit_a) - int(hit_b)
    # Match closure_common.interval, including the order of bootstrap draws.
    rng = np.random.default_rng(20261120)
    n = len(keys)
    starts = rng.integers(n, size=(10000, (n + 2) // 3, 1))
    blocks = ((starts + np.arange(3)) % n).reshape(10000, -1)[:, :n]
    seeds = rng.integers(len(SEEDS), size=(10000, len(SEEDS), 1))
    den = np.array([totals[k] for k in keys])[blocks].sum(1)
    samples = 100 * scores[blocks[:, None, :], seeds].mean(1).sum(1) / den
    return dict(pp=100 * scores.sum() / (len(SEEDS) * len(cells)),
                ci95_pp=np.quantile(samples, [.025, .975]).tolist())


def audit(archive):
    run = archive / "runs"
    report = dict(cleaning={}, splits={}, gcnsi={}, contrasts={})
    for net in ("Dept1", "Dept2", "Dept4"):
        with gzip.open(archive / "data/raw" / f"email-Eu-core-temporal-{net}.txt.gz", "rt") as f:
            contacts = [tuple(map(int, line.split())) for line in f if line.strip()]
        unique = set(contacts)
        self_loops = sum(u == v for u, v, _ in unique)
        report["cleaning"][net] = dict(raw=len(contacts), duplicates=len(contacts)-len(unique),
                                       distinct_self_loops=self_loops, kept=len(unique)-self_loops)
    for net in ("Dept2", "company", "CollegeMsg"):
        test_folder = run / "distilled_source_pairs" / net
        test = index(rows(test_folder / "public.jsonl"))
        truth = index(rows(test_folder / "truth.jsonl"))
        wp1 = index(rows(ROOT / "runs/paper_closure/wp1_budgets" / f"{net}.jsonl.gz"))
        folder = run / "matched_source_baselines" / net
        predictions = index(rows(folder / "extended_gcnsi_predictions.jsonl"))
        assert test.keys() == truth.keys() == predictions.keys() == wp1.keys()
        correct = 0
        for key, pred in predictions.items():
            chosen = set(pred["chosen"])
            assert len(chosen) == len(pred["chosen"]) == 12
            correct += set(truth[key]["sources"]) <= chosen
        source = json.loads((folder / "inputs.json").read_text("utf8"))["source"]
        model_dir = run / "matched_source_baselines" / source / "extended_gcnsi"
        protocol = json.loads((model_dir / "protocol.json").read_text("utf8"))
        report["gcnsi"][net] = dict(n=len(test), hits=correct, fit_network=source, seed=protocol["seed"],
                                     test_ids_sha256=digest(test))
        if net == "CollegeMsg":
            continue
        splits = {}
        for phase in ("fit", "cal"):
            source = run / "ordinary_source_roles" / net
            public = index(rows(source / (phase + "_public.jsonl")))
            assert public.keys() == index(rows(source / (phase + "_truth.jsonl"))).keys()
            splits[phase] = public
        splits["test"] = test
        seen_ids, seen_windows = set(), set()
        for phase, records in splits.items():
            windows = Counter(row["window"] for row in records.values())
            assert seen_ids.isdisjoint(records) and seen_windows.isdisjoint(windows)
            if seen_windows:
                assert max(seen_windows) < min(windows)
            seen_ids.update(records)
            seen_windows.update(windows)
            report["splits"].setdefault(net, {})[phase] = dict(n=len(records), windows=windows,
                                                                ids_sha256=digest(records))
        cells = rows(run / "factorial_history" / net / "cells.jsonl.gz")
        assert index(cells).keys() == test.keys()
        keys = sorted({r["window"] for r in test.values()})
        report["contrasts"][net] = {"pair_minus_additive": paired(cells, keys, "D", "C"),
                                    "additive_minus_static": paired(cells, keys, "C", "A")}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Directory containing data/raw and original runs")
    print(json.dumps(audit(parser.parse_args().archive), indent=2))

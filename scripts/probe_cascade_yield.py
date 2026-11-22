"""Measure how often training-period contacts yield a usable masked cascade."""

import gzip
import random
from itertools import groupby
from pathlib import Path
from statistics import median


RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = ("email-Eu-core-temporal-Dept3.txt.gz", "CollegeMsg.txt.gz")
TRIALS = 240
DAY = 86400


def load_events(path):
    with gzip.open(path, "rt", encoding="ascii") as stream:
        triples = {tuple(map(int, line.split())) for line in stream if line.strip()}
    return sorted((t, u, v) for u, v, t in triples if u != v)


def training_windows(events, days):
    first, last = events[0][0], events[-1][0]
    start = first + int((last - first) * 0.15)
    limit = first + int((last - first) * 0.60)
    width = days * DAY
    windows = []
    while start + width <= limit:
        roster = {v for t, u, w in events if t < start for v in (u, w)}
        contacts = [(t, u, v) for t, u, v in events
                    if start <= t < start + width and u in roster and v in roster]
        batches = [[(u, v) for _, u, v in group]
                   for _, group in groupby(contacts, key=lambda event: event[0])]
        if len(roster) >= 2:
            windows.append((sorted(roster), batches))
        start += width
    return windows


def replay(batches, sources, probability, rng):
    joined = set(sources)
    sizes = []
    for batch in batches:
        entering = set()
        for sender, receiver in batch:
            if sender in joined and receiver not in joined and rng.random() < probability:
                entering.add(receiver)
        if entering:
            sizes.append(len(entering))
            joined.update(entering)
    return sizes


def eligible(sizes, ratio):
    total = sum(sizes)
    target = int(ratio * total)
    hidden = 0
    for size in sizes:
        if hidden + size > target:
            break
        hidden += size
    return hidden > 0 and total - hidden >= 3


def main():
    print("dataset days p windows trials grown>=1 eligible50 eligible_all median_new")
    for path in FILES:
        events = load_events(RAW / path)
        for days in (7, 14):
            windows = training_windows(events, days)
            for probability in (0.1, 0.3, 0.5):
                if not windows:
                    raise ValueError(f"No training windows for {path}, duration {days}")
                # Pair the window and the two seeds across probability settings.
                design = random.Random(20261117)
                grown = at_half = at_all = 0
                counts = []
                for trial in range(TRIALS):
                    roster, batches = design.choice(windows)
                    sources = design.sample(roster, 2)
                    rng = random.Random(73000 + trial)
                    sizes = replay(batches, sources, probability, rng)
                    n = sum(sizes)
                    counts.append(n)
                    grown += n > 0
                    at_half += eligible(sizes, 0.5)
                    at_all += all(eligible(sizes, r) for r in (0.25, 0.5, 0.75))
                print(path, days, probability, len(windows), TRIALS,
                      grown, at_half, at_all, median(counts))


if __name__ == "__main__":
    main()

"""Repeat the frozen FGSSI training recipe under explicit, isolated seeds."""
import argparse
from unittest.mock import patch

import torch
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg

OUT = fg.MAIN.parent/"factorial_history"
SEEDS = (111, 314159, 431, 2027, 9001)


def weights(name, seed):
    parent = fg.OUT if seed == fg.SEED else OUT/"fgssi"/str(seed)
    folder = parent/name
    cfg = fg.load(folder/"selection.json")
    assert cfg["seed"] == seed and fg.digest(folder/"fgssi.pt") == cfg["weights"]
    return folder/"fgssi.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=("company","Dept2"))
    parser.add_argument("seeds", type=int, nargs="+", choices=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        for seed in args.seeds:
            folder = OUT/"fgssi"/str(seed)
            if seed != fg.SEED and not (folder/args.name/"selection.json").exists():
                with patch.object(fg,"OUT",folder), patch.object(fg,"SEED",seed):
                    fg.fit(args.name)
            weights(args.name,seed)

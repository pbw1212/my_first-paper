"""Fit the fixed Dept3 node and pair architectures using CollegeMsg history."""

import json
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from threadpoolctl import threadpool_limits

from build_dept3_pilot import read_rows, save_rows
from collegemsg_data import OUT, SOURCE
from evaluate_dept3_pilot import UNREACHABLE
from probe_cascade_yield import load_events
from run_dept3_candidates import FEATURES, Features, fit, rank_pool, write_json
from run_dept3_pair_ranker import pair_event, rank, train_ranker


class CollegeHistory:
    def __init__(self, graph):
        self.nodes = graph["nodes"]
        self.index = {v:i for i,v in enumerate(self.nodes)}
        self.out_degree, self.sent_count = Counter(), Counter()
        rows, cols = [], []
        for u,v,w in graph["edges"]:
            self.out_degree[u] += 1
            self.sent_count[u] += w
            rows.append(self.index[u])
            cols.append(self.index[v])
        n = len(self.nodes)
        matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n,n))
        distance = shortest_path(matrix, method="D", directed=True, unweighted=True)
        self.distance = np.where(np.isfinite(distance), distance, UNREACHABLE).astype(np.int16)


def run():
    raw = load_events(SOURCE)
    graphs = {g["window"]:g for g in read_rows(OUT/"public/graphs.jsonl")}
    train = read_rows(OUT/"public/train.jsonl")
    val = read_rows(OUT/"public/validation.jsonl")
    truth = {r["id"]:set(r["sources"]) for r in read_rows(OUT/"truth/train.jsonl")}
    samples = defaultdict(list)
    for r in train+val:
        samples[r["window"]].append(r)
    features = {}
    for k in sorted(k for k in graphs if k.startswith("train")):
        g = Features(graphs[k],raw,CollegeHistory)
        features[k] = [g.event(r) for r in samples[k]]
    del g
    nodes, models, pair_train, predictions = {}, {}, [], []
    for cut in (4,6,8,10,12):
        earlier = [r for k in sorted(features) if int(k.split("-")[1]) <= cut for r in features[k]]
        model = fit(earlier, truth, len(FEATURES))
        nodes[str(cut)] = model
        keys = ([f"train-{i:02}" for i in (cut+1,cut+2)] if cut<12 else
                sorted(k for k in graphs if k.startswith("validation")))
        joint = None
        if cut>4:
            eligible = sum(truth[e.sid] <= set(e.nodes) for e in pair_train)
            if eligible:
                joint = train_ranker(pair_train, {sid:sorted(s) for sid,s in truth.items()}, 9)
            models[str(cut)] = joint
        new = []
        for k in keys:
            g = Features(graphs[k],raw,CollegeHistory)
            for obs in samples[k]:
                e = pair_event(obs,g,model)
                if cut>4:
                    _, ordered, _, _ = rank_pool(model,g.event(obs),False)
                    predictions.append({"id":e.sid,"window":k,"cutoff":cut,"nodes":e.nodes,
                                        "first128":ordered[:128],"legal":len(ordered),"sum":e.additive.tolist(),
                                        "joint":None if joint is None else rank(e,joint).tolist()})
                if cut<12:
                    new.append(e)
            del g
        print("fit through",cut,"node events",len(earlier),"joint positives",
              joint["eligible_train_events"] if joint else 0,"target events",sum(len(samples[k]) for k in keys),flush=True)
        pair_train.extend(new)
    write_json(OUT/"candidate_models.json",{"node":nodes,"joint":models})
    save_rows(OUT/"candidate_rankings.jsonl",predictions)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()

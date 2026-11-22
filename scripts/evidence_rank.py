"""Apply the frozen CollegeMsg models to public observations of the new cohort."""

import json

import numpy as np
from numba import get_num_threads, set_num_threads
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from build_dept3_pilot import read_rows, save_rows
from collegemsg_candidates import CollegeHistory
from collegemsg_data import OUT as COLLEGE, SOURCE
from collegemsg_forward import COLUMNS, simulate_sparse
from dept3_forward_model import kernels, prefix_graph
from evidence_cohort import OUT
from probe_cascade_yield import load_events
from run_dept3_candidates import Features, rank_pool
from run_dept3_forward import design_from_kernel
from run_dept3_pair_ranker import LEFT, RIGHT, pair_event, rank


def run():
    set_num_threads(min(8,get_num_threads()))
    raw = load_events(SOURCE)
    fits = json.loads((COLLEGE/"candidate_models.json").read_text(encoding="utf-8"))
    node, joint = fits["node"]["12"], fits["joint"]["12"]
    forward = json.loads((COLLEGE/"forward_models.json").read_text(encoding="utf-8"))["12"]
    graphs = {g["window"]:g for g in read_rows(COLLEGE/"public/graphs.jsonl")}
    observations = read_rows(OUT/"observations.jsonl")
    saved = []
    folder = OUT/"scores"
    folder.mkdir(exist_ok=True)
    for key in sorted({o["window"] for o in observations}):
        g = graphs[key]
        features = Features(g,raw,CollegeHistory)
        hist = prefix_graph(g,raw)
        base,edges = kernels(hist,forward)
        sparse = csr_matrix(edges)
        obs = [o for o in observations if o["window"]==key]
        values = []
        for o in obs:
            e = pair_event(o,features,node)
            _, ordered, _, _ = rank_pool(node,features.event(o),False)
            ids = np.array([hist[1][v] for v in e.nodes])
            _,_,*rest = design_from_kernel(o,hist[1],base,edges,forward,128)
            scores = simulate_sparse(base,sparse.indptr,sparse.indices,sparse.data,
                                     np.column_stack((ids[LEFT],ids[RIGHT])),*rest)
            assert np.isfinite(scores).all()
            values.append(scores)
            saved.append({"id":o["id"],"window":key,"nodes":e.nodes,"first128":ordered[:128],
                          "sum":e.additive.tolist(),"joint":rank(e,joint).tolist()})
        np.savez_compressed(folder/f"{key}.npz",ids=[o["id"] for o in obs],scores=values,columns=COLUMNS)
        print(key,len(values),"scored",flush=True)
    save_rows(OUT/"candidates.jsonl",saved)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        run()

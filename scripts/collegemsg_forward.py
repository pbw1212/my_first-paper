"""Same forward estimator as Dept3, with sparse updates and blockwise sampling."""

import json

import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from build_dept3_pilot import read_rows
from collegemsg_data import CFG, OUT, SOURCE
from dept3_forward_model import kernels, prefix_graph
from probe_cascade_yield import load_events
from run_dept3_forward import design_from_kernel, logmean
from run_dept3_pair_ranker import LEFT, RIGHT


BLOCK = 32
COLUMNS = ("direct","prefix32","prefix128","half64a","half64b","ess128")


@njit
def add_member(rate, total, blocks, guided, node, guide, ptr, cols, weights):
    total -= rate[node]
    if guided:
        blocks[node//BLOCK] -= rate[node]*guide[node]
    rate[node] = 0.
    for p in range(ptr[node],ptr[node+1]):
        v = cols[p]
        if rate[v] > 0:
            rate[v] += weights[p]
            total += weights[p]
            if guided:
                blocks[v//BLOCK] += weights[p]*guide[v]
    return total


@njit
def suffix(rate,total,obs,ends,ptr,cols,weights):
    value, start = 0., 0
    unused = np.empty(0)
    for end in ends:
        for j in range(start,end):
            value += np.log(rate[obs[j]] / total)
        if end<len(obs):
            for j in range(start,end):
                total = add_member(rate,total,unused,False,obs[j],unused,ptr,cols,weights)
        start = end
    return value


@njit
def draw_node(rate,guide,blocks,u):
    qz = blocks.sum()
    remaining = u*qz
    last = -1
    for c in range(len(blocks)):
        mass = max(0.,blocks[c])
        if remaining >= mass:
            remaining -= mass
            continue
        for j in range(c*BLOCK,min((c+1)*BLOCK,len(rate))):
            z = rate[j]*guide[j]
            if z>0:
                last = j
                remaining -= z
                if remaining < 0:
                    return j,qz
        if last>=0:
            return last,qz
    # Roundoff at a block boundary: reconstruct the flat CDF from the actual masses.
    target = u*np.dot(rate,guide)
    for j in range(len(rate)):
        if rate[j]*guide[j]>0:
            last = j
            target -= rate[j]*guide[j]
            if target<0:
                break
    return last,qz


@njit(parallel=True)
def simulate_sparse(base,ptr,cols,weights,pairs,obs,ends,guide,lengths,uniforms):
    out = np.empty((len(pairs),6))
    for k in prange(len(pairs)):
        a,b = pairs[k]
        initial = base.copy()
        for u in (a,b):
            for p in range(ptr[u],ptr[u+1]):
                initial[cols[p]] += weights[p]
        initial[a] = initial[b] = 0.
        start_z = initial.sum()
        initial_blocks = np.zeros((len(base)+BLOCK-1)//BLOCK)
        for j in range(len(base)):
            initial_blocks[j//BLOCK] += initial[j]*guide[j]
        out[k,0] = suffix(initial.copy(),start_z,obs,ends,ptr,cols,weights)/len(obs)
        joint = np.empty(len(lengths))
        for s in range(len(lengths)):
            rate = initial.copy()
            blocks = initial_blocks.copy()
            total, logw = start_z, 0.
            for j in range(lengths[s]):
                node,qz = draw_node(rate,guide,blocks,uniforms[s,j])
                logw += np.log(qz / (total*guide[node]))
                total = add_member(rate,total,blocks,True,node,guide,ptr,cols,weights)
            joint[s] = logw+suffix(rate,total,obs,ends,ptr,cols,weights)
        out[k,1] = logmean(joint[:32])/len(obs)
        out[k,2] = logmean(joint)/len(obs)
        out[k,3] = logmean(joint[:64])/len(obs)
        out[k,4] = logmean(joint[64:])/len(obs)
        importance = np.exp(joint-joint.max())
        out[k,5] = importance.sum()**2 / np.dot(importance,importance)
    return out


def run():
    set_num_threads(min(8,get_num_threads()))
    raw = load_events(SOURCE)
    obs = {r["id"]:r for s in ("train","validation") for r in read_rows(OUT/"public"/f"{s}.jsonl")}
    models = json.loads((OUT/"forward_models.json").read_text(encoding="utf-8"))
    graphs = {g["window"]:g for g in read_rows(OUT/"public/graphs.jsonl")}
    records = read_rows(OUT/"candidate_rankings.jsonl")
    (OUT/"scores").mkdir(exist_ok=True)
    for key in sorted({r["window"] for r in records}):
        items = [r for r in records if r["window"]==key]
        model = models[str(items[0]["cutoff"])]
        graph = prefix_graph(graphs[key],raw)
        base,edges = kernels(graph,model)
        csr = csr_matrix(edges)
        saved = np.empty((len(items),len(LEFT),len(COLUMNS)))
        for i,r in enumerate(items):
            ids = np.array([graph[1][v] for v in r["nodes"]])
            _,_,*rest = design_from_kernel(obs[r["id"]],graph[1],base,edges,model,CFG["primary_particles"])
            saved[i] = simulate_sparse(base,csr.indptr,csr.indices,csr.data,np.column_stack((ids[LEFT],ids[RIGHT])),*rest)
        assert np.isfinite(saved).all()
        np.savez_compressed(OUT/"scores"/f"{key}.npz",ids=[r["id"] for r in items],
                            nodes=[r["nodes"] for r in items],columns=COLUMNS,scores=saved)
        print(key,len(items),"events scored",flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1,user_api="blas"):
        run()

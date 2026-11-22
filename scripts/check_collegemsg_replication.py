"""Audit label isolation, candidate scores, and identity ambiguity in each event week."""

import json
from collections import Counter, defaultdict
from statistics import mean

import numpy as np
from numba import set_num_threads
from threadpoolctl import threadpool_limits

from build_dept3_pilot import DAY, make_windows, mask, read_rows, simulate
from collegemsg_data import CFG, OUT, SOURCE
from dept3_forward_model import kernels, prefix_graph, transition_rows
from probe_cascade_yield import load_events
from run_dept3_candidates import write_json
from run_dept3_forward import design_from_kernel, simulate_pairs
from run_dept3_pair_ranker import COORDINATES
from score_dept3_candidates import roles


def check():
    set_num_threads(1)
    raw = load_events(SOURCE)
    models = json.loads((OUT/"forward_models.json").read_text(encoding="utf-8"))
    cand_models = json.loads((OUT/"candidate_models.json").read_text(encoding="utf-8"))
    proposals = {r["id"]:r for r in read_rows(OUT/"candidate_rankings.jsonl")}
    results = {r["id"]:r for r in read_rows(OUT/"event_results.jsonl")}
    saved_graphs = {g["window"]:g for g in read_rows(OUT/"public/graphs.jsonl")}
    cut_stats = {}
    total_checked = 0
    for split in ("train","validation"):
        graphs,streams = make_windows(raw,split,*CFG["time_splits"][split],width=7*DAY)
        obs = {r["id"]:r for r in read_rows(OUT/"public"/f"{split}.jsonl")}
        truth = {r["id"]:r for r in read_rows(OUT/"truth"/f"{split}.jsonl")}
        draws = {r["id"]:r for r in read_rows(OUT/"truth"/f"{split}_attempts.jsonl") if r["id"]}
        assert obs.keys()==truth.keys()==draws.keys()
        assert len(obs)==CFG[split+"_target"]
        weekly = {}
        differences = []
        for g in graphs:
            key = g["window"]
            assert g==saved_graphs[key]
            past = Counter((u,v) for t,u,v in raw if t<g["start"])
            assert past==Counter({(u,v):w for u,v,w in g["edges"]})
            known = set(g["nodes"])
            stream = streams[key]
            involved = {v for _,b in stream for edge in b for v in edge}
            isolated = known-involved
            ids = [i for i,r in obs.items() if r["window"]==key]
            eligible = [i for i in ids if i in proposals]
            info = {"events":len(eligible),"isolated_candidates":len(isolated),"one_isolated":0,
                    "silent_with_edges":0,"counterfactual_exact_match":0,"next_members":0,"nll_totals":[0.,0.,0.]}
            if eligible:
                model_cut = str(proposals[eligible[0]]["cutoff"])
                fwd = models[model_cut]
                joint = cand_models["joint"][model_cut]
                windows = set(fwd["windows"]+cand_models["node"][model_cut]["fit_windows"]+(joint["windows"] if joint else []))
                assert all(saved_graphs[k]["end"]<=g["start"] for k in windows)
                hist = prefix_graph(g,raw)
                base,edges = kernels(hist,fwd)
                with np.load(OUT/"scores"/f"{key}.npz") as arch:
                    score_ids, score_nodes, scores = arch["ids"].tolist(),arch["nodes"].tolist(),arch["scores"]
                assert len(score_ids)==len(set(score_ids)) and set(score_ids)==set(eligible)
                for sid,nodes,x in zip(score_ids,score_nodes,scores,strict=True):
                    p = proposals[sid]
                    assert nodes==p["nodes"] and len(set(nodes))==32
                    assert set(nodes)<=known-set(v for b in obs[sid]["observed"] for v in b)
                    assert np.isfinite(x).all() and ((x[:,-1]>=1-1e-8)&(x[:,-1]<=128+1e-8)).all()
                    source = set(truth[sid]["sources"])
                    pair = next((i for i,(a,b) in enumerate(COORDINATES) if {nodes[a],nodes[b]}==source),None)
                    tie = np.argsort(p["sum"])
                    for j,m in enumerate(("direct","prefix32","prefix128","half64a","half64b")):
                        rr = None if pair is None else int(1+sum(x[:,j]>x[pair,j])+sum((x[:,j]==x[pair,j])&(tie<tie[pair])))
                        assert rr==results[sid]["ranks"][m]
                    if sid==score_ids[0]:
                        choices = [0,183,495]
                        checked_pairs = np.array([[hist[1][nodes[v]] for v in COORDINATES[i]] for i in choices])
                        for budget in (32,128):
                            args = design_from_kernel(obs[sid],hist[1],base,edges,fwd,budget)
                            dense,_ = simulate_pairs(*args[:2],checked_pairs,*args[2:],False)
                            actual = x[choices][:,[0,1] if budget==32 else [0,2,3,4,5]]
                            ref = dense[:,[0,1] if budget==32 else [0,1,3,4,5]]
                            differences.append(float(np.max(np.abs(actual-ref))))
                            assert np.allclose(actual,ref,rtol=1e-8,atol=1e-8), (key,budget,differences[-1])
            for sid in ids:
                a,l,o = draws[sid],truth[sid],obs[sid]
                tr = simulate(stream,a["sources"],a["draw_seed"])
                cut,visible = mask(tr)
                assert tr==l["trajectory"] and l["sources"]==a["sources"] and cut==l["cut_batches"] and visible==o["observed"]
                total_checked += 1
                if sid not in proposals:
                    continue
                direct, _ = roles(l)
                iso_true = isolated.intersection(l["sources"])
                if iso_true:
                    assert len(iso_true)==1 and not (iso_true & direct)
                    other = next(v for v in l["sources"] if v not in isolated)
                    replacement = min(isolated-iso_true)
                    assert simulate(stream,[other,replacement],a["draw_seed"])==tr
                    info["one_isolated"] += 1
                    info["counterfactual_exact_match"] += 1
                elif len(direct)==1:
                    info["silent_with_edges"] += 1
                for num,den,_ in transition_rows(hist,l["sources"],tr):
                    learned = (num@fwd["weights"])/(den@fwd["weights"])
                    for j,p in enumerate((learned,num[:,1]/den[1],num[:,0]/den[0])):
                        info["nll_totals"][j] -= float(np.log(p).sum())
                    info["next_members"] += len(num)
            if eligible:
                weekly[key] = info
        n = sum(v["events"] for v in weekly.values())
        isolated_cases = sum(v["one_isolated"] for v in weekly.values())
        total_members = sum(v["next_members"] for v in weekly.values())
        cut_stats["rolling" if split=="train" else split] = {
            "n":n,"one_isolated":isolated_cases,
            "silent_with_edges":sum(v["silent_with_edges"] for v in weekly.values()),
            "counterfactual_exact_match":sum(v["counterfactual_exact_match"] for v in weekly.values()),
            "possible_isolated_range":[min(v["isolated_candidates"] for v in weekly.values()),max(v["isolated_candidates"] for v in weekly.values())],
            "oracle_top1_ceiling_isolated_stratum":sum(v["one_isolated"]/v["isolated_candidates"] for v in weekly.values())/isolated_cases,
            "oracle_top32_ceiling_isolated_stratum":sum(v["one_isolated"]*min(1,32/v["isolated_candidates"]) for v in weekly.values())/isolated_cases,
            "next_member_nll":(np.sum([v["nll_totals"] for v in weekly.values()],axis=0)/total_members).tolist(),
            "max_sparse_dense_abs_difference":max(differences),"windows":weekly}
    assert total_checked==2500
    cut_stats["trajectories_replayed"] = total_checked
    write_json(OUT/"audit.json",cut_stats)
    for k in ("rolling","validation"):
        print(k,{c:v for c,v in cut_stats[k].items() if c!="windows"},flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1,user_api="blas"):
        check()

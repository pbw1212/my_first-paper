"""Single-snapshot comparison using the authors' cluster-search function and a masked replay objective."""
import argparse
import ast
from hashlib import sha256
from itertools import combinations,groupby
from pathlib import Path
import random
import time
import warnings

import networkx as nx
import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.exceptions.warnings import InputDataWarning,NumericsWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf_discrete
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
import matched_suffix_ablation as snap
from build_dept3_pilot import mask,simulate,WIDTH
from evaluate_factorial_history import dataset
from recent_member_evidence import OUT

REV = "e891f84f8d5e1c829b511672da7d946aeba06def"
SOURCE = {"methods.py":"27b6447392b25c6b3143e7fef293154d7aa8646d47a6ec1fe73e94b7fad89bab",
          "my_util.py":"1bdf3c6ea167878995b0b699630919bb6466a56e87607df0e0c0f30841b1b1f6"}
PER_WINDOW,CLUSTERS,STEPS,SIMS = 25,20,10,10


def author_functions(checkout):
    names = {"GPSI_cluster_sampling","fourier_transfer_for_all_candidate_set","find_source_set_from_fourier"}
    namespace = dict(nx=nx,np=np,torch=torch,random=random,KMeans=KMeans,combinations=combinations,
                     recovery_rate=0.,ExpectedImprovement=ExpectedImprovement,ExactMarginalLogLikelihood=ExactMarginalLogLikelihood,
                     optimize_acqf_discrete=optimize_acqf_discrete,fit_gpytorch_model=fit_gpytorch_mll)
    for file,expected in SOURCE.items():
        code = (checkout/file).read_text(encoding="utf8")
        assert sha256(code.encode()).hexdigest()==expected
        tree = ast.parse(code)
        tree.body = [f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name in names]
        exec(compile(tree,str(checkout/file),"exec"),namespace)
    def gp(x,y):
        # Public BoTorch constructor with the author's scalar RBF and Gamma noise prior.
        like = GaussianLikelihood(noise_prior=GammaPrior(1.1,.05))
        like.noise = 2.
        return SingleTaskGP(x,y,likelihood=like,covar_module=RBFKernel(),outcome_transform=None)
    namespace["RBFSingleTaskGP"] = gp
    namespace["create_candidate_pool"] = lambda G,seen,size: sorted(set(G.nodes)-set(seen))
    namespace["Contagion"] = lambda **kw: kw["source"]
    return namespace


def past_windows(raw,g,index):
    result = []
    for k in range(6):
        end = g["start"]-k*WIDTH
        if end-WIDTH<raw[0][0]:
            continue
        selected = [(t,index[u],index[v]) for t,u,v in raw if end-WIDTH<=t<end]
        result.append([(t,[(u,v) for _,u,v in grp]) for t,grp in groupby(selected,key=lambda r:r[0])])
    assert result
    return result


def snapshot_scores(net,mean,std,hist,event):
    (nodes,_,_),(x,edge,overlap) = hist.design(snap.snapshot(event))
    feats = torch.from_numpy(np.clip((x-mean)/std,-6,6))[None]
    with torch.no_grad():
        mat = net(feats,torch.tensor(edge.astype(np.float16),dtype=torch.float32)[None],
                  torch.tensor(overlap.astype(np.float16),dtype=torch.float32)[None],torch.ones(1,len(nodes),dtype=torch.bool))[0].numpy()
    i,j = np.triu_indices(len(nodes),1)
    tie = fg.tie_order(event["id"],len(nodes))
    idx = np.lexsort((np.maximum(tie[i],tie[j]),np.minimum(tie[i],tie[j]),-mat[i,j]))[0]
    return [nodes[i[idx]],nodes[j[idx]]]


def run(name,checkout):
    author = author_functions(checkout)
    source,parent,raw,graphs = dataset(name)
    folder = OUT/"single_snapshot"/name
    folder.mkdir(parents=True,exist_ok=True)
    train = snap.OUT/source
    cfg = fg.load(train/"snapshot_selection.json")
    assert all(fg.digest(train/f)==v for f,v in cfg["hashes"].items())
    with np.load(train/"snapshot_scale.npz") as scale:
        mean,std = scale["mean"],scale["std"]
    net = snap.PairNet(len(mean),True).eval()
    net.load_state_dict(torch.load(train/"snapshot_pair.pt",weights_only=True))
    obs = fg.read_rows(parent/"public.jsonl")
    groups = []
    for g in graphs:
        population = [r for r in obs if r["window"]==g["window"]]
        selection = sorted(population,key=lambda r:sha256(("pilot-20261120-"+r["id"]).encode()).digest())[:PER_WINDOW]
        groups.append((g,selection))
    lock = dict(revision=REV,inputs=SOURCE,script=fg.digest(Path(__file__)),clusters=CLUSTERS,bo_steps=STEPS,
                sims_per_query=SIMS,snapshot_weights=fg.digest(train/"snapshot_pair.pt"),events=[o["id"] for _,es in groups for o in es])
    protocol = folder/"protocol.json"
    if protocol.exists():
        assert fg.load(protocol)==lock
    else:
        fg.write_json(protocol,lock)
    for g,events in groups:
        path = folder/(g["window"]+".jsonl")
        if path.exists():
            assert [r["id"] for r in fg.read_rows(path)]==[e["id"] for e in events]
            continue
        roster = sorted(g["nodes"])
        ix = {v:i for i,v in enumerate(roster)}
        graph = nx.Graph()
        graph.add_nodes_from(range(len(roster)))
        graph.add_edges_from((ix[u],ix[v]) for u,v,_ in g["edges"])
        past = past_windows(raw,g,ix)
        hist = snap.PublicWindow(g,raw)
        results = []
        for o in events:
            event_seed = int.from_bytes(sha256(o["id"].encode()).digest()[:4],"big")
            random.seed(event_seed)
            torch.manual_seed(event_seed)
            rng = random.Random(event_seed+1)
            visible = sorted(ix[v] for batch in o["observed"] for v in batch)
            seen = set(visible)
            calls = 0
            def coverage(pair,observed,repetitions):
                nonlocal calls
                assert set(observed)==seen and set(pair).isdisjoint(seen) and repetitions==SIMS
                calls += 1
                vals = []
                for _ in range(repetitions):
                    trace = simulate(rng.choice(past),pair,rng.getrandbits(63))
                    cut = mask(trace)
                    pred = {v for batch in cut[1] for v in batch} if cut else set()
                    vals.append(len(pred & seen)/len(pred | seen))
                scores = torch.tensor(vals)
                return scores.mean(),scores.var()
            author["source_coverage"] = coverage
            start = time.perf_counter()
            pair = author["GPSI_cluster_sampling"](graph,visible,STEPS,SIMS,.3,len(roster),"si",2,CLUSTERS)
            seconds = time.perf_counter()-start
            assert calls==CLUSTERS+STEPS and len(set(pair))==2 and set(pair).isdisjoint(seen)
            start = time.perf_counter()
            baseline = snapshot_scores(net,mean,std,hist,o)
            results.append(dict(id=o["id"],window=g["window"],bosoul=sorted(roster[j] for j in pair),
                                snapshot=sorted(baseline),bosoul_seconds=seconds,model_seconds=time.perf_counter()-start,simulations=calls*SIMS))
        tmp = path.with_suffix(".tmp")
        fg.save_rows(tmp,results)
        tmp.replace(path)
        print(g["window"],len(results),flush=True)
    rows = [r for g,_ in groups for r in fg.read_rows(folder/(g["window"]+".jsonl"))]
    truth = {r["id"]:sorted(r["sources"]) for r in fg.read_rows(parent/"truth.jsonl")}
    archive = {r["id"]:r["rank"] for r in fg.read_rows(snap.OUT/name/"snapshot_predictions.jsonl")}
    assert all((r["snapshot"]==truth[r["id"]])==(archive[r["id"]]==1) for r in rows)
    result = {}
    for m in ("snapshot","bosoul"):
        result[m] = dict(n=len(rows),top1=sum(r[m]==truth[r["id"]] for r in rows),
                         at_least_one=sum(bool(set(r[m])&set(truth[r["id"]])) for r in rows),
                         seconds=sum(r["model_seconds" if m=="snapshot" else "bosoul_seconds"] for r in rows))
    result["simulations"] = sum(r["simulations"] for r in rows)
    fg.write_json(folder/"summary.json",result)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name",choices=("Dept2","company"))
    p.add_argument("checkout",type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore",category=InputDataWarning)
    warnings.filterwarnings("ignore",category=NumericsWarning)
    with threadpool_limits(1):
        run(args.name,args.checkout)

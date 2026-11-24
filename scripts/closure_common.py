"""Read the original project archives and write only to the closure output root."""
import gzip
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
ARCH = Path(os.environ.get("CLOSURE_ARCHIVE", REPO if (REPO/"data/raw").is_dir() else REPO.parent)).resolve()
BASE = Path(os.environ.get("CLOSURE_OUTPUT", REPO/"runs/paper_closure")).resolve()
sys.path.insert(0, str(ARCH/"scripts"))
# Upstream argparse modules inspect argv on import.
import fgssi_suffix as fg
import locked_dept4 as dept4
import heldout_libc_alpha as libc
import heldout_r_lists as rl
from distilled_source_pairs import PairNet, PublicWindow
from evaluate_factorial_history import dataset, fitted
from factorial_fgssi import SEEDS
from five_frame_pairs import frames
from recent_member_evidence import PERSON, LINKS, CONTEXT, OUT as RECENT
from source_prior_shift import activity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(1)
POOL = threadpool_limits(1)
RUN = ARCH/"runs"
NAMES = ["Dept2","company","Dept4","Enron","CollegeMsg","GNU libc"]
DAY = 86400
BOOT = 20261120


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def dump(p, value):
    def conv(x):
        if isinstance(x,np.ndarray):
            return x.tolist()
        return x.item()
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=conv,allow_nan=False)+"\n",encoding="utf8")


def load(p):
    return json.loads(p.read_text(encoding="utf8"))


def save_rows(p, rows):
    with gzip.GzipFile(str(p), "wb", compresslevel=3,mtime=0) as f:
        for row in rows:
            f.write((json.dumps(row,separators=(",",":"),ensure_ascii=False,allow_nan=False)+"\n").encode())


def read_rows(p):
    with gzip.open(p,"rt",encoding="utf8") as f:
        return [json.loads(line) for line in f]


def lock(folder, plan, inputs=()):
    folder.mkdir(parents=True,exist_ok=True)
    paths=sorted(set(Path(p).resolve() for p in inputs))
    def key(p):
        return ("repo/"+p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else "archive/"+p.relative_to(ARCH).as_posix())
    cfg=dict(plan=plan,seed_order=list(SEEDS),bootstrap=dict(seed=BOOT,replicates=10000,block=3,circular=True,
            units="event-weighted; paired seed and consecutive prespecified window blocks, including empty windows"),
            inputs={key(p):sha(p) for p in paths})
    p=folder/"protocol.json"
    if p.exists():
        if load(p)!=cfg:
            raise ValueError(f"Protocol mismatch: {p}")
    else:
        dump(p,cfg)


def check_protocol(folder):
    cfg=load(folder/"protocol.json")
    for key,expected in cfg["inputs"].items():
        root,sub=key.split("/",1)
        assert sha((REPO if root=="repo" else ARCH)/sub)==expected, key


def source(name):
    return "company" if name=="company" else "Dept2"


def cohort(name):
    if name=="GNU libc":
        archive={r["id"]:r for r in fg.read_rows(libc.OUT/"events.jsonl") if r["profile"]=="flat"}
        found=set()
        for j,(g,h,stream,vol,_,_) in enumerate(rl.calendar_windows(libc.messages(),libc.NAME)):
            rng=np.random.default_rng([libc.SEED,0,j,0])
            items=[]
            for e,src,_,_ in rl.draw_events(g,stream,activity(g,h),None,rng,"flat",rl.ATTEMPTS,False):
                assert archive[e["id"]]["sources"]==src
                found.add(e["id"])
                items.append((e,src))
            yield g,h,items,vol
        assert found==archive.keys()
        return
    if name=="Dept4":
        raw=dept4.load_events(dept4.RAW)
        parent=dept4.FOLDER
        graphs=[g for g,_ in dept4.windows(raw)]
    else:
        _,parent,raw,graphs=dataset(name)
    public=fg.read_rows(parent/"public.jsonl")
    truth={r["id"]:r["sources"] for r in fg.read_rows(parent/"truth.jsonl")}
    assert {e["id"] for e in public}==truth.keys() and len(public)==len(truth)
    for g in graphs:
        items=[(e,truth[e["id"]]) for e in public if e["window"]==g["window"]]
        yield g,raw,items,None


def cohort_inputs(names):
    inputs=list((ARCH/"data/raw").glob("*"))
    for name in names:
        if name=="GNU libc":
            inputs.extend([libc.OUT/f for f in ("protocol.json","archives.json","events.jsonl")])
        else:
            parent=dept4.FOLDER if name=="Dept4" else RUN/("temporal_transfer" if name=="Enron" else "distilled_source_pairs")/name
            inputs.extend([parent/f for f in ("public.jsonl","truth.jsonl")])
    return inputs


def nets(src,methods):
    out={}
    for method in methods:
        if method in ("full","static"):
            out[method]=[fitted(src,s,"C" if method=="full" else "A") for s in SEEDS]
        else:
            path=RECENT/src
            with np.load(path/(method+".npz")) as scale:
                models=[]
                for s in SEEDS:
                    f=path/f"{method}-{s}.pt"
                    assert sha(f)==load(f.with_suffix(".json"))["sha256"]
                    m=PairNet(64,False).eval()
                    m.load_state_dict(torch.load(f,weights_only=True))
                    models.append((m,scale["mean"],scale["std"]))
                out[method]=models
    return out


def features(x,m):
    if m not in ("person","links"):
        return x
    cols=PERSON if m=="person" else PERSON+LINKS
    take=[*cols,*(c+29 for c in cols),*CONTEXT]
    a=np.zeros_like(x)
    a[:,take]=x[:,take]
    return a


@torch.inference_mode()
def logits(models,x):
    return [m.node(m.encoder(torch.from_numpy(np.clip((x-mu)/sd,-6,6)))).flatten().numpy().astype(float) for m,mu,sd in models]


def ranks(score,tie,nodes,gold):
    order=fg.sorted_nodes(np.asarray(score),tie)
    lookup={nodes[i]:j+1 for j,i in enumerate(order)}
    return [lookup.get(s,len(nodes)+1) for s in gold]


def interval(values,windows,keys):
    v=np.asarray(values,dtype=float)
    if v.ndim==1:
        v=v[:,None]
    assert len(v)==len(windows)>0 and len(set(keys))==len(keys)
    at=np.array([keys.index(w) for w in windows])
    n=len(keys)
    totals=np.bincount(at,minlength=n)
    scores=np.zeros((n,v.shape[1]))
    np.add.at(scores,at,v)
    rng=np.random.default_rng(BOOT)
    ids=rng.integers(n,size=(10000,(n+2)//3,1))
    blocks=((ids+np.arange(3))%n).reshape(10000,-1)[:,:n]
    seeds=rng.integers(v.shape[1],size=(10000,v.shape[1],1))
    den=totals[blocks].sum(1)
    ok=den>0
    samples=scores[blocks[:,None,:],seeds].mean(1).sum(1)[ok]/den[ok]
    return dict(mean=float(v.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),valid=int(ok.sum()))


def evidence(folder,extra):
    check_protocol(folder)
    files={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob("*")) if p.is_file() and p.name!="verification.json"}
    dump(folder/"verification.json",dict(passed=True,checks=extra,sha256=files))


def agg(rows,keys,reference):
    """Persist event/seed deltas alongside rates. All values use unit fractions."""
    if not rows:
        return dict(n=0,keys=keys,methods={},differences={})
    methods=list(rows[0]["scores"])
    ws=[r["window"] for r in rows]
    ans=dict(n=len(rows),keys=keys,window_n=dict(Counter(ws)),methods={},differences={})
    for m in methods:
        a=np.array([r["scores"][m] for r in rows],float)
        ans["methods"][m]=interval(a,ws,keys)
        if m==reference:
            continue
        diff=a-np.array([r["scores"][reference] for r in rows],float)
        ans["differences"][m+" minus "+reference]=interval(diff,ws,keys)
        for r,d in zip(rows,diff):
            r.setdefault("delta",{})[m+" minus "+reference]=d.tolist()
    return ans


def audit_agg(rows,stat):
    assert len(rows)==len({r["id"] for r in rows})==stat["n"]
    for m,v in stat["methods"].items():
        assert abs(np.mean([np.mean(r["scores"][m]) for r in rows])-v["mean"])<1e-12
    for k,v in stat["differences"].items():
        a,b=k.split(" minus ")
        for r in rows:
            assert np.allclose(np.subtract(r["scores"][a],r["scores"][b]),r["delta"][k],atol=1e-14,rtol=0)
        assert np.isclose(np.mean([r["delta"][k] for r in rows]),v["mean"])


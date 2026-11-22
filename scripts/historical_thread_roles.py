"""Learn role-specific rankings on Python, then test fixed models on other public communities."""
import argparse
import calendar
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
from threadpoolctl import threadpool_limits

import fgssi_suffix as fg
from public_thread_archive import LISTS,OUT as PYTHON,ROOT,download,prepare_month
from public_thread_transfer import DAY,MONTHS,marked_thread,thread_forest

OUT = ROOT/"runs/historical_thread_roles"
TARGETS = {"gcc":"https://gcc.gnu.org/pipermail/gcc/","ffmpeg-devel":"https://ffmpeg.org/pipermail/ffmpeg-devel/"}
TARGET_MONTHS = [(y,m) for y in (2021,2022) for m in range(1,13)]
ARCHIVE_MONTHS = [(2020,m) for m in range(9,13)]+TARGET_MONTHS+[(2023,1)]
WIDTHS = {"role_history":4,"direct":6,"retrieval":8}
REGULARIZERS = (.01,.1,1.)


def fetch_month(args):
    n,url,k = args
    target = OUT/n/"headers"/(k+".gz")
    manifest = target.with_suffix(".json")
    if manifest.is_file():
        entry = fg.load(manifest)
        assert entry["url"]==url and entry["metadata_sha256"]==fg.digest(target)
        return entry
    _,_,rows,entry = prepare_month(args)
    encoded = "".join(json.dumps(r,separators=(",",":"))+"\n" for r in rows).encode()
    temp = target.with_suffix(".tmp")
    temp.write_bytes(gzip.compress(encoded,mtime=0))
    temp.replace(target)
    entry["metadata_sha256"] = fg.digest(target)
    fg.write_json(manifest,entry)
    return entry


def fetch(name):
    folder = OUT/name/"headers"
    folder.mkdir(parents=True,exist_ok=True)
    base = TARGETS[name]
    links = set(re.findall(r'href="([^"]+)"',download(base).decode(),re.I))
    tasks = []
    for y,m in ARCHIVE_MONTHS:
        k = f"{y}-{calendar.month_name[m]}"
        filename = next(v for v in (k+".txt.gz",k+".txt") if v in links)
        tasks.append((name,base+filename,k))
    with ThreadPoolExecutor(max_workers=3) as pool:
        entries = list(pool.map(fetch_month,tasks))
    fg.write_json(OUT/name/"archives.json",entries)


def read_headers(name):
    paths = [PYTHON/(name+"-headers.jsonl.gz")] if name in LISTS else sorted((OUT/name/"headers").glob("*.gz"))
    assert len(paths)==(1 if name in LISTS else len(ARCHIVE_MONTHS))
    messages = {}
    conflicts = set()
    for p in paths:
        with gzip.open(p,"rt",encoding="utf8") as stream:
            for mid,a,t,parent,reply in map(json.loads,stream):
                row = (a,t,parent,reply)
                if mid in messages and messages[mid]!=row:
                    conflicts.add(mid)
                messages[mid] = row
    for k in conflicts:
        del messages[k]
    return messages


def historical_document(root,rows,t0,first):
    arrivals = {}
    for t,_,a in rows:
        if a!=first and t0<=t<t0+14*DAY:
            arrivals[a] = min(t,arrivals.get(a,t))
    order = sorted((t,a) for a,t in arrivals.items())
    second = None
    if order and order[0][0]>t0 and (len(order)==1 or order[1][0]>order[0][0]):
        second = order[0][1]
    return t0,first,second,set(arrivals)-{second}


class Month:
    def __init__(self,start,messages,contacts,docs):
        recent = [(t,a,b) for t,a,b in contacts if start-98*DAY<=t<start]
        self.roster = {v for _,a,b in recent for v in (a,b)}
        self.replies = Counter((a,b) for t,a,b in recent if t>=start-28*DAY)
        self.volume = [Counter(a for a,t,_,_ in messages.values() if start-d*DAY<=t<start) for d in (28,98)]
        self.docs = [d for d in docs if start-98*DAY<=d[0]<start-14*DAY]
        self.roles = [Counter(d[i] for d in self.docs if d[i] is not None) for i in (1,2)]
        df = Counter(v for _,_,_,late in self.docs for v in late)
        self.idf = {v:1+np.log((1+len(self.docs))/(1+n)) for v,n in df.items()}
        self.unseen_idf = 1+np.log1p(len(self.docs))
        self.norm = [sum(self.idf[v]**2 for v in late) for _,_,_,late in self.docs]

    def design(self,suffix):
        visible = self.roster.intersection(a for batch in suffix for a in batch)
        nodes = sorted(self.roster-visible)
        prior = [*self.volume,*self.roles]
        x = np.array([[c[n] for c in prior]+[sum(self.replies[n,v] for v in visible),sum(self.replies[v,n] for v in visible),0.,0.] for n in nodes],dtype=float)
        norm = sum(self.idf.get(v,self.unseen_idf)**2 for v in visible)
        votes = [Counter(),Counter()]
        for (_,first,second,late),size in zip(self.docs,self.norm,strict=True):
            shared = visible & late
            if shared and norm and size and second is not None:
                dot = sum(self.idf[v]**2 for v in sorted(shared))
                weight = dot**2/(norm*size)
                votes[0][first] += weight
                votes[1][second] += weight
        for i,vote in enumerate(votes):
            x[:,6+i] = [vote[n] for n in nodes]
        return nodes,np.log1p(x),len(visible)>=3 and len(nodes)>=12


def examples(name):
    msg = read_headers(name)
    trees,contacts = thread_forest(msg)
    docs = [historical_document(root,thread,msg[root][1],msg[root][0]) for root,thread in sorted(trees.items())]
    rows = []
    flow = []
    for y,m in (MONTHS if name in LISTS else TARGET_MONTHS):
        start = int(datetime(y,m,1,tzinfo=timezone.utc).timestamp())
        end = int(datetime(y+(m==12),m%12+1,1,tzinfo=timezone.utc).timestamp())
        month = Month(start,msg,contacts,docs)
        monthkey = f"{y}-{m:02d}"
        ct = Counter()
        for root,thread in sorted(trees.items()):
            if not start<=msg[root][1]<end:
                continue
            ct["roots"] += 1
            reason,payload = marked_thread(root,thread,msg)
            ct[reason] += 1
            if not payload:
                continue
            gold,suffix = payload
            nodes,x,ready = month.design(suffix)
            index = {v:i for i,v in enumerate(nodes)}
            ids = [index.get(v,-1) for v in gold]
            ct["scored"] += ready
            ct["two_reachable"] += ready and min(ids)>=0
            rows.append(dict(id=root,window=monthkey,sources=gold,ids=ids,nodes=nodes,x=x,ready=ready))
        flow.append(dict(window=monthkey,roster=len(month.roster),history_docs=len(month.docs),**ct))
    return rows,flow


def objective(theta,examples,reg):
    loss = 0.
    gradient = np.zeros_like(theta)
    for x,target in examples:
        s = x@theta
        loss += logsumexp(s)-s[target]
        gradient += softmax(s)@x-x[target]
    return loss/len(examples)+reg*(theta@theta)/2,gradient/len(examples)+reg*theta


def training():
    rows = []
    for n in LISTS:
        data,_ = examples(n)
        old = {r["id"]:r for r in fg.read_rows(PYTHON/n/"truth.jsonl")}
        assert len(data)==len(old)
        assert all(r["sources"]==old[r["id"]]["sources"] and r["ready"]==old[r["id"]]["scored"] for r in data)
        rows.extend(data)
    out = {}
    for name,dim in WIDTHS.items():
        out[name] = []
        for role in (0,1):
            early,late = [],[]
            for r in rows:
                if r["ready"] and r["ids"][role]>=0:
                    pair = (r["x"][:,:dim],r["ids"][role])
                    if r["window"]<"2018-06":
                        early.append(pair)
                    if r["window"]>="2018-07":
                        late.append(pair)
            trials = []
            for reg in REGULARIZERS:
                opt = minimize(objective,np.zeros(dim),args=(early,reg),method="L-BFGS-B",jac=True,bounds=[(-8.,8.)]*dim)
                assert opt.success,opt.message
                trials.append(dict(reg=reg,loss=objective(opt.x,late,0.)[0],theta=opt.x.tolist(),fit_n=len(early),val_n=len(late)))
            out[name].append(min(trials,key=lambda v:v["loss"]))
    return out


def signature():
    scripts = [Path(__file__),ROOT/"scripts/public_thread_archive.py",ROOT/"scripts/public_thread_transfer.py",ROOT/"scripts/build_dept3_pilot.py"]
    return {p.name:fg.digest(p) for p in scripts+[PYTHON/(n+"-headers.jsonl.gz") for n in LISTS]}


def freeze():
    OUT.mkdir(exist_ok=True)
    config = dict(primary="all mask-eligible real threads; both early addresses in twelve; retrieval vs volume and versus direct; evaluate projects separately",
        width=14,source="Python-Dev and Python-Ideas: 2017-01..2018-05 fit, June gap, 2018-07..12 validation",
        target=TARGETS,first="2021-01",last="2022-12",hashes=signature(),models=training())
    path = OUT/"protocol.json"
    if path.exists():
        previous = fg.load(path)
        assert config["hashes"]==previous["hashes"]
        for m in WIDTHS:
            for a,b in zip(config["models"][m],previous["models"][m],strict=True):
                assert a["reg"]==b["reg"] and np.allclose(a["theta"],b["theta"],atol=1e-7)
    else:
        fg.write_json(path,config)


def coverage(p,q,chosen):
    return p[chosen].sum()*q[chosen].sum()-(p[chosen]*q[chosen]).sum()


def choose(p,q,tie):
    """Optimize captured ordered-pair mass by greedy insertion and improving swaps."""
    marginal = fg.sorted_nodes(p*(1-q)+q*(1-p),tie)[:12]
    chosen = []
    for _ in range(12):
        gain = p*(q[chosen].sum() if chosen else 0.)+q*(p[chosen].sum() if chosen else 0.)
        if not chosen:
            gain = p*(1-q)+q*(1-p)
        gain[chosen] = -np.inf
        chosen.append(int(fg.sorted_nodes(gain,tie)[0]))
    greedy = np.array(chosen)
    best = greedy if coverage(p,q,greedy)>coverage(p,q,marginal) else marginal.copy()
    while True:
        ps,qs = p[best].sum(),q[best].sum()
        deltas = (qs-q[best,None])*(p[None,:]-p[best,None])+(ps-p[best,None])*(q[None,:]-q[best,None])
        deltas[:,best] = -np.inf
        value = deltas.max()
        if value<=1e-13:
            break
        rr,cc = np.nonzero(deltas==value)
        k = np.lexsort((tie[best[rr]],tie[cc]))[0]
        best[rr[k]] = cc[k]
    assert coverage(p,q,best)+1e-12>=coverage(p,q,marginal)
    return best,marginal


def test(name,cfg):
    folder = OUT/name
    rows,flow = examples(name)
    records = []
    for r in rows:
        lists = {}
        if r["ready"]:
            x,nodes = r["x"],r["nodes"]
            tie = fg.tie_order(r["id"],len(nodes))
            lists["volume"] = [nodes[i] for i in fg.sorted_nodes(x[:,0],tie)[:12]]
            for m,dim in WIDTHS.items():
                p,q = [softmax(x[:,:dim]@np.array(model["theta"])) for model in cfg["models"][m]]
                selected,ordinary = choose(p,q,tie)
                lists[m] = [nodes[i] for i in selected]
                lists[m+"_marginal"] = [nodes[i] for i in ordinary]
            assert all(len(set(v))==12 and set(v)<=set(nodes) for v in lists.values())
        records.append(dict(id=r["id"],window=r["window"],sources=r["sources"],ready=r["ready"],lists=lists))
    fg.save_rows(folder/"predictions.jsonl",records)
    fg.write_json(folder/"flow.json",flow)
    print(name,len(records),flush=True)


if __name__=="__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("step",choices=("freeze","fetch","test"))
    arg.add_argument("target",choices=TARGETS,nargs="?")
    a = arg.parse_args()
    with threadpool_limits(1):
        if a.step=="freeze":
            freeze()
        else:
            cfg = fg.load(OUT/"protocol.json")
            assert cfg["hashes"]==signature() and a.target
            if a.step=="fetch":
                fetch(a.target)
            else:
                test(a.target,cfg)

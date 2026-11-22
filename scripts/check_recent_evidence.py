"""Audit role effects and quantify how discriminative a single observed contact statistic is."""
import random
from collections import Counter

import numpy as np
from scipy.stats import rankdata

import fgssi_suffix as fg
from contact_rule_stress import ATTEMPTS,RULES,generate
from evaluate_factorial_history import dataset
from factorial_fgssi import SEEDS
from recent_member_evidence import OUT


def check(name):
    _,parent,raw,gs = dataset(name)
    windows,streams = fg.evaluation_windows(name,raw)
    assert [g["window"] for g in gs]==[g["window"] for g in windows]
    public = fg.read_rows(parent/"public.jsonl")
    truth = {r["id"]:r["sources"] for r in fg.read_rows(parent/"truth.jsonl")}
    out = []
    for g in gs:
        ix = {v:i for i,v in enumerate(g["nodes"])}
        counts = np.zeros((len(ix),len(ix)),int)
        for t,u,v in raw:
            if g["start"]-28*86400<=t<g["start"]:
                counts[ix[u],ix[v]] += 1
        for e in public:
            if e["window"]!=g["window"]:
                continue
            seen = {v for b in e["observed"] for v in b}
            nodes = sorted(set(ix)-seen)
            score = counts[np.ix_([ix[v] for v in nodes],[ix[v] for v in sorted(seen)])].sum(1)
            gold = np.array([v in truth[e["id"]] for v in nodes])
            assert gold.sum()==2
            ranks = rankdata(score)
            auc = (ranks[gold].sum()-3)/(2*(len(nodes)-2))
            out.append([np.mean(score[gold]>0),np.mean(score[~gold]>0),auc])
    folder = OUT/"stress"/name
    pred = {r["id"]:r for r in fg.read_rows(folder/"predictions.jsonl")}
    obs = {r["id"]:r for r in fg.read_rows(folder/"observations.jsonl")}
    protocol = fg.load(folder/"protocol.json")
    assert all(fg.digest(OUT.parents[1]/p)==h for p,h in protocol["hashes"].items())
    replayed = set()
    for j,g in enumerate(gs):
        events,labels = generate(g,streams[g["window"]],random.Random(protocol["seed"]+j))
        for e in events:
            sid = e["id"]
            assert e==obs[sid] and all(pred[sid][k]==v for k,v in labels[sid].items())
            replayed.add(sid)
    assert set(obs)==set(pred)==replayed
    groups = Counter(r["base"] for r in pred.values())
    summary = fg.load(folder/"summary.json")
    for k in RULES:
        eligible = [r for r in pred.values() if r["rule"]==k]
        for tag,rows in (("eligible",eligible),("common",[r for r in eligible if groups[r["base"]]==len(RULES)])):
            table = summary[k][tag]
            assert len(rows)==table["n"]
            for m,metrics in table["methods"].items():
                hits = [sum(set(r["sources"]).issubset(r["lists"][f"{m}/{s}"]) for r in rows) for s in SEEDS]
                assert hits==metrics["by_seed"]
    pos,neg,auc = np.mean(out,axis=0)
    return dict(events=len(out),true_source_has_contact=pos,other_candidate_has_contact=neg,
                mean_within_event_auc=auc,checks=dict(draws=ATTEMPTS*len(gs),rule_trajectories=ATTEMPTS*len(gs)*len(RULES),
                eligible_predictions=len(pred)))


if __name__=="__main__":
    fg.write_json(OUT/"contact_diagnostic_and_audit.json",{n:check(n) for n in ("Dept2","company")})

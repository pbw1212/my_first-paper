"""Aggregate external checks at the event and calendar-month levels for the experiment log."""
from collections import Counter
import numpy as np

import fgssi_suffix as fg
from heldout_r_lists import KEYS,NAMES,OUT,SEEDS
from historical_speaker_pairs import METHODS,OUT as PAIRS,margins,post


def examine(folder,scores):
    rows=fg.read_rows(folder/"events.jsonl")
    real=[r for r in rows if r["profile"]=="real"]
    active=[r for r in rows if r["profile"]=="mostly_active"]
    months=Counter(r["window"] for r in active)
    sources=Counter((r["window"],tuple(sorted(r["sources"]))) for r in active)
    c=np.array([months[k] for k in KEYS])
    gap=np.array([sum(r["hits"].get(f"mixed_offset/{s}",0)-r["hits"].get(f"mixed_plain/{s}",0) for s in SEEDS)/len(SEEDS) for r in active])
    where=np.array([r["window"] for r in active])
    loo=[float(100*gap[where!=m].mean()) for m in months if np.any(where!=m)]
    return dict(archive=fg.digest(folder/"events.jsonl"),real=dict(
        eligible=len(real),abstained=sum(not r["ready"] for r in real),
        scored_but_missing_source=sum(r["ready"] and not r["reachable"] for r in real),
        scoreable_and_reachable=sum(r["ready"] and r["reachable"] for r in real)),
        mostly_active=dict(eligible=len(active),populated_months=len(months),largest_month=months.most_common(1),
        effective_months=float(c.sum()**2/(c@c)) if len(active) else None,
        distinct_month_source_pairs=len(sources),leave_one_populated_month_out_pp=[min(loo),max(loo)] if loo else None,
        primary_results=scores["mostly_active"]["both_pct"],volume_pct=scores["mostly_active"]["volume_pct"],
        offset_minus_plain_pp=scores["mostly_active"]["offset_vs_plain"],offset_minus_simple_pp=scores["mostly_active"]["offset_vs_simple"]))


def audit_pairing():
    rng=np.random.default_rng(93271)
    worst=0.
    for n in [12,25,53]:
        v=rng.uniform(.5,500,n)
        c=rng.poisson(.35,(n,n)).astype(float)
        np.fill_diagonal(c,0.)
        prod=post(c,v,32)
        assert np.allclose(margins(prod),prod,atol=1e-10)
        paired=post(c,v,32,8.)
        for ids in [np.arange(n),np.arange(0,n,2)]:
            p=paired[np.ix_(ids,ids)]
            p/=p.sum()
            q=margins(p)
            assert np.all(np.diag(q)==0) and np.isclose(q.sum(),1.)
            # Every permitted 2x2 subtable of the independent coupling has unit cross product ratio.
            for _ in range(50):
                a,b,i,j=rng.choice(len(ids),4,replace=False)
                assert abs(np.log(q[a,i]*q[b,j]/(q[a,j]*q[b,i])))<1e-10
            worst=max(worst,np.max(abs(p.sum(1)-q.sum(1))),np.max(abs(p.sum(0)-q.sum(0))))
    summary=fg.load(PAIRS/"summary.json")
    for name,stats in summary.items():
        rows=fg.read_rows(PAIRS/(name+".jsonl"))
        assert len(rows)==stats["n"]
        for m in METHODS:
            roles=[r["roles"].get(m,0) for r in rows]
            assert sum(k==3 for k in roles)==stats["methods"][m]["both"]
            comp=[(r["roles"].get("paired",0)==3)-(r["roles"].get(m,0)==3) for r in rows]
            if m!="paired":
                delta=stats["paired_differences"][m]
                assert sum(v>0 for v in comp)==delta["gained"] and sum(v<0 for v in comp)==delta["lost"]
    return dict(max_matched_role_marginal_error=float(worst),external_discussions=sum(r["n"] for r in summary.values()),
                gained_vs_roles=sum(r["paired_differences"]["roles"]["gained"] for r in summary.values()),
                lost_vs_roles=sum(r["paired_differences"]["roles"]["lost"] for r in summary.values()))


def audit_libc_controls(folder):
    truth={z["id"]:z for z in fg.read_rows(folder/"events.jsonl") if z["profile"]=="real"}
    base=folder/"early_speaker_controls"
    rows=fg.read_rows(base/"events.jsonl")
    summary=fg.load(base/"summary.json")
    assert len(rows)==summary["n"]==len(truth)
    assert len({z["id"] for z in rows})==len(rows)
    for z in rows:
        ref=truth[z["id"]]
        assert ref["ready"]==z["ready"] and ref["window"]==z["window"]
        assert ref["hits"].get("volume",0)==z["hits"].get("volume",0)
        for s in SEEDS:
            assert ref["hits"].get(f"mixed_offset/{s}",0)==z["hits"].get(f"forward/{s}",0)
    assert all(sum(z["hits"].get(m,0) for z in rows)==n for m,n in summary["hits"].items())
    return dict(real_discussions_matched=len(rows),volume_hits=summary["hits"]["volume"],
                paired_hits=summary["hits"]["paired"],role_only_hits=summary["hits"]["roles"])


if __name__=="__main__":
    more=OUT.parent/"heldout_libc_alpha"
    files=[OUT/"protocol_v2.json",OUT/"verification.json",OUT/"direction/existing.json",OUT/"direction/heldout.json",
           PAIRS/"protocol.json",PAIRS/"summary.json",more/"protocol.json",more/"verification.json",
           more/"early_speaker_controls/protocol.json",more/"early_speaker_controls/summary.json",
           OUT/"early_only/protocol.json",OUT/"early_only/summary.json",OUT/"early_only/verification.json"]
    checks={p.relative_to(OUT.parent).as_posix():fg.digest(p) for p in files}
    main_scores=fg.load(OUT/"summary.json")
    results={n:examine(OUT/n,main_scores[n]) for n in NAMES}
    results["libc-alpha"]=examine(more,fg.load(more/"summary.json"))
    prefix=fg.load(OUT/"early_only/summary.json")
    prefix_audit=fg.load(OUT/"early_only/verification.json")
    early={}
    for name,stats in prefix.items():
        assert stats["eligible"]==prefix_audit[name]["roots"]
        early[name]=dict(eligible=stats["eligible"],ready=stats["ready"],both_reachable_and_ready=stats["scoreable_and_reachable"],
                         offset_minus_volume=stats["offset_minus_volume"],offset_minus_plain=stats["offset_minus_plain"],
                         shared_scoreable=prefix_audit[name]["same_scoreable"])
    fg.write_json(OUT/"round_review.json",dict(scope="实验准备材料",communities=results,early_two_only=early,
                  speaker_pair_audit=audit_pairing(),libc_real_controls=audit_libc_controls(more),
                  interpretation=dict(真实标签="根邮件作者和最早的另一位发言者",
                                      模拟标签="在每个月前14天的回复接触流上抽取的两位初始成员",
                                      测试设计="R 两个社区先锁定，因合格事件稀少再锁定 GNU libc；各自使用同一组冻结的 Dept2 模型",
                                      方向检验="固定活跃度先验，仅改变历史回复图的有向边",
                                      配对检验="相同角色频率下，对照引入历史首发者与最早回应者的配对计数"),
                  input_hashes=checks))

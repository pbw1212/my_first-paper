"""WP1. Compare absolute and ceil-proportional lists from the identical frozen scores."""
import itertools
from numba import njit
from closure_common import *
from factorial_history import folder_for

F=BASE/"wp1_budgets"
M=["Both-in-6","Both-in-12","Both-in-24","Both-in-5%","Both-in-10%","Any-in-12","Pair-MRR"]


@njit
def pairrank(scores,tie,a,b):
    order=np.argsort(scores)
    target=scores[a]+scores[b]
    lo,hi=min(tie[a],tie[b]),max(tie[a],tie[b])
    rank=1
    n=len(order)
    # Count strictly larger pairs with a monotone boundary. Resolve equal sums by the archived tie rule.
    left=right=n
    for i in range(n):
        v=scores[order[i]]
        while left>0 and v+scores[order[left-1]]>=target:
            left-=1
        while right>0 and v+scores[order[right-1]]>target:
            right-=1
        rank+=n-max(i+1,right)
        for j in range(max(left,i+1),right):
            x,y=tie[order[i]],tie[order[j]]
            if min(x,y)<lo or (min(x,y)==lo and max(x,y)<hi):
                rank+=1
    return rank


def metrics(n,rs,pairs):
    a=np.array(rs)
    both=a.max(1)
    ks=[6,12,24,math.ceil(n/20),math.ceil(n/10)]
    return dict(zip(M,[(both<=k).astype(int).tolist() for k in ks]+[(a.min(1)<=12).astype(int).tolist(),(1/np.asarray(pairs)).tolist()]))


def anchors(name):
    if name=="GNU libc":
        return {}
    if name=="Dept4":
        return {r["id"]:{s:r["lists"][f"C/{s}"] for s in SEEDS} for r in fg.read_rows(dept4.FOLDER/"predictions.jsonl")}
    p=RUN/"factorial_history"/name/"cells.jsonl.gz"
    return {r["id"]:{s:r["models"][f"C{s}"]["marg12"] for s in SEEDS if f"C{s}" in r["models"]} for r in read_rows(p)}


def run():
    paths=cohort_inputs(NAMES)
    for src in ("company","Dept2"):
        path=folder_for(src,"temporal")
        paths.extend([path/"scale.npz",*(path/f"add{s}.pt" for s in SEEDS)])
    lock(F,dict(networks=NAMES,model="frozen full additive; five seeds on every network, Dept2 transfer except local company",
         population="archived main eligible events; GNU libc original 800 flat draws/month, archive seed; all 24 months",
         budget="min(n,K); proportional budgets ceil(p*n); baseline Both-in-12",
         ranking="pair reciprocal rank; additive float64, tie by lexicographic sorted per-event tie indices",
         deltas="paired budget minus 12; Any and MRR reported separately; bootstrap with four windows is descriptive"),paths+[Path(__file__),Path(__file__).with_name("closure_common.py")])
    summary={}
    for name in NAMES:
        models=nets(source(name),["full"])["full"]
        cache=anchors(name)
        rows=[]
        keys=[]
        matched=0
        for g,raw,items,_ in cohort(name):
            keys.append(g["window"])
            if not items:
                continue
            h=PublicWindow(g,raw)
            for e,gold in items:
                (nodes,_,_),(x,_,_)=h.design(frames(e))
                tie=fg.tie_order(e["id"],len(nodes))
                a,b=map(nodes.index,gold)
                rs,pr=[],[]
                for seed,score in zip(SEEDS,logits(models,x)):
                    order=fg.sorted_nodes(score,tie)
                    ranks2=np.argsort(order)[[a,b]]+1
                    rs.append(ranks2.tolist())
                    pr.append(int(pairrank(score,tie,a,b)))
                    if seed in cache.get(e["id"],{}):
                        assert [nodes[i] for i in order[:12]]==cache[e["id"]][seed]
                        matched+=1
                rows.append(dict(id=e["id"],window=e["window"],n=len(nodes),ranks=rs,pair_rank=pr,scores=metrics(len(nodes),rs,pr)))
        result=agg(rows,keys,"Both-in-12")
        result.update(mean_candidates=np.mean([r["n"] for r in rows]),archived_lists_matched=matched,
                      mean_budget={str(p):np.mean([math.ceil(r["n"]*p/100) for r in rows]) for p in (5,10)})
        summary[name]=result
        save_rows(F/(name+".jsonl.gz"),rows)
        print(name,len(rows),result["methods"]["Both-in-12"]["mean"],flush=True)
    dump(F/"summary.json",summary)
    lines=["|Network|Events|Mean candidates|"+"|".join(M)+"|","|---|---:|---:|"+"---:|"*len(M)]
    fig,ax=plt.subplots(figsize=(7.4,4.5))
    for name,s in summary.items():
        vals=[s["methods"][m]["mean"] for m in M]
        lines.append(f'|{name}|{s["n"]}|{s["mean_candidates"]:.1f}|'+ '|'.join([f"{100*v:.2f}%" for v in vals[:-1]]+[f"{vals[-1]:.5f}"])+"|")
        ax.plot([6,12,24],np.array(vals[:3])*100,"o-",label=name)
    (F/"table.md").write_text("\n".join(lines)+"\n\nProportional budgets are rounded up per event. Pair-MRR is unitless. Paired intervals are in summary.json.\n",encoding="utf8")
    ax.set(xlabel="List size K",ylabel="Both sources retained (%)",xticks=[6,12,24])
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(F/"budget_curve.png",dpi=160)
    plt.close(fig)


def verify():
    summary=load(F/"summary.json")
    for name,s in summary.items():
        rows=read_rows(F/(name+".jsonl.gz"))
        audit_agg(rows,s)
        for r in rows:
            assert all(1<=p<=r["n"]*(r["n"]-1)//2 for p in r["pair_rank"])
            for j,(a,b) in enumerate(r["ranks"]):
                for m,k in zip(M[:5],[6,12,24,(r["n"]+19)//20,(r["n"]+9)//10]):
                    assert r["scores"][m][j]==int(a<=k and b<=k)
                assert r["scores"][M[5]][j]==int(min(a,b)<=12)
                assert np.isclose(r["scores"][M[6]][j],1/r["pair_rank"][j])
        expect=s["n"]*(1 if name=="CollegeMsg" else 5) if name!="GNU libc" else 0
        assert s["archived_lists_matched"]==expect
    rng=np.random.default_rng(8271)
    for _ in range(100):
        x=rng.integers(-2,3,9).astype(float)
        t=rng.permutation(9)
        pairs=sorted(itertools.combinations(range(9),2),key=lambda ab:(-sum(x[list(ab)]),*sorted(t[list(ab)])))
        assert all(pairrank(x,t,*ab)==i+1 for i,ab in enumerate(pairs))
    evidence(F,dict(events=sum(s["n"] for s in summary.values()),archive_top12=sum(s["archived_lists_matched"] for s in summary.values()),
                    brute_force_pair_ranks=3600,all_event_metrics_and_paired_means=True))


if __name__=="__main__":
    if "--verify" not in sys.argv:
        run()
    verify()


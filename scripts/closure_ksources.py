"""WP4. Fixed and hidden source counts, plus exact full-trace silent-set posterior."""
import itertools
import random
from scipy.special import softmax
from closure_common import *
from closure_cascades import Scorer,check_lists,hits,mask,pick,score_lists,silent_count,simulate,test_windows
from silent_source_oracle import CompleteTrace

F=BASE/"wp4_ksources"
DRAWS=160
ARMS=["three_uniform","three_active","hidden_uniform"]
METHODS=["full","static","volume"]


def posterior(nodes,stream,trace,known,k,prior):
    infinity=float("inf")
    when={v:-infinity for v in known}
    for t in trace:
        when.update({v:t["t"] for v,_ in t["joins"]})
    legal=sorted(set(nodes)-when.keys())
    H=Counter(); B=Counter(); internal=Counter()
    for t,batch in stream:
        for u,v in batch:
            if t<=when.get(v,infinity):
                H[u]+=1
                if when.get(u,infinity)<t:
                    B[v]+=1
            internal[u,v]+=1
    subsets=list(itertools.combinations(legal,k-len(known)))
    delta=[sum(H[u]-B[u] for u in ss)-sum(internal[u,v] for u in ss for v in ss) for ss in subsets]
    logmass=[math.log(prior(tuple(sorted([*known,*ss]))))+d*math.log(.7) for ss,d in zip(subsets,delta)]
    return subsets,softmax(logmass)


def oracle_check():
    # Explicit likelihood considers every hypothesized source set, including incompatible ones.
    nodes=list(range(7))
    stream=[(1,[(0,3),(1,2),(2,1),(4,3)]),(2,[(3,4),(0,4),(2,5),(6,3)]),(3,[(4,5),(1,6),(2,6),(5,6)])]
    maxerr=0.; cases=0; coverage=set(); old=0
    for k in (1,2,3):
        for original in itertools.combinations(nodes,k):
            for seed in range(6):
                tr=simulate(stream,original,seed)
                incoming={v for t in tr for v,_ in t["joins"]}
                known=sorted({u for t in tr for v,pp in t["joins"] for u in pp}-incoming)
                if len(known)>=k:
                    continue
                outcomes={t["t"]:{(u,v) for v,pp in t["joins"] for u in pp} for t in tr}
                for prior in (lambda ss:1.,lambda ss:float(math.prod(v+1 for v in ss))):
                    subs,p=posterior(nodes,stream,tr,known,k,prior)
                    ref={}
                    for src in itertools.combinations(nodes,k):
                        state=set(src)
                        prob=prior(src)
                        for t,batch in stream:
                            eligible={(u,v) for u,v in batch if u in state and v not in state}
                            success=outcomes.get(t,set())
                            if not success<=eligible:
                                prob=0.;break
                            prob*=.3**len(success)*.7**len(eligible-success)
                            state.update(v for _,v in success)
                        ref[src]=prob
                    total=sum(ref.values())
                    vec=[ref[tuple(sorted([*known,*ss]))]/total for ss in subs]
                    maxerr=max(maxerr,float(np.max(np.abs(p-vec))),abs(sum(vec)-1))
                    cases+=1
                if k==2 and len(known)==1:
                    world=CompleteTrace(dict(nodes=nodes),stream)
                    a,others,oldp=world.evaluate(dict(sources=list(original),trajectory=tr),return_distribution=True)
                    subs,p=posterior(nodes,stream,tr,known,k,lambda ss:1.)
                    assert a==known[0] and others==[ss[0] for ss in subs]
                    maxerr=max(maxerr,float(np.max(abs(oldp-p))))
                    old+=1
                coverage.add((k,len(known)))
    assert maxerr<1e-12 and old and (3,1) in coverage and (3,2) in coverage and (3,0) in coverage
    return dict(cases=cases,k2_legacy=old,max_abs_error=maxerr,k_m=sorted(coverage))


def generate(name):
    raw,gs=test_windows(name)
    models=nets("Dept2",["full","static"])
    rows=[]
    for j,(g,stream) in enumerate(gs):
        scorer=Scorer(g,raw,models)
        w=activity(g,raw)
        for ai,arm in enumerate(ARMS):
            rng=random.Random(913170+1000*int(name=="Enron")+10*j+ai)
            for i in range(DRAWS):
                k=rng.randint(1,3) if arm=="hidden_uniform" else 3
                src=sorted(pick(g["nodes"],w,k,rng) if arm=="three_active" else rng.sample(g["nodes"],k))
                seed=rng.getrandbits(63)
                tr=simulate(stream,src,seed)
                obs=mask(tr)
                if obs is None:
                    continue
                mid=f'{g["window"]}-{ai}-{i}'
                lists=score_lists(*scorer.predict(mid,g["window"],obs[1]))
                rows.append(dict(id=mid,window=g["window"],arm=arm,k=k,seed=seed,sources=src,silent=silent_count(src,tr),
                                 observed=obs[1],lists=lists))
    keys=[g["window"] for g,_ in gs]
    result={}
    for arm in ARMS:
        subset=[r for r in rows if r["arm"]==arm]
        for kk in (["all",1,2,3] if arm=="hidden_uniform" else ["all"]):
            rr=[r for r in subset if kk=="all" or r["k"]==kk]
            entry={}
            for metric in ("All-in-12","Recall@12"):
                tmp=[dict(id=r["id"],window=r["window"],scores=hits(r["lists"],r["sources"],metric=="Recall@12")) for r in rr]
                entry[metric]=agg(tmp,keys,"static")
                for r,t in zip(rr,tmp):
                    r[metric]=t
            entry["silent_event"]=interval([r["silent"]>0 for r in rr],[r["window"] for r in rr],keys)
            result[arm+"/"+str(kk)]=entry
    save_rows(F/(name+".jsonl.gz"),rows)
    return result


def run():
    lock(F,dict(networks=["Dept2","Enron"],windows="same WP3B 4/8 calendar windows",draws_per_window_per_arm=DRAWS,seed=913170,
         arms=ARMS,k_hidden="iid discrete uniform per attempted event; actual accepted distribution reported; no k or gold at inference",
         laws="fixed three uniform or product of w=28d sends+1 over distinct triples; unknown k uses uniform distinct nodes",
         oracle="exact labelled-parent trajectory, simultaneous SI. P(S|T,A) proportional to prior(A union S) * .7^[sum_{u in S}(H_u-B_u) - C(S,S)]; H counts outgoing to baseline susceptible, B counts baseline-active incoming, C counts contacts internal to silent set."),[Path(__file__),Path(__file__).with_name("closure_cascades.py"),BASE/"wp3_robustness/protocol.json"])
    dump(F/"summary.json",{n:generate(n) for n in ("Dept2","Enron")})


def verify():
    total=0
    for name,d in load(F/"summary.json").items():
        rows=read_rows(F/(name+".jsonl.gz"))
        total+=len(rows)
        _,gs=test_windows(name)
        streams={g["window"]:stream for g,stream in gs}
        for r in rows:
            tr=simulate(streams[r["window"]],r["sources"],r["seed"])
            assert mask(tr)[1]==r["observed"] and silent_count(r["sources"],tr)==r["silent"]
            check_lists(r)
            for metric in ("All-in-12","Recall@12"):
                assert hits(r["lists"],r["sources"],metric=="Recall@12")==r[metric]["scores"]
        for key,s in d.items():
            arm,k=key.split("/")
            part=[r for r in rows if r["arm"]==arm and (k=="all" or r["k"]==int(k))]
            for m in ("All-in-12","Recall@12"):
                audit_agg([r[m] for r in part],s[m])
            assert np.isclose(np.mean([r["silent"]>0 for r in part]),s["silent_event"]["mean"])
    evidence(F,dict(events=total,public_lists_and_trajectories=True,posterior=oracle_check()))


if __name__=="__main__":
    if "--verify" not in sys.argv:
        run()
    verify()


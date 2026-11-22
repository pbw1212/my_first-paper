"""Check three event-level oracle intersections by direct synchronous forward replay."""
from itertools import combinations, groupby

from diagnose_source_equivalence import complete_world
from masked_pair_benchmark import OUT as PREVIOUS, PLAN as PREVIOUS_PLAN, mask, roles
from partial_source_identification import NEW, OUT, company, load, load_events, make_windows, read_rows, write_json


def enumerate_pairs(g, stream, o, y):
    nodes=g["nodes"]
    index={v:j for j,v in enumerate(nodes)}
    times,senders,receivers=complete_world(stream,index,y)
    contacts=[(int(t),[(nodes[a],nodes[b]) for _,a,b in edges])
              for t,edges in groupby(zip(times,senders,receivers),key=lambda row:row[0])]
    legal=sorted(set(nodes)-{v for batch in o["observed"] for v in batch})
    accepted=[]
    for a,b in combinations(legal,2):
        active={a,b}
        trace=[]
        for t,edges in contacts:
            parents={}
            for u,v in edges:
                if u in active and v not in active:
                    parents.setdefault(v,[]).append(u)
            if parents:
                active.update(parents)
                trace.append(dict(t=t,joins=[[v,parents[v]] for v in sorted(parents)]))
        cut=mask(trace)
        if cut and cut[1]==o["observed"]:
            label=dict(sources=[a,b],trajectory=trace,cut_batches=cut[0])
            if len(roles(label)[1])==2:
                accepted.append([a,b])
    assert y["sources"] in accepted
    return dict(legal=len(legal),checked=len(legal)*(len(legal)-1)//2,pairs=accepted,
                required=sorted(set.intersection(*(set(p) for p in accepted))),sources=y["sources"])


def main():
    result={}
    for name,old in (("company",True),("company",False),("Dept2",False)):
        folder=PREVIOUS if old else OUT/name
        obs=read_rows(folder/("observations.jsonl" if old else "public.jsonl"))
        labs={y["id"]:y for y in read_rows(folder/"truth.jsonl")}
        if old:
            o=next(o for o in obs if o["id"]=="company-08-33")
        else:
            periods=load(folder/"windows.json")
            o=next(o for o in obs if periods[o["window"]]["stage"]=="test" and labs[o["id"]]["dual"])
        raw=company() if name=="company" else load_events(NEW)
        gs,streams=make_windows(raw,name,*(PREVIOUS_PLAN["span"] if old else [.15,1.]))
        key=o["window"]
        r=enumerate_pairs(next(g for g in gs if g["window"]==key),streams[key],o,labs[o["id"]])
        if not old:
            target=load(folder/"oracle.json")[o["id"]]["dual"]
            assert (r["required"],len(r["pairs"]))==(target["members"],target["pairs"])
        result[("previous/" if old else "current/")+o["id"]]=r
    write_json(OUT/"event_replay_check.json",result)
    print(result)


if __name__=="__main__":
    main()

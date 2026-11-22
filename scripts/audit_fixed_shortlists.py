"""Recompute fixed-budget outcomes and paired disagreements from saved event files."""
from collections import Counter
import hashlib

from partial_source_identification import NEW, OUT, company, load, load_events, make_windows, read_rows, write_json
from transfer_joint_shortlist import FILES


def audit(name):
    local=name in ("company","Dept2")
    folder=OUT/(name if local else "transfer_"+name)
    prefix="joint12_" if local else ""
    public=read_rows(folder/(prefix+"public.jsonl"))
    truth=read_rows(folder/(prefix+"truth.jsonl"))
    base=read_rows(folder/(prefix+"predictions.jsonl"))
    linear=read_rows(folder/"linear_predictions.jsonl")
    raw=company() if name=="company" else load_events(NEW if name=="Dept2" else OUT.parents[1]/"data/raw"/FILES[name])
    roster={g["window"]:set(g["nodes"]) for g in make_windows(raw,name,.15,1.)[0]}
    summaries=[load(folder/(prefix+"summary.json")),load(folder/"linear_summary.json")]
    if not local:
        protocol=load(folder/"protocol.json")
        for source,expected in protocol["weights"].items():
            assert hashlib.sha256((OUT/source/"models.joblib").read_bytes()).hexdigest()==expected
    result={}
    assert len({r["id"] for r in public})==len(public)
    for o,y,p,q in zip(public,truth,base,linear,strict=True):
        assert o["id"]==y["id"]==p["id"]==q["id"]
        assert o["window"]==y["window"]==p["window"]==q["window"]
        legal=roster[o["window"]]-{n for b in o["observed"] for n in b}
        sources=set(y["sources"])
        assert len(sources)==2 and sources<=legal
        all_sets=dict(**p["sets"],**q["sets"])
        hits={}
        for method,candidates in all_sets.items():
            assert len(candidates)==len(set(candidates))==12 and set(candidates)<=legal
            hits[method]=len(sources.intersection(candidates))
        for group in ("dual","ordinary"):
            if not y[group]:
                continue
            for key in ("total",o["window"]):
                row=result.setdefault(group,{}).setdefault(key,Counter())
                row["n"]+=1
                for tag,count in hits.items():
                    row[tag+"_one"]+=int(count>=1)
                    row[tag+"_both"]+=int(count==2)
                main="joint12" if local else "Dept2_joint"
                comparator="source" if local else "Dept2_source"
                for tag in (comparator,"reach","relative"):
                    row[main+"_wins_over_"+tag]+=int(hits[main]==2 and hits[tag]<2)
                    row[main+"_losses_to_"+tag]+=int(hits[main]<2 and hits[tag]==2)
    for summary in summaries:
        for group in ("dual","ordinary"):
            for key,values in summary[group].items():
                for tag,value in values.items():
                    if tag=="n" or tag.endswith(("_one","_both")) and tag!="expected_random_both":
                        assert result[group][key][tag]==value
    return result


if __name__=="__main__":
    result={name:audit(name) for name in ("company","Dept2","Dept1","CollegeMsg")}
    write_json(OUT/"fixed_budget_audit.json",result)
    print("Verified every saved 12-person shortlist, population, linked observation and outcome in all four networks.")

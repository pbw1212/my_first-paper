"""Generate six tables from the recorded feature design, sealed summaries and checked inventory."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"runs"
BASE = RUN/"paper_closure"
PAPER = ROOT/"paper_dasfaa"
NAMES = {"company":"Company","python-dev":"Python-Dev","python-ideas":"Python-Ideas",
         "r-devel":"R-devel","r-package-devel":"R-package-devel","libc-alpha":"GNU libc",
         "libc-alpha-2024":"GNU libc","samba-technical-2020":"Samba"}


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def table(name, cols, head, rows):
    lines=[r"\begin{tabular}{"+cols+"}",r"\toprule",head,r"\midrule"]
    for row in rows:
        lines.append(row if isinstance(row,str) else " & ".join(map(str,row))+r" \\")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    folder=PAPER/"tables";folder.mkdir(exist_ok=True)
    (folder/(name+".tex")).write_text("\n".join(lines)+"\n",encoding="utf8")


def rate(d):
    return f"{100*d['mean']:.2f}"


def ci(d):
    v=100*d["mean"];lo,hi=[100*x for x in d["ci95"]]
    return rf"\({v:+.2f}\;[{lo:.2f},{hi:.2f}]\)"


def features():
    tree=ast.parse((ROOT/"scripts/recent_member_evidence.py").read_text(encoding="utf8"))
    fields={}
    for stmt in tree.body:
        if isinstance(stmt,ast.Assign) and isinstance(stmt.targets[0],ast.Name) and stmt.targets[0].id in ("PERSON","LINKS","CONTEXT"):
            fields[stmt.targets[0].id]=eval(compile(ast.Expression(stmt.value),"feature masks","eval"),
                                            {"__builtins__":{},"range":range})
    p=fields["PERSON"];links=fields["LINKS"];ctx=fields["CONTEXT"]
    assert (p,links,ctx)==([*range(6),*range(19,23)],[12,23,24,25],[17,18,58,59,60,63])
    count1=len(set(p+[i+29 for i in p]+ctx));count2=count1+2*len(links)
    assert (count1,count2)==(26,34)
    labels=read(RUN/"dept3_candidates/models.json")["models"]["full100"]["features"]
    assert len(labels)==13 and labels[12]=="recent_visible_contacts_per_member"
    checks=read(BASE/"wp3_robustness/protocol.json")["plan"]["objective"]
    assert "64 columns" in checks
    table("tab1_features",r"@{}>{\raggedright\arraybackslash}p{1.64cm}r>{\raggedright\arraybackslash}p{8.53cm}@{}", r"Tier & Active & Columns and definitions (zero-based indexing) \\",
          [["Personal",count1, r"0--5: log counts (out-/in-degree, sent/received, 28-day sent/received). 19--22: fraction of quiet fortnights, mean \(0.7^{\rm sends}\), log outgoing SD, log latest sends. Each has a rank copy. Shared context: 17, 18, 58--60, 63."],
           ["+ Recent contacts",count2, r"12: log recent outgoing count per visible member. 23--25: log recency-weighted outgoing mean, incoming mean and incoming maximum. Add the four rank copies."],
           ["+ Full temporal history",64, r"6--11: reachability and reciprocal distances to all/earliest visible nodes, plus direct-neighbor fraction. 13--16: mean/max support, earliest mean support and total replay reach. 26--28: residual-weighted mean and maximum, then fraction of unweighted supports above 0.5. Add remaining ranks and context (61--62)."]])


def main():
    for path,h in read(BASE/"wp6_tables/protocol.json")["inputs"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
    features()
    budget=read(BASE/"wp1_budgets/summary.json")
    diag=read(BASE/"wp2_predictability/summary.json")["networks"]
    inv=read(PAPER/"evidence/inventory.json")
    rows=[r"\multicolumn{6}{@{}l}{\it Simulated cascades on recorded contact streams} \\"]
    for n,z in budget.items():
        v=inv["sim"][n]
        rows.append([NAMES.get(n,n),f"{v['nodes']:,}",f"{v['records']:,}",f"{v['days']:,.1f} d",len(z["keys"]),f"{z['n']:,}"])
    rows.extend([r"\midrule",r"\multicolumn{6}{@{}l}{\it Observed threads, D0 eligible before roster projection} \\"])
    bridge=read(BASE/"wp5_sim2real/summary.json")
    for n,v in inv["real"].items():
        if n.endswith("_late"):
            name=n[:-5]; z=bridge["real"][name]["0"]["all"]
        else:
            name=n;z=diag["real "+n]
        span=v["start"][:4]+"--"+v["end"][:4] if v["start"][:4]!=v["end"][:4] else v["start"][:4]
        if n.endswith("_late"):span="Feb--Apr 2019"
        rows.append([NAMES[name],f"{v['nodes']:,}",f"{v['records']:,}",span,len(z["keys"]),f"{z['n']:,}"])
    table("tab2_data","@{}lrrlrr@{}",r"Network or list & Nodes & Records & Span & Windows & Eligible \\",rows)
    hist=read(RUN/"factorial_history/summary.json")
    h4=read(RUN/"history_components/summary.json")["Dept4"]["methods"]
    tree=read(BASE/"wp3_robustness/summary.json")
    rows=[]
    for n,z in budget.items():
        static=fg=fulltree="---"
        if n in hist:
            by=hist[n]["groups"]["all"];assert all(d["n"]==z["n"] for d in by.values())
            def historical(cell):
                return sum(d[cell]["m12"]/d["n"] for d in by.values())*100/len(by)
            static=f"{historical('A'):.2f}";fg=f"{historical('F'):.2f}"
            if n=="CollegeMsg":
                static+=r"\(^*\)";fg+=r"\(^*\)"
        if n=="Dept4":
            static=f"{h4['A']['both_pct']:.2f}"
        if n in tree["tree"]:
            src=tree["tree"][n]["12"];assert src["n"]==z["n"]
            fg=rate(src["methods"]["FGSSI"]);fulltree=rate(src["methods"]["full"])
        m=z["methods"]
        rows.append([NAMES.get(n,n),f"{z['mean_candidates']:.1f}",rate(m["Any-in-12"]),fg,static,fulltree,
                     rate(m["Both-in-12"]),rate(m["Both-in-5%"]),f"{m['Pair-MRR']['mean']:.5f}"])
    table("tab3_main","@{}lrrrrrrrr@{}",
          r" & & & \multicolumn{4}{c}{Both@12 (\%)} & Both@5\% & Pair \\"+"\n"+
          r"\cmidrule(lr){4-7}Network & Pool & Any@12 & FGSSI & Static & Tree & MLP & MLP & MRR \\",rows)
    rows=[]
    for n,arms in tree["stress"].items():
        for i,(k,s) in enumerate(arms.items()):
            a,b=s["all"],s["common"]
            rows.append([n if i==0 else "",{"SI":"SI","SIR48":"SIR48","SIR72":"SIR72","noise":"External"}[k],
                         a["n"],ci(a["differences"]["links minus person"]),b["n"],ci(b["differences"]["links minus person"])])
    table("tab4_rules","@{}llrlrl@{}",r"Network & Rule & All \(n\) & \mbox{Gain [95\% CI]} & Joint \(n\) & \mbox{Gain [95\% CI]} \\",rows)
    held=read(RUN/"heldout_r_lists/early_only/role_evidence/closure/summary.json")
    assert sum(z["n"] for z in held.values())==1793
    def community(n):
        return {"libc-alpha":"GNU libc 2021--22","libc-alpha-2024":"GNU libc 2024","samba-technical-2020":"Samba 2020"}.get(n,NAMES[n])
    rows=[]
    for n,z in held.items():
        def hit(m):
            hits=z["hit_counts"][m];assert len(set(hits))==1 and abs(hits[0]*100/z["n"]-z["both_pct"][m])<1e-9
            return str(hits[0])
        rows.append([community(n),z["n"],z["scored_reachable"],*[hit(m) for m in ["volume","roles","reply","role_suffix"]]])
    table("tab5_heldout","@{}lrrrrrr@{}",r"List/period & Eligible & Reachable & Volume & Roles & Replies & Full \\",rows)
    rows=[]
    for n,ds in bridge["real"].items():
        v=[ds[d]["common"] for d in ["0","25","50"]];assert len({z["n"] for z in v})==1
        rows.append([community(n),v[0]["n"],*[rf"\({100*z['differences']['full minus volume']['mean']:+.2f}\)" for z in v]])
    table("tab6_depth","@{}lrrrr@{}",r"List/period & Common threads & D0 gain & D25 gain & D50 gain \\",rows)


if __name__=="__main__":
    main()

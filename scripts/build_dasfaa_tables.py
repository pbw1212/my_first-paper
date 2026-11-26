"""Generate six tables from the recorded feature design, sealed summaries and checked inventory."""
import gzip
from statistics import mean
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


SINGLE_FIT = r"\singlefitstar"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def table(name, cols, head, rows, *, setup=""):
    lines = [setup] if setup else []
    lines.extend([r"\begin{tabular}{"+cols+"}", r"\toprule", head, r"\midrule"])
    for row in rows:
        lines.append(row if isinstance(row,str) else " & ".join(map(str,row))+r" \\")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    folder=PAPER/"tables";folder.mkdir(exist_ok=True)
    (folder/(name+".tex")).write_text("\n".join(lines)+"\n",encoding="utf8")


def header(first, *labels):
    """Center the first heading while preserving the flush-left text column."""
    return " & ".join([rf"\multicolumn{{1}}{{@{{}}c}}{{{first}}}", *labels]) + r" \\"


def rate(d):
    return f"{100*d['mean']:.2f}"


def ci(d):
    v=100*d["mean"];lo,hi=[100*x for x in d["ci95"]]
    return rf"${v:+.2f}\;[{lo:.2f},\,{hi:.2f}]$"


def mark_coverage(values):
    """Mark the two highest distinct displayed values, including ties."""
    ordered = sorted({float(value.removesuffix(SINGLE_FIT)) for value in values}, reverse=True)
    marks = dict(zip(ordered, (r"\textbf", r"\underline")))
    result = []
    for value in values:
        number, star, _ = value.partition(SINGLE_FIT)
        mark = marks.get(float(number))
        result.append((f"{mark}{{{number}}}" if mark else number) + star)
    return result


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
    table("tab1_features", r"@{}>{\raggedright\arraybackslash}p{1.85cm}c>{\raggedright\arraybackslash}p{8.8cm}@{}",
          r"Tier & Active & Descriptors and shared event context \\",
          [["Personal", count1,
            r"\textbf{Activity (10):} log out-/in-degree, log total and 28-day sent/received counts, quiet-fortnight fraction, mean estimated no-success probability $0.7^{c_\ell}$ over up to six complete fortnights ($c_\ell$ counts sends by the candidate in fortnight $\ell$), log outgoing SD and log most recent fortnight's sends. "
            r"\textbf{Visibility variables (2):} visible count and roster share. \textbf{Shared summaries (4):} log visible and block counts, first-block share and log legal-pool size."],
           ["+ Recent contacts", count2,
            r"\textbf{Direct ties to suffix (4):} log 28-day outgoing count per visible member and log recency-weighted ($0.75^\ell$) mean outgoing, mean incoming and maximum incoming counts."],
           ["+ Full temporal history", 64,
            r"\textbf{Multi-hop (6):} For all visible members, the reachable fraction, mean reciprocal distance and direct-neighbor fraction. For the first block, the reachable fraction and the mean and maximum reciprocal distances. "
            r"\textbf{Replay (7):} mean and maximum support, mean support in the first block, total replay reach, residual-weighted mean and maximum support, and fraction of supports $>0.5$. "
            r"\textbf{Replay summaries (2):} mean support and maximum candidate mean support. The full tier also includes two tied visibility ranks."]])


def main():
    for path,h in read(BASE/"wp6_tables/protocol.json")["inputs"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
    features()
    budget=read(BASE/"wp1_budgets/summary.json")
    diag=read(BASE/"wp2_predictability/summary.json")["networks"]
    inv=read(PAPER/"evidence/inventory.json")
    rows=[r"\multicolumn{7}{@{}c@{}}{\textit{Simulated cascades}} \\"]
    for n,z in budget.items():
        v=inv["sim"][n]
        rows.append([NAMES.get(n,n),f"{v['nodes']:,}",f"{v['records']:,}",f"{v['days']:,.1f} d","1,000 / 300" if n in ("Dept2","company") else "Dept2",len(z["keys"]),f"{z['n']:,}"])
    v=inv["sim"]["Dept1"]
    z=diag["sim Dept1"]
    rows.append(["Dept1 (diagnostic)",v["nodes"],f"{v['records']:,}",f"{v['days']:.1f} d","Dept2",len(z["keys"]),z["n"]])
    rows.extend([r"\midrule",r"\multicolumn{7}{@{}c@{}}{\textit{Observed threads before roster projection}} \\",header("List", "Addresses", "Posts", r"\multicolumn{2}{c}{Period}", "Months", "Eligible")])
    bridge=read(BASE/"wp5_sim2real/summary.json")
    for n,v in inv["real"].items():
        if n.endswith("_late"):
            name = n[:-5]
            z = bridge["real"][name]["0"]["all"]
            span = "Feb--Apr 2019"
        else:
            name = n
            z = diag["real " + n]
            start_year, end_year = v["start"][:4], v["end"][:4]
            span = start_year
            if end_year != start_year:
                end_label = end_year[2:] if end_year[:2] == start_year[:2] else end_year
                span = f"{start_year}--{end_label}"
        rows.append([NAMES[name],f"{v['nodes']:,}",f"{v['records']:,}",rf"\multicolumn{{2}}{{c}}{{{span}}}",len(z["keys"]),f"{z['n']:,}"])
    table("tab2_data","@{}lcccccc@{}",header("Network", "Nodes", "Contacts", "Span", "Fit / tune", "Windows", "Test"),rows)
    gc=read(RUN/"dasfaa_gcnsi_transfer/summary.json")
    extra=read(PAPER/"evidence/top12.json")
    completed=read(RUN/"dasfaa_baseline_completion/summary.json")["results"]
    hist=read(RUN/"factorial_history/summary.json")
    h4=read(RUN/"history_components/summary.json")["Dept4"]["methods"]
    tree=read(BASE/"wp3_robustness/summary.json")
    models = ("Random", "GCNSI", "FGSSI", "Static", "Tree", "MLP")
    fit_marks = {}
    rows=[]
    for n,z in budget.items():
        static=fg=fulltree=None
        if n in hist:
            by=hist[n]["groups"]["all"];assert all(d["n"]==z["n"] for d in by.values())
            def historical(cell):
                return sum(d[cell]["m12"]/d["n"] for d in by.values())*100/len(by)
            static=f"{historical('A'):.2f}";fg=f"{historical('F'):.2f}"
            if len(by) == 1:
                static += SINGLE_FIT
                fg += SINGLE_FIT
        if n=="Dept4":
            static=f"{h4['A']['both_pct']:.2f}"
        if n in tree["tree"]:
            src=tree["tree"][n]["12"];assert src["n"]==z["n"]
            fg=rate(src["methods"]["FGSSI"]);fulltree=rate(src["methods"]["full"])
        if n == "CollegeMsg":
            fulltree = f"{completed[n]['both']:.2f}"
        if n == "GNU libc":
            static = f"{completed[n]['both']:.2f}"
        m=z["methods"]
        with gzip.open(BASE/"wp1_budgets"/(n+".jsonl.gz"),"rt") as records:
            pools=[json.loads(line)["n"] for line in records]
        assert len(pools)==z["n"]
        chance=100*mean(min(12,k)*(min(12,k)-1)/(k*(k-1)) for k in pools)
        lo,hi=m["Both-in-12"]["ci95"]
        cohort=gc[n]
        assert cohort["n"]==z["n"]
        gc_rate = f"{100*cohort['both']/cohort['n']:.2f}" + SINGLE_FIT
        values = [f"{chance:.3f}", gc_rate, fg, static, fulltree, rate(m["Both-in-12"])]
        fit_marks[n] = {method for method, value in zip(models, values) if value.endswith(SINGLE_FIT)}
        rows.append([NAMES.get(n,n),values[0],*mark_coverage(values[1:]),
                     f"[{100*lo:.2f}, {100*hi:.2f}]",f"{m['Pair-MRR']['mean']:.5f}"])
    rows.extend([r"\midrule", r"\multicolumn{9}{@{}c@{}}{\anytwelve} \\",
                 header("Network", *models, "MLP recall", "Mean pool"), r"\midrule"])
    for n, rec in extra.items():
        coverage=rec["coverage_pct"]
        values = [f"{coverage[method]['any']:.2f}" + (SINGLE_FIT if method in fit_marks[n] else "")
                  for method in models]
        rows.append([NAMES.get(n,n),values[0],*mark_coverage(values[1:]),f"{coverage['MLP']['recall']:.2f}",f"{rec['pool']:,.1f}"])
    table("tab3_main","@{}lcccccccc@{}",
          r"\multicolumn{9}{@{}c@{}}{\both} \\" + "\n" +
          header("Network", *models, r"95\% CI", "MRR"), rows,
          setup=r"\newcommand{\singlefitstar}{\raisebox{.6ex}[0pt][0pt]{\footnotesize*}}")
    rows=[]
    for n,arms in tree["stress"].items():
        for i,(k,s) in enumerate(arms.items()):
            a,b=s["all"],s["common"]
            rows.append([n if i==0 else "",{"SI":"SI","SIR48":"SIR48","SIR72":"SIR72","noise":"External"}[k],
                         a["n"],ci(a["differences"]["links minus person"]),b["n"],ci(b["differences"]["links minus person"])])
    table("tab4_rules","@{}llcccc@{}",
          header("Network", r"\multicolumn{1}{c}{Rule}", r"All $N_{\mathrm{ev}}$", r"\mbox{Gain [95\% CI]}",
                 r"Joint $N_{\mathrm{ev}}$", r"\mbox{Gain [95\% CI]}"), rows)
    held=read(RUN/"heldout_r_lists/early_only/role_evidence/closure/summary.json")
    assert sum(z["n"] for z in held.values())==1793
    def community(n):
        if n.startswith("python-"):
            period = "Feb--Apr 2019"
        else:
            dates = inv["real"][n]
            start, end = dates["start"][:4], dates["end"][:4]
            period = f"{start}--{end[2:]}" if start != end else start
        return f"{NAMES[n]} {period}"
    rows=[]
    for n,z in held.items():
        def hit(m):
            hits=z["hit_counts"][m];assert len(set(hits))==1 and abs(hits[0]*100/z["n"]-z["both_pct"][m])<1e-9
            return str(hits[0])
        contrast=z["contrasts"]["role_suffix-volume"]
        lo,hi=contrast["ci95_pp"]
        delta=rf"${contrast['pp']:+.2f}\;[{lo:.2f},\,{hi:.2f}]$"
        if lo > 0:
            delta = r"{\boldmath " + delta + "}"
        hits = [hit(method) for method in ("volume", "roles", "reply", "role_suffix")]
        full_hits = int(hits[-1])
        rows.append([community(n), z["n"], z["scored_reachable"], *hits,
                     f"{100*full_hits/z['n']:.2f}", f"{100*full_hits/z['scored_reachable']:.2f}", delta])
    table("tab5_heldout", r"@{}l@{\hspace{3pt}}cccccccc@{\hspace{3.6pt}}c@{}",
          r" & \multicolumn{2}{c}{Threads} & \multicolumn{4}{c}{Joint hits} & \multicolumn{2}{c}{Role-Full (\%)} & Role-Full $-$ Posts \\" + "\n" +
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}\cmidrule(lr){8-9}" + "\n" +
          header("List/period", "Elig.", "Reach.", "Posts", "Roles", "Replies", "Full",
                 "Elig.", "Reach.", r"pp [95\% CI]"), rows,
          setup=r"\fontsize{8.5}{10.5}\selectfont")
    rows=[]
    for n,ds in bridge["real"].items():
        v=[ds[d]["common"] for d in ["0","25","50"]];assert len({z["n"] for z in v})==1
        gains = [100 * level["differences"]["full minus volume"]["mean"] for level in v]
        signed = [f"{gain:+.2f}" if gain != 0 else "0.00" for gain in gains]
        rows.append([community(n), v[0]["n"], "/".join(str(z["n"]-z["scored"]) for z in v),
                     *[f"${gain}$" for gain in signed]])
    table("tab6_depth","@{}lccccc@{}",
          header("List/period", "Common", "Unscored", r"$D_0$ gain", r"$D_{25}$ gain", r"$D_{50}$ gain"), rows)


if __name__=="__main__":
    main()

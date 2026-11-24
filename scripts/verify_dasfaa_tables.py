"""Check the six typeset tables against the experiment records at publication precision."""
import ast
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_dasfaa"
RUN = ROOT / "runs"
CLOSURE = RUN / "paper_closure"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def summary(prefix):
    return read(next(CLOSURE.glob(prefix + "_*/summary.json")))


def table(number):
    text = next((PAPER / "tables").glob(f"tab{number}_*.tex")).read_text(encoding="utf8")
    body = text.split(r"\midrule", 1)[1].split(r"\bottomrule")[0]
    return [[cell.strip() for cell in line.removesuffix(r"\\").split("&")] for line in body.splitlines()
            if " & " in line ]


def feature_index():
    """Read column expressions from the constructors without importing training dependencies."""
    def assignment(filename, cls, fn, target):
        tree = ast.parse((ROOT / "scripts" / filename).read_text(encoding="utf8"))
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        fun = next(n for n in scope.body if isinstance(n, ast.FunctionDef) and n.name == fn)
        return next(n.value for n in fun.body if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == target)

    def columns(node):
        return [ast.unparse(x) for x in node.args[0].elts]

    base = "run_dept3_candidates.py"
    static = assignment(base, "Features", "__init__", "self.static")
    values = ["np.log1p(" + x + ")" for x in columns(static.args[0])]
    values += columns(assignment(base, "Features", "event", "x"))[1:]
    values += columns(assignment("partial_source_identification.py", "HistoricalFeatures", "event", "x"))[1:]
    roles = "ordinary_source_roles.py"
    values += columns(assignment(roles, "Window", "__init__", "self.quiet"))
    values += columns(assignment(roles, "Window", "features", "extra"))[1:]
    orig = "distilled_source_pairs.py"
    values += columns(assignment(orig, "PublicWindow", "design", "extra"))
    assert len(values) == 29
    ranks = ast.unparse(assignment(orig, "PublicWindow", "design", "relative"))
    values += [ranks + f" [column {i}]" for i in range(29)]
    values += columns(assignment(roles, "Window", "features", "ev"))
    labels = read(RUN / "dept3_candidates/models.json")["models"]["full100"]["features"]
    tiers = {}
    src = (ROOT / "scripts/recent_member_evidence.py").read_text(encoding="utf8")
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) in ("PERSON", "LINKS", "CONTEXT"):
            tiers[ast.unparse(n.targets[0])] = eval(compile(ast.Expression(n.value), "archived mask", "eval"),
                                                    {"__builtins__": {}, "range": range})
    def mask(inputs):
        return set(inputs) | {i + 29 for i in inputs} | set(tiers["CONTEXT"])
    personal = mask(tiers["PERSON"])
    contacts = mask(tiers["PERSON"] + tiers["LINKS"])
    assert len(values) == 64
    data = [dict(column=i, archived_name=labels[i] if i < len(labels) else None,
                 implementation_expression=e, first_tier="Personal" if i in personal else
                 "Recent contacts" if i in contacts else "Full temporal history") for i, e in enumerate(values)]
    (PAPER / "evidence/feature_columns.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")
    return [len(personal), len(contacts), len(values)]


def main():
    checked = {}
    def check_rows(num, expected):
        actual = table(num)
        assert len(actual) == len(expected), (num, len(actual), len(expected))
        for i, (cells, targets) in enumerate(zip(actual, expected)):
            assert cells == targets, (num, i, cells, targets)
        checked[str(num)] = {"rows": len(expected), "matched_cells": sum(map(len, expected))}

    dims = feature_index()
    assert [int(row[1]) for row in table(1)] == dims == [26, 34, 64]
    checked["1"] = {"rows": 3, "verified_active_dimensions": dims, "implementation_columns_mapped": 64}
    budget = summary("wp1")
    diagnostics = summary("wp2")["networks"]
    robustness = summary("wp3")
    real = summary("wp5")["real"]
    inv = read(PAPER / "evidence/inventory.json")
    names = {"company": "Company", "python-dev": "Python-Dev", "python-ideas": "Python-Ideas",
             "r-devel": "R-devel", "r-package-devel": "R-package-devel", "libc-alpha": "GNU libc",
             "libc-alpha-2024": "GNU libc", "samba-technical-2020": "Samba"}
    rows = []
    for name, rec in budget.items():
        v = inv["sim"][name]
        rows.append([names.get(name, name), f"{v['nodes']:,}", f"{v['records']:,}", f"{v['days']:,.1f} d",
                     str(len(rec["keys"])), f"{rec['n']:,}"])
    for key, v in inv["real"].items():
        name = key.removesuffix("_late")
        if key != name:
            z, period = real[name]["0"]["all"], "Feb--Apr 2019"
            assert v["start"] == "2019-02" and v["end"] == "2019-04"
        else:
            z = diagnostics["real " + name]
            assert (min(z["keys"]), max(z["keys"])) == (v["start"], v["end"])
            period = "--".join(dict.fromkeys((v["start"][:4], v["end"][:4])))
        rows.append([names[name], f"{v['nodes']:,}", f"{v['records']:,}", period,
                     str(len(z["keys"])), f"{z['n']:,}"])
    check_rows(2, rows)
    rows = []
    history = read(RUN / "factorial_history/summary.json")
    dept4 = read(RUN / "history_components/summary.json")["Dept4"]["methods"]
    def pct(method):
        return f"{100 * method['mean']:.2f}"
    for name, rec in budget.items():
        static = fgssi = tree = "---"
        if name in history:
            seeds = list(history[name]["groups"]["all"].values())
            assert all(r["n"] == rec["n"] for r in seeds)
            suffix = r"\(^*\)" if len(seeds) == 1 else ""
            static, fgssi = [f"{100 * mean(r[cell]['m12'] for r in seeds) / rec['n']:.2f}" + suffix
                             for cell in ("A", "F")]
        if name == "Dept4":
            static = f"{dept4['A']['both_pct']:.2f}"
        if name in robustness["tree"]:
            result = robustness["tree"][name]["12"]
            assert result["n"] == rec["n"]
            fgssi, tree = [pct(result["methods"][k]) for k in ("FGSSI", "full")]
        m = rec["methods"]
        rows.append([names.get(name, name), f"{rec['mean_candidates']:.1f}", pct(m["Any-in-12"]),
                     fgssi, static, tree, pct(m["Both-in-12"]), pct(m["Both-in-5%"]),
                     f"{m['Pair-MRR']['mean']:.5f}"])
    check_rows(3, rows)
    rows = []
    for network, arms in robustness["stress"].items():
        for i, (rule, rec) in enumerate(arms.items()):
            row = [network if i == 0 else "", "External" if rule == "noise" else rule]
            for group in ("all", "common"):
                v = rec[group]
                delta = v["differences"]["links minus person"]
                a, b = delta["ci95"]
                assert abs(delta["mean"] - v["methods"]["links"]["mean"] + v["methods"]["person"]["mean"]) < 1e-12
                row += [str(v["n"]), rf"\({100 * delta['mean']:+.2f}\;[{100*a:.2f},{100*b:.2f}]\)"]
            rows.append(row)
    check_rows(4, rows)
    qualified = read(RUN / "heldout_r_lists/early_only/role_evidence/closure/summary.json")
    periods = dict(names, **{"libc-alpha": "GNU libc 2021--22", "libc-alpha-2024": "GNU libc 2024",
                             "samba-technical-2020": "Samba 2020"})
    rows = []
    assert sum(x["n"] for x in qualified.values()) == 1793
    for name, rec in qualified.items():
        row = [periods[name], str(rec["n"]), str(rec["scored_reachable"])]
        for m in ("volume", "roles", "reply", "role_suffix"):
            hits = round(rec["both_pct"][m] * rec["n"] / 100)
            assert set(rec["hit_counts"][m]) == {hits}
            row.append(str(hits))
        rows.append(row)
    check_rows(5, rows)
    rows = []
    for name, levels in real.items():
        common = [levels[d]["common"] for d in ("0", "25", "50")]
        row = [periods[name], str(common[0]["n"])]
        for z in common:
            assert z["n"] == common[0]["n"]
            delta = z["methods"]["full"]["mean"] - z["methods"]["volume"]["mean"]
            assert abs(delta - z["differences"]["full minus volume"]["mean"]) < 1e-12
            row.append(rf"\({delta*100:+.2f}\)")
        rows.append(row)
    check_rows(6, rows)
    (PAPER / "evidence/table_audit.json").write_text(json.dumps(checked, indent=2) + "\n", encoding="utf8")
    print("Verified all six tables against feature code, inventory and archived experiment records.")


if __name__ == "__main__":
    main()

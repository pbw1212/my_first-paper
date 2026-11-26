"""Score the two missing Table 3 baselines on the existing WP1 events."""
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import sklearn

import closure_common as cc
import closure_robustness as wp3
from factorial_history import StaticWindow


DEST = cc.REPO / "runs/dasfaa_baseline_completion"
WP1 = cc.BASE / "wp1_budgets"
WP3 = cc.BASE / "wp3_robustness"


def cohort_scores(network, make_window, score):
    expected = {r["id"]: r for r in cc.read_rows(WP1 / f"{network}.jsonl.gz")}
    rows = {}
    for graph, raw, events, _ in cc.cohort(network):
        if not events:
            continue
        window = make_window(graph, raw)
        for event, gold in events:
            (nodes, _, _), (features, _, _) = window.design(cc.frames(event))
            key = event["id"]
            archived = expected[key]
            assert key not in rows and archived["window"] == graph["window"]
            assert len(nodes) == archived["n"] and set(gold) <= set(nodes)
            tie = cc.fg.tie_order(key, len(nodes))
            ranks = [cc.ranks(logit, tie, nodes, gold) for logit in score(features)]
            rows[key] = dict(id=key, window=graph["window"], n=len(nodes), ranks=ranks)
        print(network, graph["window"], len(rows), flush=True)
    assert rows.keys() == expected.keys()
    return [rows[key] for key in expected]


def main():
    cc.check_protocol(WP3)
    assert sklearn.__version__ == cc.load(WP3 / "protocol.json")["plan"]["version"]
    with TemporaryDirectory(prefix="dasfaa-baselines-") as tmp:
        stage = Path(tmp)
        wp3.F = stage
        wp3.ARMS = ["full"]
        training = wp3.train("Dept2")
        recorded = cc.load(WP3 / "training.json")["Dept2"]
        for key in ("train_ids", "validation_ids", "last_fit", "first_validation", "last_validation"):
            assert training[key] == recorded[key], key
        for key, value in training["selection"].items():
            target = recorded["selection"][key]
            assert value["iterations"] == target["iterations"], key
            for rounds, nll in value["validation_pair_nll"].items():
                assert abs(nll - target["validation_pair_nll"][str(rounds)]) < 1e-10, key
        models = joblib.load(stage / "Dept2-trees.joblib")["full"]

        def tree_score(x):
            return [model.decision_function(x) for model in models]

        tree_reference = cc.read_rows(WP3 / "tree-Dept2.jsonl.gz")
        reference = {r["id"]: r["ranks"]["full"] for r in tree_reference if r["budget"] == "12"}
        reproduced = cohort_scores("Dept2", cc.PublicWindow, tree_score)
        assert {r["id"]: r["ranks"] for r in reproduced} == reference

        static_models = cc.nets("Dept2", ["static"])["static"]
        static_window = lambda graph, _: StaticWindow(graph)
        static_score = lambda x: cc.logits(static_models, x)
        old_static = cc.ARCH / "runs/factorial_history/Dept2/cells.jsonl.gz"
        static_reference = {
            r["id"]: [r["models"][f"A{seed}"]["m"][1] for seed in cc.SEEDS]
            for r in cc.read_rows(old_static)
        }
        reproduced = cohort_scores("Dept2", static_window, static_score)
        assert {
            r["id"]: [sum(rank <= 12 for rank in pair) for pair in r["ranks"]]
            for r in reproduced
        } == static_reference

        report = {
            "training_source": "Dept2",
            "seed_order": list(cc.SEEDS),
            "tree_selection": training["selection"],
            "tree_reference_ranks_verified": len(reference),
            "static_reference_hits_verified": len(static_reference),
            "results": {},
        }
        output = stage / "results"
        output.mkdir()
        for network, method, window, score in (
            ("CollegeMsg", "Tree", cc.PublicWindow, tree_score),
            ("GNU libc", "Static", static_window, static_score),
        ):
            rows = cohort_scores(network, window, score)
            hits = (np.asarray([r["ranks"] for r in rows]) <= 12).sum(axis=2)
            path = output / f"{network}.jsonl.gz"
            cc.save_rows(path, rows)
            report["results"][network] = dict(
                method=method, n=len(rows), cohort_sha256=cc.sha(WP1 / path.name),
                ranks_sha256=cc.sha(path), both=100 * float(np.mean(hits == 2)),
                any=100 * float(np.mean(hits > 0)), recall=50 * float(np.mean(hits)),
            )
        cc.dump(output / "summary.json", report)
        if DEST.exists():
            for path in output.iterdir():
                assert cc.sha(path) == cc.sha(DEST / path.name), path.name
        else:
            shutil.copytree(output, DEST)
        print(report["results"], flush=True)


if __name__ == "__main__":
    main()

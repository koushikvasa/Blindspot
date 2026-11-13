import json

from fake_claude import FakeClient
from test_agent import MODEL, toy_world
from test_agent_b import make_responder

from blindspot.eval.experiment0 import DatasetInput
from blindspot.eval.experiment2b import aggregate, estimate_worst_case, run
from blindspot.generators.claude_client import ResponseCache


def _setup(tmp_path):
    real, splits = toy_world()
    d = DatasetInput("gss", "Toy GSS", real.subset(splits["pool"] + splits["test"]), splits)
    ref = {"coverage": 0.5, "precision": 0.9, "rcg_ratio": 1.0}
    exp1 = tmp_path / "experiment1.json"
    exp1.write_text(json.dumps({"datasets": {"gss": {"crowds": {"claude_mode": ref, "claude_sampled": ref}}}}))
    return d, splits, exp1


def test_estimate_is_positive_and_the_run_writes_all_outputs(tmp_path):
    d, splits, exp1 = _setup(tmp_path)
    client = FakeClient(make_responder(good=True))
    assert estimate_worst_case(d, client, MODEL, (1, 2), 2)["worst_case_usd"] > 0
    report = run([d], client, out_dir=tmp_path, cache=ResponseCache(tmp_path / "c"), model=MODEL, cost_limit=100.0,
                 seeds=(1, 2), rounds=2, per_round=25, workers=2, n_boot=10, exp1_path=exp1, log=lambda _m: None)
    ds = report["datasets"]["gss"]
    for metric in ("coverage", "precision"):
        agg = aggregate(ds["per_seed"], metric)
        assert set(agg) == {"h4b_verify", "h4b_ablation", "copula_on_asked", "claude_plus_asked"}
        assert all(len(budgets) == 2 and len(v) == 2 for budgets in agg.values() for v in budgets)
    md = (tmp_path / "experiment2b.md").read_text(encoding="utf-8")
    for needle in ("| method | 25 | 50 |", "Recipe accept/reject counts", "old rule", "Escalation log, seed 1",
                   "NO real respondents' rows"):
        assert needle in md
    log = json.loads((tmp_path / "agent" / "gss_h4b" / "rounds.json").read_text(encoding="utf-8"))
    asked = {i for v in log["runs"].values() for runs in v.values() for r in runs for i in r["asked_new"]}
    assert asked <= set(splits["pool"]) and not asked & (set(splits["test"]) | set(splits["holdout"]))
    assert (tmp_path / "agent" / "gss_h4b" / "verify_seed1" / "clusters_v2.json").exists()
    assert report["cost"]["actual_usd"] >= 0

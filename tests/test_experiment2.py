import json

import pytest
from fake_claude import FakeClient
from test_agent import MODEL, toy_world

from blindspot.eval.experiment0 import DatasetInput
from blindspot.eval.experiment2 import aggregate, estimate_run, run, swap_in_asked
from blindspot.generators.claude_client import CostLimitExceeded, ResponseCache


def _input():
    real, splits = toy_world()
    return DatasetInput("toy", "Toy dataset", real.subset(splits["pool"] + splits["test"]), splits), splits


def _exp1(tmp_path):
    ref = {"coverage": 0.5, "precision": 0.9, "rcg_ratio": 1.0}
    path = tmp_path / "experiment1.json"
    path.write_text(json.dumps({"datasets": {"toy": {"crowds": {"claude_mode": ref, "claude_sampled": ref}}}}))
    return path


def test_swap_in_asked_replaces_the_first_synthetic_people_with_real_ones():
    d, splits = _input()
    target, pool = d.real.subset(splits["test"]), d.real.subset(splits["pool"])
    baseline = target.subset(list(target.answers.index))                  # any crowd of the right shape
    asked = pool.subset(splits["pool"][:10])
    out = swap_in_asked(baseline, asked)
    assert len(out.answers) == len(baseline.answers)
    assert list(out.answers.index[:10]) == list(asked.answers.index)
    assert out.answers.iloc[:10].equals(asked.answers) and out.answers.iloc[10:].equals(baseline.answers.iloc[10:])
    assert list(out.demographics.columns) == list(baseline.demographics.columns)


def test_estimate_has_worst_case_above_expected_and_the_guard_stops_without_calling(tmp_path):
    d, _ = _input()
    client = FakeClient(input_tokens=4000)
    est = estimate_run([d], client, MODEL, (1, 2, 3), 4, 1.0, True)
    assert est["worst_case_usd"] > est["expected_usd"] > 0
    capped = estimate_run([d], client, MODEL, (1, 2, 3), 4, 0.5, True)
    assert capped["worst_case_usd"] < est["worst_case_usd"]
    client.calls.clear()
    with pytest.raises(CostLimitExceeded, match="worst-case estimate"):
        run([d], client, out_dir=tmp_path, cache=ResponseCache(tmp_path / "c"), model=MODEL, cost_limit=0.0001,
            exp1_path=_exp1(tmp_path))
    assert client.calls == []


def test_experiment2_runs_end_to_end_on_a_toy_dataset(tmp_path):
    d, splits = _input()
    client = FakeClient()
    report = run([d], client, out_dir=tmp_path, cache=ResponseCache(tmp_path / "c"), model=MODEL, cost_limit=100.0,
                 seeds=(1, 2), rounds=2, per_round=25, batch=False, workers=2, n_boot=10, maps=False,
                 exp1_path=_exp1(tmp_path), log=lambda _m: None)
    ds = report["datasets"]["toy"]
    assert set(ds["per_seed"]) == {"1", "2"}
    for metric in ("coverage", "precision"):
        agg = aggregate(ds["per_seed"], metric)
        assert set(agg) == {"blindspot_targeted", "blindspot_random", "copula_on_asked", "claude_plus_asked"}
        assert all(len(budgets) == 2 and len(vals) == 2 for budgets in agg.values() for vals in budgets)
    md = (tmp_path / "experiment2.md").read_text(encoding="utf-8")
    assert "| method | 25 | 50 |" in md and "claude_plus_asked" in md and "claude_mode, budget 0" in md
    assert report["cost"]["actual_usd"] >= 0
    log = json.loads((tmp_path / "agent" / "toy" / "rounds.json").read_text(encoding="utf-8"))
    asked = {i for v in log["runs"].values() for runs in v.values() for r in runs for i in r["asked_new"]}
    assert asked <= set(splits["pool"]) and not asked & (set(splits["test"]) | set(splits["holdout"]))
    assert (tmp_path / "agent" / "toy" / "targeted_seed1" / "recipes_v2.json").exists()
    assert json.loads((tmp_path / "experiment2.json").read_text(encoding="utf-8"))["settings"]["seeds"] == [1, 2]

import json

import numpy as np
import pandas as pd
import pytest
from fake_claude import FakeClient, _system_text, calls_by_kind, generic_responder
from test_agent import ITEMS, MODEL, toy_world

from blindspot.agent.asking import Segment
from blindspot.agent.assemble import assemble
from blindspot.agent.book import ClusterBook
from blindspot.agent.budget import Budget
from blindspot.agent.describe import ClaudeCaller
from blindspot.agent.detect import detection_counts, uncovered_by_percentile
from blindspot.agent.escalate import choose_segment, segment_rates
from blindspot.agent.generate import PersonaGenerator
from blindspot.agent.loop2 import LoopConfigB, run_loop_b
from blindspot.agent.propose import Variant, propose_variants
from blindspot.agent.verify import verify_variants
from blindspot.coverage import real_distances
from blindspot.data.schema import Respondents
from blindspot.eval.splits import LeakError
from blindspot.generators import claude_personas as cp
from blindspot.generators.claude_client import CostLimitExceeded, ResponseCache, Usage
from blindspot.space import encode

GOOD = "GOODMARK"


@pytest.fixture
def world():
    real, splits = toy_world()
    return real.subset(splits["pool"] + splits["test"]), splits


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(tmp_path / "cache")


def make_responder(good: bool):
    """Persona answers are all 2 (the control); with the marker and `good` they sit near the low corner
    (0 everywhere except 1 on the last two items), close to but never exactly a toy prototype."""
    def responder(params, n):
        props = params["output_config"]["format"]["schema"]["properties"]
        if "variants" in props:
            return json.dumps({"variants": [
                {"name": f"{GOOD} {k}", "description": f"Tends to answer in way number {k} overall."}
                for k in range(3)]})
        if "answers" in props:
            sys_text = _system_text(params)
            k = [line.count(" = ") for line in sys_text.splitlines() if "Options:" in line]
            low = good and GOOD in sys_text
            return json.dumps({"answers": [{"q": i, "p": [100 // c] * (c - 1) + [100 - (100 // c) * (c - 1)],
                                            "a": (1 if i > len(k) - 2 else 0) if low else 2}
                                           for i, c in enumerate(k, start=1)]})
        return generic_responder(params, n)
    return responder


def _resp(values, prefix="r"):
    ids = [f"{prefix}{i}" for i in range(len(values))]
    ans = pd.DataFrame(values, index=ids, columns=[i.code for i in ITEMS], dtype=float)
    demo = pd.DataFrame({"age_band": "30-49", "gender": "Female", "region": "South", "education": "x",
                         "income_band": "unknown", "race": "White"}, index=ids)
    return Respondents(ans, demo, ITEMS, prefix)


# ---------------- detect ----------------
def test_detection_uses_the_crowds_own_nearest_neighbour_percentile():
    rng = np.random.default_rng(0)
    crowd = _resp(rng.integers(0, 2, size=(60, 8)), "s")                 # crowd lives in the low corner
    asked = _resp(np.vstack([crowd.answers.to_numpy()[:3], np.full((4, 8), 4)]), "a")
    unc, info = uncovered_by_percentile(asked, crowd, ITEMS)
    assert unc == ["a3", "a4", "a5", "a6"]                               # copies of crowd people are covered
    assert info["n_uncovered"] == 4 and info["q"] == 95 and info["threshold"] >= 0
    _, both = detection_counts(asked, crowd, ITEMS, k=3, seed=0)
    assert both["n_uncovered_new_rule"] == 4 and "n_uncovered_old_rule" in both


def test_collapsed_crowd_makes_every_different_person_uncovered():
    crowd = _resp(np.full((40, 8), 2), "s")                              # everyone identical: threshold 0
    asked = _resp(np.vstack([np.full((2, 8), 2), np.full((3, 8), 3)]), "a")
    unc, info = uncovered_by_percentile(asked, crowd, ITEMS)
    assert info["threshold"] == 0 and unc == ["a2", "a3", "a4"]


# ---------------- propose ----------------
def test_propose_drops_copying_truncates_long_and_refuses_non_pool(world, cache):
    real, splits = world
    copying = " ".join(f"How much do you feel about {t} these days overall? agree." for t in
                       ["trust in the courts", "worry about artificial intelligence", "satisfaction with finances",
                        "views on the economy", "time spent with friends"])
    variants = {"variants": [{"name": "copy", "description": copying},
                             {"name": "long", "description": " ".join(["tends"] * 150)},
                             {"name": "fine", "description": "Tends to distrust institutions yet feel hopeful."}]}
    client = FakeClient(lambda params, n: json.dumps(variants))
    caller = ClaudeCaller(client, cache, MODEL, Usage(), log=lambda _m: None)
    got, dropped = propose_variants(caller, ITEMS, real.subset(splits["pool"][:4]), 25, splits)
    assert [v.name for v in got] == ["long", "fine"] and len(got[0].text.split()) == 120
    assert [d["name"] for d in dropped] == ["copy"] and "item by item" in dropped[0]["problems"][0]
    with pytest.raises(LeakError):
        propose_variants(caller, ITEMS, real.subset(splits["test"][:3]), 25, splits)
    prompt = client.calls[0]["messages"][0]["content"]
    assert "age_band" not in prompt                                      # answers only


# ---------------- verify ----------------
def _gen(client, cache, budget=None):
    usage = Usage()
    return PersonaGenerator(client, cache, MODEL, usage, budget or Budget(100.0), "toy", ITEMS, 1), usage


def test_verify_accepts_only_recipes_closer_than_the_control(cache):
    members = _resp(np.full((6, 8), 0), "m")                            # cluster of all-0 answers
    gen, _ = _gen(FakeClient(make_responder(good=True)), cache)
    good, bad = Variant(f"{GOOD} one", "pull toward the low end"), Variant("plain two", "no pull at all")
    rows = verify_variants(gen, [good, bad], members, ITEMS, np.random.default_rng(0), n_people=10)
    assert [r["accepted"] for r in rows] == [True, False]
    assert rows[0]["median_recipe"] < rows[0]["median_control"]
    assert rows[1]["median_recipe"] == rows[1]["median_control"]
    assert rows[0]["n_people"] == 10 and rows[0]["verified"]


def test_ablation_accepts_everything_without_calling_claude(cache):
    client = FakeClient()
    gen, _ = _gen(client, cache)
    rows = verify_variants(gen, [Variant("a", "x"), Variant("b", "y")], _resp(np.zeros((3, 8))), ITEMS,
                           np.random.default_rng(0), accept_all=True)
    assert [r["accepted"] for r in rows] == [True, True] and client.calls == []


# ---------------- escalate ----------------
def test_escalation_targets_the_highest_observed_uncovered_rate_with_min_five_people():
    demo = pd.DataFrame({"age_band": ["65+"] * 4 + ["30-49"] * 10 + ["18-29"] * 6,
                         "region": ["South"] * 20}, index=[f"p{i}" for i in range(20)])
    uncovered = {"p0", "p1", "p2", "p3",                                # all four 65+ (but only 4 people)
                 "p4", "p5", "p6",                                      # 3/10 of 30-49
                 "p14", "p15", "p16", "p17", "p18"}                     # 5/6 of 18-29
    rows, small = segment_rates(demo, uncovered)
    assert small == 1 and rows[0] == {"field": "age_band", "value": "18-29", "n": 6, "n_uncovered": 5,
                                      "rate": 0.8333}
    segment, log = choose_segment(demo, uncovered, ["C1"])
    assert segment.conditions == (("age_band", "18-29"),)
    assert log["chosen"]["rate"] == 0.8333 and "18-29" in log["next_round"] and log["triggered"]


def test_escalation_falls_back_to_random_without_signal():
    demo = pd.DataFrame({"age_band": ["a"] * 4, "region": ["South"] * 4}, index=list("wxyz"))
    segment, log = choose_segment(demo, {"w"}, ["C1"])
    assert segment is None and "random" in log["next_round"]
    none, log2 = choose_segment(demo, {"w"}, [])
    assert none is None and not log2["triggered"]


# ---------------- assemble ----------------
def test_assemble_replaces_matching_people_by_cluster_share_and_keeps_demographics(world, cache):
    real, splits = world
    pool, target = real.subset(splits["pool"]), real.subset(splits["test"])
    client = FakeClient(make_responder(good=True))
    gen, _ = _gen(client, cache)
    baseline = cp.execute(cp.plan("toy", target.demographics, ITEMS, MODEL), client, cache, Usage(), batch=False).mode
    book = ClusterBook(ITEMS)
    members = list(pool.answers.index[:10])
    entry = book.new_entry(members, 1)
    entry.variants = [Variant(f"{GOOD} one", "t"), Variant(f"{GOOD} two", "u")]
    book.refresh(dict(zip(pool.answers.index, encode(pool))))
    crowd, info = assemble(gen, baseline, book, pool, n_asked=50, rng=np.random.default_rng(0))
    want = round(10 / 50 * len(target.answers))
    assert info["n_replaced"] == want and info["per_cluster"][entry.id]["placed"] == want
    changed = np.flatnonzero((crowd.answers.to_numpy() != baseline.answers.to_numpy()).any(axis=1))
    assert len(changed) == want and (crowd.answers.iloc[changed].iloc[:, :6] == 0).all().all()
    assert crowd.demographics.equals(baseline.demographics)              # rule 3
    assert set(crowd.answers.index) == set(baseline.answers.index)       # no real person's row inserted
    book2 = ClusterBook(ITEMS)
    book2.new_entry(members, 1)                                          # no accepted variants: nothing replaced
    same, info2 = assemble(gen, baseline, book2, pool, 50, np.random.default_rng(0))
    assert info2["n_replaced"] == 0 and same is baseline


# ---------------- the loop ----------------
def _baseline(real, splits, client, cache):
    target = real.subset(splits["test"])
    crowd = cp.execute(cp.plan("toy", target.demographics, ITEMS, MODEL), client, cache, Usage(), batch=False).mode
    return target, crowd


def _run(world, cache, client, cfg):
    real, splits = world
    target, baseline = _baseline(real, splits, client, cache)
    return run_loop_b(cfg, real, splits, baseline, real_distances(target, ITEMS), client, cache, MODEL,
                      Budget(100.0), log=lambda _m: None)


def test_loop_accepts_good_recipes_and_replaces_people(world, cache, tmp_path):
    real, splits = world
    client = FakeClient(make_responder(good=True))
    cfg = LoopConfigB("toy", seed=3, rounds=3, n_boot=10, out_dir=tmp_path / "b")
    res = _run(world, cache, client, cfg)
    assert [r["budget"] for r in res.rounds] == [25, 50, 75]
    ids = [i for r in res.rounds for i in r["asked_new"]]
    assert len(ids) == len(set(ids)) == 75 and set(ids) <= set(splits["pool"])
    assert sum(r["n_variants_accepted"] for r in res.rounds) > 0
    assert any(r["assembly"]["n_replaced"] > 0 for r in res.rounds)
    assert {"n_uncovered_new_rule", "n_uncovered_old_rule", "threshold"} <= set(res.rounds[0]["detection"])
    for r in res.rounds:
        assert r["near_duplicates"]["share"] <= 0.05 and "unverified" in r["model_written_demographic_reasons"]
    kinds = calls_by_kind(client)
    assert kinds["propose"] >= 1 and kinds["seek"] == 0 and kinds["describe"] == 0
    assert (tmp_path / "b" / "clusters_v3.json").exists()
    for r in res.rounds:
        for c in r["clusters"]:
            for a in c.get("attempts", []):
                assert all(v["verified"] and v["median_recipe"] is not None for v in a["variants"])


def test_rejected_clusters_escalate_to_the_data_driven_segment(world, cache):
    real, splits = world
    client = FakeClient(make_responder(good=False))                     # no recipe ever helps
    res = _run(world, cache, client, LoopConfigB("toy", seed=3, rounds=2, n_boot=10))
    r1, r2 = res.rounds
    assert r1["n_variants_accepted"] == 0 and r1["n_clusters_cannot_imagine"] >= 1
    assert all(c["action"] == "cannot_imagine" and len(c["attempts"]) == 2 for c in r1["clusters"])
    assert r1["escalation"]["triggered"] and r1["escalation"]["chosen"]["n"] >= 5
    chosen = r1["escalation"]["chosen"]
    assert r2["asking_mode"] == "targeted" and r2["escalation_used_for_this_round"]["chosen"] == chosen
    assert r2["asking_report"][0]["label"] == f"{chosen['field']}={chosen['value']}"
    assert r1["assembly"]["n_replaced"] == 0


def test_ablation_accepts_all_recipes_without_verification_and_never_escalates(world, cache):
    client = FakeClient(make_responder(good=False))
    res = _run(world, cache, client, LoopConfigB("toy", seed=3, variant="ablation", rounds=3, n_boot=10))
    assert all(r["n_clusters_cannot_imagine"] == 0 and r["asking_mode"] == "random" for r in res.rounds)
    assert all(not r["escalation"]["triggered"] for r in res.rounds)
    assert sum(r["n_variants_accepted"] for r in res.rounds) == sum(r["n_variants_proposed"] for r in res.rounds) > 0
    assert any(r["assembly"]["n_replaced"] > 0 for r in res.rounds)
    v = next(a["variants"][0] for r in res.rounds for c in r["clusters"] for a in c.get("attempts", []))
    assert v["verified"] is False


def test_loop_b_same_seed_same_results_and_zero_new_calls(world, cache):
    cfg = LoopConfigB("toy", seed=5, rounds=2, n_boot=10)
    first = _run(world, cache, FakeClient(make_responder(good=True)), cfg)
    client2 = FakeClient(make_responder(good=True))
    second = _run(world, cache, client2, cfg)
    assert client2.calls == []
    assert [r["asked_new"] for r in first.rounds] == [r["asked_new"] for r in second.rounds]
    assert [r["metrics"]["coverage"] for r in first.rounds] == [r["metrics"]["coverage"] for r in second.rounds]


def test_loop_b_refuses_holdout_people_and_stops_at_the_budget(cache):
    real, splits = toy_world()
    target = real.subset(splits["test"])
    client = FakeClient(make_responder(good=True))
    baseline = cp.execute(cp.plan("toy", target.demographics, ITEMS, MODEL), client, cache, Usage(), batch=False).mode
    rr = real_distances(target, ITEMS)
    with pytest.raises(LeakError):                                       # holdout rows still present
        run_loop_b(LoopConfigB("toy", 1, rounds=1, n_boot=5), real, splits, baseline, rr, client, cache, MODEL,
                   Budget(100.0), log=lambda _m: None)
    pooltest = real.subset(splits["pool"] + splits["test"])
    with pytest.raises(CostLimitExceeded):
        run_loop_b(LoopConfigB("toy", 1, rounds=2, n_boot=5), pooltest, splits, baseline, rr,
                   FakeClient(make_responder(good=True)), ResponseCache(cache.dir / "fresh"), MODEL, Budget(0.0),
                   log=lambda _m: None)

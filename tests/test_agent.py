import numpy as np
import pandas as pd
import pytest
from fake_claude import FakeClient, calls_by_kind

from blindspot.agent.asking import Segment, draw_random, draw_targeted
from blindspot.agent.budget import Budget
from blindspot.agent.describe import ClaudeCaller, describe_cluster, validate_segments
from blindspot.agent.gaps import cluster_uncovered, n_clusters_for, uncovered_asked
from blindspot.agent.guards import (ArchetypeCopyError, NearDuplicateError, archetype_problems,
                                    assert_few_near_duplicates, copied_items, near_duplicates, truncate_words)
from blindspot.agent.library import RecipeLibrary, centroid
from blindspot.agent.loop import LoopConfig, run_loop
from blindspot.coverage import real_distances
from blindspot.data.schema import Item, Respondents
from blindspot.eval.splits import LeakError, make_splits
from blindspot.generators import claude_personas as cp
from blindspot.generators.claude_client import CostLimitExceeded, ResponseCache, Usage
from blindspot.space import encode

MODEL = "claude-sonnet-5-5"
LABELS = ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"]
TOPICS = ["trust in the courts", "worry about artificial intelligence", "satisfaction with finances",
          "views on the economy", "time spent with friends", "optimism about the future",
          "opinion of technology", "interest in politics"]
FAMILIES = ["trust", "trust", "money", "money", "life", "life", "tech", "tech"]
ITEMS = [Item(f"q{i}", f"How much do you feel about {t} these days overall?", "ordinal", LABELS, fam)
         for i, (t, fam) in enumerate(zip(TOPICS, FAMILIES))]


def toy_world(n: int = 300, seed: int = 0):
    """People in three tight answer clusters (plus a little noise): the fake personas, which answer
    uniformly at random, cannot cover them, so the loop finds gaps."""
    rng = np.random.default_rng(seed)
    ids = [str(i) for i in range(n)]
    protos = np.array([[0] * 8, [4] * 8, [0, 4] * 4])
    values = protos[rng.integers(0, 3, n)]
    noisy = rng.random(n) < 0.3
    values[noisy, rng.integers(0, 8, noisy.sum())] = rng.integers(0, 5, noisy.sum())
    answers = pd.DataFrame(values, index=ids, columns=[i.code for i in ITEMS], dtype=float)
    demo = pd.DataFrame({"age_band": rng.choice(["18-29", "30-49", "50-64", "65+"], n),
                         "gender": rng.choice(["Female", "Male"], n),
                         "region": rng.choice(["South", "West", "Midwest", "Northeast"], n),
                         "education": rng.choice(["High school graduate", "Postgraduate"], n),
                         "income_band": "unknown", "race": "White", "weight": 1.0}, index=ids)
    real = Respondents(answers, demo, ITEMS, "toy")
    splits = make_splits(ids, seed=1)
    return real, splits


@pytest.fixture
def world():
    real, splits = toy_world()
    return real.subset(splits["pool"] + splits["test"]), splits


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(tmp_path / "cache")


# ---------------- guards ----------------
def test_guard_catches_item_by_item_copying():
    copying = ("How much do you feel about trust in the courts these days overall? strongly agree. "
               "How much do you feel about worry about artificial intelligence these days? disagree. "
               "How much do you feel about satisfaction with finances these days overall? neutral. "
               "How much do you feel about views on the economy these days overall? strongly disagree.")
    assert len(copied_items(copying, ITEMS)) == 4
    problems = archetype_problems(copying, ITEMS)
    assert problems and "item by item" in problems[0]
    summary = ("Tends to distrust institutions yet feels hopeful about technology, so trust and optimism "
               "move in opposite directions; finances are a weak signal.")
    assert archetype_problems(summary, ITEMS) == []
    three = " ".join(f"How much do you feel about {t} these days overall? agree." for t in TOPICS[:3])
    assert archetype_problems(three, ITEMS) == []                       # exactly 3 is allowed


def test_guard_enforces_word_limit_and_truncation():
    long = " ".join(["word"] * 130)
    assert any("words" in p for p in archetype_problems(long, ITEMS))
    assert len(truncate_words(long).split()) == 120


def test_near_duplicate_detector_flags_a_copied_person(world):
    real, splits = world
    pool = real.subset(splits["pool"][:30])
    synthetic = pool.answers.iloc[:20].copy()
    synthetic.index = [f"s{i}" for i in range(20)]
    crowd = Respondents(synthetic, pool.demographics.iloc[:20].set_axis(synthetic.index), ITEMS, "toy:crowd")
    copied = near_duplicates(crowd, pool)
    assert copied.share == 1.0
    with pytest.raises(NearDuplicateError):
        assert_few_near_duplicates(copied)
    rng = np.random.default_rng(5)
    far = crowd.answers.copy()
    far[:] = rng.integers(0, 5, size=far.shape)
    clean = near_duplicates(Respondents(far, crowd.demographics, ITEMS, "toy:far"), pool)
    assert clean.share < 0.05
    assert_few_near_duplicates(clean)


# ---------------- asking ----------------
def test_targeted_asking_only_within_segments_and_only_from_pool(world):
    real, splits = world
    pool_demo = real.subset(splits["pool"]).demographics
    seg = [Segment((("age_band", "65+"),), "older"), Segment((("gender", "Female"), ("region", "South")), "x")]
    ids, report = draw_targeted(pool_demo, set(), seg, 25, np.random.default_rng(0))
    assert len(ids) == 25 and set(ids) <= set(splits["pool"])
    assert not set(ids) & (set(splits["test"]) | set(splits["holdout"]))
    in_any = seg[0].mask(pool_demo) | seg[1].mask(pool_demo)
    assert in_any[ids].all()
    assert [r["found"] for r in report] == [13, 12]


def test_targeted_asking_falls_back_to_random_for_empty_segments(world):
    real, splits = world
    pool_demo = real.subset(splits["pool"]).demographics
    already = set(splits["pool"][:10])
    empty = Segment((("age_band", "no such band"),), "none")
    ids, report = draw_targeted(pool_demo, already, [empty], 25, np.random.default_rng(0))
    assert len(ids) == 25 and not set(ids) & already and report[-1] == {"fallback_random": 25}
    assert draw_targeted(pool_demo, already, [], 5, np.random.default_rng(0))[1] == []


def test_random_asking_never_repeats_people(world):
    real, splits = world
    rng = np.random.default_rng(0)
    first = draw_random(splits["pool"], set(), 25, rng)
    second = draw_random(splits["pool"], set(first), 25, rng)
    assert not set(first) & set(second)
    with pytest.raises(ValueError):
        draw_random(splits["pool"], set(splits["pool"]), 1, rng)


def test_segment_validation_drops_unknown_fields_and_values():
    allowed = {"age_band": ["18-29", "65+"], "region": ["South", "West"]}
    raw = [{"conditions": [{"field": "age_band", "value": "65+"}, {"field": "zodiac", "value": "leo"}], "reason": "a"},
           {"conditions": [{"field": "region", "value": "mars"}], "reason": "b"},
           {"conditions": [{"field": "region", "value": "south"}], "reason": "c"},
           {"conditions": [{"field": "region", "value": "SOUTH"}], "reason": "dup"}]
    segs = validate_segments(raw, allowed)
    assert [s.conditions for s in segs] == [(("age_band", "65+"),), (("region", "South"),)]
    assert validate_segments("nonsense", allowed) == []


# ---------------- gaps and library ----------------
def test_uncovered_people_are_the_ones_far_from_the_crowd(world):
    real, splits = world
    asked = real.subset(splits["pool"][:25])
    rng = np.random.default_rng(2)
    crowd_answers = asked.answers.copy()
    crowd_answers.iloc[:, :] = rng.integers(0, 2, size=crowd_answers.shape)        # only low answers
    far = asked.answers.copy()
    far.iloc[:12] = 4                                                                # 12 real people at the top
    asked2 = Respondents(far, asked.demographics, ITEMS, "toy")
    crowd = Respondents(crowd_answers.set_axis([f"s{i}" for i in range(25)]),
                        asked.demographics.set_axis([f"s{i}" for i in range(25)]), ITEMS, "toy:crowd")
    unc = uncovered_asked(asked2, crowd, ITEMS, k=3, seed=0)
    assert set(unc) >= set(asked.answers.index[:6]) and len(unc) <= 25
    assert uncovered_asked(asked2.subset(list(asked2.answers.index[:1])), crowd, ITEMS) == []


def test_clustering_is_deterministic_and_bounded(world):
    real, splits = world
    asked = real.subset(splits["pool"][:40])
    first = cluster_uncovered(asked, list(asked.answers.index), ITEMS)
    again = cluster_uncovered(asked, list(asked.answers.index), ITEMS)
    assert first == again and 2 <= len(first) <= 5
    assert sorted(i for c in first for i in c) == sorted(asked.answers.index)
    assert [n_clusters_for(n) for n in (0, 1, 2, 8, 50, 400)] == [0, 1, 2, 2, 5, 5]
    assert cluster_uncovered(asked, [], ITEMS) == [] and cluster_uncovered(asked, ["x"], ITEMS) == [["x"]]


def test_library_merges_near_patterns_and_weights_members():
    lib = RecipeLibrary(ITEMS, merge_distance=0.12)
    rows = {f"p{i}": np.full(len(ITEMS), v) for i, v in enumerate([0.0, 0.0, 1.0, 1.0, 0.5])}
    a = lib.add("low", "tends low", ["p0", "p1"], 1)
    lib.refresh(rows)
    near, d = lib.find_match(centroid(np.vstack([rows["p0"], rows["p1"]])) + 0.01)
    assert near is a and d < 0.12
    far, _ = lib.find_match(np.full(len(ITEMS), 1.0))
    assert far is None
    b = lib.add("high", "tends high", ["p2", "p3"], 2)
    lib.merge_into(b, ["p1"])                                       # latest assignment wins
    lib.refresh(rows)
    assert a.members == {"p0"} and b.members == {"p1", "p2", "p3"}
    assert lib.weights(10) == {"A1": 0.1, "A2": 0.3}
    assert sum(lib.weights(10, max_share=0.2).values()) == pytest.approx(0.2)
    assert lib.version == 2


def test_library_recipes_use_stable_draws_and_no_archetype_remainder():
    lib = RecipeLibrary(ITEMS)
    lib.add("a", "pattern a", ["p0", "p1"], 1)
    lib.refresh({"p0": np.zeros(8), "p1": np.zeros(8)})
    u = np.array([0.0, 0.19, 0.2, 0.9])
    recipes = lib.recipes_for(u, n_asked=10)                       # weight 0.2
    assert [bool(r) for r in recipes] == [True, True, False, False]
    assert "pattern a" in recipes[0]
    assert lib.recipes_for(u, 10) == recipes


# ---------------- describe (guard + leak) ----------------
def test_describe_retries_then_discards_copying_and_refuses_non_pool_people(world, cache):
    real, splits = world
    copying = {"name": "x", "share_low_pct": 5, "share_high_pct": 9, "description": " ".join(
        f"How much do you feel about {t} these days overall? agree." for t in TOPICS[:5])}
    import json
    client = FakeClient(lambda params, n: json.dumps(copying))
    caller = ClaudeCaller(client, cache, MODEL, Usage(), log=lambda _m: None)
    cluster = real.subset(splits["pool"][:4])
    out = describe_cluster(caller, ITEMS, cluster, 25, splits)
    assert out.discarded and out.attempts == 2 and len(client.calls) == 2
    with pytest.raises(LeakError):
        describe_cluster(caller, ITEMS, real.subset(splits["test"][:3]), 25, splits)


def test_describe_returns_a_clean_archetype(world, cache):
    real, splits = world
    client = FakeClient()
    caller = ClaudeCaller(client, cache, MODEL, Usage(), log=lambda _m: None)
    out = describe_cluster(caller, ITEMS, real.subset(splits["pool"][:4]), 25, splits)
    assert not out.discarded and out.attempts == 1 and 0 <= out.share_low <= out.share_high <= 1
    prompt = client.calls[0]["messages"][0]["content"]
    assert "age_band" not in prompt and "weight" not in prompt            # answers only, no demographics


# ---------------- the loop ----------------
def _baseline(real, splits, client, cache):
    target = real.subset(splits["test"])
    plan = cp.plan("toy", target.demographics, ITEMS, MODEL)
    crowd = cp.execute(plan, client, cache, Usage(), batch=False).mode
    return target, crowd


def _run(world, cache, client, cfg, budget=None):
    real, splits = world
    target, baseline = _baseline(real, splits, client, cache)
    rr = real_distances(target, ITEMS)
    return run_loop(cfg, real, splits, baseline, rr, client, cache, MODEL, budget or Budget(100.0),
                    log=lambda _m: None)


def test_loop_runs_end_to_end_with_a_mocked_client(world, cache, tmp_path):
    real, splits = world
    client = FakeClient()
    cfg = LoopConfig("toy", seed=3, rounds=3, n_boot=10, batch=False, out_dir=tmp_path / "agent")
    res = _run(world, cache, client, cfg)

    assert len(res.rounds) == 3 and [r["budget"] for r in res.rounds] == [25, 50, 75]
    all_new = [i for r in res.rounds for i in r["asked_new"]]
    assert len(all_new) == len(set(all_new)) == 75 and set(all_new) <= set(splits["pool"])
    assert not set(all_new) & (set(splits["test"]) | set(splits["holdout"]))
    for r in res.rounds:
        assert 0 <= r["metrics"]["coverage"] <= 1 and r["near_duplicates"]["share"] <= 0.05
        assert r["library_version"] == r["round"] and r["cost_usd_cumulative"] >= 0
    assert res.rounds[0]["library"]["archetypes"], "round 1 should have found gaps and archetypes"
    assert (tmp_path / "agent" / "recipes_v3.json").exists()
    kinds = calls_by_kind(client)
    assert kinds["describe"] >= 1 and kinds["seek"] == 2 and kinds["persona"] > 90   # seek: rounds 1-2 only
    assert res.usage.calls > 0


def test_targeted_rounds_ask_inside_the_proposed_segments(world, cache):
    real, splits = world
    res = _run(world, cache, FakeClient(), LoopConfig("toy", seed=3, rounds=3, n_boot=10, batch=False))
    pool_demo = real.subset(splits["pool"]).demographics
    for prev, nxt in zip(res.rounds, res.rounds[1:]):
        segs = [Segment(tuple(s["conditions"].items()), s["reason"]) for s in prev["segments_proposed_next"]]
        assert segs and nxt["segments_requested"] == prev["segments_proposed_next"]
        in_any = pd.Series(np.logical_or.reduce([s.mask(pool_demo).to_numpy() for s in segs]), index=pool_demo.index)
        fallback = sum(r.get("fallback_random", 0) for r in nxt["asking_report"])   # logged shortfall, if any
        assert int((~in_any[nxt["asked_new"]]).sum()) <= fallback
        assert sum(r.get("found", 0) for r in nxt["asking_report"]) + fallback == 25


def test_random_variant_never_calls_seek_and_shares_round_one_with_targeted(world, cache):
    client = FakeClient()
    targeted = _run(world, cache, client, LoopConfig("toy", seed=3, rounds=2, n_boot=10, batch=False))
    before = calls_by_kind(client)
    random_ = _run(world, cache, client, LoopConfig("toy", seed=3, rounds=2, variant="random", n_boot=10, batch=False))
    after = calls_by_kind(client)
    assert after["seek"] == before["seek"]
    assert random_.rounds[0]["asked_new"] == targeted.rounds[0]["asked_new"]
    assert random_.rounds[0]["metrics"]["coverage"] == targeted.rounds[0]["metrics"]["coverage"]
    assert all(r["segments_requested"] == [] for r in random_.rounds)


def test_same_seed_gives_same_asked_ids_and_results_and_costs_nothing_more(world, cache):
    cfg = LoopConfig("toy", seed=5, rounds=3, n_boot=10, batch=False)
    client1 = FakeClient()
    first = _run(world, cache, client1, cfg)
    client2 = FakeClient()
    second = _run(world, cache, client2, cfg)
    assert client2.calls == []
    assert [r["asked_new"] for r in first.rounds] == [r["asked_new"] for r in second.rounds]
    assert [r["metrics"]["coverage"] for r in first.rounds] == [r["metrics"]["coverage"] for r in second.rounds]
    other = _run(world, cache, FakeClient(), LoopConfig("toy", seed=6, rounds=1, n_boot=10, batch=False))
    assert other.rounds[0]["asked_new"] != first.rounds[0]["asked_new"]


def test_loop_refuses_holdout_people_and_stops_at_the_budget(world, cache):
    real, splits = toy_world()
    with pytest.raises(LeakError):                                         # holdout rows still present
        _run((real, splits), cache, FakeClient(), LoopConfig("toy", seed=1, rounds=1, n_boot=5, batch=False))
    with pytest.raises(CostLimitExceeded):
        _run(world, cache, FakeClient(), LoopConfig("toy", seed=1, rounds=1, n_boot=5, batch=False),
             budget=Budget(0.0))

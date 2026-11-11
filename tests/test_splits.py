import json

import pytest

from blindspot.eval.splits import (
    LeakError, assert_no_leak, get_or_make_splits, load_splits, make_splits, save_splits,
)

IDS = [str(i) for i in range(1000)]


def test_splits_are_deterministic_and_order_independent():
    a = make_splits(IDS, seed=42)
    b = make_splits(list(reversed(IDS)), seed=42)
    assert a == b


def test_different_seed_gives_different_split():
    assert make_splits(IDS, seed=42)["pool"] != make_splits(IDS, seed=7)["pool"]


def test_splits_are_disjoint_and_cover_all_ids_with_requested_sizes():
    s = make_splits(IDS, seed=42, fractions=(0.5, 0.3, 0.2))
    pool, test, holdout = set(s["pool"]), set(s["test"]), set(s["holdout"])
    assert not (pool & test) and not (pool & holdout) and not (test & holdout)
    assert pool | test | holdout == set(IDS)
    assert (len(pool), len(test), len(holdout)) == (500, 300, 200)


@pytest.mark.parametrize("fractions", [(0.5, 0.5), (0.6, 0.3, 0.2), (1.0, 0.0, 0.0)])
def test_bad_fractions_raise(fractions):
    with pytest.raises(ValueError):
        make_splits(IDS, fractions=fractions)


def test_assert_no_leak_allows_pool_and_blocks_test_holdout_and_unknown_ids():
    s = make_splits(IDS, seed=42)
    assert_no_leak(s["pool"][:10], s)
    for name in ("test", "holdout"):
        with pytest.raises(LeakError):
            assert_no_leak([s[name][0]], s)
    with pytest.raises(LeakError):
        assert_no_leak(["not-a-real-id"], s)
    assert_no_leak(s["test"][:5], s, allowed=("pool", "test"))


def test_save_load_roundtrip_and_refuses_changed_ids(tmp_path):
    s = make_splits(IDS, seed=42)
    path = save_splits("demo", s, 42, (0.5, 0.3, 0.2), split_dir=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["n"] == 1000 and payload["seed"] == 42
    assert load_splits("demo", split_dir=tmp_path) == s
    assert get_or_make_splits("demo", IDS, split_dir=tmp_path) == s
    with pytest.raises(ValueError):
        get_or_make_splits("demo", IDS[:-1], split_dir=tmp_path)

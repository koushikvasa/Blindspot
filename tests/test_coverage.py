import numpy as np
import pandas as pd
import pytest

from blindspot.coverage import evaluate, knn_radius, subsample
from blindspot.data.schema import Item, Respondents

K_OPTIONS = 5


def _resp(values: np.ndarray, prefix: str, families=("f1", "f1", "f2", "f2")) -> Respondents:
    items = [Item(f"q{i}", f"Q{i}?", "ordinal", [str(o) for o in range(K_OPTIONS)], fam)
             for i, fam in enumerate(families)]
    ids = [f"{prefix}{i}" for i in range(len(values))]
    ans = pd.DataFrame(values, index=ids, columns=[i.code for i in items], dtype=float)
    return Respondents(ans, pd.DataFrame({"g": ["x"] * len(ids)}, index=ids), items, prefix)


@pytest.fixture
def real():
    rng = np.random.default_rng(1)
    return _resp(rng.integers(0, K_OPTIONS, size=(80, 4)), "r")


def test_identical_crowd_has_full_coverage_and_precision(real):
    crowd = _resp(real.answers.to_numpy(), "s")
    m = evaluate(real, crowd, real.items, k=5, n_boot=50)
    assert m["coverage"] == 1.0 and m["precision"] == 1.0
    assert m["uncovered_ids"] == []
    assert m["rcg_ratio"] == 0.0
    assert m["coverage_ci"][0] <= 1.0 <= m["coverage_ci"][1] + 1e-12


def test_far_away_crowd_has_zero_coverage_and_precision():
    real = _resp(np.zeros((40, 4)) + np.array([0, 0, 1, 1]), "r")
    real.answers.iloc[::2, 0] = 1  # some spread so radii are small but positive
    crowd = _resp(np.full((40, 4), K_OPTIONS - 1), "s")
    m = evaluate(real, crowd, real.items, k=3, n_boot=20)
    assert m["coverage"] == 0.0 and m["precision"] == 0.0
    assert len(m["uncovered_ids"]) == 40
    assert m["rcg_ratio"] > 1


def test_crowd_covering_one_cluster_gives_partial_scores():
    rng = np.random.default_rng(2)
    low = rng.integers(0, 2, size=(40, 4))           # cluster near the bottom of every scale
    high = rng.integers(3, 5, size=(40, 4))          # cluster near the top
    real = _resp(np.vstack([low, high]), "r")
    crowd = _resp(np.vstack([low, np.full((40, 4), 2)]), "s")  # copies of `low` + a middle profile
    m = evaluate(real, crowd, real.items, k=5, n_boot=20)
    assert m["coverage"] == pytest.approx(0.5)
    assert set(m["uncovered_ids"]) == {f"r{i}" for i in range(40, 80)}
    assert m["precision"] == pytest.approx(0.5)
    assert set(m["k_sensitivity"]) == {"3", "5", "10"}
    assert m["k_sensitivity"]["3"]["coverage"] <= m["k_sensitivity"]["10"]["coverage"]


def test_bootstrap_intervals_contain_point_estimates(real):
    rng = np.random.default_rng(4)
    noisy = np.clip(real.answers.to_numpy() + rng.integers(-1, 2, size=(80, 4)), 0, K_OPTIONS - 1)
    m = evaluate(real, _resp(noisy, "s"), real.items, k=5, n_boot=200)
    for key in ("coverage", "precision"):
        lo, hi = m[f"{key}_ci"]
        assert lo <= m[key] <= hi, (key, m[key], lo, hi)


def test_crowd_is_subsampled_to_target_size(real):
    big = _resp(np.tile(real.answers.to_numpy(), (3, 1)), "s")
    assert len(subsample(big, 80, seed=0).answers) == 80
    assert evaluate(real, big, real.items, n_boot=10)["n_synthetic"] == 80


def test_knn_radius_excludes_self():
    d = np.array([[np.inf, 1, 2], [1, np.inf, 3], [2, 3, np.inf]], dtype=float)
    assert knn_radius(d, 1).tolist() == [1, 1, 2]
    with pytest.raises(ValueError):
        knn_radius(d, 3)


def test_per_family_entropy_reported(real):
    m = evaluate(real, _resp(real.answers.to_numpy(), "s"), real.items, n_boot=10)
    assert set(m["per_family_entropy"]["crowd"]) == {"f1", "f2"}
    assert m["per_family_entropy"]["crowd"] == pytest.approx(m["per_family_entropy"]["real"])

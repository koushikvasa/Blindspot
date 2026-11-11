import numpy as np
import pandas as pd
import pytest

from blindspot.data.schema import Item, Respondents
from blindspot.eval.splits import LeakError, make_splits
from blindspot.generators.baselines import gaussian_copula, independent_marginals, other_real_people

N = 3000


@pytest.fixture(scope="module")
def world():
    """Two strongly positively correlated 5-point items and one independent binary item."""
    rng = np.random.default_rng(3)
    latent = rng.normal(size=N)
    q0 = np.digitize(latent + 0.3 * rng.normal(size=N), [-1.5, -0.5, 0.5, 1.5])
    q1 = np.digitize(latent + 0.3 * rng.normal(size=N), [-1.0, 0.0, 0.7, 1.8])
    q2 = (rng.random(N) < 0.3).astype(int)
    ids = [str(i) for i in range(N)]
    items = [Item("q0", "?", "ordinal", list("abcde"), "f1"), Item("q1", "?", "ordinal", list("abcde"), "f1"),
             Item("q2", "?", "binary", ["n", "y"], "f2")]
    ans = pd.DataFrame({"q0": q0, "q1": q1, "q2": q2}, index=ids, dtype=float)
    ans.iloc[::50, 1] = np.nan  # some missing answers
    real = Respondents(ans, pd.DataFrame({"g": ["x"] * N}, index=ids), items, "toy")
    splits = make_splits(ids, seed=1)
    pool = real.subset(splits["pool"])
    target = real.subset(splits["test"])
    return real, pool, target, splits


def _shares(s: pd.Series, k: int) -> np.ndarray:
    return s.dropna().value_counts(normalize=True).reindex(range(k), fill_value=0).to_numpy()


@pytest.mark.parametrize("make", [independent_marginals, gaussian_copula])
def test_crowd_matches_pool_marginals_and_target_demographics(world, make):
    _, pool, target, splits = world
    crowd = make(pool, target.demographics, splits, seed=0)
    assert len(crowd.answers) == len(target.answers)
    assert crowd.demographics["g"].value_counts().equals(target.demographics["g"].value_counts())
    assert not crowd.answers.isna().any().any()
    for it in pool.items:
        k = len(it.options)
        assert np.abs(_shares(crowd.answers[it.code], k) - _shares(pool.answers[it.code], k)).max() < 0.05


def test_copula_keeps_correlation_sign_and_marginals_do_not(world):
    _, pool, target, splits = world
    real_r = pool.answers[["q0", "q1"]].corr(method="spearman").iloc[0, 1]
    cop_r = gaussian_copula(pool, target.demographics, splits).answers[["q0", "q1"]].corr("spearman").iloc[0, 1]
    ind_r = independent_marginals(pool, target.demographics, splits).answers[["q0", "q1"]].corr("spearman").iloc[0, 1]
    assert real_r > 0.7 and cop_r > 0.5
    assert abs(ind_r) < 0.1


def test_other_real_people_are_pool_people(world):
    _, pool, target, splits = world
    crowd = other_real_people(pool, len(target.answers), splits)
    assert set(crowd.answers.index) <= set(splits["pool"])
    assert len(crowd.answers) == len(target.answers)


@pytest.mark.parametrize("name", ["test", "holdout"])
def test_baselines_refuse_non_pool_people(world, name):
    real, pool, target, splits = world
    leaky = real.subset(splits["pool"][:100] + splits[name][:1])
    with pytest.raises(LeakError):
        independent_marginals(leaky, target.demographics, splits)
    with pytest.raises(LeakError):
        gaussian_copula(leaky, target.demographics, splits)
    with pytest.raises(LeakError):
        other_real_people(leaky, 10, splits)


def test_baselines_ignore_test_and_holdout_answers(world):
    """Scrambling every non-pool answer leaves the fitted crowds unchanged."""
    real, pool, target, splits = world
    scrambled = real.answers.copy()
    others = splits["test"] + splits["holdout"]
    scrambled.loc[others] = 4 - scrambled.loc[others].clip(upper=4)
    real2 = Respondents(scrambled, real.demographics, real.items, "toy")
    for make in (independent_marginals, gaussian_copula):
        a = make(pool, target.demographics, splits, seed=5).answers
        b = make(real2.subset(splits["pool"]), target.demographics, splits, seed=5).answers
        assert a.equals(b)

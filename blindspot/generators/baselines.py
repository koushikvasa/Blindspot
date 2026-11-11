"""Baseline crowds (H2). All are fit on the pool split only and checked with assert_no_leak.

Rule 3 (same demographics): synthetic person j gets the demographics row of target person j, so the
crowd's demographic distribution equals the target's. The marginals and copula answers are NOT
conditioned on demographics; the demographics are attached for bookkeeping only.
other_real_people returns real pool people with their own demographics (a random sample, as the
handover specifies), so its demographic mix only approximately matches the target.
"""
import numpy as np
import pandas as pd
from scipy import stats

from blindspot.data.schema import Respondents
from blindspot.eval.splits import assert_no_leak

PSD_FLOOR = 1e-6  # eigenvalue floor when repairing the copula correlation matrix


def _check_pool(pool: Respondents, splits: dict[str, list[str]]) -> None:
    assert_no_leak(pool.answers.index, splits, allowed=("pool",))


def _crowd(values: np.ndarray, pool: Respondents, demographics: pd.DataFrame, name: str) -> Respondents:
    n = len(values)
    if len(demographics) != n:
        raise ValueError(f"need {n} demographic rows, got {len(demographics)}")
    ids = pd.Index([f"{name}_{i:05d}" for i in range(n)])
    answers = pd.DataFrame(values, index=ids, columns=pool.answers.columns, dtype="float64")
    demo = demographics.reset_index(drop=True).set_axis(ids)
    return Respondents(answers, demo, pool.items, f"baseline:{name}",
                       meta={"fit_on": "pool", "n_pool": len(pool.answers)})


def _shares(col: pd.Series, k: int) -> np.ndarray:
    counts = col.dropna().astype(int).value_counts().reindex(range(k), fill_value=0).to_numpy(float)
    if counts.sum() == 0:
        raise ValueError(f"item {col.name} has no answers in the pool")
    return counts / counts.sum()


def independent_marginals(pool: Respondents, demographics: pd.DataFrame, splits: dict[str, list[str]],
                          seed: int = 42) -> Respondents:
    """Every answer drawn independently from the pool's per-item option shares (missing ignored)."""
    _check_pool(pool, splits)
    rng = np.random.default_rng(seed)
    n = len(demographics)
    cols = [rng.choice(len(it.options), size=n, p=_shares(pool.answers[it.code], len(it.options)))
            for it in pool.items]
    return _crowd(np.column_stack(cols), pool, demographics, "independent_marginals")


def _normal_scores(col: pd.Series) -> np.ndarray:
    """Mode-impute (for fitting only), then rank -> (rank - 0.5) / n -> standard normal quantile."""
    filled = col.fillna(col.mode().iloc[0])
    ranks = stats.rankdata(filled.to_numpy(), method="average")
    return stats.norm.ppf((ranks - 0.5) / len(ranks))


def _nearest_correlation(c: np.ndarray) -> np.ndarray:
    c = np.nan_to_num(c, nan=0.0)
    np.fill_diagonal(c, 1.0)
    vals, vecs = np.linalg.eigh((c + c.T) / 2)
    fixed = vecs @ np.diag(np.clip(vals, PSD_FLOOR, None)) @ vecs.T
    d = np.sqrt(np.diag(fixed))
    return fixed / np.outer(d, d)


def gaussian_copula(pool: Respondents, demographics: pd.DataFrame, splits: dict[str, list[str]],
                    seed: int = 42) -> Respondents:
    """Gaussian copula: rank-normal scores -> correlation -> sample -> each item's empirical quantiles."""
    _check_pool(pool, splits)
    z = np.column_stack([_normal_scores(pool.answers[it.code]) for it in pool.items])
    corr = _nearest_correlation(np.corrcoef(z, rowvar=False))
    rng = np.random.default_rng(seed)
    n = len(demographics)
    u = stats.norm.cdf(rng.multivariate_normal(np.zeros(len(pool.items)), corr, size=n, method="eigh"))
    cols = []
    for j, it in enumerate(pool.items):
        cum = np.cumsum(_shares(pool.answers[it.code], len(it.options)))
        cols.append(np.minimum(np.searchsorted(cum, u[:, j], side="right"), len(it.options) - 1))
    crowd = _crowd(np.column_stack(cols), pool, demographics, "gaussian_copula")
    crowd.meta["correlation"] = corr
    return crowd


def other_real_people(pool: Respondents, n: int, splits: dict[str, list[str]], seed: int = 42) -> Respondents:
    """A random sample of n real pool people: what a perfect generator of new people would look like."""
    _check_pool(pool, splits)
    if n > len(pool.answers):
        raise ValueError(f"pool has {len(pool.answers)} people, cannot sample {n}")
    idx = np.sort(np.random.default_rng(seed).choice(len(pool.answers), size=n, replace=False))
    crowd = pool.subset(pool.answers.index[idx])
    return Respondents(crowd.answers, crowd.demographics, crowd.items, "baseline:other_real_people",
                       meta={"fit_on": "pool", "n_pool": len(pool.answers)})

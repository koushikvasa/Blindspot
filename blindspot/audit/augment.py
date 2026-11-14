"""Statistical augmentation: new respondents from a Gaussian copula fitted on the REAL sample (numpy only).

Ordered answers (numeric / ordinal) keep their real marginals and the correlations between questions; categorical
answers are drawn from their real shares independently. The output never contains a real respondent's row and is always
labelled as built from real respondents' statistics.
"""
import math

import numpy as np

from .engine import Prepared

_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
      -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
      -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
      4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)


def ndtri(p: np.ndarray) -> np.ndarray:
    """Inverse normal CDF (Acklam's rational approximation, relative error about 1e-9)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-10, 1 - 1e-10)
    out = np.empty_like(p)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(p[lo]))
    out[lo] = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    out[hi] = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    q = p[mid] - 0.5
    r = q * q
    out[mid] = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)
    return out


def ndtr(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def _mode(values: np.ndarray) -> float:
    vals, counts = np.unique(values, return_counts=True)
    return float(vals[np.argmax(counts)])


def augment(prep: Prepared, n_rows: int, seed: int = 42) -> list[list[str]]:
    """n_rows new respondents as answer labels, in the order of `prep.columns`."""
    rng = np.random.default_rng(seed)
    n, q = prep.X_real.shape
    ordered = [j for j in range(q) if not prep.cat[j]]
    out = np.empty((n_rows, q))
    if ordered:
        z = np.empty((n, len(ordered)))
        sorted_cols = []
        for c, j in enumerate(ordered):
            col = prep.X_real[:, j].copy()
            col[np.isnan(col)] = _mode(col[~np.isnan(col)]) if (~np.isnan(col)).any() else 0.0
            s = np.sort(col)
            sorted_cols.append(s)
            mid = (np.searchsorted(s, col, "left") + np.searchsorted(s, col, "right")) / 2.0 / n
            z[:, c] = ndtri(mid)
        corr = np.nan_to_num(np.corrcoef(z, rowvar=False)) if len(ordered) > 1 else np.ones((1, 1))
        np.fill_diagonal(corr, 1.0)
        vals, vecs = np.linalg.eigh((corr + corr.T) / 2)
        fixed = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
        d = np.sqrt(np.diag(fixed))
        sample = ndtr(rng.multivariate_normal(np.zeros(len(ordered)), fixed / np.outer(d, d), size=n_rows, method="eigh"))
        for c, j in enumerate(ordered):
            s = sorted_cols[c]
            out[:, j] = s[np.minimum((sample[:, c] * n).astype(int), n - 1)]
    for j in range(q):
        if prep.cat[j]:
            col = prep.X_real[:, j]
            col = col[~np.isnan(col)]
            out[:, j] = rng.choice(col, size=n_rows) if len(col) else np.nan
    return [[prep.columns[j].decode(out[i, j]) for j in range(q)] for i in range(n_rows)]

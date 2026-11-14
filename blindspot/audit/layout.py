"""2-D map of the real sample and the AI panel for the report (classical MDS, numpy only).

Real people get one fixed layout (classical multidimensional scaling of the Gower distances, spread so no dot hides
another); each AI person is drawn at the mean position of their 5 nearest real people, as in the research deck.
"""
import numpy as np

NEAREST_REAL = 5
MAX_SYN_POINTS = 600


def classical_mds(d: np.ndarray) -> np.ndarray:
    n = len(d)
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d.astype(np.float64) ** 2) @ j
    vals, vecs = np.linalg.eigh((b + b.T) / 2)
    top = np.argsort(vals)[::-1][:2]
    return vecs[:, top] * np.sqrt(np.clip(vals[top], 1e-12, None))


def declutter(xy: np.ndarray, iterations: int = 60, seed: int = 42) -> np.ndarray:
    """Push dots apart until none is closer than a minimum distance (every person stays visible)."""
    min_d = 0.55 / np.sqrt(len(xy))
    out = xy.copy()
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        delta = out[:, None, :] - out[None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, np.inf)
        i, j = np.where(np.triu(dist < min_d, 1))
        if not len(i):
            break
        vec = out[i] - out[j]
        length = np.linalg.norm(vec, axis=1, keepdims=True)
        direction = np.where(length > 1e-9, vec / np.maximum(length, 1e-9), rng.normal(size=vec.shape))
        push = 0.5 * (min_d - length) * direction * 0.6
        np.add.at(out, i, push)
        np.add.at(out, j, -push)
        out = np.clip(out, 0.0, 1.0)
    return out


def build_map(d_rr: np.ndarray, d_rs: np.ndarray, covered: np.ndarray, seed: int = 42) -> dict:
    """`d_rr` has +inf on the diagonal (self excluded); `covered` is a boolean per real person."""
    d = np.where(np.isinf(d_rr), 0.0, d_rr)
    xy = classical_mds(d) if len(d) > 2 else np.random.default_rng(seed).random((len(d), 2))
    lo, hi = np.percentile(xy, [1, 99], axis=0)
    xy = declutter(np.clip((xy - lo) / np.where(hi > lo, hi - lo, 1.0), 0.0, 1.0), seed=seed)
    n_syn = d_rs.shape[1]
    keep = np.sort(np.random.default_rng(seed).choice(n_syn, min(n_syn, MAX_SYN_POINTS), replace=False))
    k = min(NEAREST_REAL, len(xy))
    near = np.argsort(d_rs[:, keep], axis=0)[:k]                       # k x n_keep indices of the nearest real people
    syn_xy = xy[near].mean(axis=0)
    return {"real": [[round(float(x), 4), round(float(y), 4), int(c)] for (x, y), c in zip(xy, covered)],
            "synthetic": [[round(float(x), 4), round(float(y), 4)] for x, y in syn_xy]}

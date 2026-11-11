"""Answer encoding and family-weighted Gower distance (H2).

Every answer becomes a value in [0, 1]: option index / (k - 1). The distance between two people is
the weighted mean absolute difference over the items both answered. Each item family gets the same
total weight (an item weighs 1 / items-in-its-family), so a family with 10 items counts no more than
a family with 2.
"""
import numpy as np

from blindspot.data.schema import Item, Respondents

DEFAULT_CHUNK = 128  # rows of A per block: chunk x len(B) x n_items float32 temporaries


def encode(resp: Respondents) -> np.ndarray:
    """People x items array in [0, 1]; NaN stays NaN."""
    scale = np.array([max(len(it.options) - 1, 1) for it in resp.items], dtype=np.float64)
    return resp.answers.to_numpy(dtype=np.float64) / scale


def family_weights(items: list[Item]) -> np.ndarray:
    """Per-item weights summing to 1, with every family carrying equal total weight."""
    families = [it.family for it in items]
    n_families = len(set(families))
    sizes = {f: families.count(f) for f in set(families)}
    return np.array([1.0 / (n_families * sizes[f]) for f in families])


def gower(A: np.ndarray, B: np.ndarray, items: list[Item], chunk: int = DEFAULT_CHUNK) -> np.ndarray:
    """Distance matrix len(A) x len(B) in [0, 1]. NaN where two people share no answered item."""
    if A.shape[1] != len(items) or B.shape[1] != len(items):
        raise ValueError("A and B must have one column per item")
    w = family_weights(items).astype(np.float32)
    a_mask, b_mask = ~np.isnan(A), ~np.isnan(B)
    a_val = np.nan_to_num(A).astype(np.float32)
    b_val = np.nan_to_num(B).astype(np.float32)
    bw = (b_mask * w).astype(np.float32)  # len(B) x items
    out = np.empty((len(A), len(B)), dtype=np.float64)
    for start in range(0, len(A), chunk):
        stop = min(start + chunk, len(A))
        weight = a_mask[start:stop, None, :] * bw[None, :, :]               # c x nB x items
        diff = np.abs(a_val[start:stop, None, :] - b_val[None, :, :])
        num = (diff * weight).sum(axis=2)
        den = weight.sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[start:stop] = np.where(den > 0, num / den, np.nan)
    return out

"""k-NN coverage and precision of a synthetic crowd against a real target set (H2).

Following Naeem et al. (2020), "Reliable Fidelity and Diversity Metrics for Generative Models":
each real person gets a ball whose radius is the distance to their k-th nearest *real* neighbour
(same target set, self excluded).
- coverage  : share of real people with at least one synthetic person inside their ball
- precision : share of synthetic people inside at least one real ball
"Inside" means distance <= radius (answers are discrete, so ties are common).

rcg_ratio is our own simple version of the respondent-coverage-gap idea (Kumagae & Kaneko, 2026):
median over real people of (distance to nearest synthetic) / (k-NN radius). The paper's exact
formula may differ. Values <= 1 mean the person is covered.

Bootstrap CIs (percentile, radii held fixed at their full-sample values):
- coverage: resample real people.
- precision: resample synthetic people. Resampling real people for precision is biased downward
  (duplicates shrink the union of balls), so its interval would not contain the point estimate.
"""
import numpy as np

from blindspot.data.schema import Item, Respondents
from blindspot.space import encode, gower

from .spread import family_entropy

DEFAULT_K = 5
K_SENSITIVITY = (3, 5, 10)
N_BOOT = 200


def real_distances(real: Respondents, items: list[Item]) -> np.ndarray:
    """Real x real Gower distances with self and no-overlap pairs set to +inf."""
    x = encode(real)
    d = np.nan_to_num(gower(x, x, items), nan=np.inf)
    np.fill_diagonal(d, np.inf)
    return d


def knn_radius(real_dist: np.ndarray, k: int) -> np.ndarray:
    if not 1 <= k < len(real_dist):
        raise ValueError(f"k must be in [1, {len(real_dist) - 1}], got {k}")
    return np.partition(real_dist, k - 1, axis=1)[:, k - 1]


def subsample(crowd: Respondents, n: int, seed: int) -> Respondents:
    """Fixed-seed subsample to n people (no-op if the crowd is not larger)."""
    if len(crowd.answers) <= n:
        return crowd
    idx = np.sort(np.random.default_rng(seed).choice(len(crowd.answers), size=n, replace=False))
    return crowd.subset(crowd.answers.index[idx])


def _ratio(nn_syn: np.ndarray, radius: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = nn_syn / radius
    return np.where((nn_syn == 0) & (radius == 0), 0.0, r)


def _ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _bootstrap(covered: np.ndarray, syn_inside: np.ndarray, n_boot: int, seed: int) -> dict:
    """Coverage: resample real people. Precision: resample synthetic people."""
    rng = np.random.default_rng(seed)
    cov = np.array([covered[rng.integers(0, len(covered), len(covered))].mean() for _ in range(n_boot)])
    prec = np.array([syn_inside[rng.integers(0, len(syn_inside), len(syn_inside))].mean()
                     for _ in range(n_boot)])
    return {"coverage_ci": _ci(cov), "precision_ci": _ci(prec)}


def evaluate(real_target: Respondents, crowd: Respondents, items: list[Item], k: int = DEFAULT_K,
             n_boot: int = N_BOOT, seed: int = 42, k_values: tuple[int, ...] = K_SENSITIVITY,
             real_dist: np.ndarray | None = None) -> dict:
    """Coverage metrics of `crowd` against `real_target` (crowd subsampled to the target size)."""
    if [i.code for i in real_target.items] != [i.code for i in crowd.items]:
        raise ValueError("crowd and target must share items in the same order")
    crowd = subsample(crowd, len(real_target.answers), seed)
    rr = real_dist if real_dist is not None else real_distances(real_target, items)
    rs = np.nan_to_num(gower(encode(real_target), encode(crowd), items), nan=np.inf)

    radius = knn_radius(rr, k)
    nn_syn = rs.min(axis=1)
    covered = nn_syn <= radius
    inside = rs <= radius[:, None]                       # real x synthetic
    ids = real_target.answers.index
    out = {
        "k": k,
        "n_real": int(len(ids)),
        "n_synthetic": int(len(crowd.answers)),
        "coverage": float(covered.mean()),
        "precision": float(inside.any(axis=0).mean()),
        "rcg_ratio": float(np.median(_ratio(nn_syn, radius))),
        "uncovered_ids": [str(i) for i in ids[~covered]],
        "per_family_entropy": {"crowd": family_entropy(crowd), "real": family_entropy(real_target)},
        "k_sensitivity": {},
        "_real_covered": covered,
        "_synthetic_inside": inside.any(axis=0),
    }
    out.update(_bootstrap(covered, out["_synthetic_inside"], n_boot, seed))
    for kk in k_values:
        r_k = knn_radius(rr, kk)
        out["k_sensitivity"][str(kk)] = {"coverage": float((nn_syn <= r_k).mean()),
                                         "precision": float((rs <= r_k[:, None]).any(axis=0).mean())}
    return out

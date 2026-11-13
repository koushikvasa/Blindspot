"""H4b step 1: which asked real people does the current crowd fail to imagine?

New rule: an asked person is uncovered if their distance to the nearest synthetic person exceeds the
95th percentile of synthetic-to-synthetic nearest-neighbour distances in the same crowd (a person the
crowd could not have produced even by its own standards). The old H4 rule (k-NN ball of the asked
set, crowd subsampled to match) is computed alongside for comparison.
"""
import numpy as np

from blindspot.agent.gaps import DEFAULT_K, uncovered_asked
from blindspot.data.schema import Item, Respondents
from blindspot.space import encode, gower

PERCENTILE = 95.0


def synthetic_nn_distances(crowd: Respondents, items: list[Item]) -> np.ndarray:
    x = encode(crowd)
    d = np.nan_to_num(gower(x, x, items), nan=np.inf)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def uncovered_by_percentile(asked: Respondents, crowd: Respondents, items: list[Item],
                            q: float = PERCENTILE) -> tuple[list[str], dict]:
    """Ids of asked people farther from every synthetic person than the q-th percentile of the crowd's
    own nearest-neighbour distances, plus the numbers behind the call."""
    finite = synthetic_nn_distances(crowd, items)
    finite = finite[np.isfinite(finite)]
    threshold = float(np.percentile(finite, q)) if len(finite) else 0.0
    nearest = np.nan_to_num(gower(encode(asked), encode(crowd), items), nan=np.inf).min(axis=1)
    unc = [str(i) for i in asked.answers.index[nearest > threshold]]
    ok = np.isfinite(nearest)
    return unc, {"threshold": round(threshold, 5), "q": q, "n_uncovered": len(unc), "n_asked": len(nearest),
                 "median_nearest_asked": round(float(np.median(nearest[ok])), 5) if ok.any() else None}


def detection_counts(asked: Respondents, crowd: Respondents, items: list[Item], k: int = DEFAULT_K,
                     seed: int = 42) -> tuple[list[str], dict]:
    """New-rule uncovered ids (used by the loop) with the old-rule count logged for comparison."""
    new_ids, info = uncovered_by_percentile(asked, crowd, items)
    old_ids = uncovered_asked(asked, crowd, items, k, seed)
    info.update(n_uncovered_old_rule=len(old_ids), n_uncovered_new_rule=len(new_ids),
                n_both=len(set(old_ids) & set(new_ids)))
    return new_ids, info

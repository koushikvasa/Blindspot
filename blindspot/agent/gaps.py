"""Gap finding (H4): which asked real people does the current crowd not cover, and what patterns do they share?

Coverage logic is H2's, with the ASKED set as the real reference. That reference is small (25-100
people), so its k-NN radii and the coverage verdicts are noisy; as in H2 the crowd is subsampled to
the size of the reference (fixed seed) so the comparison is density-matched.
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from blindspot.coverage import knn_radius, real_distances, subsample
from blindspot.data.schema import Item, Respondents
from blindspot.space import encode, gower

DEFAULT_K = 5
MIN_CLUSTERS = 2
MAX_CLUSTERS = 5
NO_OVERLAP_DISTANCE = 1.0


def uncovered_asked(asked: Respondents, crowd: Respondents, items: list[Item], k: int = DEFAULT_K,
                    seed: int = 42) -> list[str]:
    """Ids of asked real people with no synthetic person inside their k-NN ball (k clipped to n - 1)."""
    n = len(asked.answers)
    if n < 2:
        return []
    k = min(k, n - 1)
    radius = knn_radius(real_distances(asked, items), k)
    small = subsample(crowd, n, seed)
    nearest = np.nan_to_num(gower(encode(asked), encode(small), items), nan=np.inf).min(axis=1)
    return [str(i) for i in asked.answers.index[nearest > radius]]


def n_clusters_for(n_uncovered: int) -> int:
    if n_uncovered < 2:
        return n_uncovered
    return int(min(max(round(np.sqrt(n_uncovered / 2)), MIN_CLUSTERS), MAX_CLUSTERS, n_uncovered))


def cluster_uncovered(asked: Respondents, uncovered: list[str], items: list[Item]) -> list[list[str]]:
    """Agglomerative (average-linkage) clustering of the uncovered people on Gower distance.

    Deterministic (no randomness in the algorithm). Clusters are returned largest first, ties by id.
    """
    n = len(uncovered)
    if n == 0:
        return []
    if n == 1:
        return [list(uncovered)]
    x = encode(asked.subset(uncovered))
    d = np.nan_to_num(gower(x, x, items), nan=NO_OVERLAP_DISTANCE)
    np.fill_diagonal(d, 0.0)
    labels = AgglomerativeClustering(n_clusters=n_clusters_for(n), metric="precomputed",
                                     linkage="average").fit_predict(d)
    clusters = [sorted(uncovered[i] for i in np.flatnonzero(labels == c)) for c in sorted(set(labels))]
    return sorted(clusters, key=lambda c: (-len(c), c[0]))

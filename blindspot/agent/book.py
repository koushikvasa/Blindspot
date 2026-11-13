"""H4b: the cluster book. One entry per cluster of uncovered asked people (an answer pattern):
accepted recipe variants (or "cannot imagine"), the asked members currently assigned to it (a person's
latest assignment wins), and a centroid used to merge near-identical clusters across rounds.
"""
from dataclasses import dataclass, field

import numpy as np

from blindspot.agent.library import DEFAULT_MERGE_DISTANCE, centroid
from blindspot.agent.propose import Variant
from blindspot.data.schema import Item
from blindspot.space import gower


@dataclass
class ClusterEntry:
    id: str
    round_created: int
    members: set[str] = field(default_factory=set)
    centroid: np.ndarray | None = None
    variants: list[Variant] = field(default_factory=list)   # accepted variants; empty = cannot imagine
    variant_stats: list[dict] = field(default_factory=list)
    attempts: int = 0

    @property
    def cannot_imagine(self) -> bool:
        return not self.variants


class ClusterBook:
    def __init__(self, items: list[Item], merge_distance: float = DEFAULT_MERGE_DISTANCE):
        self.items, self.merge_distance = items, merge_distance
        self.entries: list[ClusterEntry] = []
        self._counter = 0

    def find_match(self, cluster_centroid: np.ndarray) -> tuple[ClusterEntry | None, float]:
        best, best_d = None, np.inf
        for e in self.entries:
            if e.centroid is None:
                continue
            d = gower(cluster_centroid[None, :], e.centroid[None, :], self.items)[0, 0]
            if not np.isnan(d) and d < best_d:
                best, best_d = e, float(d)
        return (best, best_d) if best is not None and best_d <= self.merge_distance else (None, best_d)

    def _claim(self, ids: set[str], target: ClusterEntry) -> None:
        for e in self.entries:
            if e is not target:
                e.members -= ids
        target.members |= ids

    def new_entry(self, ids: list[str], round_index: int) -> ClusterEntry:
        self._counter += 1
        e = ClusterEntry(f"C{self._counter}", round_index)
        self.entries.append(e)
        self._claim(set(ids), e)
        return e

    def merge_into(self, entry: ClusterEntry, ids: list[str]) -> None:
        self._claim(set(ids), entry)

    def refresh(self, encoded_by_id: dict[str, np.ndarray]) -> None:
        self.entries = [e for e in self.entries if e.members]
        for e in self.entries:
            e.centroid = centroid(np.vstack([encoded_by_id[i] for i in sorted(e.members)]))

    def accepted(self) -> list[ClusterEntry]:
        return [e for e in self.entries if e.variants]

    def snapshot(self, n_asked: int) -> list[dict]:
        return [{"id": e.id, "round_created": e.round_created, "n_members": len(e.members),
                 "share_of_asked": round(len(e.members) / max(n_asked, 1), 4), "attempts": e.attempts,
                 "status": "cannot_imagine" if e.cannot_imagine else "accepted",
                 "variants": [{"name": v.name, "text": v.text} for v in e.variants],
                 "variant_stats": e.variant_stats} for e in self.entries]

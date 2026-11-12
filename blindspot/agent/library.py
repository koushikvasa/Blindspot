"""The recipe library (H4): archetypes with weights, merged across rounds, versioned.

An archetype is a plain-language answer pattern (never answer rows). Its members are the asked real
people currently assigned to it (a person's latest assignment wins); its weight is
members / asked people, and the remaining weight means "no archetype" (plain claude_mode).
New clusters whose answer-pattern centroid is within `merge_distance` of an existing archetype are
merged into it instead of creating a near-duplicate archetype.
"""
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from blindspot.data.schema import Item
from blindspot.space import gower

DEFAULT_MERGE_DISTANCE = 0.12
RECIPE_FRAME = ("Background on this respondent: their answers tend to follow a particular pattern, described "
                "below. Use it to decide how their answers go together; it is a tendency, not a list of answers.\n")


@dataclass
class Archetype:
    id: str
    name: str
    text: str
    round_created: int
    members: set[str] = field(default_factory=set)
    centroid: np.ndarray | None = None
    share_low: float | None = None      # the agent's own estimate (fractions), from the description call
    share_high: float | None = None

    def recipe(self) -> str:
        return f"{RECIPE_FRAME}{self.name}: {self.text}"


def centroid(encoded_rows: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN columns stay NaN
        return np.nanmean(encoded_rows, axis=0)


class RecipeLibrary:
    def __init__(self, items: list[Item], merge_distance: float = DEFAULT_MERGE_DISTANCE):
        self.items = items
        self.merge_distance = merge_distance
        self.archetypes: list[Archetype] = []
        self.version = 0
        self._counter = 0

    # ---- learning ----
    def find_match(self, cluster_centroid: np.ndarray) -> tuple[Archetype | None, float]:
        """Nearest existing archetype by centroid distance, if within `merge_distance`."""
        best, best_d = None, np.inf
        for a in self.archetypes:
            if a.centroid is None:
                continue
            d = gower(cluster_centroid[None, :], a.centroid[None, :], self.items)[0, 0]
            if not np.isnan(d) and d < best_d:
                best, best_d = a, float(d)
        return (best, best_d) if best is not None and best_d <= self.merge_distance else (None, best_d)

    def _claim(self, ids: set[str], target: Archetype) -> None:
        for a in self.archetypes:
            if a is not target:
                a.members -= ids
        target.members |= ids

    def merge_into(self, archetype: Archetype, ids: list[str]) -> None:
        self._claim(set(ids), archetype)

    def add(self, name: str, text: str, ids: list[str], round_index: int,
            share_low: float | None = None, share_high: float | None = None) -> Archetype:
        self._counter += 1
        a = Archetype(f"A{self._counter}", name, text, round_index, set(), None, share_low, share_high)
        self.archetypes.append(a)
        self._claim(set(ids), a)
        return a

    def refresh(self, encoded_by_id: dict[str, np.ndarray]) -> None:
        """Recompute centroids from current members; drop archetypes that lost all members."""
        self.archetypes = [a for a in self.archetypes if a.members]
        for a in self.archetypes:
            a.centroid = centroid(np.vstack([encoded_by_id[i] for i in sorted(a.members)]))
        self.version += 1

    # ---- using ----
    def weights(self, n_asked: int, max_share: float = 1.0) -> dict[str, float]:
        """Archetype id -> weight; scaled down proportionally if the total exceeds `max_share`."""
        if n_asked <= 0:
            return {}
        raw = {a.id: len(a.members) / n_asked for a in self.archetypes}
        total = sum(raw.values())
        scale = min(1.0, max_share / total) if total > 0 else 1.0
        return {k: v * scale for k, v in raw.items()}

    def recipes_for(self, uniforms: np.ndarray, n_asked: int, max_share: float = 1.0) -> list[str]:
        """Recipe text per synthetic person from fixed uniforms in [0, 1): archetype by cumulative
        weight, '' (no archetype) for the remaining weight. Stable draws mean small weight changes
        move few people, so most calls stay cached across rounds."""
        w = self.weights(n_asked, max_share)
        bounds = np.cumsum([w[a.id] for a in self.archetypes]) if self.archetypes else np.array([])
        out = []
        for u in uniforms:
            idx = int(np.searchsorted(bounds, u, side="right"))
            out.append(self.archetypes[idx].recipe() if idx < len(self.archetypes) else "")
        return out

    def snapshot(self, n_asked: int, max_share: float = 1.0) -> dict:
        w = self.weights(n_asked, max_share)
        return {"version": self.version, "n_asked": n_asked, "no_archetype_weight": round(1 - sum(w.values()), 4),
                "archetypes": [{"id": a.id, "name": a.name, "text": a.text, "weight": round(w[a.id], 4),
                                "n_members": len(a.members), "round_created": a.round_created,
                                "share_estimate_by_agent": [a.share_low, a.share_high]}
                               for a in self.archetypes]}

    def save(self, path: Path, n_asked: int, max_share: float = 1.0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(n_asked, max_share), indent=1), encoding="utf-8")

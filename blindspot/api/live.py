"""Live "can the AI imagine you?" scoring. Answers are scored in memory and never written anywhere."""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

K = 5


class LiveScorer:
    def __init__(self, bundle: dict):
        self.items = bundle["items"]
        self.codes = [it["code"] for it in self.items]
        fams = [it.get("family", "all") for it in self.items]
        fw = bundle.get("family_weights") or {}
        uniq = sorted(set(fams))
        w = np.array([fw.get(f, 1 / len(uniq)) / fams.count(f) for f in fams], float)
        self.w = w / w.sum()
        self.real = np.array(bundle["real"], float)       # NaN for null
        self.real_xy = np.array(bundle["real_xy"], float)
        self.crowd = np.array(bundle["crowd"], float)
        self.crowd_label = bundle.get("crowd_label", "AI crowd")
        self.base_rate = bundle.get("base_rate_uncovered")
        # each real person's own nearest-synthetic distance, for a percentile
        self.real_min_syn = np.array([self._dist(r, self.crowd).min() for r in self.real])

    def _dist(self, v: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Weighted Gower distance from vector v to each row of M, ignoring missing values."""
        mask = ~np.isnan(M) & ~np.isnan(v)[None, :]
        diff = np.where(mask, np.abs(M - np.nan_to_num(v)[None, :]), 0.0)
        wsum = (mask * self.w[None, :]).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = (diff * self.w[None, :]).sum(axis=1) / wsum
        return np.where(wsum > 0, d, 1.0)

    def encode(self, answers: dict) -> np.ndarray:
        v = np.full(len(self.items), np.nan)
        for j, it in enumerate(self.items):
            a = answers.get(it["code"])
            k = len(it["options"])
            if a is None:
                continue
            a = int(a)
            if not 0 <= a < k:
                raise ValueError(f"answer out of range for {it['code']}")
            v[j] = a / (k - 1) if k > 1 else 0.0
        if np.isnan(v).sum() > len(v) // 2:
            raise ValueError("please answer at least half of the questions")
        return v

    def score(self, answers: dict) -> dict:
        v = self.encode(answers)
        d_real = self._dist(v, self.real)
        order = np.argsort(d_real)
        radius = d_real[order[K - 1]]
        d_syn = self._dist(v, self.crowd).min()
        x, y = self.real_xy[order[:K]].mean(axis=0)
        return {
            "covered": bool(d_syn <= radius),
            "distance_ratio": float(d_syn / radius) if radius > 0 else None,
            "percentile_farther_than": float((self.real_min_syn < d_syn).mean()),
            "x": float(x), "y": float(y),
            "base_rate_uncovered": self.base_rate,
            "crowd_label": self.crowd_label,
        }


class VisitorFeed:
    """Last N anonymous visitor points for the stage view (position + covered only)."""

    def __init__(self, size: int = 30):
        self._q: deque = deque(maxlen=size)
        self._seq = 0
        self._lock = threading.Lock()

    def add(self, x: float, y: float, covered: bool) -> int:
        with self._lock:
            self._seq += 1
            self._q.append({"seq": self._seq, "x": x, "y": y, "covered": covered, "t": time.time()})
            return self._seq

    def since(self, after: int) -> list[dict]:
        with self._lock:
            return [p for p in self._q if p["seq"] > after]

    def reset(self):
        with self._lock:
            self._q.clear()

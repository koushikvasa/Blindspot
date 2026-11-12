"""Asking for real people (H4): the agent draws people from the POOL split only.

Before asking, the agent sees only demographics (like a panel quota); after asking it sees the
people's answers. These functions receive the pool's demographics table and nothing else, so they
cannot return a test or holdout id; callers still run assert_no_leak on the result.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_SEGMENTS = 3
MAX_CONDITIONS = 2


@dataclass(frozen=True)
class Segment:
    """A demographic quota cell, e.g. age_band == '65+' and region == 'South'."""
    conditions: tuple[tuple[str, str], ...]
    reason: str = ""

    def label(self) -> str:
        return " & ".join(f"{f}={v}" for f, v in self.conditions)

    def mask(self, demographics: pd.DataFrame) -> pd.Series:
        m = pd.Series(True, index=demographics.index)
        for field, value in self.conditions:
            if field not in demographics:
                return pd.Series(False, index=demographics.index)
            m &= demographics[field].astype(str).str.casefold() == value.casefold()
        return m

    def to_dict(self) -> dict:
        return {"conditions": dict(self.conditions), "reason": self.reason}


def draw_random(pool_ids: list[str], already: set[str], n: int, rng: np.random.Generator) -> list[str]:
    """n pool people not asked yet, uniformly at random."""
    eligible = sorted(set(pool_ids) - already)
    if n > len(eligible):
        raise ValueError(f"only {len(eligible)} pool people left, need {n}")
    return sorted(rng.choice(eligible, size=n, replace=False).tolist())


def _split_evenly(n: int, parts: int) -> list[int]:
    base, extra = divmod(n, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def draw_targeted(pool_demo: pd.DataFrame, already: set[str], segments: list[Segment], n: int,
                  rng: np.random.Generator) -> tuple[list[str], list[dict]]:
    """n pool people drawn from the requested segments (split evenly), falling back to random pool
    people for any shortfall. Returns (ids, per-segment report)."""
    if not segments:
        return draw_random(list(pool_demo.index), already, n, rng), []
    chosen: list[str] = []
    report = []
    for segment, want in zip(segments, _split_evenly(n, len(segments))):
        eligible = sorted(set(pool_demo.index[segment.mask(pool_demo)]) - already - set(chosen))
        take = min(want, len(eligible))
        picked = sorted(rng.choice(eligible, size=take, replace=False).tolist()) if take else []
        chosen += picked
        report.append({**segment.to_dict(), "label": segment.label(), "requested": want, "found": take,
                       "available": len(eligible)})
    shortfall = n - len(chosen)
    if shortfall:
        chosen += draw_random(list(pool_demo.index), already | set(chosen), shortfall, rng)
    return sorted(chosen), report + ([{"fallback_random": shortfall}] if shortfall else [])

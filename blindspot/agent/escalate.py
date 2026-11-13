"""H4b step 4: when a cluster cannot be imagined, ask for more people where the data says gaps are densest.

The segment is chosen from DATA only: the observed uncovered rate (uncovered / asked) of each
single-field demographic segment among the people asked so far, requiring at least MIN_SEGMENT_N people.
If no segment qualifies (or none has any uncovered person), the next round asks at random. The rate
behind every choice is logged. Nothing model-written is used here; any model-written demographic
reason elsewhere is unverified and must never be shown in the UI.
"""
import pandas as pd

from blindspot.agent.asking import Segment
from blindspot.generators.claude_personas import PERSONA_FIELDS

MIN_SEGMENT_N = 5
TOP_LOGGED = 8
MISSING = {"", "unknown", "nan", "none"}


def segment_rates(asked_demo: pd.DataFrame, uncovered: set[str], fields: tuple[str, ...] = PERSONA_FIELDS,
                  min_n: int = MIN_SEGMENT_N) -> tuple[list[dict], int]:
    """Qualifying segments (n >= min_n) sorted by rate desc, n desc, label; plus how many were too small."""
    rows, too_small = [], 0
    for f in fields:
        if f not in asked_demo:
            continue
        values = asked_demo[f].astype(str)
        for v in sorted(values.unique()):
            if v.lower() in MISSING:
                continue
            ids = asked_demo.index[values == v]
            if len(ids) < min_n:
                too_small += 1
                continue
            n_unc = sum(i in uncovered for i in ids)
            rows.append({"field": f, "value": v, "n": len(ids), "n_uncovered": n_unc,
                         "rate": round(n_unc / len(ids), 4)})
    rows.sort(key=lambda r: (-r["rate"], -r["n"], r["field"], r["value"]))
    return rows, too_small


def choose_segment(asked_demo: pd.DataFrame, uncovered: set[str], cannot_imagine: list[str]
                   ) -> tuple[Segment | None, dict]:
    """Escalation decision for the next round, with the log behind it."""
    log: dict = {"triggered": bool(cannot_imagine), "cannot_imagine_clusters": list(cannot_imagine)}
    if not cannot_imagine:
        log["next_round"] = "random (no cluster needs escalation)"
        return None, log
    rows, too_small = segment_rates(asked_demo, uncovered)
    log.update(n_uncovered_asked=len(uncovered), n_asked=len(asked_demo), segments_below_min_n=too_small,
               top_segments=rows[:TOP_LOGGED], min_segment_n=MIN_SEGMENT_N)
    if not rows or rows[0]["rate"] == 0:
        log["next_round"] = "random (no segment with >= %d people and a non-zero uncovered rate)" % MIN_SEGMENT_N
        return None, log
    best = rows[0]
    log["next_round"] = f"targeted: {best['field']}={best['value']} (observed uncovered rate {best['rate']:.1%} " \
                        f"= {best['n_uncovered']}/{best['n']})"
    log["chosen"] = best
    return Segment(((best["field"], best["value"]),), f"highest observed uncovered rate {best['rate']:.1%}"), log

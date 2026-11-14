"""Room version of "can the AI imagine you?" (Hackathon Working Doc): 7 short questions (v2), scored live against TWO AI crowds.

- Claude crowd: the claude_mode crowd from results/ui/live_gss.json (334 synthetic people, test split, 20 live items).
- Flashpoint panel: the 200-persona Flashpoint synthetic panel shipped with the audit sample
  (blindspot/audit/sample/ai_panel_flashpoint.csv, same 20 GSS items, verbatim wording).

Items: the 7 (version 2; version 1 had 10) whose answers differ most between real people the Claude crowd covers and the ones it misses
(total-variation distance on the 334 test-split people), ordered easy first (work-life, tech) and money/trust last.
Short version caveat from the doc: coverage on 10 items is a different number from the 20-item headline, so every
rate this module returns is labelled as the room version (room-7) and must not be mixed with the 20-item results.

Visitor answers are scored in memory and never stored; only aggregates (shares, percentiles, rates) are returned.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audit.parse import is_missing, parse_csv
from .live import K, LiveScorer

FP_CSV = Path(__file__).resolve().parents[1] / "audit" / "sample" / "ai_panel_flashpoint.csv"

# code -> (question shown, topic for the result line, choices as (label, index into the official GSS options)).
# Version 2 after 8 simulated testers: 7 questions (the 7 that best separate covered from missed people), plain short
# wording, and 5 labelled buttons instead of 0-10 rows (each button maps to a point on the official 0-10 scale).
SCALE = lambda lo, a, mid, b, hi: [(lo, 0), (a, 3), (mid, 5), (b, 7), (hi, 10)]
ROOM = {
    "aidrive":  ("Ride in a driverless car?", "driverless cars", SCALE("No way", "Uneasy", "Not sure", "Fine", "Love it")),
    "aiworry":  ("Worried AI will take people's jobs?", "AI taking jobs",
                 [("Very", 0), ("Somewhat", 1), ("Neutral", 2), ("Not really", 3), ("Not at all", 4)]),
    "trppl":    ("Can most people be trusted?", "trusting people",
                 SCALE("No, be careful", "Mostly not", "Depends", "Mostly", "Yes")),
    "aimed":    ("A robot doing your surgery?", "robot surgery", SCALE("No way", "Uneasy", "Not sure", "Fine", "Love it")),
    "econstat": ("How's the US economy right now?", "the US economy",
                 [("Very good", 0), ("Good", 1), ("So-so", 2), ("Bad", 3), ("Very bad", 4)]),
    "satfin":   ("Happy with your money situation?", "your money situation", [("Yes", 0), ("Kind of", 1), ("No", 2)]),
    "trcourts": ("Trust US courts?", "the courts",
                 SCALE("Not at all", "A little", "Somewhat", "Mostly", "Completely")),
}
CODES = list(ROOM)


def _norm(s: object) -> str:
    return " ".join(str(s).replace("’", "'").replace("‘", "'").lower().split()).rstrip(".")


def load_flashpoint(items: list[dict], path: Path = FP_CSV) -> np.ndarray:
    """Flashpoint labelled export -> people x items, encoded like live_gss (option index / (k - 1), NaN if blank)."""
    table = parse_csv(path.read_text(encoding="utf-8"), path.name)
    col = {_norm(h): j for j, h in enumerate(table.headers)}
    out = np.full((table.n_rows, len(items)), np.nan)
    for j, it in enumerate(items):
        c = col.get(_norm(it["wording"]))
        if c is None:
            raise ValueError(f"Flashpoint panel has no column for {it['code']}")
        k = len(it["options"])
        lut = {_norm(o): i for i, o in enumerate(it["options"])}
        if k == 11:
            lut.update({str(i): i for i in range(11)})
        for r, row in enumerate(table.rows):
            if not is_missing(row[c]) and _norm(row[c]) in lut:
                out[r, j] = lut[_norm(row[c])] / (k - 1)
    return out


class RoomScorer:
    def __init__(self, bundle: dict, fp_path: Path = FP_CSV):
        by_code = {it["code"]: j for j, it in enumerate(bundle["items"])}
        cols = [by_code[c] for c in CODES]
        self.items = [bundle["items"][j] for j in cols]
        real = np.array(bundle["real"], float)[:, cols]
        crowds = {
            "claude": (np.array(bundle["crowd"], float)[:, cols], "Claude crowd"),
            "flashpoint": (load_flashpoint(bundle["items"], fp_path)[:, cols], "Flashpoint panel"),
        }
        self.scorers = {name: LiveScorer({**bundle, "items": self.items, "real": real.tolist(), "crowd": M.tolist(),
                                          "crowd_label": label})
                        for name, (M, label) in crowds.items()}
        self.real = real
        self.crowd_n = {name: len(M) for name, (M, _) in crowds.items()}
        self.miss_rate = {name: self._miss_rate(s) for name, s in self.scorers.items()}
        self.real_share = [self._bucket_shares(real[:, j], it) for j, it in enumerate(self.items)]

    @staticmethod
    def _shares(col: np.ndarray, k: int) -> np.ndarray:
        col = col[~np.isnan(col)]
        counts = np.bincount(np.rint(col * (k - 1)).astype(int), minlength=k)[:k].astype(float)
        return counts / max(1.0, counts.sum())

    @staticmethod
    def bucket(code: str, idx: int) -> int:
        """Which button (position in ROOM choices) an official option index falls into: the nearest button."""
        vals = [v for _, v in ROOM[code][2]]
        return min(range(len(vals)), key=lambda b: (abs(vals[b] - idx), b))

    def _bucket_shares(self, col: np.ndarray, it: dict) -> np.ndarray:
        k, n = len(it["options"]), len(ROOM[it["code"]][2])
        idx = np.rint(col[~np.isnan(col)] * (k - 1)).astype(int)
        counts = np.bincount([self.bucket(it["code"], int(i)) for i in idx], minlength=n).astype(float)
        return counts / max(1.0, counts.sum())

    @staticmethod
    def _miss_rate(s: LiveScorer) -> float:
        """Share of the 334 real people this crowd cannot imagine on these 10 items (each person left out of their own
        neighbourhood, the same rule the visitor is scored with)."""
        missed = 0
        for r in s.real:
            radius = np.sort(s._dist(r, s.real))[K]          # index 0 is the person themself
            missed += s._dist(r, s.crowd).min() > radius
        return round(float(missed) / len(s.real), 3)

    def questions(self) -> list[dict]:
        return [{"code": it["code"], "question": ROOM[it["code"]][0], "wording": it["wording"],
                 "choices": [{"label": lab, "value": v} for lab, v in ROOM[it["code"]][2]]} for it in self.items]

    def score(self, answers: dict) -> dict:
        verdicts, raw = {}, {}
        for name, s in self.scorers.items():
            r = raw[name] = s.score(answers)                               # validates answers (range, at least half answered)
            verdicts[name] = {"label": s.crowd_label, "covered": r["covered"],
                              "harder_than": round(r["percentile_farther_than"], 3), "miss_rate": self.miss_rate[name]}

        # one line of why: where the visitor and the closest Claude person disagree most
        s = self.scorers["claude"]
        v = s.encode(answers)
        twin = s.crowd[int(np.argmin(s._dist(v, s.crowd)))]
        diffs = np.where(np.isnan(v) | np.isnan(twin), -1.0, np.abs(np.nan_to_num(twin) - np.nan_to_num(v)))
        j = int(np.argmax(diffs))
        it, k = self.items[j], len(self.items[j]["options"])
        code, choices = it["code"], ROOM[it["code"]][2]
        mine, twin_b = self.bucket(code, int(round(v[j] * (k - 1)))), self.bucket(code, int(round(twin[j] * (k - 1))))
        answered = int((~np.isnan(v)).sum())
        why = {"question": ROOM[code][0], "topic": ROOM[code][1], "yours": choices[mine][0], "ai": choices[twin_b][0],
               "real_share": round(float(self.real_share[j][mine]), 3), "differs": mine != twin_b, "answered": answered}
        return {"version": "room-7", "verdicts": verdicts, "why": why, "x": raw["claude"]["x"], "y": raw["claude"]["y"],
                "real_n": len(self.real), "crowd_n": self.crowd_n}

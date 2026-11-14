"""Adaptive room survey: each next question depends on the answers so far, chosen from the 20 live GSS items.

Two rules, both from the survey literature, decide the next question:

1. Follow the energy (laddering; Reynolds & Gutman 1988, J. Advertising Research). A strong answer (first or last button)
   among the last two answers means the next question stays on that topic and moves one rung up the ladder from concrete
   experience to personal consequence to values ("a driverless car?" -> "a robot doing YOUR surgery?" -> "AI taking
   jobs?" -> "does tech do more harm than good?"). At most MAX_PER_TOPIC questions per topic so one passion does not
   take over the whole survey.
2. Otherwise, ask where people like you split (computerized adaptive testing: ask the most informative item next;
   Toubia, Simester, Hauser & Dahan 2003, Marketing Science, for adaptive question design with few questions). Among the
   K_NEIGHBOURS real GSS respondents closest to the visitor so far, pick the unanswered question whose answers among
   them are most spread out (normalised entropy over the buttons). That question best separates the visitor from the
   people who look like them, which is also what decides whether an AI crowd has someone like them.

Length stays at N_QUESTIONS = 7 (Galesic & Bosnjak 2009, Public Opinion Quarterly: longer surveys lose finishers).
Everyone starts with the same two openers so first impressions are comparable.

Scoring uses the same rule as the research (k-NN coverage, family-weighted Gower) on exactly the questions the visitor
answered, so the crowd miss rates shown are computed on that same set of questions ("adaptive-7"); they are not
comparable with the fixed room-7 or the 20-question headline. Answers are scored in memory and never stored.
"""
from __future__ import annotations

from collections import OrderedDict
from threading import Lock

import numpy as np

from .live import LiveScorer
from .room import BANK, RoomScorer, extreme_values, has_ends

N_QUESTIONS = 7
OPENERS = ("aidrive", "trppl")
MAX_PER_TOPIC = 4
K_NEIGHBOURS = 40
SCORER_CACHE = 128

# Laddering order per question family: concrete experience -> personal consequence -> value/worldview
# (attributes -> consequences -> values in means-end terms). "Deeper" moves one rung on from the strong answer.
LADDERS = {
    "tech_science": ["aidrive", "aimed", "aiworry", "harmgood1", "techesy", "natsci"],
    "trust_institutions": ["intcnfrm", "trcourts", "trppl"],
    "money_economy": ["finalter", "satfin", "finrela", "class", "econstat"],
    "work_life": ["socfrend", "socrel", "workless", "meovrwrk", "life", "happy"],
}
ENERGY_WINDOW = 2          # a strong answer among the last 2 still steers the next question
TOPIC_NAMES = {"tech_science": "tech and AI", "trust_institutions": "trust", "money_economy": "money",
               "work_life": "life and family"}


class PathError(ValueError):
    """The path or answers sent by the page are not a valid adaptive path."""


class AdaptivePlanner:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.all = LiveScorer({**bundle, "crowd": bundle["crowd"][:1]})      # distance on all 20 items
        self.codes = [it["code"] for it in bundle["items"]]
        self.index = {c: j for j, c in enumerate(self.codes)}
        self.family = {it["code"]: it.get("family", "all") for it in bundle["items"]}
        self.k_options = {it["code"]: len(it["options"]) for it in bundle["items"]}
        missing = set(self.codes) - set(BANK)
        if missing:
            raise ValueError(f"no short wording for {sorted(missing)}")
        self._scorers: OrderedDict[tuple, RoomScorer] = OrderedDict()
        self._lock = Lock()

    # ---------- validation ----------
    def check(self, answers: dict, path: list[str]) -> None:
        if len(path) > N_QUESTIONS or len(set(path)) != len(path):
            raise PathError("the question path is too long or repeats a question")
        if any(c not in self.index for c in path):
            raise PathError("unknown question in the path")
        if set(answers) - set(path):
            raise PathError("answers must belong to the question path")
        for c, v in answers.items():
            if v is not None and not (isinstance(v, int) and 0 <= v < self.k_options[c]):
                raise PathError(f"answer out of range for {c}")

    # ---------- choosing the next question ----------
    def question(self, code: str, reason: str = "opener", step: int = 0) -> dict:
        short, topic, choices = BANK[code]
        it = self.bundle["items"][self.index[code]]
        fam = self.family[code]
        return {"code": code, "question": short, "topic": topic, "wording": it["wording"],
                "family": fam, "family_name": TOPIC_NAMES.get(fam, fam), "reason": reason, "step": step,
                "choices": [{"label": lab, "value": v} for lab, v in choices]}

    def _neighbours(self, answers: dict) -> np.ndarray:
        v = np.full(len(self.codes), np.nan)
        for c, a in answers.items():
            if a is not None:
                v[self.index[c]] = a / (self.k_options[c] - 1)
        d = self.all._dist(v, self.all.real)
        return np.argsort(d, kind="stable")[:K_NEIGHBOURS]

    def _spread(self, code: str, rows: np.ndarray) -> float:
        """Normalised entropy of the neighbours' answers on this question, over its buttons."""
        col = self.all.real[rows, self.index[code]]
        col = col[~np.isnan(col)]
        n_buttons = len(BANK[code][2])
        if len(col) < 2 or n_buttons < 2:
            return 0.0
        idx = np.rint(col * (self.k_options[code] - 1)).astype(int)
        counts = np.bincount([RoomScorer.bucket(code, int(i)) for i in idx], minlength=n_buttons).astype(float)
        p = counts[counts > 0] / counts.sum()
        return float(-(p * np.log(p)).sum() / np.log(n_buttons))

    def _ladder_rank(self, code: str) -> int:
        ladder = LADDERS.get(self.family[code], [])
        return ladder.index(code) if code in ladder else 99

    @staticmethod
    def _strong(answers: dict, code: str) -> bool:
        a = answers.get(code)
        return a is not None and has_ends(code) and a in extreme_values(code)

    def _deeper(self, answers: dict, path: list[str], per_topic: dict) -> str | None:
        """Next rung up the ladder from the most recent strong answer among the last ENERGY_WINDOW answers."""
        for src in reversed(path[-ENERGY_WINDOW:]):
            fam = self.family[src]
            if not self._strong(answers, src) or per_topic.get(fam, 0) >= MAX_PER_TOPIC:
                continue
            ladder = LADDERS.get(fam, [])
            pos = ladder.index(src) if src in ladder else -1
            rungs = [c for c in ladder[pos + 1:] + ladder[:max(pos, 0)] if c not in path]
            if rungs:
                return rungs[0]
        return None

    def next(self, answers: dict, path: list[str]) -> dict | None:
        """The next question for this path, or None when the survey is complete."""
        self.check(answers, path)
        step = len(path)
        if step >= N_QUESTIONS:
            return None
        if step < len(OPENERS):
            return self.question(OPENERS[step], "opener", step)
        per_topic = {f: sum(self.family[c] == f for c in path) for f in LADDERS}
        deeper = self._deeper(answers, path, per_topic)
        if deeper:
            return self.question(deeper, "deeper", step)
        rows = self._neighbours(answers)
        candidates = [c for c in self.codes if c not in path and per_topic.get(self.family[c], 0) < MAX_PER_TOPIC]
        if not candidates:
            return None
        best = max(candidates, key=lambda c: (round(self._spread(c, rows), 6), -self._ladder_rank(c)))
        return self.question(best, "split", step)

    # ---------- scoring a finished path ----------
    def scorer(self, path: list[str]) -> RoomScorer:
        key = tuple(sorted(path))
        with self._lock:
            if key in self._scorers:
                self._scorers.move_to_end(key)
                return self._scorers[key]
        s = RoomScorer(self.bundle, codes=list(path), version=f"adaptive-{N_QUESTIONS}")
        with self._lock:
            self._scorers[key] = s
            while len(self._scorers) > SCORER_CACHE:
                self._scorers.popitem(last=False)
        return s

    def score(self, answers: dict, path: list[str]) -> dict:
        self.check(answers, path)
        if len(path) < N_QUESTIONS:
            raise PathError(f"answer all {N_QUESTIONS} questions (Skip counts) before the verdict")
        result = self.scorer(path).score({c: answers.get(c) for c in path})
        result["path"] = self.describe(answers, path)
        return result

    def describe(self, answers: dict, path: list[str]) -> list[dict]:
        """What the visitor was asked and answered, in order, with how many real people gave the same button."""
        out = []
        for k, c in enumerate(path):
            short, topic, choices = BANK[c]
            a = answers.get(c)
            share, label = None, None
            if a is not None:
                b = RoomScorer.bucket(c, a)
                label = choices[b][0]
                col = self.all.real[:, self.index[c]]
                idx = np.rint(col[~np.isnan(col)] * (self.k_options[c] - 1)).astype(int)
                share = round(float(np.mean([RoomScorer.bucket(c, int(i)) == b for i in idx])), 3) if len(idx) else None
            before = {p: answers.get(p) for p in path[:k] if p in answers}
            planned = self.next(before, path[:k])             # replay the planner to label why this question came
            reason = planned["reason"] if planned and planned["code"] == c else "chosen"
            out.append({"code": c, "question": short, "topic": topic, "family": self.family[c],
                        "family_name": TOPIC_NAMES.get(self.family[c], self.family[c]), "answer": label,
                        "real_share": share, "reason": reason})
        return out

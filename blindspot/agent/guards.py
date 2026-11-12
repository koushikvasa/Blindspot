"""Guards for the self-improving loop (H4): no copying (CLAUDE.md rule 2) and no leaks.

1. Archetype copy check: an archetype describes tendencies and how answers go together. It must not
   restate answers item by item. An item counts as "copied" when the text mentions it (the item code
   or any 4 consecutive words of its wording that no other item shares, so common stems such as
   "would you support or oppose" do not count) AND quotes one of its answer labels verbatim (labels of
   at least MIN_LABEL_CHARS characters, so "yes"/"no" cannot be attributed to a specific item).
   More than MAX_ITEMS_COPIED copied items is a violation.
2. Near-duplicate check: after regeneration, the share of synthetic people closer than
   NEAR_DUPLICATE_DISTANCE (family-weighted Gower) to any asked real person. Above
   MAX_NEAR_DUPLICATE_SHARE the round fails loudly.
"""
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from blindspot.data.schema import Item, Respondents
from blindspot.space import encode, gower

MAX_ARCHETYPE_WORDS = 120
MAX_ITEMS_COPIED = 3
WORDING_WINDOW = 4
MIN_LABEL_CHARS = 4
NEAR_DUPLICATE_DISTANCE = 0.02
MAX_NEAR_DUPLICATE_SHARE = 0.05


class ArchetypeCopyError(ValueError):
    """An archetype restates answers item by item."""


class NearDuplicateError(RuntimeError):
    """Too many synthetic people are near-copies of asked real people; the round is invalid."""


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _wording_windows(wording: str) -> list[str]:
    words = _norm(wording).split()
    if len(words) <= WORDING_WINDOW:
        return [" ".join(words)] if words else []
    return [" ".join(words[i:i + WORDING_WINDOW]) for i in range(len(words) - WORDING_WINDOW + 1)]


def _distinctive_windows(items: list[Item]) -> list[set[str]]:
    """Per item, the wording windows that occur in no other item (shared stems like 'Would you support
    or oppose' must not make every item look mentioned)."""
    per_item = [set(_wording_windows(it.wording)) for it in items]
    counts = Counter(w for windows in per_item for w in windows)
    return [{w for w in windows if counts[w] == 1} for windows in per_item]


def copied_items(text: str, items: list[Item]) -> list[str]:
    """Codes of items whose code or distinctive wording is mentioned together with one of their answer labels."""
    haystack = f" {_norm(text)} "
    out = []
    for it, windows in zip(items, _distinctive_windows(items)):
        mentioned = f" {_norm(it.code)} " in haystack or any(f" {w} " in haystack for w in windows)
        if not mentioned:
            continue
        labels = [_norm(label) for label in it.options]
        if any(len(label) >= MIN_LABEL_CHARS and f" {label} " in haystack for label in labels):
            out.append(it.code)
    return out


def word_count(text: str) -> int:
    return len(text.split())


def archetype_problems(text: str, items: list[Item]) -> list[str]:
    """Empty list = the archetype passes both checks."""
    problems = []
    if word_count(text) > MAX_ARCHETYPE_WORDS:
        problems.append(f"{word_count(text)} words (limit {MAX_ARCHETYPE_WORDS})")
    copied = copied_items(text, items)
    if len(copied) > MAX_ITEMS_COPIED:
        problems.append(f"restates answers item by item for {len(copied)} items ({', '.join(copied[:6])}); "
                        f"limit {MAX_ITEMS_COPIED}")
    return problems


def truncate_words(text: str, limit: int = MAX_ARCHETYPE_WORDS) -> str:
    words = text.split()
    return text if len(words) <= limit else " ".join(words[:limit]).rstrip(",;:") + "..."


@dataclass(frozen=True)
class DuplicateReport:
    share: float
    n_duplicates: int
    n_crowd: int
    threshold: float


def near_duplicates(crowd: Respondents, asked: Respondents, threshold: float = NEAR_DUPLICATE_DISTANCE
                    ) -> DuplicateReport:
    """Share of synthetic people within `threshold` Gower distance of any asked real person."""
    n = len(crowd.answers)
    if len(asked.answers) == 0 or n == 0:
        return DuplicateReport(0.0, 0, n, threshold)
    d = np.nan_to_num(gower(encode(crowd), encode(asked), crowd.items), nan=np.inf)
    dup = d.min(axis=1) < threshold
    return DuplicateReport(float(dup.mean()), int(dup.sum()), n, threshold)


def assert_few_near_duplicates(report: DuplicateReport, max_share: float = MAX_NEAR_DUPLICATE_SHARE) -> None:
    if report.share > max_share:
        raise NearDuplicateError(
            f"{report.n_duplicates}/{report.n_crowd} synthetic people ({report.share:.1%}) are within "
            f"{report.threshold} of an asked real person; limit {max_share:.0%}. The crowd is copying.")

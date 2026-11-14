"""Column typing for the audit: numeric, ordinal (recognised scales or numeric labels) or categorical.

Distance rules (equal weight per question): numeric/ordinal answers are scaled to [0, 1]; categorical answers are
a 0/1 mismatch. Answer options are the UNION of both files so the two panels are always compared on one scale.
"""
import re
from dataclasses import dataclass, field

from .parse import is_missing

MAX_TEXT_OPTIONS = 30      # more distinct text answers than this = free text or an id, not a survey question
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_PREFIX = re.compile(r"^\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*(?:[-:.)/%]|out of\b|\b|$)", re.I)

# Ordered answer scales we recognise (lower case). Direction does not matter: distance is symmetric.
SCALES: tuple[tuple[str, ...], ...] = (
    ("strongly agree", "agree", "somewhat agree", "neither agree nor disagree", "neutral", "somewhat disagree", "disagree",
     "strongly disagree"),
    ("very happy", "pretty happy", "not too happy"),
    ("very satisfied", "pretty well satisfied", "satisfied", "somewhat satisfied", "more or less satisfied",
     "neither satisfied nor dissatisfied", "neutral", "somewhat dissatisfied", "dissatisfied", "not satisfied at all",
     "very dissatisfied"),
    ("almost daily", "once or twice a week", "several times a month", "about once a month", "several times a year",
     "about once a year", "never"),
    ("always", "very often", "often", "sometimes", "rarely", "never"),
    ("daily", "weekly", "monthly", "yearly", "never"),
    ("excellent", "very good", "good", "neither bad nor good", "fair", "bad", "poor", "very poor", "very bad"),
    ("very worried", "somewhat worried", "neither worried nor unworried", "not very worried", "not at all worried"),
    ("far below average", "below average", "average", "above average", "far above average"),
    ("lower class", "working class", "middle class", "upper class"),
    ("exciting", "pretty routine", "routine", "dull"),
    ("better", "stayed same", "same", "worse"),
    ("too little", "about right", "too much"),
    ("extremely likely", "very likely", "somewhat likely", "not very likely", "not at all likely"),
    ("very interested", "somewhat interested", "not very interested", "not at all interested"),
    ("very important", "somewhat important", "not very important", "not at all important"),
    ("a great deal", "a lot", "some", "a little", "not at all"),
    ("strongly oppose", "oppose", "somewhat oppose", "neither", "somewhat favor", "favor", "strongly favor"),
)


def normalize_label(value: str) -> str:
    text = str(value).translate(_QUOTES).lower().strip().rstrip(".")
    return re.sub(r"\s+", " ", text)


def _to_float(value: str) -> float | None:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str                                   # "numeric" | "ordinal" | "categorical"
    labels: dict[str, str] = field(default_factory=dict)       # normalized answer -> display label
    positions: dict[str, float] = field(default_factory=dict)  # ordinal: normalized answer -> position in [0, 1]
    categories: tuple[str, ...] = ()            # categorical: normalized answers in a fixed order
    lo: float = 0.0                             # numeric range
    hi: float = 1.0

    @property
    def n_options(self) -> int:
        return len(self.positions) or len(self.categories) or int(self.hi - self.lo + 1)

    def encode(self, value: str) -> float:
        """Answer text -> number in [0, 1] (ordinal/numeric) or category index (categorical); NaN if missing/unknown."""
        if is_missing(value):
            return float("nan")
        if self.kind == "numeric":
            v = _to_float(value)
            return float("nan") if v is None else (v - self.lo) / ((self.hi - self.lo) or 1.0)
        key = normalize_label(value)
        if self.kind == "ordinal":
            return self.positions.get(key, float("nan"))
        return float(self.categories.index(key)) if key in self.categories else float("nan")

    def decode(self, code: float) -> str:
        """Encoded value -> a human label (for tables); numeric values are shown as numbers."""
        if code != code:
            return "(missing)"
        if self.kind == "numeric":
            return f"{self.lo + code * (self.hi - self.lo):.4g}"
        if self.kind == "ordinal":
            best = min(self.positions, key=lambda k: abs(self.positions[k] - code))
            return self.labels.get(best, best)
        return self.labels.get(self.categories[int(code)], self.categories[int(code)])


def analyse_column(name: str, real_values: list[str], panel_values: list[str]) -> tuple[ColumnSpec | None, str | None]:
    """Type one question from both files. Returns (spec, None) or (None, why it is ignored)."""
    answers = [v for v in real_values + panel_values if not is_missing(v)]
    if not answers:
        return None, "no answers in either file"
    labels: dict[str, str] = {}
    for v in answers:
        labels.setdefault(normalize_label(v), v.strip())
    keys = list(labels)
    if len(keys) == 1:
        return None, "everyone gives the same answer (no information)"
    nums = [_to_float(v) for v in answers]
    if all(n is not None for n in nums):
        lo, hi = min(nums), max(nums)
        return ColumnSpec(name, "numeric", labels, lo=lo, hi=hi), None
    prefix = {k: _PREFIX.match(labels[k]) for k in keys}
    if all(prefix.values()) and len(keys) <= MAX_TEXT_OPTIONS:
        vals = {k: float(prefix[k].group(1).replace(",", "")) for k in keys}
        lo, hi = min(vals.values()), max(vals.values())
        if hi > lo:
            return ColumnSpec(name, "ordinal", labels, {k: (v - lo) / (hi - lo) for k, v in vals.items()}), None
    for scale in SCALES:
        if all(k in scale for k in keys):
            pos = {k: scale.index(k) / (len(scale) - 1) for k in keys}
            return ColumnSpec(name, "ordinal", labels, pos), None
    if len(keys) > MAX_TEXT_OPTIONS:
        return None, f"{len(keys)} different text answers (looks like free text or an id)"
    real_counts: dict[str, int] = {}
    for v in real_values:
        if not is_missing(v):
            real_counts[normalize_label(v)] = real_counts.get(normalize_label(v), 0) + 1
    ordered = sorted(keys, key=lambda k: (-real_counts.get(k, 0), k))
    return ColumnSpec(name, "categorical", labels, categories=tuple(ordered)), None

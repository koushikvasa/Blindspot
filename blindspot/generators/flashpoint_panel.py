"""Load saved Flashpoint synthetic-panel rows into Respondents (H5).

No network and no payments here: the rows were exported earlier (CSV, one row per synthetic
respondent: persona attributes + one answer label per item) into data/raw/flashpoint/ (git-ignored).
Answers arrive as option LABELS and are mapped to our option indices (index into Item.options).
Anything that does not match is NaN and is counted, never guessed.

Source string is "flashpoint:uncalibrated": the panels were run without calibration.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from blindspot.data.gss import AGE_BINS, AGE_LABELS
from blindspot.data.schema import Item, Respondents

SOURCE = "flashpoint:uncalibrated"
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalize(label: object) -> str:
    """Lower-case, straighten quotes, collapse whitespace, drop trailing punctuation."""
    text = str(label).translate(_QUOTES).lower().strip()
    return re.sub(r"\s+", " ", text).rstrip(".")


def label_lookup(item: Item) -> dict[str, int]:
    """normalized option label -> index. 0-10 scales also accept the bare number ("7")."""
    lookup = {normalize(label): i for i, label in enumerate(item.options)}
    if len(item.options) == 11:
        lookup.update({str(i): i for i in range(11)})
    return lookup


@dataclass
class MappingReport:
    """Per-item counts of answers that could not be mapped to one of the item's options."""
    n_rows: int
    n_items: int
    missing: dict[str, int] = field(default_factory=dict)             # blank / NaN
    unmapped: dict[str, dict[str, int]] = field(default_factory=dict)  # item -> {raw text: count}

    @property
    def n_missing(self) -> int:
        return sum(self.missing.values())

    @property
    def n_unmapped(self) -> int:
        return sum(sum(v.values()) for v in self.unmapped.values())

    @property
    def n_invalid(self) -> int:
        return self.n_missing + self.n_unmapped

    def to_dict(self) -> dict:
        total = self.n_rows * self.n_items
        return {"n_rows": self.n_rows, "n_items": self.n_items, "n_answers": total,
                "missing": self.n_missing, "unmapped": self.n_unmapped, "invalid_share": self.n_invalid / total,
                "missing_by_item": self.missing, "unmapped_by_item": self.unmapped}


def map_answers(raw: pd.DataFrame, items: list[Item]) -> tuple[pd.DataFrame, MappingReport]:
    """Option labels -> option indices (float, NaN where missing or unmapped)."""
    absent = [it.code for it in items if it.code not in raw.columns]
    if absent:
        raise ValueError(f"export is missing item columns {absent}")
    report = MappingReport(n_rows=len(raw), n_items=len(items))
    cols: dict[str, pd.Series] = {}
    for it in items:
        lookup = label_lookup(it)
        values = raw[it.code]
        blank = values.isna() | values.astype(str).str.strip().eq("")
        mapped = values.where(~blank).map(lambda v: lookup.get(normalize(v), np.nan) if pd.notna(v) else np.nan)
        bad = ~blank & mapped.isna()
        if blank.any():
            report.missing[it.code] = int(blank.sum())
        if bad.any():
            report.unmapped[it.code] = {str(k): int(v) for k, v in values[bad].astype(str).value_counts().items()}
        cols[it.code] = mapped.astype("float64")
    return pd.DataFrame(cols, index=raw.index), report


def demographics_from(raw: pd.DataFrame) -> pd.DataFrame:
    """Persona attributes in our column names, "unknown" where the export has none (the survey export
    carries no persona attributes and no persona-to-response key, so exports from H5 have none).
    Flashpoint's own education/income vocabulary is kept: it is coarser than GSS, descriptive only."""
    def col(name: str) -> pd.Series:
        return raw[name].astype(str).str.lower() if name in raw.columns else pd.Series("unknown", index=raw.index)

    age = pd.to_numeric(raw.get("age", pd.Series(np.nan, index=raw.index)), errors="coerce")
    return pd.DataFrame({
        "age_band": pd.cut(age, AGE_BINS, right=False, labels=AGE_LABELS).astype(object).fillna("unknown"),
        "gender": col("gender"), "education": col("education"),
        "income_band": col("income_bracket"), "race": col("ethnicity"),
    }, index=raw.index)


def load(csv_path: Path, items: list[Item], tag: str, meta: dict | None = None) -> tuple[Respondents, MappingReport]:
    """One Flashpoint run (CSV export) -> Respondents with ids fp_<tag>_0000... and a mapping report."""
    raw = pd.read_csv(csv_path, dtype=str)
    raw.index = pd.Index([f"fp_{tag}_{i:04d}" for i in range(len(raw))])
    answers, report = map_answers(raw, items)
    resp = Respondents(answers, demographics_from(raw), items, SOURCE,
                       meta={"tag": tag, "file": Path(csv_path).name, **(meta or {})})
    return resp, report

"""Load a COMPLETED Flashpoint real-respondent export (Prolific) into Respondents.

Input: the labelled CSV export of survey "GSS live items (20) + demographics": two-row header (variable
names, then question text), one row per respondent, option label text in the answer cells, the 20 live
items plus age_band, gender, education, income_band. Only COMPLETE rows are kept; rows flagged
`Quality Excluded` are dropped (counted). Vendor ids are never kept: people get ids fp_real_0000....
Source string: "flashpoint:prolific_real". These are REAL respondents, but never GSS people, so the
experiment must not mix them into pool/test splits.

Demographics use the GSS vocabulary of docs/DATA.md; region and race were not asked ("unknown").
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from blindspot.data.gss import AGE_LABELS, INCOME_BANDS
from blindspot.data.schema import Item, Respondents
from blindspot.generators.flashpoint_panel import MappingReport, map_answers, normalize

SOURCE = "flashpoint:prolific_real"
MIN_RESPONDENTS = 10
DEMO_COLS = ("age_band", "gender", "education", "income_band")
AGE_OK = {normalize(a): a for a in AGE_LABELS}
GENDER_OK = {"male": "male", "female": "female"}
EDU_OK = {normalize(e): e for e in ("less than high school", "high school", "associate/junior college",
                                    "bachelor's", "graduate")}
INCOME_OK = {normalize(label): label for _, _, label in INCOME_BANDS}
_VOCAB = {"age_band": AGE_OK, "gender": GENDER_OK, "education": EDU_OK, "income_band": INCOME_OK}
_TRUE = {"true", "1", "yes", "y"}


@dataclass
class RealLoadReport:
    n_rows_in_file: int
    n_complete: int
    n_quality_excluded: int
    n_kept: int
    answers: MappingReport
    demographics_unmapped: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"n_rows_in_file": self.n_rows_in_file, "n_complete": self.n_complete,
                "n_quality_excluded": self.n_quality_excluded, "n_kept": self.n_kept,
                "answers": self.answers.to_dict(), "demographics_unmapped": self.demographics_unmapped}


def read_export(path: Path) -> pd.DataFrame:
    """Read the CSV; drop the second header row (question text) when present."""
    raw = pd.read_csv(path, dtype=str)
    if len(raw) and "#" in raw.columns and str(raw["#"].iloc[0]).strip() == "#":
        raw = raw.iloc[1:]
    return raw.reset_index(drop=True)


def map_demographics(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Option labels -> GSS vocabulary; anything unmatched becomes "unknown" and is counted."""
    cols, bad = {}, {}
    for col in DEMO_COLS:
        if col not in raw.columns:
            raise ValueError(f"export is missing demographic column {col!r}")
        vocab = _VOCAB[col]
        mapped = raw[col].map(lambda v: vocab.get(normalize(v), "unknown") if pd.notna(v) else "unknown")
        miss = mapped.eq("unknown")
        if miss.any():
            bad[col] = {str(k): int(n) for k, n in raw[col][miss].fillna("<blank>").value_counts().items()}
        cols[col] = mapped
    demo = pd.DataFrame(cols, index=raw.index)
    demo["region"], demo["race"], demo["weight"] = "unknown", "unknown", np.nan
    return demo, bad


def load(csv_path: Path, items: list[Item], min_n: int = MIN_RESPONDENTS) -> tuple[Respondents, RealLoadReport]:
    """Completed, non-excluded respondents -> Respondents (ids fp_real_0000...), plus a report."""
    raw = read_export(Path(csv_path))
    n_file = len(raw)
    complete = raw["Status"].astype(str).str.upper().eq("COMPLETE") if "Status" in raw.columns else pd.Series(True, index=raw.index)
    excluded = (raw["Quality Excluded"].astype(str).str.lower().isin(_TRUE) if "Quality Excluded" in raw.columns
                else pd.Series(False, index=raw.index))
    kept = raw[complete & ~excluded].reset_index(drop=True)
    if len(kept) < min_n:
        raise ValueError(f"only {len(kept)} usable respondents in {Path(csv_path).name}; need at least {min_n}")
    kept.index = pd.Index([f"fp_real_{i:04d}" for i in range(len(kept))])
    answers, answer_report = map_answers(kept, items)
    demo, bad = map_demographics(kept)
    report = RealLoadReport(n_file, int(complete.sum()), int((complete & excluded).sum()), len(kept), answer_report, bad)
    return Respondents(answers, demo, items, SOURCE, meta={"file": Path(csv_path).name}), report

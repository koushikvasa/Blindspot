"""Load GSS 2024 (Stata) into Respondents using pyreadstat (H1). See docs/DATA.md.

The target set is the main cross-section (sample == 13, n = 3,309, the codebook's case base) on
ballot 1: n = 1,116. The 677 sample == 14 rows are the AmeriSpeak Black/Hispanic/AANHPI oversample
(no wtssps weight) and are excluded. The wtssps weight is kept as a demographics column (`weight`)
for later; it is not used for coverage. Reserved codes (don't know, IAP,
no answer, skipped on web, ...) are Stata extended missing values; pyreadstat loads them as NaN.
Option labels come from the file's value labels; for 0-10 scales only the endpoints are labelled.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from .items import GSS_BALLOT, GSS_ITEMS, GSS_SAMPLE, GSS_WEIGHT, GSS_WORDING, ItemSpec
from .schema import Item, Respondents

ROOT = Path(__file__).resolve().parents[2]
DTA = ROOT / "data/raw/gss/2024/GSS2024.dta"
ID_COL = "id"
DEMO_COLS = ["age", "sex", "degree", "race", "region", "income16"]

AGE_BINS = [18, 30, 50, 65, 200]
AGE_LABELS = ["18-29", "30-49", "50-64", "65+"]
# income16 codes -> bands. 21 = $75,000-89,999 and 22 = $90,000-109,999, so the top cut is 110k, not 100k.
INCOME_BANDS = [(1, 15, "Less than $30,000"), (16, 18, "$30,000-$49,999"), (19, 20, "$50,000-$74,999"),
                (21, 22, "$75,000-$109,999"), (23, 26, "$110,000 or more")]


def read_raw() -> tuple[pd.DataFrame, object]:
    cols = [ID_COL, "sample", "ballot", GSS_WEIGHT] + DEMO_COLS + [s.source for s, _ in GSS_ITEMS]
    df, meta = pyreadstat.read_dta(str(DTA), usecols=cols)
    if not df[ID_COL].is_unique:
        raise ValueError("GSS id is not unique")
    return df.set_index(df[ID_COL].astype(int).astype(str)).drop(columns=ID_COL), meta


def options_for(meta: object, spec: ItemSpec, codes: tuple[int, ...]) -> list[str]:
    labels = meta.variable_value_labels.get(spec.source, {})
    if len(codes) == 11:  # 0-10 scale: only the endpoints carry labels (some already start with the number)
        def scale_label(c: int) -> str:
            text = str(labels.get(c, c))
            return text if text.startswith(str(c)) else f"{c} - {text}"
        return [scale_label(c) for c in codes]
    missing = [c for c in codes if c not in labels]
    if missing:
        raise ValueError(f"{spec.source}: no value label for codes {missing}")
    return [str(labels[c]) for c in codes]


def encode(raw: pd.Series, spec: ItemSpec, codes: tuple[int, ...]) -> pd.Series:
    """Substantive code -> option index (position in `codes`); NaN stays NaN; other codes raise."""
    x = pd.to_numeric(raw, errors="raise")
    bad = x.notna() & ~x.isin(codes)
    if bad.any():
        raise ValueError(f"{spec.source}: unexpected codes {sorted(x[bad].unique())}")
    return x.map({c: float(i) for i, c in enumerate(codes)}).astype("float64")


def _band_income(code: float) -> str:
    if np.isnan(code):
        return "unknown"
    return next(label for lo, hi, label in INCOME_BANDS if lo <= code <= hi)


def demographics(df: pd.DataFrame, meta: object) -> pd.DataFrame:
    num = df[DEMO_COLS].apply(pd.to_numeric, errors="coerce")

    def labelled(var: str) -> pd.Series:
        labels = meta.variable_value_labels.get(var, {})
        return num[var].map(lambda v: "unknown" if np.isnan(v) else str(labels.get(int(v), int(v))))

    age = pd.cut(num["age"], AGE_BINS, right=False, labels=AGE_LABELS).astype(object).fillna("unknown")
    return pd.DataFrame({
        "age_band": age,
        "gender": labelled("sex"),
        "education": labelled("degree"),
        "income_band": num["income16"].map(_band_income),
        "region": labelled("region"),
        "race": labelled("race"),
        "weight": pd.to_numeric(df[GSS_WEIGHT], errors="coerce"),
    }, index=df.index)


def target_mask(df: pd.DataFrame, sample: int = GSS_SAMPLE, ballot: int = GSS_BALLOT) -> pd.Series:
    return (pd.to_numeric(df["sample"]) == sample) & (pd.to_numeric(df["ballot"]) == ballot)


def load(sample: int = GSS_SAMPLE, ballot: int = GSS_BALLOT) -> Respondents:
    df, meta = read_raw()
    df = df[target_mask(df, sample, ballot)]
    items = [Item(code=s.code, wording=GSS_WORDING[s.code], kind=s.kind, options=options_for(meta, s, codes),
                  theme=s.theme, sensitive=s.sensitive, verified=s.verified, family=s.theme)
             for s, codes in GSS_ITEMS]
    answers = pd.DataFrame({s.code: encode(df[s.source], s, codes) for s, codes in GSS_ITEMS}, index=df.index)
    return Respondents(answers, demographics(df, meta), items, "gss:2024",
                       meta={"file": DTA.name, "sample": sample, "ballot": ballot, "n_file_rows": meta.number_rows})

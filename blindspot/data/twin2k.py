"""Load Twin-2K-500 held-out answers into Respondents (H1). See docs/DATA.md for the schema.

Three CSVs share the same 168 columns and the same 2,058 TWIN_IDs:
  responses_wave1_3_formatted.csv     -> "real"           (waves 1-3 human answers)
  responses_llm_imputed_formatted.csv -> "gpt41mini_twin" (GPT-4.1-mini digital-twin answers)
  responses_wave4_formatted.csv       -> "retest"         (same people, wave 4)
Row 0 of every CSV is question wording, not a respondent.

Demographics and option labels come from full_persona/persona_json. Only the Demographics
block and the option *labels* are read: persona_json also holds each person's wave-4 answers
to the scored items, so persona text must never reach a generator prompt.
"""
import json
import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .items import PERCENT_BIN_LABELS, TWIN_ITEMS, TWIN_OPTION_QID, ItemSpec
from .schema import Item, Respondents

ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = ROOT / "data/raw/twin2k/LLM_simulation_results/GPT4.1-mini-simulation-llm-vs-human"
PERSONA_DIR = ROOT / "data/raw/twin2k_parquet/full_persona/data"
DEMO_CACHE = ROOT / "data/processed/twin2k_demographics.csv"

FILES = {
    "real": "responses_wave1_3_formatted.csv",
    "gpt41mini_twin": "responses_llm_imputed_formatted.csv",
    "retest": "responses_wave4_formatted.csv",
}
ID_COL = "TWIN_ID"
DEMO_QIDS = {  # persona_json Demographics block
    "QID11": "region", "QID12": "gender", "QID13": "age_band",
    "QID14": "education", "QID15": "race", "QID21": "income_band",
}


def read_raw(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return (answers as strings indexed by TWIN_ID, row-0 wording by column)."""
    df = pd.read_csv(CSV_DIR / FILES[name], dtype=str, encoding="utf-8")
    wording = df.iloc[0]
    if wording[ID_COL] != ID_COL:
        raise ValueError(f"{FILES[name]}: row 0 is not the wording row")
    body = df.iloc[1:].set_index(ID_COL)
    if not body.index.is_unique:
        raise ValueError(f"{FILES[name]}: duplicate {ID_COL}")
    return body, wording


def _questions(persona_json: str) -> Iterator[dict]:
    for block in json.loads(persona_json):
        yield from block.get("Questions", [])


def _persona_files() -> list[Path]:
    files = sorted(PERSONA_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no persona parquet files in {PERSONA_DIR}")
    return files


def option_labels() -> dict[str, list[str]]:
    """Option labels per persona QID, read from the first persona (the question text is shared)."""
    first = pq.read_table(_persona_files()[0], columns=["persona_json"]).slice(0, 1).to_pylist()[0]
    wanted = set(TWIN_OPTION_QID.values())
    labels = {q["QuestionID"]: list(q.get("Columns") or q.get("Options"))
              for q in _questions(first["persona_json"]) if q["QuestionID"] in wanted}
    missing = wanted - labels.keys()
    if missing:
        raise ValueError(f"persona_json lacks option labels for {sorted(missing)}")
    return labels


def _extract_demographics() -> pd.DataFrame:
    records = {}
    for path in _persona_files():
        for row in pq.read_table(path, columns=["pid", "persona_json"]).to_pylist():
            rec = {}
            for q in _questions(row["persona_json"]):
                col = DEMO_QIDS.get(q["QuestionID"])
                if col:
                    rec[col] = (q.get("Answers") or {}).get("SelectedText")
            records[str(row["pid"])] = rec
    demo = pd.DataFrame.from_dict(records, orient="index").reindex(columns=list(DEMO_QIDS.values()))
    demo.index.name = ID_COL
    return demo.assign(region=demo["region"].str.replace(r"\s*\(.*\)$", "", regex=True))


def demographics(refresh: bool = False) -> pd.DataFrame:
    """Demographics per TWIN_ID from the persona Demographics block (cached in data/processed)."""
    if DEMO_CACHE.exists() and not refresh:
        return pd.read_csv(DEMO_CACHE, dtype=str).set_index(ID_COL)
    demo = _extract_demographics()
    DEMO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(DEMO_CACHE)
    return demo


def _options_for(spec: ItemSpec, labels: dict[str, list[str]]) -> list[str]:
    if spec.bins is not None:
        return list(PERCENT_BIN_LABELS)
    prefix = next(p for p in TWIN_OPTION_QID if spec.source.startswith(p))
    return labels[TWIN_OPTION_QID[prefix]]


def encode(raw: pd.Series, spec: ItemSpec, n_options: int) -> pd.Series:
    """Raw strings ('3', '3.0', '57.0') -> option index 0..k-1 or NaN. Raises on unexpected codes."""
    x = pd.to_numeric(raw, errors="raise")
    if spec.bins is not None:
        if ((x < 0) | (x > 100)).any():
            raise ValueError(f"{spec.source}: percentage outside 0-100")
        idx = pd.cut(x, list(spec.bins), right=False, labels=False)
    else:
        idx = x - 1
        bad = idx.notna() & ~idx.isin(range(n_options))
        if bad.any():
            raise ValueError(f"{spec.source}: codes outside 1..{n_options}: {sorted(x[bad].unique())}")
    return idx.astype("float64")


def clean_wording(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_items(wording: pd.Series, labels: dict[str, list[str]]) -> list[Item]:
    return [
        Item(code=s.code, wording=clean_wording(wording[s.source]), kind=s.kind,
             options=_options_for(s, labels), theme=s.theme, sensitive=s.sensitive)
        for s in TWIN_ITEMS
    ]


def load() -> dict[str, Respondents]:
    """Keys real / gpt41mini_twin / retest; all share ids (real's order), items and demographics."""
    labels = option_labels()
    demo = demographics()
    real_body, wording = read_raw("real")
    ids = real_body.index
    if set(demo.index) != set(ids):
        raise ValueError("persona pids do not match TWIN_IDs")
    items = build_items(wording, labels)
    out = {}
    for name in FILES:
        body, w = read_raw(name)
        if not w.fillna("").equals(wording.fillna("")):
            raise ValueError(f"{FILES[name]}: wording row differs from the real file")
        if set(body.index) != set(ids):
            raise ValueError(f"{FILES[name]}: TWIN_IDs differ from the real file")
        body = body.loc[ids]
        answers = pd.DataFrame(
            {it.code: encode(body[s.source], s, len(it.options)) for s, it in zip(TWIN_ITEMS, items)},
            index=ids,
        )
        out[name] = Respondents(answers, demo.loc[ids], items, f"twin2k:{name}", meta={"file": FILES[name]})
    return out

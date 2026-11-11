# H1: Load the data, pick the items, make the splits

Read `CLAUDE.md` and `docs/PLAN.md` first. Scope is ONLY this handover; do not start H2.

## Inputs (already placed by me)
- `data/raw/twin2k/` : Twin-2K-500 parquet files from Hugging Face
- `data/raw/gss/` : GSS 2024 Stata zip (`2024_stata.zip`) or the extracted `.dta`

## Tasks
1. **Inspect, then document.** Open both datasets. In `docs/DATA.md` record: files, row counts, respondent ID, demographic columns, how answers are stored, missing-value codes. Do not guess column meanings; quote what the files/codebook say.
2. **Loaders.** Implement `blindspot/data/twin2k.py: load() -> Respondents` and `blindspot/data/gss.py: load() -> Respondents` returning the contract in `blindspot/data/schema.py`. Demographics normalized to: `age_band, gender, education, income_band, region` (use NaN if unavailable; note it in DATA.md).
3. **Item selection.** In `blindspot/data/items.py`, propose 24 items per dataset across 4 themes (e.g. trust/institutions, money/economics, risk & personality, technology/AI) with: missingness < 10%, no option above 85% share, ordinal or binary preferred, nothing about sexual behaviour, health conditions or immigration status. Write a short table of chosen items into DATA.md and STOP for my review before finalizing.
4. **Splits.** `blindspot/eval/splits.py: make_splits(ids, seed=42, fractions=(0.5,0.3,0.2))` → saved JSON in `data/processed/splits.json` per dataset. Deterministic. Add a guard function `assert_no_leak(ids_used, split_name)`.
5. **Tests.** Loader returns Respondents with matching shapes; splits are deterministic and disjoint; no item exceeds the missingness rule.
6. **Script.** `python -m blindspot.data.inspect` prints, per dataset: n respondents, n items chosen, missingness per item, answer-share range.

## Done when
- `pytest -q` passes
- `python -m blindspot.data.inspect` runs on both datasets
- `docs/DATA.md` filled, item table waiting for my approval

## Out of scope
Distances, generators, Claude calls, UI.

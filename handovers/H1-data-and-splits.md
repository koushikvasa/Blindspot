# H1: Load the data, pick the items, make the splits

Read `CLAUDE.md` and `docs/PLAN.md` first. Scope is ONLY this handover; do not start H2.

## Inputs (already on disk)

**Twin-2K-500 held-out answers** (three files, same 168 columns, joined on `TWIN_ID`):
`data/raw/twin2k/LLM_simulation_results/GPT4.1-mini-simulation-llm-vs-human/`
- `responses_wave1_3_formatted.csv` = REAL answers → `Respondents(source="twin2k:real")`
- `responses_llm_imputed_formatted.csv` = GPT-4.1-mini digital-twin answers for the same people → `Respondents(source="twin2k:gpt41mini_twin")`
- `responses_wave4_formatted.csv` = same people retested weeks later → `Respondents(source="twin2k:retest")`

Known facts (verify, don't trust blindly):
- pandas reads 2,059 rows; the dataset has 2,058 people. Row 0 very likely holds question wording, not answers. Confirm, use it as `Item.wording`, then drop it.
- Metadata columns to drop: `StartDate, EndDate, Status, Progress, Duration (in seconds), Finished, RecordedDate, attention_1, attention_2, STUDY_ID, SESSION_ID, tasktime, worktimeArray, offTask, onTask, totalOffTask, totalOnTask, perPagePT, pageNo, randDollars, randDollarsString, randDollarsString_test, WAVE`, and free-text columns like `Q259_3_TEXT`.
- Question families seen: `False Cons. self _1.._10` (own policy support), `False cons. others _*` (estimates of what others think), `1_Q295.._40_Q295` (40-item block, identify what it is), `nonseparabilty bene _*` / `nonseparability ris _*`, reasoning tasks (`base_rate, outcome_bias, sunk_cost, Allais, disease, linda, african, redwood, absolute, Myside, less_is_more, Thaler, matching, Omission bias , Denominator neglect `), and `Q156..Q203_*` (identify from row 0 wording). Note trailing spaces in some column names.
- The retest file may have more missing values; report missingness per column for all three files.

**Twin-2K-500 personas (for demographics):** `data/raw/twin2k_parquet/full_persona/data/0000.parquet … 0006.parquet` (ignore `.cache/`). Inspect the schema; find the ID that joins to `TWIN_ID` and extract `age_band, gender, education, income_band, region` (plus any other demographics present). If demographics are inside a text/JSON blob, parse them; document how.

**GSS 2024:** `data/raw/gss/2024/GSS2024.dta` (3,986 rows, 989 columns). Load with `pyreadstat` (value labels on). Use the codebook labels; never guess a variable's meaning.

## Tasks
1. **Inspect, then document.** Fill `docs/DATA.md`: files, row counts, ID columns, the row-0 finding, each question family with example wording and answer scale, missing-value codes, how demographics were extracted, license/citation (Twin-2K-500: CC BY 4.0, Toubia et al. 2025; GSS: NORC).
2. **Loaders.**
   - `blindspot/data/twin2k.py: load() -> dict[str, Respondents]` with keys `real`, `gpt41mini_twin`, `retest`; all three share ids, items, demographics.
   - `blindspot/data/gss.py: load() -> Respondents`.
   - Answers stored as integer option index (0..k-1) or NaN; keep the original label list in `Item.options`. If a column holds text labels in one file and numbers in another, map both to the same index and test it.
3. **Item selection.** `blindspot/data/items.py`:
   - Twin: propose ~30 items across families (policy views, reasoning tasks, risk/benefit judgments, the Q295 block, others), ordinal/binary preferred, missingness < 10% in `real`, no option above 85% share. Tag each item `sensitive=True` if it is politically charged (the policy items are). Sensitive items are allowed in analysis but never in live mode.
   - GSS: propose 24 items across 4 themes (trust/institutions, money/economy, work/life, technology/science), same rules; exclude sexual behaviour, health conditions, immigration status, religion-identity questions.
   - Write both tables into `docs/DATA.md` and **STOP for my review** before finalizing.
   - Add `sensitive: bool = False` to `Item` in `schema.py`.
4. **Splits.** `blindspot/eval/splits.py: make_splits(ids, seed=42, fractions=(0.5, 0.3, 0.2))` → `data/processed/splits_{dataset}.json` with `pool`, `test`, `holdout`. Deterministic, disjoint. Add `assert_no_leak(ids_used, allowed=("pool",))`. For Twin, the same split applies to `real`, `gpt41mini_twin` and `retest` (they share ids).
5. **Tests.** Shapes match; row 0 dropped; the three Twin files share ids and items; label/number mapping is consistent; splits deterministic and disjoint; item rules hold.
6. **Script.** `python -m blindspot.data.inspect` prints, per dataset and file: n people, n items chosen, missingness per item, answer-share range, and for Twin the simple per-item agreement between `real` and `gpt41mini_twin` and between `real` and `retest` (a sanity check only; no coverage metrics yet).

## Done when
- `pytest -q` passes
- `python -m blindspot.data.inspect` runs on both datasets
- `docs/DATA.md` filled, item tables waiting for my approval

## Out of scope
Distances, coverage, generators, Claude calls, UI. (H2 will run "Experiment 0": does the GPT-4.1-mini twin crowd leave real people uncovered, compared with the retest ceiling and simple baselines?)

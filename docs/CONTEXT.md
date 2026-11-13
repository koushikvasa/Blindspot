# Blindspot: full project context (read this first in a new session)

Last updated: Fri 2026-11-13, ~1 PM ET. Hackathon is **Sat 2026-11-14**, Fordham "AI Hackathon: Hack the Unreachable", Self-Improving AI track. Presentations at 3:30 PM. Repo: `D:\blindspot` (GitHub `koushikvasa/Blindspot`, currently **private**; must be public by submission after cleanup).

Read with: `CLAUDE.md` (hard rules), `docs/PLAN.md`, `docs/DATA.md`, `docs/METRICS.md`, `docs/FLASHPOINT.md`, `docs/UI_DATA_CONTRACT.md`, and the `results/*.md` files.

---

## 1. What Blindspot is (plain words)

Companies and researchers increasingly use AI to *simulate survey respondents* ("silicon sampling" / synthetic panels). Blindspot audits a synthetic crowd against real survey respondents and finds **the real people the AI crowd cannot imagine**.

A real person is **covered** if at least one synthetic person is within that real person's distance to their 5th nearest *real* neighbour (k-NN coverage, Naeem et al. 2020), using family-weighted Gower distance over survey answers. **Precision** = share of synthetic people inside some real person's k-NN ball (how realistic the fakes are).

Hackathon theme: "people say one thing and do another; find what people actually think." Track 3: "models that recognize their own knowledge gaps and seek answers from the right people."

## 2. The story the evidence supports (as of now)

1. **Every AI crowd we tested covers real people no better than randomly shuffled answers**, across 3 generators and 2 national surveys. This agrees with Kumagae & Kaneko (arXiv 2610.30900, Oct 2026), who found every LLM did worse than a Gaussian copula fitted to human data.
2. AI crowds fail in **three different ways**:
   - Claude "typical answers" (mode): too alike. Realistic but collapsed.
   - Claude "random answers" (sampled from its own probabilities): right spread, incoherent people.
   - Flashpoint synthetic panel: right spread, but answers don't combine like real people's (roughly shuffled-answer level on test, 0.61 vs 0.63; on holdout 0.759 and 0.674 for two panels, still below shuffled answers' 0.853). Run-to-run agreement was close on test but not on holdout, so do not call it stable.
3. **Prompting cannot fix it.** Two agent designs that taught Claude text "archetypes" of missing people (H4, H4b) did not improve coverage.
4. **Real data fixes it.** Statistics fitted on real people, or adding a small number of real people, raises coverage a lot.
5. Open question being tested now: can the agent **learn whom to ask** (targeted recruitment) and **select** realistic diverse candidates (select-from-many)? These are the self-improving pieces.

## 3. Data

- **Twin-2K-500** (Toubia et al. 2025, CC BY 4.0, Hugging Face `LLM-Digital-Twin/Twin-2K-500`): 2,058 US people. Files in `data/raw/twin2k/LLM_simulation_results/GPT4.1-mini-simulation-llm-vs-human/`: `responses_wave1_3_formatted.csv` (real), `responses_llm_imputed_formatted.csv` (dataset authors' GPT-4.1-mini digital twins), `responses_wave4_formatted.csv` (same people retested). Row 0 is question wording. 30 items chosen in 4 families: `policy_self`, `perception`, `risk_benefit`, `decisions`. Demographics from `data/raw/twin2k_parquet/full_persona/data/*.parquet`. **Persona text contains wave-4 answers: never put it in any prompt (rule 9).**
- **GSS 2024** (`data/raw/gss/2024/GSS2024.dta`): main cross-section (sample==13), Ballot 1 → **n = 1,116**. 24 items in 4 families (trust_institutions, money_economy, work_life, tech_science), all wording verified from the official questionnaire; 20 are `live_ok` (verified and not sensitive). Oversample (sample==14, 677 people) excluded.
- **Splits** (seed 42, 50/30/20): Twin pool 1,029 / test 617 / holdout 412; GSS pool 558 / test 334 / holdout 224. Files in `data/processed/splits_*.json`. **Holdout has never been touched; it is reserved for one final headline run.**
- All raw data and processed splits are git-ignored.

## 4. Results so far (test split)

### Experiment 0/1: Twin-2K-500 (n=617), k=5
| crowd | coverage | precision |
| --- | --- | --- |
| retest (same people) | 0.984 | 0.958 |
| other_real_people | 0.981 | 0.908 |
| gaussian_copula (pool) | 0.888 | 0.862 |
| independent_marginals | 0.512 | 0.577 |
| claude_mode (Haiku 5.5) | 0.345 | 0.963 |
| gpt41mini_twin | 0.344 | 0.956 |
| claude_sampled | 0.170 | 0.269 |
Twin answer spread (entropy) collapses for gpt41mini_twin: risk_benefit 0.33 vs real 0.81; decisions 0.43 vs 0.96.

### Experiment 1: GSS (n=334), k=5
| crowd | coverage | precision |
| --- | --- | --- |
| other_real_people | 0.961 | 0.895 |
| gaussian_copula | 0.880 | 0.880 |
| independent_marginals | 0.740 | 0.778 |
| claude_mode | 0.638 | 0.997 |
| claude_sampled | 0.371 | 0.521 |

### Experiment 2 (H4) and 2b (H4b): agent loop on GSS, coverage by real people asked (25/50/75/100), 3 seeds
| method | 25 | 50 | 75 | 100 |
| --- | --- | --- | --- | --- |
| H4 Blindspot targeted | 0.641 | 0.662 | 0.655 | 0.656 |
| H4b verify | 0.621 | 0.610 | 0.610 | 0.614 |
| copula_on_asked | ~0.76–0.81 | ~0.83–0.85 | ~0.83–0.87 | ~0.83–0.86 |
| claude_plus_asked | ~0.72–0.74 | ~0.82 | ~0.85 | ~0.89 |
Why they failed: H4's gap detection used the tiny asked set as reference (found almost no gaps). H4b's rule (95th pct of synthetic-to-synthetic NN distance) flagged everyone because claude_mode people are nearly identical; it then replaced the whole crowd with recipe people who were barely closer to targets (0.269 vs 0.292). Conclusion: text descriptions do not teach the LLM missing structure.

### Experiment FP: Flashpoint synthetic panel (GSS live items, n=200, uncalibrated, gender × age × education matched via chat quota plan; race/income/geography invented)
| crowd (n=200) | coverage | precision |
| --- | --- | --- |
| other_real_people | 0.915 | 0.934 |
| gaussian_copula | 0.791 | 0.883 |
| independent_marginals | 0.631 | 0.779 |
| flashpoint run 1 | 0.611 | 0.765 |
| flashpoint run 2 | 0.617 | 0.740 |
| claude_mode | 0.567 | 0.997 |
Flashpoint: no bunching, entropy close to real; the two test panels agreed (0.611 vs 0.617) but the two holdout panels did not (0.759 vs 0.674), so run-to-run stability is not established; per-item distributions ~2× farther from real than baselines (TV 0.14 vs 0.07). Calibration (as described by Flashpoint's agent, unverified) only reweights age/gender composition.

### Final holdout run (once, 2026-11-13, code commit a828ae1, `results/final_holdout.md`; never rerun)
Fit and filter on pool only; Claude holdout personas from demographics only (cost $0.10). k=5.
| GSS holdout (n=224, 20 live items) | coverage | precision |
| --- | --- | --- |
| other_real_people | 0.982 | 0.938 |
| gaussian_copula (pool) | 0.915 | 0.933 |
| copula on 50 asked (10 seeds) | 0.893 | 0.902 |
| independent_marginals | 0.853 | 0.817 |
| copula on 25 asked (10 seeds) | 0.849 | 0.900 |
| select_from_50, lambda 0.5, pool filter | 0.818 | 0.874 |
| Flashpoint panel 1 / 2 (n=200) | 0.759 / 0.674 | 0.840 / 0.845 |
| claude_mode | 0.634 | 1.000 |
| claude_sampled | 0.487 | 0.647 |

| Twin holdout (n=412, 30 items) | coverage | precision |
| --- | --- | --- |
| retest | 0.990 | 0.961 |
| other_real_people | 0.976 | 0.922 |
| gaussian_copula | 0.886 | 0.859 |
| independent_marginals | 0.600 | 0.636 |
| claude + GPT 50/50 mix | 0.438 | 0.976 |
| gpt41mini_twin | 0.354 | 0.981 |
| claude_mode | 0.350 | 0.973 |
| claude_sampled | 0.180 | 0.408 |
The holdout agrees with test: every AI-only crowd is at or below shuffled answers (Flashpoint panel 1 is the closest at 0.759 vs 0.853), statistics fit on real people win, selection from AI candidates helps but stays under the copulas. claude_mode puts 35% of GSS items above 80% on one option.

Costs so far: about $2.6 of Anthropic API (Haiku 5.5). Flashpoint: $0 (synthetic panels free; no paid calls ever made).

## 5. Code map
- `blindspot/data/`: `schema.py` (`Item`, `Respondents`), `twin2k.py`, `gss.py`, `items.py`, `inspect.py`
- `blindspot/eval/`: `splits.py` (`make_splits`, `assert_no_leak`), `experiment0.py`, `experiment1.py`, `experiment2.py`, `experiment2b.py`, `experiment_fp.py`
- `blindspot/space/`: encoding, family-weighted Gower, maps (UMAP)
- `blindspot/coverage/`: `evaluate()` with coverage, precision, rcg_ratio, CIs, k sensitivity
- `blindspot/generators/`: `baselines.py` (independent_marginals, gaussian_copula, other_real_people), `claude_personas.py` (cached, cost guard), `flashpoint_panel.py`
- `blindspot/agent/`: H4 and H4b loops (parked)
- `blindspot/api/`: `main.py`, `live.py` (demo server; do not change except for bugs)
- `web/`: static demo UI (story `/`, phone `/join`, projector `/stage`), sample data in `web/mock/`; real data goes to `results/ui/` via H8 (`handovers/H8-export-ui-data.md`)
- `results/`: experiment JSON/MD, `results/agent/`, `results/cache/` (ignored), `results/maps/` (ignored)
- Tests: `pytest -q` (107 passing as of H5)
- Environment: Windows, Cursor, `.venv` (Anaconda Python 3.12). `CLAUDE_MODEL=claude-haiku-5-5` in `.env`. Flashpoint MCP connected at user scope (OAuth).

## 6. Hard rules (summary; full text in CLAUDE.md)
1. Splits sacred: agent/generators may read **pool** only; scores on **test**; **holdout** only in one final run.
2. No copying real answer rows into a synthetic crowd unless the method is explicitly labeled "includes real respondents" (e.g. claude_plus_asked). Near-duplicate guard (< 0.02 Gower) reported always.
3. Synthetic crowds match target demographics where possible; report when they don't.
4. Report all baselines next to ours, even when ours loses.
5. Name gaps by answer patterns, never demographic labels. Model-written demographic reasons are unverified and must never appear in the UI or public files.
6. No raw data, ids or secrets in git.
7. Cache every LLM call; cost guard $5 per run.
8. Inspect before assuming.
9. Twin persona text never enters any prompt.
10. **Never call paid Flashpoint actions** (recruit, quotes, checkout, payment_confirmation) without the user's explicit approval.

## 7. Working method
- Small, scoped handovers. Each ends with: commit, report tables, **stop**.
- Before building a new design, a fresh reviewer ("goldfish") reads the spec and lists ways it could mislead or fail.
- Tune on test; report holdout once at the end.

## 8. CURRENT TASK (do this now)

Two zero-cost experiments, back to back, GSS test split, never holdout, **no new Claude or Flashpoint calls**.

### Part 1: Bake-off round 1 (select-from-many + model mix)
- For each GSS test persona, draw 50 candidates from the cached `claude_sampled` per-item probabilities (seeded, independent per item).
- `realism` = mean Gower distance to the 5 nearest **pool** real people (lower = more realistic). `novelty` = distance to the nearest already-selected crowd member.
- Greedily build a same-size crowd: for each persona pick the candidate minimising `realism − λ·novelty`, λ ∈ {0, 0.5, 1, 2}.
- Report coverage, precision, CIs, k=3/5/10 per λ next to claude_mode, claude_sampled, gaussian_copula (pool), other_real_people. Report near-duplicate share (realism uses pool data as a filter, not copying).
- Repeat the selection using Flashpoint's two saved panels as the candidate pool (select 200 from 400 with the same score) and evaluate at n=200 like Experiment FP.
- Twin model mix: crowd = 50% claude_mode + 50% gpt41mini_twin (same n, seeded) and the union upper bound (a real person covered if either full crowd covers them).
- Output `results/bakeoff1.json` and `results/bakeoff1.md`. **Commit, then continue to Part 2.**

### Part 2: H4c, learning whom to ask
- Crowd at each budget = claude_mode with the asked real pool people swapped in (label: "includes real respondents"; near-duplicate guard exempts those rows). Budgets 25/50/75/100 (rounds of 25), **10 seeds**.
- **random**: random pool people (equals claude_plus_asked).
- **targeted**: round 1 random. After each round: miss score per asked person = Gower distance to the nearest **claude_mode** crowd member; fit logistic regression on one-hot demographics with label = miss score above the asked-people median; next round asks 70% of the unasked pool people with highest predicted miss probability (only demographics visible before asking) + 30% random. Log coefficients per round.
- Same comparison for gaussian_copula fit on targeted vs random asked people.
- **Self-knowledge**: each round, the agent's estimated share of the population its claude_mode crowd misses (from asked people, reweighted to pool demographics) next to the true test uncovered share.
- **Fairness audit**: claude_mode coverage on test by demographic group (age band, gender, education, income band, region), only groups with n ≥ 30, bootstrap CIs. Plain table, no interpretation.
- Output `results/h4c.json` and `results/h4c.md`. Commit.

### Report back
Both parts' tables (mean and seed range), near-duplicate shares, runtime. Then **stop**.

## 9. After this (not now)
- H8: export real results into `results/ui/` for the demo (`handovers/H8-export-ui-data.md`), updated to the final story.
- Clean public files (remove model-written demographic reasons from `results/agent/` logs or move them out of the repo), README, then make the repo public.
- Holdout headline run (once).

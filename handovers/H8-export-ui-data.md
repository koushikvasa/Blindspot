# H8: Export real results for the UI

Run this in **Claude Code** after the H4b run finishes. Read `CLAUDE.md` and `docs/UI_DATA_CONTRACT.md` first. The UI (`web/`, `blindspot/api/`) is already built and tested; do not change it except for bugs, and say if you do. Never touch the holdout split. No new API spend.

## Task
Write `blindspot/eval/export_ui.py` (`python -m blindspot.eval.export_ui`) that writes the files in `docs/UI_DATA_CONTRACT.md` to `results/ui/`, from existing results only:

1. `summary.json` from `results/experiment0.json` and `results/experiment1.json` (test split). Use these crowd keys and labels: retest "Same people, retested", other_real_people "Other real people", gaussian_copula "Statistics (copula)", independent_marginals "Shuffled answers", claude_mode "Claude, typical answers", claude_sampled "Claude, random answers", gpt41mini_twin "GPT-4.1-mini twins".
2. **One fixed layout per dataset:** fit 2-D coordinates once on the real test set (UMAP on precomputed family-weighted Gower, seed 42; scale to [0, 1]). Place each synthetic person at the mean (x, y) of their 5 nearest real people (Gower). Real points in the same order in every file.
3. `map_{dataset}_{crowd}.json` for every crowd in summary, with `covered01` from the H2 rule (k = 5). Check: the share of covered points equals the crowd's coverage in summary (assert to 3 decimals).
4. `agent_gss.json` and `map_gss_agent_r{n}.json` from the H4b run, **seed 42, verified method** (or the best H4b method if verification was renamed). Round 0 = claude_mode. Gaps: name, one-sentence answer-pattern summary (from the archetype text, shortened, no demographics), size, status (`learned` / `rejected` / `cannot_imagine`), recipes tried/accepted, recruited, observed_rate. **Drop every model-written demographic reason.** If a gap name or summary mentions a demographic group (race, gender, age, income, region, religion), rewrite it to describe answers only and list what you changed in the report.
5. `scoreboard_gss.json` from H4/H4b Experiment 2 (mean and seed range at 25/50/75/100) for: the best Blindspot method, Blindspot hybrid (if run), claude_plus_asked, copula_on_asked. References: claude_mode at budget 0 and other_real_people. Write the `caption` from the numbers in one plain sentence that says who wins.
6. `live_gss.json`: GSS live items (`live_ok=True`) in a fixed order with verified wording and options; `real` = test split encoded on those items; `real_xy` from the GSS layout; `crowd` = the final-round crowd of the method in `agent_gss.json`; `base_rate_uncovered` = 1 − that crowd's test coverage **computed on the live items only** (report it next to the full-item coverage). No ids anywhere.

## Checks
- No respondent ids, persona text or demographic reasons in any `results/ui/` file (add a test).
- `pytest -q` passes, including `tests/test_api.py`.
- Start the server (`uvicorn blindspot.api.main:app --port 8000`), open `/api/health`: all expected files listed. Open `/` and confirm the "Sample data in use" badge is gone.

## Report
The list of files written, the coverage assertion results, any gap names you rewrote, the live-items base rate vs the full-item coverage, then stop. `results/ui/` should be committed (no ids, small files) unless a file exceeds 5 MB.

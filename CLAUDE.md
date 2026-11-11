# Blindspot: rules for Claude Code

## What this is
An agent that audits synthetic survey respondents against real ones, finds uncovered real people, and learns to cover them. Python backend (`blindspot/`), Next.js UI later (`web/`). Read `docs/PLAN.md` first.

## Rules that never bend
1. **Splits are sacred.** Real respondents are split once (seeded) into `pool` (agent may request from), `test` (used for round-by-round scores), `holdout` (headline number, touched only by `eval/final.py`). No code path may read `test` or `holdout` rows into a generator prompt.
2. **No copying.** The agent improves generators by editing persona recipes (text rules/descriptions), never by inserting real respondents' answer rows into the synthetic crowd. The "add N real people" baseline exists to prove we beat copying.
3. **Same demographics.** Every synthetic crowd matches the real target set's demographic distribution, so gaps reflect how answers combine, not who was sampled.
4. **Honest numbers.** Report all baselines (independent marginals, Gaussian copula, add-N-real) and the human test-retest ceiling next to ours, even when ours loses.
5. **Name gaps by answer patterns,** not demographic labels. Demographics only as descriptive footnotes.
6. **No raw data or secrets in git.** `data/raw`, `data/processed`, `.env` are ignored. Never print API keys.
7. **Cache every LLM call** (`results/cache/`, keyed by prompt hash) so experiments are reproducible and cheap.
8. **Inspect before assuming.** Dataset schemas are not documented here; read the files and write what you find into `docs/DATA.md` before coding against them.

## Environment
Windows + Cursor. Use `pathlib`, no shell-specific scripts. Python 3.11+. Run tests with `pytest -q`.

# Handoff: Blindspot after H1–H3 (2026-11-12)

For whoever continues in Cursor (you or Cursor's agent). Read `CLAUDE.md` first (9 rules that never bend), then `docs/PLAN.md`, `docs/DATA.md`, `docs/METRICS.md`, `results/experiment1.md`.

## Where things stand
| Step | Status | Key output |
| --- | --- | --- |
| H1 data + splits | done, approved | loaders, 30 Twin + 24 GSS items, `docs/DATA.md` |
| H2 coverage + Experiment 0 | done, approved | `results/experiment0.md` |
| H3 Claude crowds + Experiment 1 | done, **awaiting review** | `results/experiment1.md`, `results/prompts/` |
| H4 agent loop + Experiment 2 | **next**: spec in `handovers/H4-agent-loop-and-experiment2.md` (file exists, **not yet committed**) | n/a |

Last commit: `25b54fd`. `pytest -q` passes (59 tests). Working tree is clean apart from the H4 handover file and this file.

### Headline numbers so far (test split, k = 5)
| crowd | Twin coverage / precision | GSS coverage / precision |
| --- | --- | --- |
| retest (same people, later) | 0.984 / 0.958 | n/a |
| other_real_people | 0.981 / 0.908 | 0.961 / 0.895 |
| gaussian_copula | 0.888 / 0.862 | 0.880 / 0.880 |
| independent_marginals | 0.512 / 0.577 | 0.740 / 0.778 |
| claude_mode (Haiku 5.5) | 0.345 / 0.963 | 0.638 / 0.997 |
| gpt41mini_twin | 0.344 / 0.956 | n/a |
| claude_sampled | 0.170 / 0.269 | 0.371 / 0.521 |

Robustness (Twin, unweighted Gower): ranking changes only by `claude_mode` and `independent_marginals` swapping places. Bunching (>80% of people on one option): real 0%, claude_mode 20% (Twin) / 29% (GSS).

## Environment and commands (Windows)
- Python 3.12 venv in `.venv` (Anaconda base). Activate: `.\.venv\Scripts\Activate.ps1`. Install: `pip install -r requirements.txt`.
- Tests: `pytest -q` (data tests skip automatically if `data/raw` is missing).
- `python -m blindspot.data.inspect`: H1 sanity report; also (re)creates `data/processed/splits_*.json` (deterministic, seed 42).
- `python -m blindspot.eval.experiment0`: H2, ~85 s (≈80 s is UMAP maps).
- `python -m blindspot.eval.experiment1 [--estimate-only] [--max-cost 5] [--no-batch]`: H3. Fully cached now, so it costs $0. ~9 min (robustness check + maps).
- Printing Unicode in a Windows shell: set `PYTHONIOENCODING=utf-8`. For background runs use `python -u` (otherwise the log stays empty until exit).
- `.env` (never print it): `ANTHROPIC_API_KEY`, `CLAUDE_MODEL` (**currently `claude-haiku-5-5`**, not the Sonnet shown in `.env.example`), others unused so far.

## Code map
| Path | What |
| --- | --- |
| `blindspot/data/schema.py` | `Item` (code, wording, kind, options, theme, sensitive, verified, family, `live_ok`), `Respondents` (+ `subset(ids)`) |
| `blindspot/data/twin2k.py`, `gss.py`, `items.py` | loaders and the approved item specs |
| `blindspot/eval/splits.py` | `make_splits`, `get_or_make_splits`, `load_splits`, `assert_no_leak(ids, splits, allowed=("pool",))` |
| `blindspot/space/` | `encode`, family-weighted `gower`, `map_coords` (UMAP) |
| `blindspot/coverage/` | `evaluate` (coverage, precision, rcg_ratio, CIs, k-sensitivity, per-family entropy), `spread` helpers |
| `blindspot/generators/baselines.py` | marginals, Gaussian copula, other real people (pool-only, leak-guarded) |
| `blindspot/generators/claude_personas.py` | persona prompts, `plan` → `estimate` → `execute`, `recipe` parameter (for H4) |
| `blindspot/generators/claude_client.py` | response cache, pricing, cost guard, Message Batches (resumable) |
| `blindspot/eval/experiment0.py`, `experiment1.py` | experiment runners |

## Facts and gotchas worth knowing before H4
**Data**
- Twin target is all 2,058 people. GSS target = main cross-section (`sample == 13`) on ballot 1 = **1,116** people. The 677 extra `.dta` rows are the AmeriSpeak oversample (no `wtssps`); GSS weight kept as `demographics["weight"]`.
- Splits: Twin 1029 / 617 / 412, GSS 558 / 334 / 224 (pool / test / holdout). **Never read holdout**; experiment runners drop it right after loading and assert it never reaches crowds or outputs.
- **Rule 9:** Twin `persona_*` columns contain wave-4 answers to the scored items. Loaders read only demographics + option labels. Persona prompts use only `PERSONA_FIELDS` (age_band, gender, race, education, income_band, region); everything else is ignored.
- All 24 GSS items have verified wording (official 2024 Ballot 1 questionnaire); `satdemoc` keeps a `{response to SATDEMOC_fill}` marker. Live-mode items: 10 Twin, 20 GSS.

**Metrics**
- Coverage = k-NN (Naeem et al. 2020), ties count as inside. Precision bootstrap resamples *synthetic* people (resampling real people biased it; documented in `docs/METRICS.md`).
- `results/maps/` and `results/cache/` are git-ignored (regenerable; maps hold per-person ids).

**Claude API (verified against the current API, Nov 2026)**
- Current models **reject forced `tool_choice`** → we use structured output (`output_config.format` JSON schema).
- The schema must stay small: one property per question compiled to a grammar the API rejected as too large. We use `{"answers": [{"q", "p", "a"}, ...]}`.
- **No `temperature` field** (API version 2026-10-21 rejects it; the default 1.0 applies).
- Thinking off is model-specific: Haiku 5.5 `{"type": "disabled"}`, Sonnet 5.5 `{"type": "between_tools"}`, Opus 5.5 cannot disable (use low effort). See `claude_personas.thinking_off`.
- Cache key = hash of (model, system, user, item codes, recipe version, prompt version, replicate, attempt). `replicate` keeps identical personas independent. H4 recipes change the system prompt, so they get their own cache entries automatically.
- Message Batches: the batch id is saved in `results/cache/_batches/` until results are collected, so an interrupted run **resumes** the same batch instead of paying twice; polling waits out network drops (the first H3 run died on a DNS drop).
- Cost guard: `count_tokens` (free) before any paid call, $5 default. H3 actual cost: **$0.23** for 1,078 responses.
- GSS claude responses: 1.09% of answers still missing after one retry (left as NaN).

## Working agreements (how the user wants this run)
- Do one handover at a time, exactly as written; **stop and report** at the handover's checkpoint and wait for approval.
- Inspect files before coding against them; record findings in `docs/`.
- Report honest numbers, including when ours loses; don't interpret beyond the numbers unless asked.
- Before any paid API run, show the estimate; stop if it's over $5.
- Small commits with conventional messages (`feat:`, `fix:`, `docs:`, `results:`). Never commit `data/raw`, `data/processed`, `.env`, or `results/cache`.

## Suggested first steps in Cursor
1. `pytest -q` to confirm the environment.
2. Commit `handovers/H4-agent-loop-and-experiment2.md` (and this file if you want it kept).
3. Start H4 from its handover. Its loop reuses `claude_personas.plan/estimate/execute` (pass the archetype as `recipe`), `coverage.evaluate`, `baselines.gaussian_copula`, and `assert_no_leak` with `allowed=("pool",)`.

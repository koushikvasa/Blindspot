# H2: Distance, coverage metrics, baselines, and Experiment 0

Read `CLAUDE.md`, `docs/PLAN.md`, `docs/DATA.md` first. Scope is ONLY this handover; do not start H3 (no Claude API calls in H2).

## Goal
Answer one question with honest numbers: **does a synthetic crowd leave real people uncovered, and how does it compare with simple baselines and with the same people retested?** Run it on Twin-2K-500 (where the dataset authors' GPT-4.1-mini twins exist) and run the baselines on GSS.

## 0. Side task (small, do first): GSS wording
For the 10 GSS items with `verified=False` (trppl, trcourts, trcong, intcnfrm, satdemoc, techesy, harmgood1, aiworry, aidrive, aimed), try to get the exact question text and answer labels from the GSS Data Explorer variable pages (gssdataexplorer.norc.org) or other official NORC/ISSP 2024 documentation. If you get official text, store it and set `verified=True`. If not reachable, leave them and say so. Do not invent wording.

## 1. `blindspot/space`
- `encode(resp) -> np.ndarray`: each answer becomes a value in [0, 1] (ordinal/binary: index / (k-1)); NaN stays NaN.
- `gower(A, B, items) -> np.ndarray` distance matrix: mean absolute difference over items both people answered, with **each `Item.family` weighted equally** (items weighted 1/(items in family), then normalized). Vectorized numpy; must handle ~2,000 × ~2,000 quickly on a laptop (chunk if needed).
- `map_coords(real, synthetic_by_name, seed=42)`: 2-D coordinates for the UI using UMAP on the precomputed Gower distance of the combined set (real target + each crowd). Save to `results/maps/{dataset}_{crowd}.json` as `{id, x, y, kind: real|synthetic, crowd, covered}`.

## 2. `blindspot/coverage`
Use the k-nearest-neighbour coverage idea from generative-model evaluation (Naeem et al., 2020, "Reliable Fidelity and Diversity Metrics for Generative Models"): a real person is **covered** if at least one synthetic person lies within that real person's distance to their k-th nearest *real* neighbour (within the same real target set, excluding themselves).
Implement `evaluate(real_target, crowd, items, k=5) -> dict` returning:
- `coverage`: share of real people covered (the headline).
- `uncovered_ids`: list of real ids not covered.
- `precision`: share of synthetic people that fall inside at least one real person's k-NN ball (how realistic the fakes are).
- `rcg_ratio`: median over real people of (distance to nearest synthetic) / (distance to k-th nearest real). Our own simple version of the "respondent coverage gap" idea (Kumagae & Kaneko, 2026); name it `rcg_ratio` and say in docs that the paper's exact formula may differ.
- `per_family_entropy` for the crowd vs the real target (from H1's inspect logic).
- Bootstrap 95% CIs (200 resamples of real people, seeded) for coverage and precision.
Crowds are compared at the **same size as the real target set** (subsample if larger, with a fixed seed). Also report coverage at k = 3, 5, 10 as a sensitivity check.

## 3. `blindspot/generators/baselines.py`
All baselines are fit on the **pool** split only and generate a crowd the size of the target set. Use `assert_no_leak`.
- `independent_marginals(pool)`: each answer sampled independently from the pool's per-item shares.
- `gaussian_copula(pool)`: map each item to normal scores via ranks, estimate the correlation matrix, sample, map back through each item's empirical quantiles. Handle NaN (pairwise or impute with mode for fitting only).
- `other_real_people(pool)`: a random sample of real pool people, as a reference for "what a perfect generator of new people would look like".
Twin-only crowds (no fitting; same people as the target): `gpt41mini_twin` and `retest` restricted to the target ids.

## 4. `blindspot/eval/experiment0.py`
`python -m blindspot.eval.experiment0` runs, with target = **test** split:
- Twin: crowds = gpt41mini_twin, retest, independent_marginals, gaussian_copula, other_real_people.
- GSS: crowds = independent_marginals, gaussian_copula, other_real_people.
Outputs:
- `results/experiment0.json` (all metrics, CIs, k-sensitivity, settings, git commit hash).
- `results/experiment0.md`: one table per dataset (crowd | coverage [CI] | precision [CI] | rcg_ratio), sorted by coverage, plus the per-family entropy table.
- Map JSON files for Twin test vs gpt41mini_twin and vs gaussian_copula.
Do NOT touch the holdout split anywhere in H2.

## 5. Tests
- gower: identical people → 0; opposite ends of every item → 1; family weighting works on a toy example; NaN handled.
- coverage: crowd identical to target → coverage 1.0; crowd far away → 0.0; precision behaves likewise.
- baselines: marginals match pool shares within tolerance; copula preserves marginals and roughly the correlation sign; nothing reads test/holdout ids.
- experiment0 runs end to end on a tiny synthetic fixture (fast test, no real data needed).

## Done when
- `pytest -q` passes
- `python -m blindspot.eval.experiment0` finishes and writes the files above
- Commit, then report: the two tables from `results/experiment0.md`, the k-sensitivity, runtime, and anything surprising. Then STOP. Do not interpret beyond the numbers; I will review.

## Out of scope
Claude/Flashpoint generators, the agent loop, API, UI.

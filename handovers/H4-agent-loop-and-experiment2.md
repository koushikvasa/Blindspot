# H4: The self-improving agent loop and Experiment 2

Read `CLAUDE.md`, `docs/PLAN.md`, `docs/DATA.md`, `docs/METRICS.md`, and `results/experiment1.md` first. Scope is ONLY this handover; do not start H5. Never touch the holdout split.

## What we learned (context for design)
- `claude_mode` is realistic but collapsed (high precision, low coverage).
- `claude_sampled` restores spread but makes incoherent people (low precision AND lower coverage).
- So the gap is **structure**: which answers go together in one person. The agent must learn answer *patterns* (archetypes), not add noise.

## The scenario we simulate
The agent starts with only its synthetic crowd and a **budget of real people it may ask** (default 100 total, in rounds of 25). It does NOT know the real population. Asking = drawing people from the **pool** split. Before asking, it can see only demographics (like a panel quota); after asking, it sees their answers. Every round is scored on the **test** split, which the agent never sees.

## Loop (per round r = 1..4)
1. **Ask.**
   - Round 1: draw 25 pool people uniformly at random ("scouting").
   - Rounds 2+: draw 25 pool people **targeted** by demographic segments the agent chose in step 4 of the previous round (sample within those segments; fall back to random if a segment is empty). Log which segments it asked for and why.
2. **Find gaps.** Using all real people asked so far (pool only), compute which of them are uncovered by the current crowd (H2 coverage logic, k=5, with the asked set as the real reference; document that this reference is small and noisy). Cluster the uncovered ones by answer pattern (Gower distance + agglomerative clustering, 2–5 clusters, seeded).
3. **Describe.** For each cluster, one Claude call sees that cluster's answers (pool data, allowed) and writes an **archetype**: a short plain-language description of the answer pattern in at most 120 words, naming tendencies and how answers go together, not listing item-by-item answers. Also returns: a name for the pattern (by answers, not demographics, per rule 5), and its estimated share of the population (from the asked sample, with the agent's uncertainty).
4. **Seek.** One Claude call sees the archetypes plus the demographics of the asked people and proposes up to 3 demographic segments to ask next round where it expects to find more people like the uncovered ones, with a reason each. (This is the "seeks answers from the right people" step.)
5. **Learn.** Update the **recipe library**: a list of archetypes with weights (shares). Merge near-duplicate archetypes across rounds (one Claude call, or text similarity) and re-estimate weights from all asked people. Version the library (`results/agent/{dataset}/recipes_v{r}.json`).
6. **Regenerate.** Build a new crowd for the test demographics using `claude_personas.py`: each synthetic person gets their demographics plus ONE archetype drawn by weight (or "no archetype" with the remaining weight), passed as the `recipe`. Use the chosen answer (mode), not sampling.
7. **Measure.** Evaluate the new crowd on the test split (coverage, precision, rcg_ratio, CIs, entropy). Append to the round log.

## Guards (must exist and be tested)
- **No copying (rule 2):** archetype text must not contain answer labels for more than 3 specific items verbatim (check), and after regeneration report the share of synthetic people that are near-duplicates (Gower distance < 0.02) of any asked real person. Fail the round loudly if above 5%.
- **No leak:** `assert_no_leak` on every set of ids passed to any prompt; only pool ids allowed.
- **Cost guard:** estimate per round; stop and ask if the full run's estimate exceeds $5. Cache everything (keys include the recipe library version).

## Experiment 2: `python -m blindspot.eval.experiment2`
Run on **GSS first** (live demo dataset), then Twin. Seeds: run the whole loop with 3 seeds and report mean ± range.
Compare, at the **same budget of real people asked** (25, 50, 75, 100):
- **Blindspot (targeted)**: the full loop above.
- **Blindspot (random asking)**: same loop, but every round asks random pool people (ablation: does targeting matter?).
- **Copula on the same people**: Gaussian copula fit only on the real people Blindspot asked at that budget.
- **Claude + the asked people**: claude_mode crowd with the asked real people swapped in for the same number of synthetic people (the "just add real people" baseline).
- Fixed references: claude_mode (budget 0), claude_sampled, copula on the full pool, other_real_people, and retest for Twin.
Outputs:
- `results/experiment2.json`, `results/experiment2.md` (coverage vs budget table per dataset, plus precision).
- `results/agent/{dataset}/rounds.json`: per round, the asked ids (pool only), segments requested and reasons, clusters, archetypes, library version, metrics, near-duplicate share, cost. This file feeds the UI.
- Map JSON for GSS: test vs the crowd at each round (for the round slider), with each real person's covered flag per round.

## Tests
- Archetype guard catches item-by-item copying (fixture).
- Near-duplicate detector flags a copied person.
- Targeted asking samples only within requested segments and only from pool.
- Loop runs end to end on a tiny fixture with a mocked Claude client.
- Same seed → same asked ids and same results (with cache).

## Done when
- `pytest -q` passes
- `python -m blindspot.eval.experiment2` completes on GSS and Twin within the cost guard
- Commit, then report the coverage-vs-budget tables, one example round from rounds.json for GSS (segments + reasons + archetype names), near-duplicate shares, and actual cost. Then STOP.

## Out of scope
Flashpoint, API, UI, holdout evaluation.

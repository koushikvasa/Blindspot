# H3: Claude persona crowds and Experiment 1

Read `CLAUDE.md`, `docs/PLAN.md`, `docs/DATA.md`, `docs/METRICS.md` first. Scope is ONLY this handover; do not start H4. Never touch the holdout split.

## Goal
Build the plain "silicon sampling" generator with Claude and measure it with the H2 metrics, in two variants, so H4's agent has honest baselines to beat:
- **claude_mode**: Claude answers as the persona; we take its single answer.
- **claude_sampled**: Claude gives a probability for every option of every item; we sample one answer from those probabilities (seeded).

## 1. `blindspot/generators/claude_personas.py`
- Input: a target demographics table (the **test** split's demographics) and the item list. Output: a `Respondents` crowd with one synthetic person per target person, same demographics (rule 3 in CLAUDE.md).
- Persona = demographics only, written as a short first-person backstory from the demographics fields (age band, gender, education, income band, region, plus any other demographic fields in DATA.md). **No persona text from Twin-2K-500, no answers from any split** (rule 9).
- One call per synthetic person answers all items at once via forced tool use returning JSON: for each item code, a probability for each option (summing to 1) and the single answer it would choose. Validate with pydantic; retry once on invalid output; record failures.
- Model from `CLAUDE_MODEL` in `.env`. Temperature 1.0. Use the Message Batches API if available in the installed SDK, else bounded concurrency (8) with backoff.
- **Cache every response** in `results/cache/` keyed by a hash of (model, prompt, item set, recipe version). Re-runs must cost nothing.
- **Cost guard:** before calling the API, print the number of calls and an estimated token count and cost; if the estimate exceeds $5 for the run, stop and ask me.
- Expose `recipe` as a parameter (a text block appended to the system prompt, default empty). H4 will edit recipes; H3 only uses the empty one. Save the exact prompts used to `results/prompts/`.

## 2. `blindspot/eval/experiment1.py`
`python -m blindspot.eval.experiment1` with target = **test** split, for both datasets:
- Generate `claude_mode` and `claude_sampled` crowds (both from the same cached calls: one call gives both the chosen answer and the probabilities).
- Evaluate with H2's `evaluate` (k = 3, 5, 10, bootstrap CIs) next to all H2 crowds (load H2 results; don't recompute baselines unless needed).
- **Robustness check:** recompute the main Twin table with unweighted Gower (every item equal) and report whether the crowd ranking changes.
- Write `results/experiment1.json` and `results/experiment1.md` (same table format as H2, with the new rows added and sorted by coverage), per-family entropy including the new crowds, and the call count, tokens and actual cost.
- Write map JSON for GSS test vs `claude_mode` and vs `claude_sampled` (the live demo will use GSS).
- Also compute and report: the share of items where `claude_mode` puts more than 80% of people on one option, versus the real target.

## 3. Tests
- Prompt builder never includes persona text or answer data (test with a fake persona string and a sentinel answer value).
- Probability parsing: normalizes, rejects negative values, handles a missing item by retry then NaN.
- Sampling is deterministic for a fixed seed.
- Cache hit makes zero API calls (mock the client).
- Cost guard stops above the threshold (mock).

## Done when
- `pytest -q` passes
- `python -m blindspot.eval.experiment1` finishes on both datasets within the cost guard
- Commit (no cache files, prompts are fine to commit), then report the two tables, the robustness check, the >80% bunching comparison, and the actual cost. Then STOP.

## Out of scope
The agent loop (gap finding, describing, requesting, recipe editing), Flashpoint, API, UI.

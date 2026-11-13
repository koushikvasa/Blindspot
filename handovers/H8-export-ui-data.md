# H8: Export real results for the UI (final story)

Run in **Claude Code** after the holdout run. Read `docs/CONTEXT.md`, `CLAUDE.md` and `docs/UI_DATA_CONTRACT.md` first. The UI (`web/`, `blindspot/api/`) is built and tested; change it only for bugs and say if you do. No new API spend. Do not run anything on the holdout split again; you may READ `results/final_holdout.json`.

## Which numbers
Use **test-split** results for everything in the UI (they have maps, seeds and CIs), except: where `results/final_holdout.json` has the same crowd, add `"holdout_coverage"` next to it in `summary.json` so the presenter can quote it. Never mix item sets in one table without a `note`.

## Files to write to `results/ui/` (formats in the contract)
1. `summary.json`: Twin crowds from experiment0/1 (30 items, n=617). GSS crowds on the **20 live items** (n=334) from bakeoff1, plus `flashpoint_panel` from experiment_fp (n=200, add `"n": 200`; use run 1, or the mean of both runs with its range in a note). Labels: retest "Same people, retested", other_real_people "Other real people", gaussian_copula "Statistics (copula)", independent_marginals "Shuffled answers", flashpoint_panel "Flashpoint synthetic panel", claude_mode "Claude, typical answers", claude_sampled "Claude, random answers", gpt41mini_twin "GPT-4.1-mini twins".
2. Maps: one fixed real-person layout per dataset (UMAP on precomputed family-weighted Gower over the real test set, seed 42, scaled to [0, 1]); synthetic people placed at the mean (x, y) of their 5 nearest real people. `map_{dataset}_{crowd}.json` for every crowd in summary, including `map_gss_flashpoint_panel.json`. `covered01` from the H2 rule; assert the covered share equals the summary coverage (3 decimals) for the same item set and n.
3. `repairs_gss.json`, in this order, coverage on GSS test:
   - Random answers from Claude (`ai`)
   - Flashpoint synthetic panel (`ai`, note "n=200")
   - Teach Claude archetypes, H4 at 100 real people (`ai_filtered`, note "24 items")
   - Teach Claude verified recipes, H4b at 100 (`ai_filtered`, note "24 items")
   - Pick best of 50 AI drafts, H4d select_claude at B=100 (`ai_filtered`, note "100 real people as filter")
   - Pick best drafts + Flashpoint, H4d select_claude_plus_fp at B=100 (`ai_filtered`, note "100 real people as filter")
   - 25 real people + statistics, H4d copula_on_asked at B=25 (`real`)
   - 100 real people + statistics, H4d copula_on_asked at B=100 (`real`)
   Use seed ranges as `lo`/`hi`. `baseline` = claude_mode on the 20 live items. Caption written from the numbers.
4. `scoreboard_gss.json` from H4d (budgets 25/50/75/100, mean + seed min/max): copula_on_asked "Statistics on the real people", claude_plus_asked "AI crowd + the real people", select_claude_plus_fp "AI drafts filtered by them (+Flashpoint)", select_claude "AI drafts filtered by them". References: claude_mode "AI crowd alone", other_real_people "Real people ceiling". Caption from the numbers.
5. `fairness_gss.json` from the H4c fairness audit (test split, claude_mode, groups n ≥ 30, bootstrap CIs). `overall` = claude_mode coverage on the same item set. Caption must say whether intervals overlap.
6. `live_gss.json`: GSS live items (verified wording, options), `real` = test split encoded on those items, `real_xy` from the GSS layout, `crowd` = **claude_mode** (the live question is "can the AI imagine you?"), `crowd_label` "Claude AI crowd", `base_rate_uncovered` = 1 − claude_mode coverage on the live items.
7. Do NOT write `realrecruit_gss.json` (that comes after the real Flashpoint study).
8. Delete nothing in `web/mock/`; real files simply take priority.

## Privacy and cleanup checks
- No respondent ids, persona text or model-written demographic reasons in any `results/ui/` file (add a test that scans them).
- Separately, list every committed file under `results/` that contains model-written demographic reasons (e.g. the H4 `rounds.json` segment reasons) or respondent ids, and propose how to remove them before the repo goes public (move to an ignored folder and add a short note). Do not rewrite git history yet; just report.

## Done when
- `pytest -q` passes.
- Server running: `/api/health` lists all files above; `/` shows no "Sample data in use" badge (the real-people scene shows its "recruiting" message, which is expected).
- Commit `results/ui/` and the export script. Report: files written, coverage assertion results, the captions you wrote, the cleanup list, then stop.

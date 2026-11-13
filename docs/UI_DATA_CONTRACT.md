# UI data contract

The web app reads JSON files from `results/ui/`. If a file is missing, the API serves the matching file from `web/mock/` and marks the response `"mock": true` (the page then shows a "sample data" badge). Exporting real files is H8's job (`handovers/H8-export-ui-data.md`).

**Layout rule:** every map of a dataset uses ONE fixed layout for real people (2-D coordinates fitted once on the real test set). Synthetic people are placed at the mean (x, y) of their 5 nearest real people (Gower). This keeps real dots still when switching crowds or rounds, so only colours and the synthetic glow change. Coordinates are scaled to [0, 1].

**Privacy rule:** no respondent ids anywhere in these files. Points are anonymous and in a stable order.

## `summary.json`
```json
{
  "model": "claude-haiku-5-5",
  "datasets": {
    "twin": {"label": "Twin-2K-500", "split": "test", "n": 617,
      "crowds": {"gpt41mini_twin": {"label": "GPT-4.1-mini twins", "coverage": 0.344, "coverage_ci": [0.301, 0.383],
                                     "precision": 0.956, "precision_ci": [0.937, 0.972]}}},
    "gss": {"label": "GSS 2024", "split": "test", "n": 334, "crowds": {}}
  }
}
```
Crowd keys used by the UI (any subset may be present): `retest, other_real_people, gaussian_copula, independent_marginals, flashpoint_panel, claude_mode, claude_sampled, gpt41mini_twin`. Each crowd may add `"n"` and `"items"` (e.g. Flashpoint is n=200 on the 20 live items).

## `map_{dataset}_{crowd}.json`
```json
{"dataset": "gss", "crowd": "claude_mode",
 "real": [[x, y, covered01], ...],
 "synthetic": [[x, y], ...]}
```
`covered01` is 1 if that real person is covered by this crowd (k = 5 rule), else 0. Real points are in the same order in every file of a dataset.

## `scoreboard_gss.json`
```json
{"budgets": [25, 50, 75, 100],
 "methods": {"blindspot_verified": {"label": "Blindspot", "mean": [...], "lo": [...], "hi": [...]}},
 "references": {"claude_mode": {"label": "Claude alone", "value": 0.638},
                "other_real_people": {"label": "Real people ceiling", "value": 0.961}},
 "caption": "One plain sentence saying who wins, written from the numbers."}
```

## `live_gss.json` (for `/join` scoring; no ids)
```json
{"items": [{"code": "happy", "wording": "...", "options": ["...", "..."], "family": "work_life"}],
 "family_weights": {"work_life": 0.25},
 "real": [[v0, v1, ...], ...],
 "real_xy": [[x, y], ...],
 "crowd": [[v0, v1, ...], ...],
 "crowd_label": "Blindspot crowd (round 4)",
 "base_rate_uncovered": 0.34}
```
Values are encoded answers in [0, 1] (`index / (k-1)`) or null, in the item order given. `real` is the GSS test split restricted to the live items. Item weights inside a family are equal; families are weighted by `family_weights` (equal by default).

## `repairs_gss.json` (scene "We tried five ways")
```json
{"baseline": 0.608,
 "items": [{"label": "Pick best of 50 AI drafts", "coverage": 0.765, "lo": 0.74, "hi": 0.79,
            "kind": "ai_filtered", "note": "100 real people as filter"}],
 "caption": "One plain sentence from the numbers."}
```
`kind`: `ai` (no real data), `ai_filtered` (AI output steered or filtered with real people), `hybrid` (includes real respondents), `real` (built from real people). Order = display order (top to bottom). `lo`/`hi` optional (seed range or CI). `note` says anything that makes a row not directly comparable (different item set, crowd size, number of real people used).

## `fairness_gss.json` (scene "Who is missed")
```json
{"overall": 0.608,
 "groups": [{"dimension": "Household income", "label": "Under $30k", "n": 64, "coverage": 0.47, "lo": 0.36, "hi": 0.58}],
 "caption": "One plain sentence; say clearly if intervals overlap."}
```
Only groups with n ≥ 30. These are factual coverage rates by group from data, not model-written claims.

## `realrecruit_gss.json` (scene "Real people", optional)
Only written once real Flashpoint respondents exist. If absent, the scene shows a "recruiting" message (no sample data is ever shown here).
```json
{"n": 50, "provider": "Prolific", "cost_usd": 188.12,
 "coverage_before": 0.608, "coverage_after": 0.85, "method": "statistics fitted on recruits, reweighted",
 "caption": "One plain sentence, including how this compares with 50 random pool people."}
```

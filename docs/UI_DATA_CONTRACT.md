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
Crowd keys used by the UI (any subset may be present): `retest, other_real_people, gaussian_copula, independent_marginals, claude_mode, claude_sampled, gpt41mini_twin`.

## `map_{dataset}_{crowd}.json`
```json
{"dataset": "gss", "crowd": "claude_mode",
 "real": [[x, y, covered01], ...],
 "synthetic": [[x, y], ...]}
```
`covered01` is 1 if that real person is covered by this crowd (k = 5 rule), else 0. Real points are in the same order in every file of a dataset.

## `agent_gss.json`
```json
{"method": "blindspot_verified", "seed": 42,
 "rounds": [
   {"round": 0, "asked": 0, "coverage": 0.638, "map": "map_gss_agent_r0.json",
    "gaps": []},
   {"round": 1, "asked": 25, "coverage": 0.66, "map": "map_gss_agent_r1.json",
    "gaps": [{"id": "g1", "name": "Distrustful economic pessimists",
              "summary": "One short sentence about the answer pattern.",
              "size": 6, "status": "learned",
              "recipes_tried": 3, "recipes_accepted": 1,
              "recruited": 0, "observed_rate": null}]}
 ]}
```
`status` is one of `learned` (a recipe was accepted), `rejected` (recipes tried, none accepted yet), `cannot_imagine` (escalated; `recruited` = real people asked for this gap, `observed_rate` = the uncovered rate in the segment that triggered it, from data). **No demographic labels or model-written demographic reasons in this file.**

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

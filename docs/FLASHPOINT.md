# Flashpoint synthetic panel: capabilities and audit notes (H5)

Facts from probing the MCP server on 2026-11-13. Nothing paid was called (no recruit, quote, checkout, payment_confirmation, calibration). Results: `results/experiment_fp.md`.

## What works
- Synthetic panels cost no money by Flashpoint's description; 200 personas per panel is our account's cap (stated by the `chat` agent, not tested above 200).
- A 20-item survey with verbatim wording and fixed options builds through `tool_survey`. A panel runs through it in about 2.5 minutes per 200 personas. Inference model reported: `gpt-5.4-mini`; run metadata says `calibration_version = uncalibrated`.
- `panel(action="create")` ignores `audience.quota_plan`: only gender was sampled, age, education and income were invented.
- The `chat` orchestrator accepts a `quota_plan` (gender x age band cells, plus an education marginal) and reported exact matches to the plan. Education vocabulary is coarser than GSS: no "less than high school", no PhD tier (we collapsed less-than-HS into High School, associate into Some College, graduate into Master's). Education is a marginal only, independent of the gender x age skeleton. Race, income and geography are invented.
- Labelled CSV export of the responses works through `tool_survey` (presigned URL, expires in 1 hour). The survey agent's text misreported "10 rows"; the file had 200.

## Limits found
- Exports carry no persona attributes and no persona-to-response key (`Panel Respondent ID` is blank), so per-respondent demographics cannot be joined to answers.
- The survey agent once returned a presigned URL with its query string cut off; ask for it again in a code block.
- Answers are drawn from per-item distributions the model emits (`sampled_answers` in the run result), then sampled.

## Calibration (answers from the `chat` agent, information only, unverified)
- What it does: demographic re-weighting of panel composition. It builds a new sibling panel whose age and gender marginals match a real survey's Results file. It is not fine-tuning, not a change to persona prompts, and not a post-hoc reweighting of answers.
- Input: a survey Results artifact already on the team (`results_artifact_id`), read by the tool `calibrate_panel_from_results(panel_id, results_artifact_id)`. Whether a raw CSV of our own rows can serve as that artifact: unknown. Column schema and minimum N: unknown.
- Cost: unknown (described as free LLM compute; no price found). It shows an Approve / Cancel prompt before running.
- Scope: per panel x per survey. Storage and isolation of uploaded data: unknown; ask info@flashpoint.ai.
- Consequence for us: calibration can only fix demographic composition, which we already control with the quota plan. It does not change how answers combine, so it is unlikely to move coverage. We did not run it.

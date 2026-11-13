# Experiment FP: Flashpoint synthetic panel vs our baselines (GSS test split)

Target = GSS test split (n = 334), 20 live items, family-weighted Gower, k = 5. Flashpoint: two separate n = 200 panels, model gpt-5.4-mini (inference_backend openai, reported by Flashpoint), calibration `uncalibrated`, run 2026-11-13T17:51:57Z to 2026-11-13T17:54:35Z. Other crowds: mean over 20 seeded draws of n = 200 (range = 2.5-97.5 percentile across seeds, not a bootstrap CI). Commit `793f1e4f78e800d36dd342cfd2eb940bb71c6940`.

## Main table (n = 200 crowds)

| crowd | coverage | precision | rcg_ratio | items >80% on one option | mean per-item TV vs real |
| --- | --- | --- | --- | --- | --- |
| **flashpoint_q1** | 0.611 [0.554, 0.671] | 0.765 [0.695, 0.815] | 0.959 | 0.00 | 0.143 |
| **flashpoint_q2** | 0.617 [0.569, 0.677] | 0.740 [0.680, 0.795] | 0.967 | 0.00 | 0.147 |
| other_real_people | 0.915 (sd 0.014, [0.895, 0.940]) | 0.934 (sd 0.014, [0.912, 0.958]) | 0.835 | 0.00 | 0.070 |
| gaussian_copula | 0.791 (sd 0.030, [0.752, 0.853]) | 0.883 (sd 0.022, [0.839, 0.908]) | 0.897 | 0.00 | 0.077 |
| independent_marginals | 0.631 (sd 0.032, [0.576, 0.686]) | 0.779 (sd 0.032, [0.722, 0.820]) | 0.955 | 0.00 | 0.078 |
| claude_mode | 0.567 (sd 0.012, [0.549, 0.585]) | 0.997 (sd 0.003, [0.995, 1.000]) | 0.968 | 0.28 | 0.404 |

Real target: 0.00 of items have >80% on one option.

## k-sensitivity (coverage / precision)

| crowd | k=3 | k=5 | k=10 |
| --- | --- | --- | --- |
| **flashpoint_q1** | 0.419 / 0.590 | 0.611 / 0.765 | 0.844 / 0.910 |
| **flashpoint_q2** | 0.398 / 0.620 | 0.617 / 0.740 | 0.862 / 0.925 |
| claude_mode | 0.437 / 0.995 | 0.567 / 0.997 | 0.730 / 0.997 |
| gaussian_copula | 0.585 / 0.744 | 0.791 / 0.883 | 0.963 / 0.975 |
| other_real_people | 0.746 / 0.842 | 0.915 / 0.934 | 0.990 / 0.983 |
| independent_marginals | 0.425 / 0.616 | 0.631 / 0.779 | 0.884 / 0.933 |

## Per-family mean normalized entropy

| family | real target | flashpoint_q1 | flashpoint_q2 | claude_mode | gaussian_copula | other_real_people | independent_marginals |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trust_institutions | 0.91 | 0.92 | 0.91 | 0.48 | 0.90 | 0.90 | 0.90 |
| money_economy | 0.88 | 0.87 | 0.87 | 0.57 | 0.89 | 0.89 | 0.90 |
| work_life | 0.87 | 0.86 | 0.87 | 0.32 | 0.87 | 0.87 | 0.87 |
| tech_science | 0.84 | 0.88 | 0.90 | 0.43 | 0.82 | 0.82 | 0.82 |

Run-to-run: coverage 0.611 vs 0.617, precision 0.765 vs 0.740; mean per-item TV between the two panels' answer distributions 0.083.

## Answer mapping (labels -> option indices)

- q1: 4000 answers, 0 missing, 0 unmapped (0.00% invalid)
- q2: 4000 answers, 0 missing, 0 unmapped (0.00% invalid)

## Flashpoint panel spec

- Sampled to plan: gender x age band (8 cells, exact), education marginal (4 levels, exact)
- Education deviation: Flashpoint has no 'less than high school' or PhD level: GSS less-than-HS + HS collapsed to High School (116), associate -> Some College (16), graduate -> Master's (27)
- Invented by the generator: race/ethnicity, income, geography, all other traits
- Joint: gender x age is a joint skeleton; education is a marginal only (independent of the skeleton)
- Reached via: chat orchestrator only; panel(action=create) ignored audience.quota_plan (only gender sampled)

claude_mode: H3 answers generated with all 24 items in one prompt; the 4 sensitive items are dropped afterwards. Subsampled to 200 from the 334 test personas. 71 answers on the live items stay NaN as in H3 (unparseable after retry).

# Experiment 1b: does a bigger model reduce the coverage gap? (test split only)

Models: haiku = `claude-haiku-5-5`, sonnet = `claude-sonnet-5-5`, opus = `claude-opus-5-5`. Same H3 generator and prompts (p1, demographics only, API-default sampling), standard API with bounded concurrency. **Bigger models (Sonnet 5.5, Opus 5.5) were tested on GSS only; Twin-2K-500 was not run.** Target = test split, holdout never used. k = 5, 200 bootstrap resamples; multi-seed rows show the mean (seed range) with the first seed's CI. Commit `f8f553a2eadd048b22c6ca6368a2986d95114612`. Runtime 718.0 s.

Cost this run: haiku $0.00 (0 calls; all responses used $0.17); sonnet $2.37 (334 calls; all responses used $2.37); opus $4.76 (334 calls; all responses used $4.76).

## gss, 24 items (target n = 334, pool n = 558)

| crowd | n | coverage [95% CI, first seed]; seed range | precision [95% CI] | near-dup vs pool | k=3 cov/prec | k=5 | k=10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| haiku: claude_mode | 334 | 0.638 [0.581, 0.698] | 0.997 [0.988, 1.000] | 0.000 | 0.548 / 0.997 | 0.638 / 0.997 | 0.787 / 0.997 |
| haiku: claude_sampled | 334 | 0.371 [0.326, 0.413] | 0.521 [0.461, 0.575] | 0.000 | 0.272 / 0.335 | 0.371 / 0.521 | 0.623 / 0.731 |
| sonnet: claude_mode | 334 | 0.539 [0.485, 0.596] | 1.000 [1.000, 1.000] | 0.000 | 0.425 / 1.000 | 0.539 / 1.000 | 0.716 / 1.000 |
| sonnet: claude_sampled | 334 | 0.587 [0.536, 0.638] | 0.620 [0.578, 0.668] | 0.000 | 0.419 / 0.482 | 0.587 / 0.620 | 0.835 / 0.826 |
| opus: claude_mode | 334 | 0.710 [0.659, 0.758] | 1.000 [1.000, 1.000] | 0.000 | 0.608 / 1.000 | 0.710 / 1.000 | 0.850 / 1.000 |
| opus: claude_sampled | 334 | 0.746 [0.692, 0.788] | 0.704 [0.656, 0.758] | 0.003 | 0.548 / 0.551 | 0.746 / 0.704 | 0.934 / 0.877 |
| ref: shuffled_answers | 334 | 0.740 [0.701, 0.788] | 0.778 [0.722, 0.826] | 0.003 | 0.581 / 0.632 | 0.740 / 0.778 | 0.916 / 0.919 |
| ref: copula_on_pool | 334 | 0.880 [0.841, 0.916] | 0.880 [0.847, 0.910] | 0.000 | 0.710 / 0.763 | 0.880 / 0.880 | 0.982 / 0.976 |
| ref: other_real_people | 334 | 0.961 [0.940, 0.979] | 0.895 [0.856, 0.928] | 1.000 | 0.856 / 0.808 | 0.961 / 0.895 | 0.997 / 0.961 |
| ref: copula_on_25_asked (mean of 10) | 334 | 0.778 [0.738, 0.818]; seeds 0.734-0.859 | 0.867 [0.839, 0.904] | 0.001 | 0.614 / 0.754 | 0.778 / 0.867 | 0.937 / 0.961 |

Per-family normalized entropy and bunching:

| family | real | haiku:claude_mode | haiku:claude_sampled | sonnet:claude_mode | sonnet:claude_sampled | opus:claude_mode | opus:claude_sampled |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trust_institutions | 0.91 | 0.43 | 0.92 | 0.33 | 0.92 | 0.34 | 0.92 |
| money_economy | 0.84 | 0.56 | 0.91 | 0.47 | 0.85 | 0.50 | 0.85 |
| work_life | 0.87 | 0.32 | 0.89 | 0.24 | 0.89 | 0.28 | 0.89 |
| tech_science | 0.84 | 0.42 | 0.93 | 0.19 | 0.90 | 0.32 | 0.88 |

Items with >80% of people on one option: haiku:claude_mode 0.29, haiku:claude_sampled 0.00, sonnet:claude_mode 0.50, sonnet:claude_sampled 0.00, opus:claude_mode 0.46, opus:claude_sampled 0.00

Paired note (does the crowd beat the references?):

- haiku claude_mode: 0.638; vs shuffled answers -0.102 (not above), vs copula on 25 asked -0.141 (not above)
- haiku claude_sampled: 0.371; vs shuffled answers -0.368 (not above), vs copula on 25 asked -0.407 (not above)
- sonnet claude_mode: 0.539; vs shuffled answers -0.201 (not above), vs copula on 25 asked -0.240 (not above)
- sonnet claude_sampled: 0.587; vs shuffled answers -0.153 (not above), vs copula on 25 asked -0.192 (not above)
- opus claude_mode: 0.710; vs shuffled answers -0.030 (not above), vs copula on 25 asked -0.069 (not above)
- opus claude_sampled: 0.746; vs shuffled answers +0.006 (not above), vs copula on 25 asked -0.033 (not above)

People with a missing item after retry: {'haiku': 44, 'sonnet': 0, 'opus': 0}

## gss, 20 live items (target n = 334, pool n = 558)

| crowd | n | coverage [95% CI, first seed]; seed range | precision [95% CI] | near-dup vs pool | k=3 cov/prec | k=5 | k=10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| haiku: claude_mode | 334 | 0.608 [0.554, 0.662] | 0.997 [0.988, 1.000] | 0.000 | 0.488 / 0.994 | 0.608 / 0.997 | 0.766 / 0.997 |
| haiku: claude_sampled | 334 | 0.473 [0.428, 0.527] | 0.548 [0.497, 0.602] | 0.000 | 0.335 / 0.365 | 0.473 / 0.548 | 0.716 / 0.766 |
| haiku: select_from_50_lambda_0.0 | 334 | 0.747 [0.730, 0.823]; seeds 0.710-0.775 | 0.928 [0.895, 0.949] | 0.002 | 0.599 / 0.797 | 0.747 / 0.928 | 0.897 / 0.990 |
| haiku: select_from_50_lambda_0.5 | 334 | 0.789 [0.772, 0.853]; seeds 0.763-0.823 | 0.855 [0.826, 0.892] | 0.002 | 0.619 / 0.688 | 0.789 / 0.855 | 0.947 / 0.969 |
| sonnet: claude_mode | 334 | 0.503 [0.443, 0.566] | 1.000 [1.000, 1.000] | 0.000 | 0.359 / 1.000 | 0.503 / 1.000 | 0.689 / 1.000 |
| sonnet: claude_sampled | 334 | 0.653 [0.605, 0.704] | 0.659 [0.599, 0.710] | 0.000 | 0.461 / 0.512 | 0.653 / 0.659 | 0.886 / 0.874 |
| sonnet: select_from_50_lambda_0.0 | 334 | 0.857 [0.850, 0.919]; seeds 0.841-0.886 | 0.968 [0.961, 0.991] | 0.005 | 0.700 / 0.862 | 0.857 / 0.968 | 0.960 / 0.997 |
| sonnet: select_from_50_lambda_0.5 | 334 | 0.896 [0.856, 0.928]; seeds 0.877-0.907 | 0.906 [0.877, 0.934] | 0.005 | 0.734 / 0.768 | 0.896 / 0.906 | 0.989 / 0.989 |
| opus: claude_mode | 334 | 0.701 [0.662, 0.754] | 1.000 [1.000, 1.000] | 0.000 | 0.533 / 1.000 | 0.701 / 1.000 | 0.820 / 1.000 |
| opus: claude_sampled | 334 | 0.778 [0.728, 0.820] | 0.772 [0.725, 0.820] | 0.003 | 0.563 / 0.614 | 0.778 / 0.772 | 0.964 / 0.925 |
| opus: select_from_50_lambda_0.0 | 334 | 0.894 [0.883, 0.940]; seeds 0.862-0.913 | 0.986 [0.970, 0.997] | 0.004 | 0.766 / 0.938 | 0.894 / 0.986 | 0.968 / 1.000 |
| opus: select_from_50_lambda_0.5 | 334 | 0.935 [0.925, 0.970]; seeds 0.919-0.949 | 0.949 [0.937, 0.979] | 0.007 | 0.787 / 0.854 | 0.935 / 0.949 | 0.995 / 0.995 |
| ref: shuffled_answers | 334 | 0.823 [0.787, 0.862] | 0.799 [0.757, 0.841] | 0.000 | 0.656 / 0.659 | 0.823 / 0.799 | 0.961 / 0.934 |
| ref: copula_on_pool | 334 | 0.922 [0.892, 0.946] | 0.886 [0.856, 0.916] | 0.000 | 0.754 / 0.740 | 0.922 / 0.886 | 0.991 / 0.970 |
| ref: other_real_people | 334 | 0.967 [0.949, 0.982] | 0.910 [0.877, 0.937] | 1.000 | 0.847 / 0.805 | 0.967 / 0.910 | 0.988 / 0.979 |
| ref: copula_on_25_asked (mean of 10) | 334 | 0.815 [0.759, 0.832]; seeds 0.743-0.889 | 0.873 [0.837, 0.906] | 0.000 | 0.629 / 0.747 | 0.815 / 0.873 | 0.952 / 0.968 |

Per-family normalized entropy and bunching:

| family | real | haiku:claude_mode | haiku:claude_sampled | sonnet:claude_mode | sonnet:claude_sampled | opus:claude_mode | opus:claude_sampled |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trust_institutions | 0.91 | 0.48 | 0.94 | 0.25 | 0.95 | 0.28 | 0.94 |
| money_economy | 0.88 | 0.57 | 0.92 | 0.51 | 0.86 | 0.52 | 0.87 |
| work_life | 0.87 | 0.32 | 0.89 | 0.24 | 0.89 | 0.28 | 0.89 |
| tech_science | 0.84 | 0.42 | 0.93 | 0.19 | 0.90 | 0.32 | 0.88 |

Items with >80% of people on one option: haiku:claude_mode 0.30, haiku:claude_sampled 0.00, sonnet:claude_mode 0.55, sonnet:claude_sampled 0.00, opus:claude_mode 0.45, opus:claude_sampled 0.00

Paired note (does the crowd beat the references?):

- haiku claude_mode: 0.608; vs shuffled answers -0.216 (not above), vs copula on 25 asked -0.207 (not above)
- haiku claude_sampled: 0.473; vs shuffled answers -0.350 (not above), vs copula on 25 asked -0.342 (not above)
- haiku select_from_50_lambda_0.0: 0.747; vs shuffled answers -0.077 (not above), vs copula on 25 asked -0.068 (not above)
- haiku select_from_50_lambda_0.5: 0.789; vs shuffled answers -0.035 (not above), vs copula on 25 asked -0.026 (not above)
- sonnet claude_mode: 0.503; vs shuffled answers -0.320 (not above), vs copula on 25 asked -0.312 (not above)
- sonnet claude_sampled: 0.653; vs shuffled answers -0.171 (not above), vs copula on 25 asked -0.162 (not above)
- sonnet select_from_50_lambda_0.0: 0.857; vs shuffled answers +0.034 (CI above), vs copula on 25 asked +0.042 (CI above)
- sonnet select_from_50_lambda_0.5: 0.896; vs shuffled answers +0.072 (CI above), vs copula on 25 asked +0.081 (CI above)
- opus claude_mode: 0.701; vs shuffled answers -0.123 (not above), vs copula on 25 asked -0.114 (not above)
- opus claude_sampled: 0.778; vs shuffled answers -0.045 (not above), vs copula on 25 asked -0.036 (not above)
- opus select_from_50_lambda_0.0: 0.894; vs shuffled answers +0.071 (CI above), vs copula on 25 asked +0.079 (CI above)
- opus select_from_50_lambda_0.5: 0.935; vs shuffled answers +0.112 (CI above), vs copula on 25 asked +0.121 (CI above)

People with a missing item after retry: {'haiku': 44, 'sonnet': 0, 'opus': 0}


# Experiment 1: plain Claude persona crowds

Model `claude-haiku-5-5`, prompt p1, empty recipe, thinking off, API-default sampling, Message Batches API. Target = test split; holdout untouched. H2 rows copied from `experiment0.json`; new rows in bold. k = 5, 200 bootstrap resamples. Commit `62e68bbbe0d98b1cfdc0e5d02159bfdd921c0b69`. Runtime 107.5 s.

Cost: estimate $0.00 for 0 uncached calls (limit $5.00); actual $0.00 for 0 calls, 0 input + 0 cache-write + 0 cache-read + 0 output tokens; 0 API errors.
Total cost of every response this experiment uses (all runs, incl. interrupted ones and retries): $0.23 for 1078 responses.

## Twin-2K-500 (target n = 617, pool n = 1029)

| crowd | n | coverage [95% CI] | precision [95% CI] | rcg_ratio |
| --- | --- | --- | --- | --- |
| retest | 617 | 0.984 [0.974, 0.994] | 0.958 [0.940, 0.974] | 0.769 |
| other_real_people | 617 | 0.981 [0.968, 0.990] | 0.908 [0.885, 0.930] | 0.817 |
| gaussian_copula | 617 | 0.888 [0.864, 0.909] | 0.862 [0.831, 0.885] | 0.868 |
| independent_marginals | 617 | 0.512 [0.470, 0.549] | 0.577 [0.535, 0.616] | 1.000 |
| **claude_mode** | 617 | 0.345 [0.308, 0.383] | 0.963 [0.948, 0.977] | 1.103 |
| gpt41mini_twin | 617 | 0.344 [0.301, 0.383] | 0.956 [0.937, 0.972] | 1.104 |
| **claude_sampled** | 617 | 0.170 [0.141, 0.198] | 0.269 [0.237, 0.305] | 1.198 |

k-sensitivity (coverage / precision):

| crowd | k=3 | k=5 | k=10 |
| --- | --- | --- | --- |
| retest | 0.922 / 0.885 | 0.984 / 0.958 | 1.000 / 0.992 |
| other_real_people | 0.888 / 0.830 | 0.981 / 0.908 | 0.998 / 0.969 |
| gaussian_copula | 0.742 / 0.733 | 0.888 / 0.862 | 0.985 / 0.961 |
| independent_marginals | 0.335 / 0.394 | 0.512 / 0.577 | 0.742 / 0.809 |
| claude_mode | 0.253 / 0.911 | 0.345 / 0.963 | 0.494 / 0.990 |
| gpt41mini_twin | 0.232 / 0.861 | 0.344 / 0.956 | 0.475 / 0.995 |
| claude_sampled | 0.084 / 0.141 | 0.170 / 0.269 | 0.295 / 0.509 |

Per-family mean normalized entropy:

| family | real target | retest | other_real_people | gaussian_copula | independent_marginals | claude_mode | gpt41mini_twin | claude_sampled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| policy_self | 0.88 | 0.87 | 0.89 | 0.89 | 0.89 | 0.71 | 0.81 | 0.95 |
| perception | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.37 | 0.78 | 0.93 |
| risk_benefit | 0.81 | 0.80 | 0.80 | 0.81 | 0.80 | 0.55 | 0.33 | 0.94 |
| decisions | 0.96 | 0.96 | 0.95 | 0.95 | 0.96 | 0.49 | 0.43 | 0.96 |

Share of items with more than 80% of people on one option: real_target 0.00, claude_mode 0.20, claude_sampled 0.00, gpt41mini_twin 0.13
Parsing: 0 synthetic people with a missing item after retry; 0 missing answers (0.00%).

## GSS 2024 (main cross-section, ballot 1) (target n = 334, pool n = 558)

| crowd | n | coverage [95% CI] | precision [95% CI] | rcg_ratio |
| --- | --- | --- | --- | --- |
| other_real_people | 334 | 0.961 [0.940, 0.979] | 0.895 [0.856, 0.928] | 0.824 |
| gaussian_copula | 334 | 0.880 [0.841, 0.916] | 0.880 [0.847, 0.910] | 0.871 |
| independent_marginals | 334 | 0.740 [0.701, 0.788] | 0.778 [0.722, 0.826] | 0.920 |
| **claude_mode** | 334 | 0.638 [0.581, 0.698] | 0.997 [0.988, 1.000] | 0.925 |
| **claude_sampled** | 334 | 0.371 [0.326, 0.413] | 0.521 [0.461, 0.575] | 1.051 |

k-sensitivity (coverage / precision):

| crowd | k=3 | k=5 | k=10 |
| --- | --- | --- | --- |
| other_real_people | 0.856 / 0.808 | 0.961 / 0.895 | 0.997 / 0.961 |
| gaussian_copula | 0.710 / 0.763 | 0.880 / 0.880 | 0.982 / 0.976 |
| independent_marginals | 0.581 / 0.632 | 0.740 / 0.778 | 0.916 / 0.919 |
| claude_mode | 0.548 / 0.997 | 0.638 / 0.997 | 0.787 / 0.997 |
| claude_sampled | 0.272 / 0.335 | 0.371 / 0.521 | 0.623 / 0.731 |

Per-family mean normalized entropy:

| family | real target | other_real_people | gaussian_copula | independent_marginals | claude_mode | claude_sampled |
| --- | --- | --- | --- | --- | --- | --- |
| trust_institutions | 0.91 | 0.91 | 0.90 | 0.91 | 0.43 | 0.92 |
| money_economy | 0.84 | 0.85 | 0.84 | 0.86 | 0.56 | 0.91 |
| work_life | 0.87 | 0.88 | 0.87 | 0.87 | 0.32 | 0.89 |
| tech_science | 0.84 | 0.83 | 0.82 | 0.82 | 0.42 | 0.93 |

Share of items with more than 80% of people on one option: real_target 0.00, claude_mode 0.29, claude_sampled 0.00
Parsing: 44 synthetic people with a missing item after retry; 87 missing answers (1.09%).

## Robustness: Twin with unweighted Gower (every item equal)

| crowd | coverage, family-weighted | coverage, unweighted |
| --- | --- | --- |
| retest | 0.984 | 0.989 |
| other_real_people | 0.981 | 0.977 |
| gaussian_copula | 0.888 | 0.854 |
| independent_marginals | 0.512 | 0.382 |
| claude_mode | 0.345 | 0.399 |
| gpt41mini_twin | 0.344 | 0.345 |
| claude_sampled | 0.170 | 0.109 |

Ranking, family-weighted: retest > other_real_people > gaussian_copula > independent_marginals > claude_mode > gpt41mini_twin > claude_sampled
Ranking, unweighted: retest > other_real_people > gaussian_copula > claude_mode > independent_marginals > gpt41mini_twin > claude_sampled
Ranking changed: **yes**

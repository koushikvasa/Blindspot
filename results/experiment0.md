# Experiment 0: does a synthetic crowd leave real people uncovered?

Target = test split; baselines fit on pool; holdout untouched. k = 5, 200 bootstrap resamples (coverage: real people; precision: synthetic people; radii fixed), seed 42, family-weighted Gower distance, inside = distance <= k-NN radius. Commit `e92a4650e8444f146e5ac896a7e4ba9773a07a56`. Runtime 85.2 s.

## Twin-2K-500 (target n = 617, pool n = 1029)

| crowd | n | coverage [95% CI] | precision [95% CI] | rcg_ratio |
| --- | --- | --- | --- | --- |
| retest | 617 | 0.984 [0.974, 0.994] | 0.958 [0.940, 0.974] | 0.769 |
| other_real_people | 617 | 0.981 [0.968, 0.990] | 0.908 [0.885, 0.930] | 0.817 |
| gaussian_copula | 617 | 0.888 [0.864, 0.909] | 0.862 [0.831, 0.885] | 0.868 |
| independent_marginals | 617 | 0.512 [0.470, 0.549] | 0.577 [0.535, 0.616] | 1.000 |
| gpt41mini_twin | 617 | 0.344 [0.301, 0.383] | 0.956 [0.937, 0.972] | 1.104 |

k-sensitivity (coverage / precision):

| crowd | k=3 | k=5 | k=10 |
| --- | --- | --- | --- |
| retest | 0.922 / 0.885 | 0.984 / 0.958 | 1.000 / 0.992 |
| other_real_people | 0.888 / 0.830 | 0.981 / 0.908 | 0.998 / 0.969 |
| gaussian_copula | 0.742 / 0.733 | 0.888 / 0.862 | 0.985 / 0.961 |
| independent_marginals | 0.335 / 0.394 | 0.512 / 0.577 | 0.742 / 0.809 |
| gpt41mini_twin | 0.232 / 0.861 | 0.344 / 0.956 | 0.475 / 0.995 |

Per-family mean normalized entropy (0 = everyone gives the same answer, 1 = uniform):

| family | real target | retest | other_real_people | gaussian_copula | independent_marginals | gpt41mini_twin |
| --- | --- | --- | --- | --- | --- | --- |
| policy_self | 0.88 | 0.87 | 0.89 | 0.89 | 0.89 | 0.81 |
| perception | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.78 |
| risk_benefit | 0.81 | 0.80 | 0.80 | 0.81 | 0.80 | 0.33 |
| decisions | 0.96 | 0.96 | 0.95 | 0.95 | 0.96 | 0.43 |

## GSS 2024 (main cross-section, ballot 1) (target n = 334, pool n = 558)

| crowd | n | coverage [95% CI] | precision [95% CI] | rcg_ratio |
| --- | --- | --- | --- | --- |
| other_real_people | 334 | 0.961 [0.940, 0.979] | 0.895 [0.856, 0.928] | 0.824 |
| gaussian_copula | 334 | 0.880 [0.841, 0.916] | 0.880 [0.847, 0.910] | 0.871 |
| independent_marginals | 334 | 0.740 [0.701, 0.788] | 0.778 [0.722, 0.826] | 0.920 |

k-sensitivity (coverage / precision):

| crowd | k=3 | k=5 | k=10 |
| --- | --- | --- | --- |
| other_real_people | 0.856 / 0.808 | 0.961 / 0.895 | 0.997 / 0.961 |
| gaussian_copula | 0.710 / 0.763 | 0.880 / 0.880 | 0.982 / 0.976 |
| independent_marginals | 0.581 / 0.632 | 0.740 / 0.778 | 0.916 / 0.919 |

Per-family mean normalized entropy (0 = everyone gives the same answer, 1 = uniform):

| family | real target | other_real_people | gaussian_copula | independent_marginals |
| --- | --- | --- | --- | --- |
| trust_institutions | 0.91 | 0.91 | 0.90 | 0.91 |
| money_economy | 0.84 | 0.85 | 0.84 | 0.86 |
| work_life | 0.87 | 0.88 | 0.87 | 0.87 |
| tech_science | 0.84 | 0.83 | 0.82 | 0.82 |

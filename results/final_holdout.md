# Final holdout run (holdout split)

Target = holdout split; every fit, filter and draw on the POOL split only. Commit `a828ae14e4d3a660f13e6c458dbfb7732a238a5e`. Model `claude-haiku-5-5`, Claude holdout personas from demographics only. k = 5, 200 bootstrap resamples, seed 42. Multi-seed rows: mean over seeds, bootstrap CI from the first seed, seed range in the coverage cell (mix 10 seeds, select 5, asked 10). Cost: estimate $0.15, actual $0.10 for 478 calls. Runtime 1188.9 s. Claude people with a missing item after retry: {'twin2k': 1, 'gss': 16}.

## Twin-2K-500 (target n = 412, pool n = 1029, 30 items)

| crowd | n | coverage [95% CI, first seed]; seed range | precision [95% CI] | near-dup vs pool | k=3 cov/prec | k=5 | k=10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| retest | 412 | 0.990 [0.981, 0.998] | 0.961 [0.944, 0.978] | 0.000 | 0.947 / 0.900 | 0.990 / 0.961 | 1.000 / 0.993 |
| other_real_people | 412 | 0.976 [0.964, 0.988] | 0.922 [0.891, 0.949] | 1.000 | 0.905 / 0.845 | 0.976 / 0.922 | 0.998 / 0.966 |
| gaussian_copula | 412 | 0.886 [0.852, 0.915] | 0.859 [0.827, 0.886] | 0.000 | 0.735 / 0.738 | 0.886 / 0.859 | 0.993 / 0.968 |
| independent_marginals | 412 | 0.600 [0.558, 0.641] | 0.636 [0.590, 0.680] | 0.000 | 0.388 / 0.481 | 0.600 / 0.636 | 0.820 / 0.859 |
| claude_mode | 412 | 0.350 [0.308, 0.391] | 0.973 [0.956, 0.988] | 0.000 | 0.262 / 0.910 | 0.350 / 0.973 | 0.532 / 1.000 |
| gpt41mini_twin | 412 | 0.354 [0.303, 0.408] | 0.981 [0.964, 0.993] | 0.000 | 0.231 / 0.954 | 0.354 / 0.981 | 0.566 / 0.993 |
| claude_sampled | 412 | 0.180 [0.141, 0.214] | 0.408 [0.362, 0.466] | 0.000 | 0.129 / 0.252 | 0.180 / 0.408 | 0.354 / 0.653 |
| mix_50_50_claude_mode_gpt41mini | 412 | 0.438 [0.404, 0.483]; seeds 0.417-0.464 | 0.976 [0.964, 0.992] | 0.000 | 0.307 / 0.928 | 0.438 / 0.976 | 0.652 / 0.997 |

Per-family mean normalized entropy:

| family | real target | retest | other_real_people | gaussian_copula | independent_marginals | claude_mode | gpt41mini_twin | claude_sampled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| policy_self | 0.87 | 0.87 | 0.89 | 0.89 | 0.89 | 0.72 | 0.79 | 0.95 |
| perception | 0.90 | 0.90 | 0.91 | 0.91 | 0.89 | 0.36 | 0.77 | 0.93 |
| risk_benefit | 0.80 | 0.80 | 0.81 | 0.81 | 0.81 | 0.53 | 0.33 | 0.94 |
| decisions | 0.96 | 0.96 | 0.96 | 0.95 | 0.94 | 0.46 | 0.46 | 0.96 |

Share of items with more than 80% of people on one option: retest 0.00, other_real_people 0.00, gaussian_copula 0.00, independent_marginals 0.00, claude_mode 0.23, gpt41mini_twin 0.13, claude_sampled 0.00

## GSS 2024, 20 live items (target n = 224, pool n = 558, 20 items)

| crowd | n | coverage [95% CI, first seed]; seed range | precision [95% CI] | near-dup vs pool | k=3 cov/prec | k=5 | k=10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| claude_mode | 224 | 0.634 [0.571, 0.696] | 1.000 [1.000, 1.000] | 0.000 | 0.500 / 1.000 | 0.634 / 1.000 | 0.795 / 1.000 |
| claude_sampled | 224 | 0.487 [0.424, 0.554] | 0.647 [0.589, 0.705] | 0.000 | 0.326 / 0.446 | 0.487 / 0.647 | 0.737 / 0.826 |
| independent_marginals | 224 | 0.853 [0.808, 0.893] | 0.817 [0.768, 0.857] | 0.000 | 0.634 / 0.683 | 0.853 / 0.817 | 0.987 / 0.973 |
| gaussian_copula | 224 | 0.915 [0.888, 0.946] | 0.933 [0.897, 0.969] | 0.000 | 0.772 / 0.821 | 0.915 / 0.933 | 0.991 / 0.991 |
| other_real_people | 224 | 0.982 [0.969, 1.000] | 0.938 [0.911, 0.964] | 1.000 | 0.893 / 0.853 | 0.982 / 0.938 | 1.000 / 0.982 |
| select_from_50_lambda_0.5_pool_filter | 224 | 0.818 [0.754, 0.844]; seeds 0.781-0.871 | 0.874 [0.848, 0.924] | 0.003 | 0.639 / 0.727 | 0.818 / 0.874 | 0.948 / 0.963 |
| copula_on_asked_B25 | 224 | 0.849 [0.757, 0.846]; seeds 0.790-0.906 | 0.900 [0.868, 0.942] | 0.000 | 0.690 / 0.764 | 0.849 / 0.900 | 0.976 / 0.974 |
| copula_on_asked_B50 | 224 | 0.893 [0.888, 0.953]; seeds 0.857-0.920 | 0.902 [0.875, 0.938] | 0.000 | 0.747 / 0.774 | 0.893 / 0.902 | 0.991 / 0.980 |
| flashpoint_q1_n200 | 200 | 0.759 [0.705, 0.808] | 0.840 [0.780, 0.890] | 0.000 | 0.562 / 0.670 | 0.759 / 0.840 | 0.906 / 0.975 |
| flashpoint_q2_n200 | 200 | 0.674 [0.625, 0.737] | 0.845 [0.795, 0.890] | 0.000 | 0.518 / 0.700 | 0.674 / 0.845 | 0.879 / 0.965 |

Per-family mean normalized entropy:

| family | real target | claude_mode | claude_sampled | independent_marginals | gaussian_copula | other_real_people | flashpoint_q1_n200 | flashpoint_q2_n200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trust_institutions | 0.91 | 0.51 | 0.92 | 0.90 | 0.90 | 0.91 | 0.92 | 0.91 |
| money_economy | 0.90 | 0.59 | 0.92 | 0.90 | 0.90 | 0.88 | 0.87 | 0.87 |
| work_life | 0.89 | 0.31 | 0.91 | 0.88 | 0.87 | 0.87 | 0.86 | 0.87 |
| tech_science | 0.84 | 0.43 | 0.92 | 0.82 | 0.83 | 0.83 | 0.88 | 0.90 |

Share of items with more than 80% of people on one option: claude_mode 0.35, claude_sampled 0.00, independent_marginals 0.00, gaussian_copula 0.00, other_real_people 0.00, flashpoint_q1_n200 0.00, flashpoint_q2_n200 0.00


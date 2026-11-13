# H4d-big: select from Sonnet / Opus candidates with B asked people as the realism filter

Zero API calls (cached responses from experiment 1b), GSS 20 live items, test n = 334, 10 seeds, same asked people and settings as H4d (k = min(5, B//5), lambda by half-split from [0.0, 0.25, 0.5, 1.0]). Cells: mean (min-max over seeds). Haiku rows are H4d's saved numbers. Near-dup = synthetic rows within 0.02 Gower of an asked person. Commit `b5303e581b37e9d3e22bd5479bbbb2c848c76e2e`. Runtime 428.0 s.

## Coverage

| method | B=25 | B=50 | B=75 | B=100 |
| --- | --- | --- | --- | --- |
| select_claude, Opus 5.5 probs | 0.807 (0.763-0.877) | 0.851 (0.793-0.913) | 0.860 (0.820-0.895) | 0.877 (0.826-0.901) |
| select_claude, Sonnet 5.5 probs | 0.775 (0.692-0.829) | 0.797 (0.683-0.856) | 0.825 (0.751-0.868) | 0.854 (0.793-0.913) |
| select_claude, Haiku 5.5 probs (H4d) | 0.699 (0.626-0.766) | 0.698 (0.632-0.740) | 0.750 (0.713-0.793) | 0.765 (0.695-0.808) |
| copula_on_asked (same asked people) | 0.815 (0.743-0.889) | 0.868 (0.805-0.889) | 0.885 (0.856-0.910) | 0.893 (0.862-0.925) |

## Precision

| method | B=25 | B=50 | B=75 | B=100 |
| --- | --- | --- | --- | --- |
| select_claude, Opus 5.5 probs | 0.981 (0.955-0.991) | 0.985 (0.961-1.000) | 0.965 (0.811-0.997) | 0.962 (0.787-0.997) |
| select_claude, Sonnet 5.5 probs | 0.958 (0.928-0.979) | 0.965 (0.952-0.985) | 0.957 (0.919-0.973) | 0.954 (0.889-0.982) |
| select_claude, Haiku 5.5 probs (H4d) | 0.896 (0.844-0.928) | 0.879 (0.707-0.934) | 0.893 (0.859-0.943) | 0.835 (0.713-0.925) |
| copula_on_asked (same asked people) | 0.873 (0.841-0.904) | 0.890 (0.850-0.922) | 0.882 (0.856-0.916) | 0.886 (0.853-0.919) |

## Near-duplicate share

| method | B=25 | B=50 | B=75 | B=100 |
| --- | --- | --- | --- | --- |
| select_claude, Opus 5.5 probs | 0.000 (0.000-0.003) | 0.001 (0.000-0.009) | 0.001 (0.000-0.009) | 0.001 (0.000-0.009) |
| select_claude, Sonnet 5.5 probs | 0.000 (0.000-0.003) | 0.001 (0.000-0.003) | 0.001 (0.000-0.003) | 0.001 (0.000-0.003) |
| select_claude, Haiku 5.5 probs (H4d) | 0.000 (0.000-0.003) | 0.000 (0.000-0.003) | 0.000 (0.000-0.003) | 0.000 (0.000-0.003) |
| copula_on_asked (same asked people) | 0.000 (0.000-0.003) | 0.000 (0.000-0.003) | 0.000 (0.000-0.003) | 0.000 (0.000-0.000) |

## Paired vs copula_on_asked (coverage, same seeds and asked people)

| budget | model | mean diff (select - copula) | sd | select wins |
| --- | --- | --- | --- | --- |
| 25 | Opus | -0.008 | 0.048 | 4/10 |
| 25 | Sonnet | -0.039 | 0.051 | 2/10 |
| 25 | Haiku (H4d) | -0.116 | 0.073 | 0/10 |
| 50 | Opus | -0.016 | 0.034 | 2/10 |
| 50 | Sonnet | -0.071 | 0.039 | 0/10 |
| 50 | Haiku (H4d) | -0.169 | 0.030 | 0/10 |
| 75 | Opus | -0.025 | 0.026 | 1/10 |
| 75 | Sonnet | -0.060 | 0.040 | 0/10 |
| 75 | Haiku (H4d) | -0.135 | 0.028 | 0/10 |
| 100 | Opus | -0.016 | 0.024 | 2/10 |
| 100 | Sonnet | -0.039 | 0.043 | 3/10 |
| 100 | Haiku (H4d) | -0.128 | 0.042 | 0/10 |

## Chosen lambdas (count of seeds)

| model | B | λ=0.0 | λ=0.25 | λ=0.5 | λ=1.0 |
| --- | --- | --- | --- | --- | --- |
| Opus | 25 | 10 | 0 | 0 | 0 |
| Opus | 50 | 8 | 1 | 1 | 0 |
| Opus | 75 | 7 | 1 | 1 | 1 |
| Opus | 100 | 5 | 3 | 1 | 1 |
| Sonnet | 25 | 9 | 1 | 0 | 0 |
| Sonnet | 50 | 10 | 0 | 0 | 0 |
| Sonnet | 75 | 7 | 2 | 1 | 0 |
| Sonnet | 100 | 5 | 2 | 3 | 0 |
| Haiku | 25 | 9 | 0 | 1 | 0 |
| Haiku | 50 | 6 | 2 | 1 | 1 |
| Haiku | 75 | 2 | 8 | 0 | 0 |
| Haiku | 100 | 1 | 2 | 4 | 3 |

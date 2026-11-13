# Screener targeting simulation (zero cost, GSS test split for scoring)

20 live items, 10 seeds. Missed = pool person not covered by claude_mode (pool 5-NN radius; 48.9% of the pool). Per seed the pool is split in random halves: one half trains the screener (greedy forward selection, CV AUC), the other is the recruiting pool. Threshold accepts ~40%. Coverage and precision on the test split (n = 334), `claude_plus_recruits` INCLUDES REAL RESPONDENTS. Coverage and precision cells: mean over seeds (2.5-97.5 percentile across seeds); the bootstrap 95% CI belongs to the first seed's point estimate. Commit `32b2e4b1e6b4a4cc09c703a7dddda0c2f918d948`. Runtime 120.4 s.

## Screener quality (evaluated on the other half of the pool)

| screener | AUC | accept rate | base rate missed | precision | recall | items chosen (seeds) |
| --- | --- | --- | --- | --- | --- | --- |
| screened_3 | 0.780 (0.737-0.817) | 0.42 | 0.49 | 0.736 (0.693-0.789) | 0.638 (0.575-0.712) | workless 10, trcourts 9, happy 7, satfin 2, trppl 1, aidrive 1 |
| screened_4 | 0.802 (0.759-0.834) | 0.41 | 0.49 | 0.761 (0.725-0.818) | 0.636 (0.563-0.708) | trcourts 10, workless 10, happy 8, aidrive 4, intcnfrm 3, trppl 2 |

## gaussian_copula fit on the recruits: test coverage / precision

| budget | recruits | coverage | precision | near-dup |
| --- | --- | --- | --- | --- |
| 25 | random | 0.800 (seeds 0.693-0.872); first seed 0.677 [0.635, 0.725] | 0.850 (seeds 0.767-0.907); first seed 0.754 [0.707, 0.808] | 0.000 |
| 25 | screened_3 | 0.638 (seeds 0.565-0.688); first seed 0.581 [0.527, 0.626] | 0.845 (seeds 0.773-0.917); first seed 0.796 [0.757, 0.835] | 0.000 |
| 25 | screened_4 | 0.639 (seeds 0.569-0.717); first seed 0.581 [0.533, 0.638] | 0.842 (seeds 0.756-0.894); first seed 0.844 [0.811, 0.880] | 0.000 |
| 50 | random | 0.855 (seeds 0.774-0.930); first seed 0.760 [0.716, 0.808] | 0.867 (seeds 0.815-0.922); first seed 0.814 [0.769, 0.851] | 0.000 |
| 50 | screened_3 | 0.675 (seeds 0.617-0.780); first seed 0.656 [0.611, 0.707] | 0.842 (seeds 0.806-0.893); first seed 0.802 [0.763, 0.847] | 0.000 |
| 50 | screened_4 | 0.685 (seeds 0.618-0.769); first seed 0.769 [0.724, 0.808] | 0.850 (seeds 0.808-0.883); first seed 0.874 [0.838, 0.910] | 0.000 |
| 75 | random | 0.875 (seeds 0.830-0.915); first seed 0.829 [0.799, 0.871] | 0.871 (seeds 0.832-0.906); first seed 0.850 [0.814, 0.883] | 0.000 |
| 75 | screened_3 | 0.702 (seeds 0.674-0.783); first seed 0.677 [0.629, 0.725] | 0.852 (seeds 0.793-0.891); first seed 0.850 [0.820, 0.883] | 0.000 |
| 75 | screened_4 | 0.696 (seeds 0.622-0.742); first seed 0.707 [0.656, 0.761] | 0.847 (seeds 0.783-0.895); first seed 0.880 [0.847, 0.916] | 0.000 |
| 100 | random | 0.871 (seeds 0.824-0.921); first seed 0.820 [0.781, 0.859] | 0.866 (seeds 0.828-0.902); first seed 0.826 [0.790, 0.868] | 0.000 |
| 100 | screened_3 | 0.712 (seeds 0.660-0.761); first seed 0.716 [0.668, 0.758] | 0.853 (seeds 0.819-0.882); first seed 0.868 [0.832, 0.904] | 0.000 |
| 100 | screened_4 | 0.704 (seeds 0.654-0.739); first seed 0.689 [0.641, 0.740] | 0.861 (seeds 0.804-0.910); first seed 0.880 [0.847, 0.910] | 0.000 |

## claude_mode + recruits swapped in: test coverage / precision

| budget | recruits | coverage | precision | near-dup |
| --- | --- | --- | --- | --- |
| 25 | random | 0.718 (seeds 0.692-0.752); first seed 0.707 [0.659, 0.752] | 0.991 (seeds 0.986-0.996); first seed 0.991 [0.979, 1.000] | 0.000 |
| 25 | screened_3 | 0.739 (seeds 0.687-0.778); first seed 0.734 [0.683, 0.776] | 0.991 (seeds 0.986-0.994); first seed 0.991 [0.979, 1.000] | 0.000 |
| 25 | screened_4 | 0.737 (seeds 0.686-0.781); first seed 0.728 [0.674, 0.767] | 0.991 (seeds 0.986-0.994); first seed 0.991 [0.982, 1.000] | 0.000 |
| 50 | random | 0.796 (seeds 0.764-0.827); first seed 0.778 [0.730, 0.820] | 0.986 (seeds 0.979-0.993); first seed 0.985 [0.973, 0.994] | 0.000 |
| 50 | screened_3 | 0.804 (seeds 0.762-0.849); first seed 0.796 [0.757, 0.838] | 0.985 (seeds 0.980-0.993); first seed 0.982 [0.967, 0.994] | 0.000 |
| 50 | screened_4 | 0.800 (seeds 0.766-0.838); first seed 0.793 [0.754, 0.835] | 0.985 (seeds 0.980-0.993); first seed 0.985 [0.973, 0.997] | 0.000 |
| 75 | random | 0.842 (seeds 0.816-0.868); first seed 0.814 [0.769, 0.856] | 0.980 (seeds 0.971-0.990); first seed 0.979 [0.964, 0.991] | 0.000 |
| 75 | screened_3 | 0.846 (seeds 0.813-0.891); first seed 0.841 [0.811, 0.880] | 0.979 (seeds 0.974-0.987); first seed 0.979 [0.964, 0.991] | 0.000 |
| 75 | screened_4 | 0.842 (seeds 0.801-0.881); first seed 0.835 [0.802, 0.880] | 0.980 (seeds 0.973-0.987); first seed 0.982 [0.970, 0.994] | 0.000 |
| 100 | random | 0.878 (seeds 0.854-0.901); first seed 0.856 [0.823, 0.889] | 0.973 (seeds 0.964-0.984); first seed 0.973 [0.955, 0.985] | 0.000 |
| 100 | screened_3 | 0.878 (seeds 0.853-0.912); first seed 0.874 [0.847, 0.907] | 0.973 (seeds 0.962-0.981); first seed 0.973 [0.955, 0.988] | 0.000 |
| 100 | screened_4 | 0.874 (seeds 0.842-0.905); first seed 0.874 [0.844, 0.910] | 0.972 (seeds 0.962-0.978); first seed 0.976 [0.958, 0.991] | 0.000 |

## Screening effort and who got recruited

| budget | recruits | people screened (mean, range) | budget reached (of 10 seeds) | share missed among recruits |
| --- | --- | --- | --- | --- |
| 25 | random | 25.000 (25.000-25.000) | 10 | 0.49 |
| 25 | screened_3 | 58.400 (47.450-72.975) | 10 | 0.78 |
| 25 | screened_4 | 61.000 (50.125-71.425) | 10 | 0.79 |
| 50 | random | 50.000 (50.000-50.000) | 10 | 0.49 |
| 50 | screened_3 | 117.100 (95.800-138.325) | 10 | 0.73 |
| 50 | screened_4 | 121.600 (107.450-139.425) | 10 | 0.77 |
| 75 | random | 75.000 (75.000-75.000) | 10 | 0.49 |
| 75 | screened_3 | 178.200 (152.125-202.200) | 10 | 0.73 |
| 75 | screened_4 | 186.400 (162.350-212.775) | 10 | 0.76 |
| 100 | random | 100.000 (100.000-100.000) | 10 | 0.50 |
| 100 | screened_3 | 237.000 (205.450-261.100) | 10 | 0.73 |
| 100 | screened_4 | 246.300 (214.800-277.200) | 9 | 0.76 |

## Paired: screened minus random (test coverage, same seeds)

| budget | screener | method | mean diff | sd | screened wins |
| --- | --- | --- | --- | --- | --- |
| 25 | screened_3 | copula_on_recruits | -0.162 | 0.062 | 0/10 |
| 25 | screened_3 | claude_plus_recruits | +0.022 | 0.030 | 9/10 |
| 25 | screened_4 | copula_on_recruits | -0.161 | 0.057 | 0/10 |
| 25 | screened_4 | claude_plus_recruits | +0.019 | 0.031 | 9/10 |
| 50 | screened_3 | copula_on_recruits | -0.180 | 0.056 | 0/10 |
| 50 | screened_3 | claude_plus_recruits | +0.008 | 0.022 | 6/10 |
| 50 | screened_4 | copula_on_recruits | -0.170 | 0.086 | 1/10 |
| 50 | screened_4 | claude_plus_recruits | +0.004 | 0.024 | 5/10 |
| 75 | screened_3 | copula_on_recruits | -0.173 | 0.039 | 0/10 |
| 75 | screened_3 | claude_plus_recruits | +0.004 | 0.022 | 5/10 |
| 75 | screened_4 | copula_on_recruits | -0.179 | 0.058 | 0/10 |
| 75 | screened_4 | claude_plus_recruits | +0.001 | 0.023 | 6/10 |
| 100 | screened_3 | copula_on_recruits | -0.159 | 0.059 | 0/10 |
| 100 | screened_3 | claude_plus_recruits | -0.000 | 0.011 | 4/10 |
| 100 | screened_4 | copula_on_recruits | -0.167 | 0.035 | 0/10 |
| 100 | screened_4 | claude_plus_recruits | -0.004 | 0.017 | 4/10 |

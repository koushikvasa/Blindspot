# H4c: learning whom to ask (zero cost, GSS test split)

Crowd = claude_mode with the asked pool people swapped in: **includes real respondents (asked pool people)**. 24 items, target n = 334, pool n = 558, 10 seeds. Cells show the mean (2.5-97.5 percentile across seeds). k = 5. Budget 0 (claude_mode): coverage 0.638, precision 0.997. Commit `1dbd7f6031993e2952400138c2ed3c12ad8b6d4a`. Runtime 27.7 s.

## claude_mode + asked real people: coverage / precision

| budget | random coverage | targeted coverage | random precision | targeted precision |
| --- | --- | --- | --- | --- |
| 25 | 0.739 (0.720-0.769) | 0.739 (0.720-0.769) | 0.990 (0.986-0.994) | 0.990 (0.986-0.994) |
| 50 | 0.808 (0.777-0.829) | 0.819 (0.792-0.853) | 0.985 (0.979-0.990) | 0.984 (0.974-0.993) |
| 75 | 0.849 (0.820-0.873) | 0.864 (0.821-0.892) | 0.979 (0.973-0.990) | 0.975 (0.965-0.982) |
| 100 | 0.884 (0.862-0.903) | 0.892 (0.860-0.924) | 0.973 (0.967-0.984) | 0.967 (0.958-0.976) |

## gaussian_copula fit on the asked people: coverage / precision

| budget | random coverage | targeted coverage | random precision | targeted precision |
| --- | --- | --- | --- | --- |
| 25 | 0.778 (0.738-0.846) | 0.778 (0.738-0.846) | 0.867 (0.792-0.916) | 0.867 (0.792-0.916) |
| 50 | 0.835 (0.791-0.879) | 0.819 (0.781-0.848) | 0.863 (0.832-0.894) | 0.870 (0.831-0.893) |
| 75 | 0.859 (0.834-0.883) | 0.825 (0.756-0.874) | 0.872 (0.829-0.899) | 0.849 (0.814-0.882) |
| 100 | 0.857 (0.823-0.891) | 0.838 (0.788-0.906) | 0.869 (0.842-0.894) | 0.850 (0.807-0.887) |

## Targeted minus random (coverage, paired by seed)

| budget | method | mean diff | sd | seeds targeted better |
| --- | --- | --- | --- | --- |
| 25 | claude_plus_asked | +0.000 | 0.000 | 0/10 |
| 25 | copula_on_asked | +0.000 | 0.000 | 0/10 |
| 50 | claude_plus_asked | +0.011 | 0.024 | 6/10 |
| 50 | copula_on_asked | -0.016 | 0.036 | 4/10 |
| 75 | claude_plus_asked | +0.015 | 0.028 | 6/10 |
| 75 | copula_on_asked | -0.034 | 0.031 | 1/10 |
| 100 | claude_plus_asked | +0.008 | 0.023 | 6/10 |
| 100 | copula_on_asked | -0.019 | 0.039 | 4/10 |

## Near-duplicate share (synthetic rows within 0.02 Gower of an asked person; swapped-in rows exempt)

| budget | random | targeted |
| --- | --- | --- |
| 25 | 0.000 | 0.000 |
| 50 | 0.000 | 0.000 |
| 75 | 0.000 | 0.000 |
| 100 | 0.000 | 0.000 |

## Self-knowledge: estimated share of people claude_mode misses

True uncovered share on the test split: **0.362** (1 - coverage). Auxiliary truth on the pool with full-pool radii: 0.496. Estimate = asked people farther from every claude_mode member than from their 5th nearest asked neighbour, reweighted to pool gender x age-band shares.

| budget | estimate (random asked) | estimate (targeted asked) |
| --- | --- | --- |
| 25 | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) |
| 50 | 0.040 (0.000-0.097) | 0.033 (0.001-0.067) |
| 75 | 0.096 (0.020-0.193) | 0.090 (0.019-0.168) |
| 100 | 0.144 (0.101-0.187) | 0.125 (0.073-0.197) |

## Targeted: mean logistic coefficients (top 8 by size; positive = predicts a miss)

- fit_before_round_2 (10 seeds): income_band_$110,000 or more -0.24, race_white -0.23, age_band_65+ -0.23, income_band_Less than $30,000 +0.21, education_associate/junior college +0.20, age_band_unknown +0.20, income_band_$30,000-$49,999 +0.19, education_graduate -0.19
- fit_before_round_3 (10 seeds): income_band_Less than $30,000 +0.34, education_associate/junior college +0.29, income_band_$110,000 or more -0.28, age_band_30-49 +0.27, race_black +0.24, income_band_$75,000-$109,999 -0.24, income_band_$30,000-$49,999 +0.22, age_band_65+ -0.18
- fit_before_round_4 (10 seeds): income_band_$75,000-$109,999 -0.41, income_band_Less than $30,000 +0.36, race_black +0.34, education_associate/junior college +0.28, age_band_65+ -0.25, income_band_$30,000-$49,999 +0.23, region_midwest -0.23, race_unknown -0.21

## Fairness audit: claude_mode coverage by group (test, overall 0.638, groups with n >= 30, bootstrap 95% CI)

| group | n | coverage [CI] |
| --- | --- | --- |
| age_band=18-29 | 52 | 0.558 [0.423, 0.673] |
| age_band=30-49 | 111 | 0.676 [0.603, 0.757] |
| age_band=50-64 | 83 | 0.542 [0.434, 0.663] |
| age_band=65+ | 76 | 0.724 [0.632, 0.816] |
| gender=female | 190 | 0.589 [0.521, 0.663] |
| gender=male | 143 | 0.699 [0.629, 0.776] |
| education=bachelor's | 68 | 0.662 [0.544, 0.779] |
| education=graduate | 45 | 0.800 [0.711, 0.911] |
| education=high school | 162 | 0.599 [0.531, 0.685] |
| education=less than high school | 32 | 0.562 [0.375, 0.719] |
| income_band=$110,000 or more | 81 | 0.802 [0.704, 0.877] |
| income_band=$30,000-$49,999 | 46 | 0.500 [0.369, 0.632] |
| income_band=$50,000-$74,999 | 42 | 0.619 [0.476, 0.762] |
| income_band=$75,000-$109,999 | 57 | 0.719 [0.596, 0.842] |
| income_band=Less than $30,000 | 74 | 0.473 [0.378, 0.568] |
| income_band=unknown | 34 | 0.676 [0.500, 0.824] |
| region=midwest | 84 | 0.548 [0.440, 0.667] |
| region=northeast | 52 | 0.712 [0.596, 0.827] |
| region=south | 129 | 0.636 [0.558, 0.714] |
| region=west | 69 | 0.696 [0.594, 0.812] |

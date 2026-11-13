# H4d: B real people, select from AI candidates vs fit statistics (zero cost, GSS test split)

20 live items, target n = 334, 10 seeds, same asked people for every method. Cells: mean (2.5-97.5 percentile across seeds). k = 5. Realism filter = the B asked people only (k = min(5, B//5)); lambda chosen per seed from [0.0, 0.25, 0.5, 1.0] on a held-out half of the asked people. `select_claude_plus_fp`: the 400 saved Flashpoint people form a shared pool competing in every slot with that persona's 50 Claude candidates, each used at most once. `claude_plus_asked` and `select_plus_asked` INCLUDE REAL RESPONDENTS (asked people swapped in). Near-dup = synthetic rows within 0.02 Gower of any ASKED person (swapped-in rows exempt). Commit `bdf4e83bde4c756b8089a345dee817634712844e-dirty`. Runtime 917.3 s.

## Coverage

| method | B=25 | B=50 | B=75 | B=100 |
| --- | --- | --- | --- | --- |
| select_claude | 0.699 (0.629-0.758) | 0.698 (0.632-0.738) | 0.750 (0.715-0.790) | 0.765 (0.699-0.806) |
| select_claude_plus_fp | 0.771 (0.698-0.851) | 0.783 (0.678-0.849) | 0.807 (0.741-0.838) | 0.830 (0.764-0.874) |
| copula_on_asked | 0.815 (0.748-0.886) | 0.868 (0.815-0.889) | 0.885 (0.857-0.909) | 0.893 (0.862-0.922) |
| claude_plus_asked | 0.724 (0.701-0.755) | 0.799 (0.767-0.824) | 0.840 (0.809-0.864) | 0.888 (0.863-0.912) |
| select_plus_asked | 0.748 (0.688-0.792) | 0.796 (0.745-0.832) | 0.855 (0.826-0.883) | 0.885 (0.848-0.904) |

## Precision

| method | B=25 | B=50 | B=75 | B=100 |
| --- | --- | --- | --- | --- |
| select_claude | 0.896 (0.852-0.927) | 0.879 (0.735-0.933) | 0.893 (0.860-0.935) | 0.835 (0.715-0.922) |
| select_claude_plus_fp | 0.921 (0.885-0.952) | 0.931 (0.902-0.962) | 0.902 (0.731-0.961) | 0.922 (0.859-0.960) |
| copula_on_asked | 0.873 (0.843-0.902) | 0.890 (0.853-0.920) | 0.882 (0.859-0.915) | 0.886 (0.855-0.918) |
| claude_plus_asked | 0.991 (0.986-0.996) | 0.987 (0.977-0.991) | 0.981 (0.974-0.990) | 0.976 (0.968-0.982) |
| select_plus_asked | 0.900 (0.859-0.935) | 0.887 (0.776-0.941) | 0.902 (0.883-0.936) | 0.869 (0.782-0.937) |

## Near-duplicate share

| method | B=25 | B=50 | B=75 | B=100 |
| --- | --- | --- | --- | --- |
| select_claude | 0.000 (0.000-0.002) | 0.000 (0.000-0.002) | 0.000 (0.000-0.002) | 0.000 (0.000-0.002) |
| select_claude_plus_fp | 0.000 (0.000-0.002) | 0.000 (0.000-0.002) | 0.000 (0.000-0.002) | 0.000 (0.000-0.002) |
| copula_on_asked | 0.000 (0.000-0.002) | 0.000 (0.000-0.002) | 0.000 (0.000-0.002) | 0.000 (0.000-0.000) |
| claude_plus_asked | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) |
| select_plus_asked | 0.000 (0.000-0.003) | 0.000 (0.000-0.003) | 0.000 (0.000-0.003) | 0.000 (0.000-0.003) |

## k-sensitivity (mean coverage / precision at k=3 / 5 / 10)

| method | B | k=3 | k=5 | k=10 |
| --- | --- | --- | --- | --- |
| select_claude | 25 | 0.540 / 0.741 | 0.699 / 0.896 | 0.857 / 0.988 |
| select_claude | 50 | 0.537 / 0.718 | 0.698 / 0.879 | 0.866 / 0.971 |
| select_claude | 75 | 0.586 / 0.754 | 0.750 / 0.893 | 0.890 / 0.981 |
| select_claude | 100 | 0.578 / 0.683 | 0.765 / 0.835 | 0.918 / 0.956 |
| select_claude_plus_fp | 25 | 0.596 / 0.773 | 0.771 / 0.921 | 0.911 / 0.992 |
| select_claude_plus_fp | 50 | 0.616 / 0.786 | 0.783 / 0.931 | 0.925 / 0.994 |
| select_claude_plus_fp | 75 | 0.629 / 0.773 | 0.807 / 0.902 | 0.945 / 0.981 |
| select_claude_plus_fp | 100 | 0.657 / 0.792 | 0.830 / 0.922 | 0.958 / 0.988 |
| copula_on_asked | 25 | 0.629 / 0.747 | 0.815 / 0.873 | 0.952 / 0.968 |
| copula_on_asked | 50 | 0.694 / 0.755 | 0.868 / 0.890 | 0.985 / 0.979 |
| copula_on_asked | 75 | 0.705 / 0.760 | 0.885 / 0.882 | 0.987 / 0.977 |
| copula_on_asked | 100 | 0.710 / 0.763 | 0.893 / 0.886 | 0.989 / 0.983 |
| claude_plus_asked | 25 | 0.577 / 0.982 | 0.724 / 0.991 | 0.877 / 0.996 |
| claude_plus_asked | 50 | 0.651 / 0.972 | 0.799 / 0.987 | 0.934 / 0.994 |
| claude_plus_asked | 75 | 0.696 / 0.956 | 0.840 / 0.981 | 0.960 / 0.992 |
| claude_plus_asked | 100 | 0.734 / 0.944 | 0.888 / 0.976 | 0.981 / 0.991 |
| select_plus_asked | 25 | 0.579 / 0.750 | 0.748 / 0.900 | 0.899 / 0.987 |
| select_plus_asked | 50 | 0.628 / 0.741 | 0.796 / 0.887 | 0.940 / 0.972 |
| select_plus_asked | 75 | 0.695 / 0.769 | 0.855 / 0.902 | 0.967 / 0.981 |
| select_plus_asked | 100 | 0.722 / 0.735 | 0.885 / 0.869 | 0.984 / 0.967 |

## Chosen lambdas (count of seeds) and Flashpoint share of the crowd

| method | B | λ=0.0 | λ=0.25 | λ=0.5 | λ=1.0 | FP share |
| --- | --- | --- | --- | --- | --- | --- |
| select_claude | 25 | 9 | 0 | 1 | 0 | 0.000 |
| select_claude | 50 | 6 | 2 | 1 | 1 | 0.000 |
| select_claude | 75 | 2 | 8 | 0 | 0 | 0.000 |
| select_claude | 100 | 1 | 2 | 4 | 3 | 0.000 |
| select_claude_plus_fp | 25 | 9 | 0 | 1 | 0 | 0.233 |
| select_claude_plus_fp | 50 | 9 | 1 | 0 | 0 | 0.213 |
| select_claude_plus_fp | 75 | 7 | 0 | 2 | 1 | 0.238 |
| select_claude_plus_fp | 100 | 4 | 3 | 3 | 0 | 0.236 |

## Paired comparison (coverage, same seeds)

| budget | A vs B | mean diff (A - B) | sd | A wins |
| --- | --- | --- | --- | --- |
| 25 | select_claude vs copula_on_asked | -0.116 | 0.073 | 0/10 |
| 25 | select_claude_plus_fp vs copula_on_asked | -0.044 | 0.067 | 2/10 |
| 25 | select_plus_asked vs claude_plus_asked | +0.024 | 0.032 | 8/10 |
| 50 | select_claude vs copula_on_asked | -0.169 | 0.030 | 0/10 |
| 50 | select_claude_plus_fp vs copula_on_asked | -0.085 | 0.046 | 0/10 |
| 50 | select_plus_asked vs claude_plus_asked | -0.003 | 0.024 | 5/10 |
| 75 | select_claude vs copula_on_asked | -0.135 | 0.028 | 0/10 |
| 75 | select_claude_plus_fp vs copula_on_asked | -0.079 | 0.039 | 0/10 |
| 75 | select_plus_asked vs claude_plus_asked | +0.014 | 0.014 | 8/10 |
| 100 | select_claude vs copula_on_asked | -0.128 | 0.042 | 0/10 |
| 100 | select_claude_plus_fp vs copula_on_asked | -0.063 | 0.046 | 0/10 |
| 100 | select_plus_asked vs claude_plus_asked | -0.003 | 0.018 | 5/10 |

## References (n = 334, 20 items, from bakeoff1; selection rows use the FULL pool as filter)

- claude_mode: coverage 0.608, precision 0.997
- select_from_50_lambda_0.0: coverage 0.747, precision 0.928
- select_from_50_lambda_0.5: coverage 0.789, precision 0.855
- select_from_50_lambda_1.0: coverage 0.702, precision 0.686
- gaussian_copula: coverage 0.922, precision 0.886
- other_real_people: coverage 0.967, precision 0.910

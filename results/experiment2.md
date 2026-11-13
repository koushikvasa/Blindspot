# Experiment 2: the self-improving loop vs baselines at equal numbers of real people asked

Model `claude-haiku-5-5`, seeds [42, 43, 44], 4 rounds of 25 people, max archetype share 1.0, k = 5, 50 bootstrap resamples per point. Target = test split, agent asks pool only, holdout untouched. Commit `c907528050a85da99b74baa0b3deedfc934b5250`.

Cost: worst-case estimate $2.26, expected $1.13; actual $0.21 (limit $5.00). Runtime 61012.5 s.

## GSS 2024 (main cross-section, ballot 1) (target n = 334, pool n = 558, 3 seeds)

**coverage**, mean (min-max) over seeds, by number of real people asked:

| method | 25 | 50 | 75 | 100 |
| --- | --- | --- | --- | --- |
| blindspot_targeted | 0.641 (0.638-0.647) | 0.662 (0.647-0.677) | 0.655 (0.638-0.674) | 0.656 (0.647-0.671) |
| blindspot_random | 0.641 (0.638-0.647) | 0.652 (0.635-0.662) | 0.648 (0.632-0.680) | 0.659 (0.635-0.671) |
| copula_on_asked | 0.764 (0.716-0.817) | 0.848 (0.808-0.868) | 0.829 (0.808-0.841) | 0.834 (0.826-0.841) |
| claude_plus_asked | 0.721 (0.698-0.734) | 0.818 (0.814-0.823) | 0.853 (0.835-0.862) | 0.895 (0.880-0.907) |

**precision**, mean (min-max) over seeds, by number of real people asked:

| method | 25 | 50 | 75 | 100 |
| --- | --- | --- | --- | --- |
| blindspot_targeted | 0.998 (0.997-1.000) | 0.999 (0.997-1.000) | 0.999 (0.997-1.000) | 0.999 (0.997-1.000) |
| blindspot_random | 0.998 (0.997-1.000) | 0.999 (0.997-1.000) | 0.999 (0.997-1.000) | 0.998 (0.994-1.000) |
| copula_on_asked | 0.862 (0.817-0.934) | 0.871 (0.835-0.892) | 0.818 (0.805-0.832) | 0.833 (0.808-0.865) |
| claude_plus_asked | 0.989 (0.988-0.991) | 0.984 (0.979-0.988) | 0.978 (0.970-0.982) | 0.974 (0.970-0.976) |

Fixed references (test split, k = 5):

| reference | coverage | precision |
| --- | --- | --- |
| claude_mode, budget 0 | 0.638 | 0.997 |
| claude_mode | 0.638 | 0.997 |
| claude_sampled | 0.371 | 0.521 |
| gaussian_copula | 0.880 | 0.880 |
| other_real_people | 0.961 | 0.895 |

Near-duplicate share (synthetic people within 0.02 of an asked real person), worst over all runs and rounds: 0.00% (limit 5%). Seed 42 targeted archetypes per round: [1, 3, 4, 6].
Spend for this dataset: $0.21 over 1017 billed calls (0 errors). Runtime 61011.9 s.

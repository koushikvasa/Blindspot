# Experiment 2b: the redesigned loop (detect, propose, verify, escalate, assemble) on GSS

Model `claude-haiku-5-5`, seeds [42, 43, 44], 4 rounds of 25 people, k = 5. Target = test split (n = 334); agent asks pool only; holdout untouched. Commit `4d75e12a74d9eda8ee9bec2c03da3d7ab7857b76`. Cost: actual $2.18 (hard stop $5.00; worst-case estimate $7.59). The H4b crowds contain NO real respondents' rows; claude_plus_asked does (the rule-2 baseline).

**coverage**, mean (min-max) over seeds, by number of real people asked:

| method | 25 | 50 | 75 | 100 |
| --- | --- | --- | --- | --- |
| h4b_verify | 0.621 (0.608-0.632) | 0.610 (0.593-0.623) | 0.610 (0.602-0.626) | 0.614 (0.581-0.647) |
| h4b_ablation | 0.634 (0.611-0.659) | 0.612 (0.590-0.644) | 0.618 (0.599-0.629) | 0.626 (0.593-0.656) |
| copula_on_asked | 0.809 (0.722-0.889) | 0.834 (0.808-0.853) | 0.865 (0.859-0.871) | 0.860 (0.832-0.874) |
| claude_plus_asked | 0.738 (0.716-0.766) | 0.817 (0.787-0.844) | 0.855 (0.847-0.868) | 0.890 (0.877-0.907) |
| (earlier H4 loop, for reference) | 0.641 (0.638-0.647) | 0.662 (0.647-0.677) | 0.655 (0.638-0.674) | 0.656 (0.647-0.671) |

**precision**, mean (min-max) over seeds, by number of real people asked:

| method | 25 | 50 | 75 | 100 |
| --- | --- | --- | --- | --- |
| h4b_verify | 1.000 (1.000-1.000) | 0.998 (0.994-1.000) | 0.998 (0.994-1.000) | 0.999 (0.997-1.000) |
| h4b_ablation | 0.998 (0.997-1.000) | 0.998 (0.997-1.000) | 0.998 (0.994-1.000) | 0.999 (0.997-1.000) |
| copula_on_asked | 0.889 (0.850-0.925) | 0.856 (0.838-0.874) | 0.862 (0.847-0.883) | 0.856 (0.844-0.865) |
| claude_plus_asked | 0.992 (0.985-0.997) | 0.981 (0.973-0.988) | 0.973 (0.961-0.979) | 0.966 (0.958-0.976) |
| (earlier H4 loop, for reference) | 0.998 (0.997-1.000) | 0.999 (0.997-1.000) | 0.999 (0.997-1.000) | 0.999 (0.997-1.000) |

Fixed references (test split):

| reference | coverage | precision |
| --- | --- | --- |
| claude_mode, budget 0 | 0.638 | 0.997 |
| claude_mode | 0.638 | 0.997 |
| claude_sampled | 0.371 | 0.521 |
| gaussian_copula | 0.880 | 0.880 |
| other_real_people | 0.961 | 0.895 |

## Recipe accept/reject counts (summed over rounds 1-4)

| seed | variant | clusters | merged | proposed | accepted | rejected | dropped by guard | cannot imagine | people replaced (last round) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | verify | 18 | 7 | 31 | 25 | 6 | 0 | 0 | 334 |
| 42 | ablation | 18 | 7 | 31 | 31 | 0 | 0 | 0 | 334 |
| 43 | verify | 19 | 5 | 40 | 38 | 2 | 0 | 0 | 333 |
| 43 | ablation | 19 | 5 | 40 | 40 | 0 | 0 | 0 | 333 |
| 44 | verify | 19 | 10 | 30 | 20 | 10 | 0 | 0 | 331 |
| 44 | ablation | 19 | 10 | 27 | 27 | 0 | 0 | 0 | 330 |

## Uncovered asked people per round: old rule vs new rule (verify loop)

| seed | round | asked | old rule (k-NN ball) | new rule (95th pct NN) | new-rule threshold |
| --- | --- | --- | --- | --- | --- |
| 42 | 1 | 25 | 2 | 23 | 0.05069 |
| 42 | 2 | 50 | 4 | 49 | 0.05326 |
| 42 | 3 | 75 | 11 | 74 | 0.06042 |
| 42 | 4 | 100 | 26 | 100 | 0.05282 |
| 43 | 1 | 25 | 1 | 25 | 0.05069 |
| 43 | 2 | 50 | 7 | 50 | 0.05625 |
| 43 | 3 | 75 | 12 | 75 | 0.05423 |
| 43 | 4 | 100 | 33 | 100 | 0.05816 |
| 44 | 1 | 25 | 0 | 25 | 0.05069 |
| 44 | 2 | 50 | 6 | 49 | 0.06032 |
| 44 | 3 | 75 | 15 | 74 | 0.05217 |
| 44 | 4 | 100 | 16 | 99 | 0.05069 |

## Escalation log, seed 42 (verify loop)

- **Round 1** asked random: random (round 1)
  - decided after this round: random (no cluster needs escalation)
- **Round 2** asked random: random (no cluster needs escalation)
  - decided after this round: random (no cluster needs escalation)
- **Round 3** asked random: random (no cluster needs escalation)
  - decided after this round: random (no cluster needs escalation)
- **Round 4** asked random: random (no cluster needs escalation)
  - decided after this round: n/a (last)

Near-duplicate share, worst over all runs and rounds: 0.00% (limit 5%). Billed calls: 5532, errors: 0. Wall-clock (includes any laptop sleep): 1163.5 s.

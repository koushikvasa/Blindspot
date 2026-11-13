# UI export (H8)

Files in `results/ui/`, test-split numbers, runtime 48 s. Coverage assertion (map covered share equals summary coverage, 3 decimals):

| dataset | crowd | map covered share | summary coverage | equal |
| --- | --- | --- | --- | --- |
| twin | retest | 0.984 | 0.984 | yes |
| twin | other_real_people | 0.981 | 0.981 | yes |
| twin | gaussian_copula | 0.888 | 0.888 | yes |
| twin | independent_marginals | 0.512 | 0.512 | yes |
| twin | claude_mode | 0.345 | 0.345 | yes |
| twin | gpt41mini_twin | 0.344 | 0.344 | yes |
| twin | claude_sampled | 0.17 | 0.17 | yes |
| gss | other_real_people | 0.967 | 0.967 | yes |
| gss | gaussian_copula | 0.922 | 0.922 | yes |
| gss | independent_marginals | 0.823 | 0.823 | yes |
| gss | flashpoint_panel | 0.611 | 0.611 | yes |
| gss | claude_mode | 0.608 | 0.608 | yes |
| gss | claude_sampled | 0.473 | 0.473 | yes |

## Captions

- repairs_gss: Claude alone covers 61% of real people. With only 25 real people as the filter, Opus drafts reach 81% and statistics fitted on the same 25 reach 82%: a tie within noise (the seed ranges overlap), and Opus never beats statistics on average (with 100 people: 88% vs 89%). Opus's picks look more realistic (precision 0.96+ vs 0.87 for statistics); Haiku drafts, archetypes and recipes fall short. Bars show the range over seeds (or a 95% interval for single runs); scored on GSS test people none of the methods saw.
- scoreboard_gss: Statistics fitted on the real people came first at 4 of 4 budgets (82% with 25 people, 89% with 100). Opus drafts filtered by the same people came close (81% and 88%) without beating them on average; Sonnet drafts reached 85% and Haiku drafts at most 83% with 100 people. Lines show the range over 10 seeds.
- fairness_gss: Share of each group the AI crowd can imagine (overall 64%, all 24 items, groups of 30+ people, bootstrap 95% intervals). 4 pairs of groups within a dimension have clearly separated intervals, the widest being under $30k (47%) vs $110k and over (80%); the other intervals overlap.

live_gss: 20 items, 334 real people, crowd = claude_mode, base_rate_uncovered = 0.392.

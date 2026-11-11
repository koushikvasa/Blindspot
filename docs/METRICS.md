# Distance and coverage metrics (H2)

## Distance (`blindspot/space`)
- Each answer becomes a value in [0, 1]: option index / (k − 1). NaN stays NaN.
- Gower distance = weighted mean absolute difference over the items both people answered.
- Each `Item.family` carries equal total weight: an item weighs 1 / (items in its family), and weights are renormalized over the items both people answered. Two people who share no answered item get NaN (treated as +inf in coverage and as 1.0 in the maps).

## Coverage (`blindspot/coverage`)
Following Naeem et al. (2020), *Reliable Fidelity and Diversity Metrics for Generative Models*:
- Each real person in the target set gets a ball with radius = distance to their k-th nearest **real** neighbour (same target set, self excluded). Default k = 5; k = 3 and 10 are reported as a sensitivity check.
- **coverage**: share of real people with at least one synthetic person inside their ball (distance ≤ radius; ties are common with discrete answers).
- **precision**: share of synthetic people inside at least one real ball.
- **rcg_ratio**: median over real people of (distance to nearest synthetic) / (k-NN radius). Values ≤ 1 mean covered. This is **our own simple version** of the respondent-coverage-gap idea (Kumagae & Kaneko, 2026). The paper's exact formula may differ.
- **per_family_entropy**: mean normalized entropy (0 = everyone gives the same answer, 1 = uniform) per family, for the crowd and the real target.
- **Bootstrap CIs**: 200 seeded percentile resamples with radii held fixed. Coverage resamples real people. Precision resamples synthetic people, because resampling real people biases precision downward (duplicates shrink the union of balls) and gave intervals that excluded the point estimate.
- Crowds are compared at the size of the real target set (fixed-seed subsample if larger).

## Baselines (`blindspot/generators/baselines.py`)
All are fit on the **pool** split (checked with `assert_no_leak`) and generate a crowd the size of the target.
- `independent_marginals`: each answer drawn independently from the pool's option shares.
- `gaussian_copula`: mode-impute (fitting only) → rank normal scores → correlation matrix (repaired to positive definite) → multivariate normal sample → each item's empirical quantiles.
- `other_real_people`: a random sample of real pool people.
- Synthetic person j gets target person j's demographics row (rule 3). Marginals/copula answers are not conditioned on demographics. `other_real_people` keeps its own demographics, so its demographic mix only approximately matches the target.

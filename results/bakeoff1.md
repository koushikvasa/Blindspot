# Bake-off round 1: select-from-many and model mix (zero cost, test split only)

No Claude or Flashpoint calls. Candidates: 50 per persona from cached claude_sampled probabilities; realism = mean Gower to the 5 nearest pool people; novelty = distance to the nearest already selected member; pick = argmin realism - lambda*novelty. 5 seeds for selection rows (mean; seed range), bootstrap CI from the first seed. Near-dup = share within 0.02 Gower of any pool person. Commit `4abe0eeff19170cac40135ad16cf0189d69423f1`. Runtime 48.7 s.

GSS parts use the 20 live items (H3's 24-item numbers differ). Pool used only as a realism filter.

## A. GSS select-from-many (target n = 334, 20 live items)

| crowd | n | coverage [95% CI, first seed]; seed range | precision [95% CI] | near-dup vs pool | k=3 cov/prec | k=5 | k=10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| claude_mode | 334 | 0.608 [0.554, 0.662] | 0.997 [0.988, 1.000] | 0.000 | 0.488 / 0.994 | 0.608 / 0.997 | 0.766 / 0.997 |
| claude_sampled | 334 | 0.473 [0.428, 0.527] | 0.548 [0.497, 0.602] | 0.000 | 0.335 / 0.365 | 0.473 / 0.548 | 0.716 / 0.766 |
| gaussian_copula | 334 | 0.922 [0.892, 0.946] | 0.886 [0.856, 0.916] | 0.000 | 0.754 / 0.740 | 0.922 / 0.886 | 0.991 / 0.970 |
| independent_marginals | 334 | 0.823 [0.787, 0.862] | 0.799 [0.757, 0.841] | 0.000 | 0.656 / 0.659 | 0.823 / 0.799 | 0.961 / 0.934 |
| other_real_people | 334 | 0.967 [0.949, 0.982] | 0.910 [0.877, 0.937] | 1.000 | 0.847 / 0.805 | 0.967 / 0.910 | 0.988 / 0.979 |
| select_from_50_lambda_0.0 | 334 | 0.747 [0.730, 0.823]; seeds 0.710-0.775 | 0.928 [0.895, 0.949] | 0.002 | 0.599 / 0.797 | 0.747 / 0.928 | 0.897 / 0.990 |
| select_from_50_lambda_0.5 | 334 | 0.789 [0.772, 0.853]; seeds 0.763-0.823 | 0.855 [0.826, 0.892] | 0.002 | 0.619 / 0.688 | 0.789 / 0.855 | 0.947 / 0.969 |
| select_from_50_lambda_1.0 | 334 | 0.702 [0.665, 0.749]; seeds 0.686-0.737 | 0.686 [0.659, 0.763] | 0.001 | 0.504 / 0.519 | 0.702 / 0.686 | 0.911 / 0.880 |
| select_from_50_lambda_2.0 | 334 | 0.477 [0.437, 0.542]; seeds 0.446-0.497 | 0.494 [0.449, 0.563] | 0.000 | 0.317 / 0.338 | 0.477 / 0.494 | 0.747 / 0.738 |

## B. Flashpoint candidates, choose 200 of 400 (n = 200)

| crowd | n | coverage [95% CI, first seed]; seed range | precision [95% CI] | near-dup vs pool | k=3 cov/prec | k=5 | k=10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flashpoint_q1 | 200 | 0.611 [0.554, 0.671] | 0.765 [0.695, 0.815] | 0.000 | 0.419 / 0.590 | 0.611 / 0.765 | 0.844 / 0.910 |
| flashpoint_q2 | 200 | 0.617 [0.569, 0.677] | 0.740 [0.680, 0.795] | 0.000 | 0.398 / 0.620 | 0.617 / 0.740 | 0.862 / 0.925 |
| flashpoint_random_200_of_400 | 200 | 0.608 [0.582, 0.695]; seeds 0.557-0.662 | 0.749 [0.710, 0.830] | 0.000 | 0.412 / 0.608 | 0.608 / 0.749 | 0.862 / 0.919 |
| flashpoint_all_400 (n=400, subsampled to 334 by evaluate) | 334 | 0.763 [0.716, 0.808] | 0.778 [0.733, 0.818] | 0.000 | 0.569 / 0.629 | 0.763 / 0.778 | 0.949 / 0.928 |
| flashpoint_select_200_of_400_lambda_0.0 | 200 | 0.740 [0.698, 0.787] | 0.900 [0.850, 0.940] | 0.000 | 0.530 / 0.760 | 0.740 / 0.900 | 0.937 / 0.985 |
| flashpoint_select_200_of_400_lambda_0.5 | 200 | 0.740 [0.701, 0.788] | 0.860 [0.810, 0.905] | 0.000 | 0.536 / 0.690 | 0.740 / 0.860 | 0.943 / 0.950 |
| flashpoint_select_200_of_400_lambda_1.0 | 200 | 0.716 [0.674, 0.766] | 0.795 [0.740, 0.850] | 0.000 | 0.470 / 0.600 | 0.716 / 0.795 | 0.928 / 0.930 |
| flashpoint_select_200_of_400_lambda_2.0 | 200 | 0.614 [0.572, 0.665] | 0.685 [0.620, 0.735] | 0.000 | 0.404 / 0.515 | 0.614 / 0.685 | 0.850 / 0.860 |

Reference at n = 200 (mean of 20 draws, from experiment_fp): claude_mode 0.567 / 0.997; gaussian_copula 0.791 / 0.883; other_real_people 0.915 / 0.934; independent_marginals 0.631 / 0.779

## C. Twin model mix (target n = 617, 30 items)

| crowd | n | coverage [95% CI, first seed]; seed range | precision [95% CI] | near-dup vs pool | k=3 cov/prec | k=5 | k=10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| claude_mode | 617 | 0.345 [0.308, 0.383] | 0.963 [0.948, 0.977] | 0.000 | 0.253 / 0.911 | 0.345 / 0.963 | 0.494 / 0.990 |
| gpt41mini_twin | 617 | 0.344 [0.301, 0.383] | 0.956 [0.937, 0.972] | 0.000 | 0.232 / 0.861 | 0.344 / 0.956 | 0.475 / 0.995 |
| mix_50_50 | 617 | 0.422 [0.391, 0.465]; seeds 0.405-0.439 | 0.959 [0.932, 0.969] | 0.000 | 0.310 / 0.888 | 0.422 / 0.959 | 0.575 / 0.993 |

Union upper bound (real person covered if either full crowd covers them): coverage **0.502**. Reference (experiment1): gaussian_copula 0.888 / 0.862; other_real_people 0.981 / 0.908; independent_marginals 0.512 / 0.577

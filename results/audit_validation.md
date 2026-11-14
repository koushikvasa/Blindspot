# Audit estimator: small-sample validation

GSS 2024, 20 live items. Real samples of 25, 50, 100, 200 people drawn from the POOL split (20 draws each); AI crowds = Haiku 5.5 typical answers (claude_mode), Haiku 5.5 random answers (claude_sampled) and Flashpoint panel 1. The tool's estimate is compared with the TRUE coverage of the same crowd: the same engine on the whole TEST split (334 real people, 5th-nearest-neighbour radius). Error = estimate - truth (coverage = share of real people covered; missed = 1 - coverage, so a positive coverage bias means the tool is too optimistic). Zero API cost, holdout never loaded. Commit `97dec96da8474c5d140d6e9433f50f110f8f3161`. Runtime 18 s.

True coverage on the test split: claude_mode 0.623 (precision 0.997), claude_sampled 0.515 (precision 0.608), flashpoint_panel_1 0.635 (precision 0.770)

## Estimators (pooled over the three crowds; coverage error in points = 100 x error)

| estimator | n=25: bias / sd / rmse | n=50: bias / sd / rmse | n=100: bias / sd / rmse | n=200: bias / sd / rmse | mean rmse |
| --- | --- | --- | --- | --- | --- |
| k=1 | +14.7 / 11.2 / 18.4 | -2.3 / 9.9 / 10.0 | -18.3 / 5.1 / 19.0 | -33.3 / 5.7 / 33.8 | 20.3 |
| k=3 | +38.5 / 7.0 / 39.1 | +30.8 / 7.7 / 31.8 | +16.2 / 6.3 / 17.4 | -1.3 / 4.9 / 5.0 | 23.3 |
| k=5 | +40.7 / 5.7 / 41.0 | +38.4 / 6.2 / 38.8 | +29.4 / 6.1 / 30.0 | +15.6 / 5.5 / 16.5 | 31.6 |
| k=1, rescaled x0.5 | +1.3 / 13.1 / 13.0 | -6.2 / 9.9 / 11.6 | -12.6 / 5.5 / 13.7 | -18.7 / 4.7 / 19.3 | 14.4 |
| k=3, rescaled x0.5 | +20.7 / 11.7 / 23.8 | +12.9 / 8.7 / 15.5 | +4.9 / 5.5 / 7.3 | -1.5 / 4.9 / 5.1 | 12.9 |
| k=5, rescaled x0.5 | +29.1 / 10.0 / 30.8 | +21.2 / 9.0 / 23.0 | +13.8 / 6.4 / 15.2 | +6.3 / 5.4 / 8.3 | 19.3 |
| k=1, rescaled x0.75 | -6.5 / 13.4 / 14.8 | -8.1 / 9.8 / 12.7 | -10.0 / 5.5 / 11.4 | -10.7 / 5.1 / 11.8 | 12.7 |
| k=3, rescaled x0.75 **(chosen)** | +3.3 / 14.7 / 14.9 | +0.7 / 11.3 / 11.2 | -0.8 / 5.2 / 5.2 | -1.5 / 4.9 / 5.0 | 9.1 |
| k=5, rescaled x0.75 | +8.9 / 14.2 / 16.7 | +3.7 / 11.7 / 12.2 | +4.4 / 6.0 / 7.4 | +1.8 / 5.0 / 5.2 | 10.4 |
| k=1, rescaled x1 | -14.2 / 14.6 / 20.3 | -10.4 / 10.0 / 14.4 | -7.5 / 5.6 / 9.3 | -2.0 / 5.4 / 5.7 | 12.4 |
| k=3, rescaled x1 | -13.7 / 16.6 / 21.5 | -10.6 / 11.0 / 15.3 | -6.0 / 5.6 / 8.2 | -1.5 / 4.9 / 5.0 | 12.5 |
| k=5, rescaled x1 | -12.9 / 17.6 / 21.7 | -11.1 / 12.1 / 16.4 | -6.6 / 6.2 / 9.0 | -3.0 / 5.2 / 5.9 | 13.3 |

## Chosen estimator: k=3, rescaled x0.75

Per crowd (coverage error in points: bias / sd / rmse):

| crowd | n=25 | n=50 | n=100 | n=200 |
| --- | --- | --- | --- | --- |
| claude_mode | +0.9 / 14.2 / 13.9 | -1.0 / 8.7 / 8.5 | -1.3 / 4.4 / 4.5 | -3.2 / 2.6 / 4.1 |
| claude_sampled | +5.9 / 15.7 / 16.4 | +6.6 / 12.5 / 13.8 | +3.4 / 4.6 / 5.6 | +4.2 / 2.6 / 4.9 |
| flashpoint_panel_1 | +3.1 / 14.4 / 14.4 | -3.5 / 10.3 / 10.6 | -4.3 / 3.6 / 5.6 | -5.4 / 2.5 / 6.0 |

## Precision (share of AI respondents that look realistic)

Precision depends on how many real people the AI cloud can be compared with, so small samples understate it. Pooled error in points (bias / sd / rmse):

| estimator | n=25 | n=50 | n=100 | n=200 | mean rmse |
| --- | --- | --- | --- | --- | --- |
| k=1 | -23.8 / 13.7 / 27.4 | -27.9 / 15.4 / 31.8 | -31.3 / 15.7 / 35.0 | -33.1 / 17.4 / 37.3 | 32.9 |
| k=3 **(chosen for precision)** | +4.4 / 6.9 / 8.1 | -2.8 / 5.5 / 6.2 | -6.1 / 5.8 / 8.4 | -8.7 / 6.5 / 10.9 | 8.4 |
| k=5 | +14.4 / 11.4 / 18.3 | +8.7 / 7.5 / 11.5 | +4.8 / 4.8 / 6.8 | +2.9 / 3.5 / 4.5 | 10.3 |
| k=1, rescaled x0.5 | -39.1 / 15.0 / 41.9 | -32.5 / 16.3 / 36.3 | -25.6 / 14.3 / 29.3 | -17.1 / 11.6 / 20.6 | 32.0 |
| k=3, rescaled x0.5 | -26.9 / 14.7 / 30.6 | -23.9 / 14.3 / 27.8 | -16.5 / 11.2 / 19.9 | -8.8 / 6.6 / 11.0 | 22.3 |
| k=5, rescaled x0.5 | -18.8 / 13.8 / 23.2 | -18.4 / 12.3 / 22.1 | -11.7 / 9.0 / 14.7 | -5.2 / 4.5 / 6.8 | 16.7 |
| k=1, rescaled x0.75 | -46.2 / 15.1 / 48.5 | -34.4 / 16.5 / 38.1 | -22.9 / 13.3 / 26.4 | -8.3 / 7.1 / 10.9 | 31.0 |
| k=3, rescaled x0.75 | -42.9 / 16.0 / 45.8 | -34.1 / 16.4 / 37.8 | -21.7 / 13.6 / 25.5 | -8.8 / 6.6 / 11.0 | 30.0 |
| k=5, rescaled x0.75 | -38.6 / 17.0 / 42.1 | -32.3 / 16.9 / 36.4 | -20.2 / 13.3 / 24.1 | -9.0 / 6.7 / 11.2 | 28.5 |
| k=1, rescaled x1 | -52.2 / 14.3 / 54.1 | -36.5 / 16.6 / 40.0 | -20.0 / 12.2 / 23.4 | +0.4 / 4.4 / 4.3 | 30.5 |
| k=3, rescaled x1 | -55.6 / 14.9 / 57.6 | -43.2 / 15.9 / 46.0 | -26.3 / 15.3 / 30.3 | -8.9 / 6.6 / 11.1 | 36.2 |
| k=5, rescaled x1 | -54.3 / 15.6 / 56.4 | -44.2 / 16.8 / 47.2 | -27.8 / 16.4 / 32.3 | -13.0 / 9.2 / 15.9 | 38.0 |

## Table shipped with the report (`blindspot/audit/calibration.json`)

| real sample size | bias | sd | typical error (rmse) |
| --- | --- | --- | --- |
| 25 | +3.3 pts | 14.7 pts | ±14.9 pts |
| 50 | +0.7 pts | 11.3 pts | ±11.2 pts |
| 100 | -0.8 pts | 5.2 pts | ±5.2 pts |
| 200 | -1.5 pts | 4.9 pts | ±5.0 pts |

Precision error table shipped with the report: n=25: +4.4 / ±8.1 pts, n=50: -2.8 / ±6.2 pts, n=100: -6.1 / ±8.4 pts, n=200: -8.7 / ±10.9 pts

Reliable minimum: **100 real people** (the smallest size whose typical coverage error is at most 7 points). Below it the report warns that the numbers are only a rough guide.

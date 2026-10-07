# U05 lab keys: execution-verified outputs

Seed 0, NumPy, float64.

## Task 1

All 5 records pass: one prompt, two distinct non-empty responses each.

## Task 2

w = [0.300, 12.227]. Margins: 6.39, 3.37, 2.11, 2.17, all positive. The quality feature dominates, as constructed.

## Task 3

| beta | KL | E[reward] |
| --- | --- | --- |
| 0.2 | 1.058 | 1.993 |
| 0.5 | 0.658 | 1.851 |
| 1.0 | 0.266 | 1.575 |
| 2.0 | 0.078 | 1.320 |
| 5.0 | 0.013 | 1.132 |
| 20.0 | 0.001 | 1.033 |

Monotone frontier: less KL buys less reward.

## Task 4

Beta 5: [0.40, 0.33, 0.27], sum 1.000. Beta 0.5: [0.87, 0.12, 0.02], sum 1.000.

## Task 5

Loss at margin 2.0: 0.313. At 0.0: 0.693. At -1.0: 0.974.

## Task 6

Apparent gap = log(0.65/0.35) = 0.619 utils of pure bias.

## Task 7

True peak at step 12 (value 0.62). Stopping rule: halt when the proxy rises for 5 straight steps while the human-eval probe falls twice in a row.

## Task 8

Majority agreements: 0.86, 0.90, 0.86, 0.88, 0.81. Mean pairwise agreement 0.80.

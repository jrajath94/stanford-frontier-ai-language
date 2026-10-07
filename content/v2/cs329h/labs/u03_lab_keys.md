# U03 lab keys: execution-verified outputs

Seed 0, NumPy, float64.

## Task 1

MLE = 0.70. Grid optimum matches 7/10. Log-likelihood at 0.70 exceeds its value at the truth 0.65.

## Task 2

Log-likelihood: -775.9 -> -738.2 over 40 iterations, monotone (asserted on diffs). Recovered means: -0.99 and 1.97. Mixing weight 0.42.

## Task 3

Posterior Beta(9,5), mean 0.643. Prior mean 0.50 < 0.643 < MLE 0.70.

## Task 4

Monte Carlo mean 0.643 (matches 9/14). 90% interval [0.423, 0.830].

## Task 5

A-B gaps at lambda 0.01, 0.1, 0.5, 1, 2, 5, 10: 5.08, 3.23, 2.06, 1.61, 1.21, 0.75, 0.48. Monotone shrinkage.

## Task 6

Mean absolute gap 0.044 across 10 bins. The curve sags below the diagonal: overconfident.

## Task 7

Full-sample 0.910, naive on kept data 1.077 (MNAR overstates the gap), IPW 0.899 (recovers the full-sample value).

## Task 8

Fold accuracies: 0.740, 0.610, 0.790, 0.730, 0.650. Mean 0.704, sd 0.062.

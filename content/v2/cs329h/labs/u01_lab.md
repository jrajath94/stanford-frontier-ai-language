# U01 lab: from choices to scores

Environment: Python 3 with NumPy and matplotlib. No installs beyond the standard scientific stack. Seed 0 everywhere unless noted. Run each task as a script and keep the outputs. The keys file shows the verified numbers.

## Task 1: separate preference from context

Generate 400 comparisons: 200 with A shown first, 200 with B shown first. True s = 1.1, context effect b = -0.8 on the B-first group. Fit (s, b) by grid search on the log-likelihood. Report s_hat, b_hat, and the pooled naive fraction.

## Task 2: add a wording feature

Take the task 1 data. Add a binary wording flag w (half the records). Give w a true effect of 0.5 utils. Extend the design matrix and refit (s, b, w_coef) by gradient ascent. Report all three estimates.

## Task 3: win-count matrix

From the record list [(A,B,1), (A,B,1), (B,A,1), (B,C,1), (B,C,1), (B,C,1)] build the 3x3 win-count matrix. Verify antisymmetry of the pair totals.

## Task 4: fit three players

Fit Bradley-Terry scores on the task 3 records with sum-zero anchoring, 2000 gradient steps, lr 0.1. Report the scores and the predicted P(A beats B).

## Task 5: reproduce the likelihood plate

Grid search the negative log-likelihood for the 5-1 toy over d in [-3, 3]. Report the MLE gap and the minimum NLL. Confirm the closed form log(5).

## Task 6: noise attenuation curve

True gap 1.5, n = 2000. For q in [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], flip labels and estimate the gap by log-odds. Report the seven estimates.

## Task 7: train/test curves

True gap 1.0. For n in [20, 50, 100, 200, 500, 1000], fit the gap and score train and test log-loss (test n = 5000). Report both series.

## Task 8: annotator quality report

Simulate 5 annotators with gold accuracies [0.98, 0.95, 0.90, 0.80, 0.70] and self-consistency [0.93, 0.90, 0.85, 0.72, 0.65]. Apply the rule: drop below 0.85 gold. Report who stays and the overhead of 10% repeats plus 5% golds.

## Task 9: per-group report card

Two groups, n = 600 each, true gaps 2.0 and 0.4. Fit one pooled majority predictor. Report pooled accuracy and per-group accuracies with the gap standard error.

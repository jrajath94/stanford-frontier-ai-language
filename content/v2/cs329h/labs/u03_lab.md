# U03 lab: estimation and Bayes in code

Environment: Python 3 with NumPy and matplotlib. Seed 0. Keep outputs. Keys show verified numbers.

## Task 1: MLE peak

Grid search the log-likelihood for 7 wins in 10 trials over p in [0.01, 0.99]. Report the MLE and confirm 0.70.

## Task 2: EM on a Gaussian mixture

Generate 400 scores from the two-Gaussian mixture (means -1.0, 2.0. Sd 1.0. Weight 0.4). Run 40 EM iterations from (0.5, 0.0, 0.5, 1.5, 1.5). Assert the log-likelihood never decreases. Report start/end likelihood and recovered means.

## Task 3: Beta posterior

Compute the Beta(9,5) posterior from a Beta(2,2) prior and 7 wins in 10 trials. Report the mean and verify it sits between the prior mean and the MLE.

## Task 4: predictive interval

Draw 5000 samples from Beta(9,5). Report the mean and the 90% interval. Confirm the mean matches 9/14.

## Task 5: regularization path

Fit ridge-penalized Bradley-Terry on the lopsided three-item data for lambda in [0.01, 0.1, 0.5, 1, 2, 5, 10]. Report the A-B gap path.

## Task 6: reliability diagram

Simulate 2000 pairs with true gaps N(0, 1.2) and an overconfident model (gaps x 1.3). Bin predictions into 10 bins. Report the mean absolute gap.

## Task 7: MNAR and IPW

Simulate the C10 missingness toy. Report the full-sample, naive, and IPW gap estimates.

## Task 8: five-fold CV

Run five-fold CV on 500 comparisons with true gap 1.0. Report per-fold accuracies, mean, and sd.

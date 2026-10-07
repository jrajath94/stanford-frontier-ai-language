# U04 lab: information and design in code

Environment: Python 3 with NumPy and matplotlib. Seed 0. Keep outputs. Keys show verified numbers.

## Task 1: score zero

Plot the Bernoulli score for 7 wins in 10 trials. Report the zero crossing and the score values at p = 0.5 and p = 0.9.

## Task 2: curvature match

Overlay the quadratic with curvature I = 47.62 on the true log-likelihood. Report the max gap over p in [0.55, 0.85].

## Task 3: non-regular rate

Simulate the Uniform(0, 2) MLE (sample max) at n = 50, 200, 800, 3200, 300 trials each. Fit the log-log slope of mean error versus n. Report the slope.

## Task 4: two designs

Build the information matrices for design 1 (10 labels on one pair) and design 2 (5/5 split) at gap 0.5. Report determinants, A-optimality, and the null direction of design 1.

## Task 5: sample sizing

For Bernoulli p = 0.7, compute the n for SE <= 0.05 from the formula. Verify by simulation at that n.

## Task 6: query ranking

Rank candidate pairs with gaps [0, 0.5, 1.0, 2.0, 3.0] by Fisher information. Report the ranking and values.

## Task 7: greedy versus random

Reproduce the 30-step variance curves for greedy and random selection. Report end variances and the label savings ratio.

## Task 8: noise multiplier

Compute the relative information and the label multiplier 1/rel for q in [0, 0.1, 0.2, 0.3, 0.4]. Report both series.

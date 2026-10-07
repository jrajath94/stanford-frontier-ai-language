# U08 lab: inversion, style, and bounded rationality in code

Environment: Python 3 with NumPy and matplotlib. Seed 0. Keep outputs. Keys show verified numbers.

## Task 1: surrogate gap

quality = 0.8 - 0.6t, clickbait = t, S = quality + 2*clickbait, t in [0,1] on a 101 grid. Report argmax of G and of S, with G and S at both.

## Task 2: style/content split

Responses with (content, style): (0.9, 0.3) and (0.4, 0.9). Compute totals under additivity. Report which response wins and which component decided.

## Task 3: satisficing

100 options, utilities uniform[0,1], budget 10, threshold 0.8, 20000 trials, seed 0. Report P(find above threshold) and mean chosen utility.

## Task 4: bias fits

Simulate 2000 anchored choices: true gap 0, bias b = 0.619, seed 0. Fit the gap with and without the bias term. Report both estimates.

## Task 5: misspecification demo

5 options utils [1.0, 0.8, 0.6, 0.4, 0.2], random order, threshold 0.7, budget 3, 20000 trials, seed 0. Report choice shares.

## Task 6: MAP vs MLE

A beats B twice, B beats C once, N(0,1) prior on scores. Fit MAP by gradient ascent. Report gap_AB and gap_BC. State what MLE does.

## Task 7: shaping invariance

2-state MDP, gamma 0.9, Phi(goal)=1, Phi(start)=0.5. Compute Q under r and under r'. Report both vectors and the argmax.

## Task 8: debias table

True gap 0.5, observed margin 0.8. Compute corrected gaps for b_hat in {0.0, 0.3, 0.9}. Report each with its error.

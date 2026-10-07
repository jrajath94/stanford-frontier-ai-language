# U05 lab: RLHF and DPO in code

Environment: Python 3 with NumPy and matplotlib. Seed 0. Keep outputs. Keys show verified numbers.

## Task 1: dataset validation

Build a toy preference dataset of 5 records (x, y_w, y_l). Validate: every record has one prompt, two distinct responses, no empties. Report pass/fail per rule.

## Task 2: linear reward head

Four pairs with features [length, quality] for winner and loser (see keys). Train w by gradient ascent on the logistic margin loss, 2000 steps, lr 0.05. Report w and the four margins.

## Task 3: frontier trace

Rewards [2, 1, 0], uniform reference. For beta in [0.2, 0.5, 1, 2, 5, 20], compute the closed-form optimum, its KL, and expected reward. Report the frontier table.

## Task 4: closed form

Implement pi*(y) proportional to pi_ref(y) exp(r(y)/beta). Report the distributions for beta 5 and 0.5. Verify they sum to 1.

## Task 5: DPO loss

Implement the DPO loss for one pair with beta 0.5. Report the loss at implicit margins 2.0, 0.0, -1.0.

## Task 6: bias measurement

On a toy dataset where longer answers win 65% regardless of quality, compute the apparent gap in utils. Report it.

## Task 7: hacking monitor

Simulate proxy = 0.1t and true = 0.1t - 0.004t^2 for t in 0..50. Find the true peak step. Write the stopping rule: stop when the proxy rises for 5 straight steps while a human-eval probe falls twice.

## Task 8: agreement matrix

Five annotators, 200 pairs, true gap 0.5, 15% flips. Compute per-annotator agreement with the majority and the mean pairwise agreement. Report both.

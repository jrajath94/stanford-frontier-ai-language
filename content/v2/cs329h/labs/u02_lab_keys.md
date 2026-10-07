# U02 lab keys: execution-verified outputs

Seed 0, NumPy, float64.

## Task 1

Shares: [0.547, 0.331, 0.122], sum 1.0. Adding 10 to all utilities leaves them unchanged.

## Task 2

Simulated: [0.511, 0.308, 0.181]. Formula: [0.506, 0.307, 0.186]. Max gap 0.006, consistent with 1/sqrt(20000) Monte Carlo error.

## Task 3

Cycle detected: True.

## Task 4

Shares: [0.636, 0.234, 0.086, 0.032, 0.012]. Top option takes nearly two-thirds.

## Task 5

Max reconstruction error at rank 2: 0.0 (exact by construction).

## Task 6

r_new = [14.26, 6.17], item-4 prediction -12.42. Diagnosis: items 0 and 1 have nearly collinear factors, so two ratings leave the position almost undetermined. This is the C08 failure case in action: fold-in needs varied item factors.

## Task 7

Pooled 0.499, group 1 0.900, group 2 0.098. The pooled number represents nobody.

## Task 8

Pair coverage 3 of 3, no majority cycle, split-half win rates agree: all clauses pass on clean simulated data.

## Task 9

Max residual of the naive fit on the cycle: 0.30. Battery: cycle fails naive (residual 0.30), buses fail IIA (car error 0.17), pooling fails (gap 0.40), scale is unidentified. Each passes its repair (pairwise model, nested logit, mixture, scale fix).

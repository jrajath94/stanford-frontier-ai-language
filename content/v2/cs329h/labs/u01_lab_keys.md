# U01 lab keys: execution-verified outputs

Seed 0, NumPy, float64. Each output below was produced by running the task code. Reruns with the same seed reproduce them.

## Task 1

s_hat = 0.80, b_hat = -0.66, pooled naive fraction = 0.613. The pooled fraction mixes the two contexts. The two-parameter fit separates them.

## Task 2

s = 1.10, b = -0.87, wording coefficient = 0.57. True values were 1.1, -0.8, 0.5. All three recovered within noise.

## Task 3

Win-count matrix (rows/cols A, B, C):

```
[[0 2 0]
 [1 0 3]
 [0 0 0]]
```

Pair totals: A-B 3, B-C 3, A-C 0. Antisymmetry of pair totals holds.

## Task 4

Scores: [2.72, 2.03, -4.76] (sum zero). P(A beats B) = 0.666. C never wins, so its score is strongly negative.

## Task 5

MLE gap = 1.61, minimum NLL = 2.703, closed form log(5) = 1.609. Grid and closed form agree.

## Task 6

Estimated gaps at q = 0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30: 1.49, 1.33, 1.11, 0.95, 0.88, 0.65, 0.53. Monotone shrinkage toward zero.

## Task 7

Train log-loss at n = 20, 50, 100, 200, 500, 1000: 0.562, 0.641, 0.619, 0.598, 0.606, 0.595. Test log-loss: 0.579, 0.591, 0.583, 0.579, 0.580, 0.579. Train is optimistic at small n. Both settle near 0.58.

## Task 8

Annotators kept (gold >= 0.85): 0, 1, 2. Dropped: 3, 4. Quality-control overhead: 15% (10% repeats + 5% gold).

## Task 9

Pooled accuracy = 0.747. Group 1 = 0.910, group 2 = 0.583. Gap standard error = 0.023. The gap is far above noise.

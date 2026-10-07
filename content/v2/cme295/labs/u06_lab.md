# U06 lab , reasoning RL in code

Prerequisites: the U06 lesson. Runner: `u06_lab_run.py` (numpy, CPU,
deterministic). Work each task by hand first, then verify with the
runner. Answers and verified outputs: `u06_lab_key.md`.

## Task 1 , group normalization

1. Compute normalized advantages for rewards [1, 1, 0, 0] and
   [3, 1, 1, 1].
2. State what the lone winner gets in each case.

## Task 2 , pass@k

1. Compute pass@k for p = 0.2, k in {1, 2, 4, 8, 16}.
2. State where doubling k stops paying.

## Task 3 , compute allocation

Toy model: p(T) = 1 - exp(-T/60), k = (100 - T)/10,
score = 1 - (1 - p)^k. Curves are hypothetical, labeled as such.

1. Compute the score for T in {20, 40, 60, 80}.
2. State the flat-top zone.

## Task 4 , verifier gap

1. Strict grade 0.41, verifier pass 0.93. Compute the hack gap.
2. FP rate 0.18 on 200 wrong answers. How many false passes?
3. State the decision.

## Task 5 , ablation read

Deltas vs full config: group norm -0.17, KL -0.04, small G -0.09.
Seed std 0.02.

1. Compute z-scores for the three deltas.
2. State which deltas are real and which are suggestive.

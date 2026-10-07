# U03 lab , generation in code

Prerequisites: the U03 lesson. Runner: `u03_lab_run.py` (numpy, CPU,
deterministic). Work each task by hand first, then verify with the
runner. Answers and verified outputs: `u03_lab_key.md`.

## Task 1 , temperature

Logits [3, 1, 0, -1].

1. Compute the distributions at T = 0.5, 1.0, 2.0 and their
   entropies.
2. State the limits T -> 0 and T -> infinity.

## Task 2 , truncation

Distribution [0.5, 0.3, 0.15, 0.05].

1. Apply top-k = 2 and renormalize.
2. Apply top-p = 0.8 and renormalize.
3. State when the two coincide.

## Task 3 , MoE routing

8 experts, top-2, seed 11 (as in the runner).

1. State the chosen experts and their renormalized weights.
2. State the active compute fraction.

## Task 4 , majority vote

Votes [391, 391, 382, 391, 390].

1. State the winner and the count.
2. Compute the majority accuracy for p = 0.6, N = 5.

## Task 5 , context memory wall

1. Compute fp16 attention-score memory per layer for
   (T = 32768, h = 8) and (T = 131072, h = 32).
2. State which wall hits first: memory or position codes.

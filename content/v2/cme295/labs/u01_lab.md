# U01 lab , foundations in code

Prerequisites: the U01 lesson. Runner: `u01_lab_run.py` (numpy, CPU,
deterministic). Work each task by hand first, then verify with the
runner. Answers and verified outputs: `u01_lab_key.md`.

## Task 1 , BPE merges by hand

Corpus: low x2, lower, lowest, new, newer.

1. Count all adjacent character pairs with frequencies. State the top
   pair and its count.
2. Apply merge 1 everywhere. Write the new splits.
3. Recompute pair counts. State merge 2 and its count.
4. Compute fertility (tokens per word) before and after each merge.

## Task 2 , masked softmax

Raw 4x4 scores are in the runner.

1. Build the causal mask (-inf above the diagonal).
2. Apply masked softmax. Verify every row sums to 1 and the future
   triangle is exactly 0.
3. State row 0. Explain why it must be [1, 0, 0, 0].

## Task 3 , gradient scales

1. Compute the gradient scale rho^steps for (0.5, 10), (0.8, 20),
   (1.2, 10).
2. Label each case vanishing, healthy, or exploding.
3. Explain in one sentence why clipping fixes the third case but not
   the first.

## Task 4 , LSTM cell trace

c_0 = 1.0. Step 1: f = 1.0, i*g = 0.0. Step 2: f = 0.5, i*g = 0.6.

1. Trace c_1, c_2.
2. Compute dc_2/dc_0 and compare with 0.5^2.
3. State what the comparison proves about the carousel.

## Task 5 , shapes and parameter counts

d = 64, h = 4, V = 1000, B = 2, T = 8.

1. State the shapes of the attention scores and the logits.
2. Count per-layer parameters (attention + FFN with 4d hidden +
   LayerNorms) and embedding parameters.
3. Identify which term dominates as T grows: the T^2 scores or the
   d^2 weights.

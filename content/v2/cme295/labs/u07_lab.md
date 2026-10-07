# U07 lab , retrieval and agents in code

Prerequisites: the U07 lesson. Runner: `u07_lab_run.py` (numpy, CPU,
deterministic). Work each task by hand first, then verify with the
runner. Answers and verified outputs: `u07_lab_key.md`.

## Task 1 , recall and precision

1. Compute recall and precision for (R, K, H) = (8, 20, 6)
   and (8, 40, 7).
2. State the tradeoff.

## Task 2 , hybrid fusion

BM25 = [0.9, 0.3, 0.5], dense = [0.4, 0.8, 0.5].

1. Fuse with alpha = 0.5 and alpha = 0.8. Rank the docs.
2. RRF with k = 60: doc X ranks (1, 4), doc Y ranks (2, 2).
   Who wins?

## Task 3 , context pack

W = 8000, S = 600, R = 1000. Docs = [1500, 1200, 900, 800, 700].

1. Do all 5 fit? Show the arithmetic.
2. Add a 6th doc of 2000 tokens. What gets cut?

## Task 4 , stopping rule

Trace actions: ["search", "search", "read", "read", "read"].

1. With cap 10 and no-repeat rule, at which step does the
   guard fire?
2. State the runaway signature.

## Task 5 , failure separation

100 wrong answers: 47 retrieval, 31 model, 22 tool.

1. Print the histogram shares.
2. State the first fix and why.

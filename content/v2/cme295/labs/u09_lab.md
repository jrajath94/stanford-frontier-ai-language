# U09 lab , synthesis in code

Prerequisites: the U09 lesson. Runner: `u09_lab_run.py` (numpy, CPU,
deterministic). Work each task by hand first, then verify with the
runner. Answers and verified outputs: `u09_lab_key.md`.

## Task 1 , diffusion unmasking

Confidences [0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], 8 tokens.

1. Step 1 unmasks the top 4. How many stay masked?
2. Step 2 unmasks 2 more (next most confident). How many
   stay masked?

## Task 2 , tiny forward pass

E = [[1,0],[0,1],[1,1],[-1,0]], x = [0,1], identity Q/K/V,
Wo = [[1,0,-1,0],[0,1,0,-1]].

1. Compute the attention matrix (softmax of q k^T / sqrt(2)).
2. Verify rows sum to 1.
3. Compute logits and the argmax per position.

## Task 3 , objective coverage

24 objectives, 21 mapped.

1. Compute coverage.
2. State what the 3 orphans become.

## Task 4 , keystones

Downstream counts: attention 15, policygrad 9, retrieval 7.

1. Rank the keystones.
2. State the study implication.

## Task 5 , architecture decision

AR [3,2,3], diffusion [2,3,1] on (quality, latency, maturity).

1. Equal weights: who wins?
2. Weights [0.2, 0.6, 0.2]: who wins?

# U02 lab , attention variants in code

Prerequisites: the U02 lesson. Runner: `u02_lab_run.py` (numpy, CPU,
deterministic). Work each task by hand first, then verify with the
runner. Answers and verified outputs: `u02_lab_key.md`.

## Task 1 , FLOP ratios

1. Compute exact vs linear-attention FLOPs for T = 4096, d = 512
   (feature width d).
2. Repeat for T = 1024, d = 512, m = 256.
3. State how the ratio scales with T.

## Task 2 , RoPE by hand

1. Rotate x = (1, 0) by angle m*theta = 1.0. State the result.
2. Verify the relative property: R(1) R(2)^T == R(-1) to 1e-12.
3. Verify the norm is preserved.

## Task 3 , KV cache bytes

32 layers, 32 query heads, d_k = 128, T = 2048, fp16.

1. Compute the cache for MHA (32 KV heads), GQA-8, MQA.
2. State the MQA/MHA ratio.

## Task 4 , ALiBi penalty

Scores [5, 0, 0, 0], slope m = 0.5, distances [0, 1, 2, 3].

1. Apply the penalty and softmax. State the weights.
2. State the penalty factor at distance 3.

## Task 5 , attention sink

Scores [3, 0, 0].

1. Compute the softmax weights and the sink mass on position 0.
2. Explain in one sentence why the sink forms.

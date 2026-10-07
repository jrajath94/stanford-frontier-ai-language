# Lab key , U05

Execution-verified outputs from `labs/u05_lab_run.py`, run 2026-10-06
(CPython, numpy 1.26.4, CPU). Re-run the script to confirm.

## T1 , attention toy

Scores (0.707, 0.000, -0.707). Weights (0.576, 0.284, 0.140), sum
1.000. Output (0.872, 0.568) = 0.576 v1 + 0.284 v2 + 0.140 v3. The
query (1,0) favors the parallel key and downweights the opposite
one.

## T2 , scale demo

Unscaled: max weight 0.866, entropy 0.571 (near one-hot). Scaled by
1/8: max weight 0.218, entropy 1.974 (spread). The scale is the
difference between routing and collapsing.

## T3 , causal mask

Row sums (1, 1, 1, 1). Upper-triangle max 0.0 exactly. Every row is
a distribution over its own past and present only.

## T4 , MHA shapes

Per-head scores (8, 8), concat (8, 32), W_o (32, 32). Four heads at
d_k = 8 tile the d = 32 width.

## T5 , layer norm

Mean -0.0, variance 0.999995 (the eps). Zero sublayer: the residual
block returns its input unchanged, the identity path is exact.

## T6 , KV cache

No cache 23936, cached 2176, ratio 11.0x. The cache trades a linear
memory pad for a factor-of-n time saving.

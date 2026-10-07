# Answer key , U05 Attention and transformers

Attempt the exercises before reading. Ladders are oral: answer aloud,
then check.

## Remediation

R1. (1,0).(1,0) = 1, (1,0).(0,1) = 0, (1,0).(-1,0) = -1. Similarity
is the dot product, attention scores inherit the geometry.
R2. (0.576, 0.284, 0.140), sum 1.000. Every attention row is a
distribution over positions.
R3. (n, d) @ (d, d_k) = (n, d_k). (n, n) @ (n, d_v) = (n, d_v).
Typecheck every product.

## Breadth

A1. Q asks, K indexes, V carries. Scores = QK^T/sqrt(d_k), weights =
softmax, out = weights @ V. Toy: scores (0.707, 0, -0.707), weights
(0.576, 0.284, 0.140), out (0.872, 0.568).
A2. Raw dot variance = d_k, at d_k = 64 the softmax saturates to
one-hot and gradients die. Dividing by sqrt(d_k) restores unit
variance. Toy: max 0.866/entropy 0.571 unscaled, 0.218/1.974 scaled.
A3. h heads, d_k = d/h, each with its own QKV, concat and project
with W_o. Same FLOPs as one wide head, h independent routings.
Shapes (n=8,d=32,h=4): scores (8,8), concat (8,32), W_o (32,32).
A4. Causal: -inf above the diagonal, added to scores before softmax.
Padding: -inf at pad columns. Mask after softmax leaks (the future
shaped the distribution). Toy: row sums 1, upper max 0.0.
A5. Residual: y = x + F(x), gradient keeps the identity path.
Layer norm: per-vector mean 0 var 1. Pre-norm is the stable default
past ~12 layers.
A6. Attention is permutation-equivariant, so order needs an explicit
signal. Sinusoid: fixed waves, learned: a table to n_max. PE(0) =
(0,1,0,1), PE(1) = (0.841, 0.540, 0.010, 1.000) at d = 4.
A7. Per-position two-layer MLP: (n,d) -> (n,4d) -> (n,d), 8d^2
params per layer (2/3 of the block). Toy d=32: 8320 params.
A8. Encoder-only: no mask, understanding. Decoder-only: causal mask,
generation. Enc-dec: cross-attention (Q decoder, K/V encoder),
conditioned generation.
A9. One layer (n=8,d=32,h=4): X (8,32) -> QKV (8,32) -> split
(4,8,8) -> scores (4,8,8) -> heads (4,8,8) -> concat (8,32) -> W_o
(8,32) -> FFN hidden (8,128) -> out (8,32).
A10. Per layer: attention 2n^2d, FFN 16nd^2, projections 4nd^2.
Toy n=512,d=1024: 0.54 + 8.59 + 2.15 = 11.3 GFLOP. Training
parallel, inference serial, 4x context = 16x attention FLOPs.
A11. Cache past K/V (frozen under causality): step cost O(t^2d) ->
O(td), total O(n^3d) -> O(n^2d). Toy: 23936 -> 2176, ratio 11.0x.
Price: 2Lnd floats (4.0 GiB at 32 layers, 8k, 4096, fp16).
A12. The six lines: PE add, LN, MHA, residual, LN, FFN, residual,
every shape asserted.

## Oral ladders

L1 (QKV). Define the three roles. Compute the toy. Derive the convex
blend. Diagnose the explanation failure (weights are one factor).
Design the frozen-values test.

L2 (scale). Derive Var = d_k. Compute both regimes. Explain
saturation. Diagnose the stalled tiny transformer. Design the
scale ablation.

L3 (MHA). Write the head split. Trace the five shapes. Derive 4d^2
params. Diagnose head redundancy. Design the pruning test.

L4 (masks). Write both masks. Prove mask-before-softmax (exp(-1e9)
= 0, post-softmax renormalizes a shaped distribution). Compute the
toy. Diagnose the peek (validation looks too good). Design the
mask-order test.

L5 (residual/norm). Write both formulas. Compute the toy norm.
Derive dy/dx = I + dF/dx. Diagnose the no-warmup post-norm stall.
Design the placement test.

L6 (positions). Write the sinusoid. Compute PE(1). Prove
permutation equivariance (scores depend on content pairs only).
Diagnose the chance reversal score. Design the reversal test.

L7 (FFN). Write the two layers. Count 8320. Derive the 2/3 param
share. Diagnose the removal (capacity collapse). Design the
expansion sweep.

L10 (cost). Write the three terms. Compute the toy. Derive the n =
8d crossover. Diagnose flat step time (bandwidth bound). Design the
scaling fit.

L11 (KV cache). Define the cache. Compute 23936/2176. Derive the
2Lnd memory price. Diagnose the long-context OOM. Design the timing
test.

## Exercises

E1. `attention` reproduces scores, weights, output.
E2. Random Q, K: every row sums to 1 (assert allclose).
E3. `scaled_scores` reproduces the entropy pair 0.571/1.974.
E4. d_k = 256: unscaled score variance ~256, scaled ~1.
E5. `mha` asserts all five shapes.
E6. h = 1 output matches single-head attention with d_k = d.
E7. Both masks: row sums 1, upper triangle exactly 0.
E8. Post-softmax masking gives different weights than pre-softmax
masking on the toy (report both).
E9. `layer_norm` gives mean -0.0, var 0.999995 on (1,2,3,4,5).
E10. F = 0: block output equals input exactly.
E11. `sinusoid` reproduces PE(0) and PE(1).
E12. Permute input without PE: output permutes identically
(equivariance holds numerically).
E13. `ffn` asserts (8,32) -> (8,128) -> (8,32).
E14. d = 64: 16640 + 16448 = 33088 params.
E15. `cross_attention` asserts scores (6,10), out (6,32).
E16. Table: encoder-only none, decoder-only causal, enc-dec
causal (decoder) + padding (cross).
E17. `layer_trace` reproduces all 12 shapes.
E18. Transposed scores: shapes pass, causal pattern fails (row 1
no longer (1,0,0,0)).
E19. `flops` reproduces 0.54/8.59/2.15/11.3 GFLOP.
E20. 2n^2d = 16nd^2 gives n = 8d = 8192 at d = 1024.
E21. `decode_step` matches full recompute exactly.
E22. 2 x 32 x 8192 x 4096 x 2 bytes = 4,294,967,296 bytes = 4.0
GiB.
E23. `transformer_layer` cold: shape (4,16), finite, one backward
step no nan.
E24. With PE the permutation check fails (order matters), without
PE it passes (equivariance).

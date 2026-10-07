# Interview key , U05

## Breadth

A1. Q asks, K indexes, V carries. Three projections let asking,
indexing, and carrying use different geometry.
A2. Raw dot variance = d_k, at d_k = 64 softmax saturates to one-hot
and gradients vanish. The scale restores unit variance.
A3. h independent (n,n) routings in separate subspaces at the same
total FLOPs, the output projection mixes the specialists.
A4. Added to scores before softmax (-inf -> 0 weight). After softmax
the future already shaped the distribution, renormalizing leaks.
A5. Residual: the identity gradient path across depth. Layer norm:
per-vector mean 0 var 1, stops scale drift.
A6. Attention is permutation-equivariant (scores depend on content
pairs only), without positions "dog bites man" = "man bites dog".

## Deep ladders

L1. (1) S = QK^T/sqrt(d_k) (n,n), P = softmax(S), out = PV (n,d_v).
(2) (0.707,0,-0.707), (0.576,0.284,0.140), (0.872,0.568). (3) Softmax
rows sum to 1, so each output row is a convex combination of value
rows. (4) Values carry the content, same weights on different values
give different outputs. (5) Freeze values to position codes, swap
inputs, check whether high-weight positions predict the change.

L2. (1) Sum of d_k independent unit-variance products. (2) 0.571 ->
1.974 entropy. (3) exp(-1e9) underflows to 0, row renormalizes over
allowed positions. (4) The old mask leaked future context, the fix
removed the peek, so the honest score is worse. (5) Train with the
mask deliberately after softmax: it should look better than the
correct one.

## Analytical

A7. Post-norm puts the norm after the residual add, so early in
training the gradient path through 24 layers is unscaled, pre-norm
keeps the identity path clean. Cheapest fix: learning-rate warmup
(or switch to pre-norm).
A8. The 2n^2d attention term: 4x context = 16x attention FLOPs (14x
observed, rest is overhead). Cuts: KV cache (already assumed? then
flash-style tiling), shorter context, or a linear-attention
variant.

## Implementation/debugging

A9. (1) Check temperature/top-k: too low -> raise temperature. (2)
Check the KV cache: stale or misaligned cache -> verify against
recompute. (3) Check the causal mask: leak -> fix order. (4) Check
training data: repetitive data -> dedupe. (5) Check repetition
penalty: add one.

## Changed-constraint

A10. (1) KV cache quantization (fp16 -> int8/int4). (2) Smaller
batch / shorter context. (3) Eviction (keep a window) or offload to
CPU, accepting the speed hit.
A11. Mask: none needed (bidirectional ok) or keep padding mask.
Readout: a [CLS]-style pooled vector instead of last-token logits.
Loss: classification CE over labels instead of next-token CE.

## Research critique

A12. Steelman: weights are the only input-dependent routing, high
weight means the output blended that position's value heavily.
Counterexample: identical weights on different values give
different outputs, and different weights can give the same output,
weights are one factor, not the explanation. Faithfulness needs
intervention, not inspection.

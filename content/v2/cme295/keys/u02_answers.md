# U02 answer key

Answers to the lesson exercises. Do not open before attempting.

## C01

- R1. Exact: T^2 d = 1024^2 * 512 = 5.37e8. Approx: T m d =
  1024 * 256 * 512 = 1.34e8 (plus the m x d terms). Ratio ~4.0.
- R2. The normalizer is sum_v phi(q) . phi(k_v) = phi(q) .
  sum_v phi(k_v): the sum runs over keys (the V side). Summing
  phi(Q) would normalize over queries, which is meaningless.
- R3. Negative features allow negative "weights" and a zero
  normalizer. The attention-as-distribution reading breaks, use
  nonnegative features (exp-based) or a different normalizer.

## C02

- R4. 4 * 128^2 = 65536 (Q, K, V, output projections).
- R5. q . k ~ N(0, d_k), without the 1/sqrt(d_k) the softmax input
  has std sqrt(d_k), saturates on the max entry, and gradients
  vanish at init.
- R6. d_k = 768 / 12 = 64.

## C03

- R7. MQA: kv_heads = 1. Bytes = 2 * 24 * 1 * 64 * 4096 * 2 =
  25,165,824 = 24 MiB.
- R8. MHA with 16 heads: 16x the KV heads, so 16x the bytes.
  Ratio 16.0.
- R9. The query projections and the per-head score math are
  unchanged, only the K/V projections shrink (h -> 1) and the cache
  traffic falls. Arithmetic is dominated by Q and the scores.

## C04

- R10. 2 * 32 * 4 * 128 * 2048 * 2 = 134,217,728 = 128 MiB.
- R11. G = h = 32: every query head gets a private KV head, which
  is MHA exactly.
- R12. A view shares storage: materializing h copies of each KV
  head would multiply the cache by h/G for no reason. The view
  keeps the C03 byte counts honest.

## C05

- R13. lambda_0 = 2 pi * 10000^0 = 2 pi ~= 6.28.
- R14. 512 * 768 = 393,216 parameters.
- R15. The model learns separate subspaces for content and
  position, addition keeps the shape (T, d) and all downstream
  shapes unchanged. Concat would add d dimensions everywhere.

## C06

- R16. Angle = 2 * 0.5 = 1.0 rad. R(1)(0,1) = (-sin 1, cos 1) =
  (-0.841, 0.540).
- R17. R^T R = I, so |Rx|^2 = x^T R^T R x = |x|^2. Rotations are
  orthogonal.
- R18. V carries content, not relations. Rotating V injects a
  position-dependent distortion into the values with no
  corresponding benefit, the relative property lives in the QK
  dot product.

## C07

- R19. Weight ratio = exp(-1) / exp(-4) = exp(3) ~= 20.1. Distance
  1 gets 20x the weight of distance 4.
- R20. The bias must shift logits before the softmax normalizes,
  after softmax it would need renormalization and the exponential
  penalty reading breaks.
- R21. Offsets past 128 share log-spaced buckets: exact large
  distances are lost, only the rough magnitude survives.

## C08

- R22. (2, 4, 16, 32): batch 2, 4 heads, 16 target queries, 32
  source keys.
- R23. The encoder runs once over the source, the decoder runs one
  step per target token (its self-cache grows).
- R24. BERT is bidirectional: its representations already contain
  future tokens, so left-to-right generation would leak the target.

## C09

- R25. 512 * 0.15 = 76.8, about 77 positions.
- R26. [MASK] never appears at fine-tune time, the 80/10/10 rule
  (mask/random/unchanged) forces the model to also represent
  unmasked tokens well.
- R27. Higher temperature softens the teacher distribution: more
  dark knowledge (relative confidences), less peak signal on the
  top class.

## C10

- R28. 2 * 12 * 4 * 64 * 1024 * 2 = 12,582,912 = 12 MiB.
- R29. Without cache, step t projects t tokens: total ~T^2/2
  token-projections. With cache: T. Ratio ~T/2 = 256 at T = 512
  (projection FLOPs).
- R30. The new query must dot all T cached keys: one dot per past
  token is unavoidable. The cache removes the repeated K/V
  projections, not the attention itself.

## C11

- R31. Per layer 2 * 2048^2 * 1024 = 8.59e9 (QK^T + AV). Times 24
  layers: 2.06e11 FLOPs.
- R32. 8 * 8192^2 * 2 bytes = 1,073,741,824 = 1 GiB (fp16).
- R33. FlashAttention computes the same softmax attention, it
  tiles the work into SRAM to cut memory IO. FLOPs are identical,
  wall time is not.

## C12

- R34. Softmax [0.909, 0.045, 0.045]. Sink mass 0.909 on
  position 0.
- R35. The residual adds x directly to the output, so each token
  keeps a private channel that attention averaging cannot wash
  out. Rank collapse needs the attention path to dominate, the
  residual prevents that.
- R36. RoPE base: sharp cliff at a multiple of trained length,
  position-sensitive errors. Data shift: gradual, domain-dependent
  degradation. Run a length-matched in-domain control first.

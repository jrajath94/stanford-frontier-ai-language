# U02 interview key

Minimum sufficient explanation, strong answer, common red flags,
scoring rubric, remediation per question. Keep it closed-book.

## Breadth

- B1. Strong: the K and V heads are shared (one head), each query
  head keeps its private Q projection. Red flag: "everything is
  shared". Rubric: shared vs private (2). Remediation: C03.
- B2. Strong: pair (x0, x1) -> R(m theta_i)(x0, x1), property:
  (R(mt)q).(R(nt)k) = q^T R((n-m)t) k. Red flag: rotation applied to
  V. Rubric: formula (1), property (1). Remediation: C06.
- B3. Strong: q.k has std sqrt(d_k) at init, the scale restores unit
  variance so the softmax starts diffuse. Red flag: "it is a
  convention". Rubric: variance argument (2). Remediation: C02.
- B4. Strong: stored K/V of past tokens per layer, GQA changes
  kv_heads (G). Red flag: "it stores queries". Rubric: content (1),
  term (1). Remediation: C10.
- B5. Strong: learned stops at T_max (no trained rows beyond),
  sinusoid extends by formula but keeps absolute residue, neither
  truly extrapolates. Red flag: "sinusoid extrapolates perfectly".
  Rubric: both limits (2). Remediation: C05.
- B6. Strong: encoder (unmasked/bidirectional), decoder (causal),
  encoder-decoder (causal self + full cross). Red flag: "BERT is
  causal". Rubric: three masks (2). Remediation: C08.

## Deep ladders

- L1. Strong path: angle = m * theta_i, theta_i = base^{-2i/d},
  (0.5403, 0.8415), proof via R^T R composition, rope reshapes to
  pairs and rotates, checks m = 0 identity, norm, 1e-12 relative
  property, ALiBi penalizes distance without angles, collapse past
  trained length = angle combinations unseen, "extrapolates" needs
  the base/interpolation qualifier, experiment = sweep base, find
  the perplexity cliff. Rubric: 2 per rung, 10 total.
- L2. Strong path: per layer K/V (B, kv_heads, T, d_k), 32 MiB,
  recompute costs O(T^2) projections vs O(T) with cache,
  decode_step appends and matches uncached forward to 1e-5, MHA
  1024 / GQA-8 256 / MQA 32 MiB, OOM at long context = cache,
  fix = fewer KV heads or shorter T, "more heads" critiqued via
  d_k starvation and cache, crossover = measure latency
  cache vs recompute over T. Rubric: 2 per rung, 10 total.

## Analytical exercises

- E1. Strong: per-layer 2 * 4096^2 * 1024 = 3.44e10 FLOPs, scores
  fp16 = 16 * 4096^2 * 2 = 512 MiB. Crossover: T^2 d = 4 T d^2 ->
  T = 4d = 4096. Red flag: forgetting the factor 2 or the head
  count. Rubric: FLOPs (2), bytes (2), crossover (1).
  Remediation: C11.
- E2. Strong: cache = 2*32*4*128*8192*2 = 0.5 GiB. B = 1: 14.5 GB
  fits. B = 4: 2 GiB cache, 16 GB total, fits. Red flag: dropping
  the factor 2 for K and V. Rubric: bytes (2), both verdicts (2).
  Remediation: C10.

## Implementation/debug

- D1. Strong order: (1) verify G matches and divides h, (2) check
  the pooling (mean over the right heads), (3) compare logits of
  converted vs original on a canary set. Culprit: mean-pooling
  without the recovery continued-training run. Fix: short
  continued training, or train GQA natively. Asserts: cache bytes
  equal the formula, GQA(G=h) reproduces MHA logits exactly.
  Red flag: "retrain from scratch" as the first move. Rubric:
  checks (3), culprit (2), asserts (2). Remediation: C04.

## Changed-constraint scenarios

- S1. Strong: (1) larger RoPE base (breaks trained-length angle
  calibration, metric: perplexity at 128k vs 4k), (2) position
  interpolation, scale positions by 4k/128k (breaks fine-grained
  local resolution, metric: passkey retrieval at 128k). Red flag:
  "it just works". Rubric: two fixes (2), assumptions (2),
  metrics (1).
- S2. Strong: one KV head = 32 MiB exactly -> MQA (G = 1). Risk:
  quality drop, worst on retrieval-heavy tasks. Monitor: canary
  eval set (retrieval + long context) before/after, plus live
  perplexity. Red flag: no arithmetic. Rubric: variant + math (3),
  risk + monitor (2).

## Research-critique

- R1. Strong: claim = same expressivity at linear cost, counterexample
  = unbiased with high variance at small m diverges in practice,
  experiment = fix T and a copy task, sweep m, find the m where
  accuracy matches exact attention, falsification = no m below the
  exact-attention FLOP count reaches parity (then the claim fails
  on that task). Red flag: "unbiased means equal". Rubric: steelman
  (2), counterexample (2), experiment + falsification (3).
  Remediation: C01 item 11.

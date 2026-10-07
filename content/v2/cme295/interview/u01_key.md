# U01 interview key

Minimum sufficient explanation, strong answer, common red flags,
scoring rubric, remediation per question. Keep it closed-book.

## Breadth

- B1. Strong: an id is an integer index, arbitrary and unordered, an
  embedding is the learned dense vector at that row, and geometry
  carries similarity. Red flag: "embeddings are one-hot". Rubric: both
  defined and distinguished (2), one defined (1). Remediation: C03.
- B2. Strong: frequent pairs give the most compression per merge, the
  greedy choice maximizes tokens saved per vocabulary slot. Red flag:
  "random order works the same". Rubric: compression argument (2).
  Remediation: C02.
- B3. Strong: h_t = tanh(W x_t + U h_{t-1} + b), W input-to-hidden, U
  hidden-to-hidden, b bias. Red flag: missing U or the nonlinearity.
  Rubric: equation plus roles (2). Remediation: C05.
- B4. Strong: it multiplies the previous cell c_{t-1} elementwise,
  dc_t/dc_{t-1} = f_t, so f_t near 1 preserves gradients (the
  carousel). Red flag: "it multiplies the input". Rubric: equation
  and gradient consequence (2). Remediation: C06.
- B5. Strong: -inf scores become zero weight with rows still summing
  to 1, masking after softmax steals mass or leaks gradients. Red
  flag: "it is just convention". Rubric: softmax math (2).
  Remediation: C09.
- B6. Strong: scores (B, h, T, T), logits (B, T, V). Red flag:
  scores as (B, T, d). Rubric: both exact (2). Remediation: C10.

## Deep ladders

- L1. Strong path: pair count = sum over words of freq * adjacent
  occurrences, merge 1 = (o, w) count 4 (tie with (l, o) at
  4, broken by the runner's lexicographic-max rule, (l, o)
  accepted when the rule is stated), rank order makes encoding
  deterministic, rewrite pass O(corpus), word-level fails on OOV,
  cross-space merge bug = broken pretokenizer, fixed vocab criticized
  via multilingual fertility, experiment = train on code vs prose,
  compare cross fertility. Red flags: merges in discovery order,
  "vocabulary size does not matter". Rubric: 2 per rung, 10 total.
- L2. Strong path: M[i,j] = -inf for j > i, softmax [0.269, 0,
  0.731], exp(-inf) = 0 keeps row sums at 1, checks = row sums 1 and
  triangle 0, post-softmax zeroing mis-scales, disabled mask =
  train/eval gap with great train loss, "the mask ran" needs the
  row-0 test, experiment = train with/without mask, compare
  generation. Rubric: 2 per rung, 10 total.

## Analytical exercises

- E1. Strong: pairs: (t,h): 150, (h,e): 150, (e,</w>): 110... (with
  counts the:100, then:40, their:10): (t,h) = 100+40+10 = 150,
  (h,e) = 150, merge 1 = (t,h) or (h,e), tie broken by rule (first
  max). Fertility before: (3*100 + 4*40 + 5*10)/150 = 510/150 = 3.4
  chars/word, after merging (t,h): (2*100 + 3*40 + 4*10)/150 =
  360/150 = 2.4. Red flag: ignoring frequencies. Rubric: counts (2),
  fertility (2). Remediation: C02, lab task 1.
- E2. Strong: d_k = 64, scores (4, 8, 1024, 1024) = 33.5M floats =
  134 MB fp32, embeddings 32000*512*2 = 32.8 MB. At T = 8192:
  scores (4, 8, 8192, 8192) = 2.1B floats = 8.6 GB >> weights.
  Red flag: "weights dominate". Rubric: shapes (2), bytes (2),
  conclusion (1). Remediation: C10.

## Implementation/debug

- D1. Strong order: (1) mask test (row sums, triangle), (2) shape
  trace, (3) data pipeline (train/inference tokenizer match, label
  shift by one). Most likely: mask disabled or applied after
  softmax (great train loss = reads the future, garbage generation =
  no future at inference). Three-line test:
  A = softmax(randn(4,4) + causal_mask(4)), assert row sums 1,
  assert future triangle 0. Red flag: "more data" as the fix.
  Rubric: ordered checks (3), culprit (2), test (2). Remediation:
  C09, C12.

## Changed-constraint scenarios

- S1. Strong: retrain BPE on the new language with the same V,
  breaks: all ids shift, old embeddings misalign, detect with the
  roundtrip test plus a canary eval set before/after. Alternative:
  byte fallback keeps V fixed but fertility explodes, state the
  trade. Red flag: "just add tokens" (violates the constraint).
  Rubric: proposal (2), breakage (2), detection (1).
- S2. Strong: embeddings = 50000*1024*2 = 102.4 MB. Fits in 256 MB
  alone, but barely with the rest of the model. Changes: (1) smaller
  d or V (cost: capacity), (2) quantized int8 embeddings (cost:
  quality, needs dequant). Red flag: no arithmetic. Rubric: bytes
  (2), fit verdict (1), two changes with costs (2).

## Research-critique

- R1. Strong: claim = high weight means the model used that token,
  counterexample = different weight patterns, same output (weights
  are not uniquely determined by behavior), experiment = perturb
  inputs to flip weights without changing the output (or vice
  versa), falsification = output changes while weights stay fixed,
  or weights change with output fixed, breaking the "explanation"
  reading. Red flag: "attention is explanation, period". Rubric:
  steelman (2), counterexample (2), experiment + falsification (3).
  Remediation: C07 item 11.

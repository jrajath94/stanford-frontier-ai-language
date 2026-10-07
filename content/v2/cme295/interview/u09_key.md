# U09 interview key

Minimum sufficient explanation, strong answer, common red flags,
scoring rubric, remediation per question. Keep it closed-book.

## Breadth

- B1. Strong: start fully masked, unmask the most confident
  subset per step, refine in parallel. Autoregressive writes
  left to right, one token per step. Red flag: "diffusion
  adds noise at inference". Rubric: mechanism (1), contrast
  (1). Remediation: C01.
- B2. Strong: objectives pointed at lesson homes plus
  assessments. An orphan has no home. Red flag: "a study
  list". Rubric: definition (1), orphan (1). Remediation:
  C03.
- B3. Strong: bands, contamination, saturation, gaming,
  construct validity. Red flag: naming three. Rubric: five
  (2). Remediation: C12 (U08).
- B4. Strong: the concept whose removal collapses the most
  downstream material, most transitive dependents. Red
  flag: "the hardest concept". Rubric: definition (2).
  Remediation: C07.
- B5. Strong: data match, latency, cost, shadow, rollback.
  Red flag: wrong order. Rubric: five in order (2).
  Remediation: C11.
- B6. Strong: define, toy, derive, implement/complexity,
  compare, debug, critique, design. Red flag: missing
  rungs. Rubric: eight (2). Remediation: C12.

## Deep ladders

- L1. Strong path: schedule defined (how many unmask per
  step), top-4 unmasked by confidence, re-masking gives
  wrong tokens another chance with more context, k = 1
  reduces to order-free serial generation, speculative
  decoding drafts parallel but verifies serial while
  diffusion refines parallel, collapse at few steps means
  miscalibrated confidence, "free" ignores full-sequence
  cost per step, experiment = quality vs step count, find
  the knee. Rubric: 2 per rung, 10 total.
- L2. Strong path: embed, q/k/v, scores, softmax, weighted
  sum, logits, rows [0.67, 0.33] and [0.33, 0.67], sqrt(d)
  keeps dot products in the softmax sweet spot, row sums
  equal 1, wrong logit shape means a transposed W_o,
  the toy teaches connection not systems, experiment = add
  the causal mask, watch position 1 change. Rubric: 2 per
  rung, 10 total.

## Analytical exercises

- E1. Strong: coverage 24/30 = 0.80. Allocate: 4 hours to
  the 2 L8 orphans (diffusion, high risk times high
  weight), 4 hours to keystone review (attention, RL),
  2 hours to the remaining orphans. Red flag: equal hours
  per objective. Rubric: coverage (1), allocation (2),
  justification (1). Remediation: C03, C07.
- E2. Strong: A: 3 (B, C, D). B: 1 (D). C: 0. Keystone is
  A. Red flag: counting only direct edges. Rubric: three
  counts (2), keystone (1). Remediation: C07.

## Implementation/debug task

- D1. Strong: checks in order: (1) bands on the 92 (n?
  interval?), (2) contamination log (quarantined? gap
  measured?), (3) eval config (swapped? pinned? same as
  baseline?). Culprit if hollow: contamination or a
  saturated benchmark, the number measures the leak.
  Honest claim: "92 +/- band on benchmark X, contamination
  status Y, n = Z". Three lines: score + band, n and
  config, contamination status. Red flag: announcing 92.
  Rubric: ordered checks (2), culprit (1), claim (1),
  lines (1). Remediation: C12 (U08).

## Changed-constraint scenarios

- S1. Strong: trust the syllabus bullets plus primary
  papers (masked diffusion LMs, 2024-2025). Do not trust
  secondary summaries or dated blog claims. First
  derivation from scratch: the masked training objective
  and the unmask-by-confidence loop. Red flag: "skip L8,
  it will not be tested". Rubric: trusted (1), untrusted
  (1), first derivation (2). Remediation: C01, C05.
- S2. Strong: cut k first if the accuracy target holds at
  lower k (measure the k-vs-accuracy curve, find the
  knee). Distill only if k = 1 still misses latency.
  Renegotiate only with the curve as evidence. The
  measurement that decides: accuracy at k in {1, 2, 4, 8}
  vs p99 latency. Red flag: cutting k without measuring.
  Rubric: choice (2), deciding measurement (2).
  Remediation: C07 (U06), C11.

## Research-critique question

- R1. Strong: strongest claim: parallel refinement wins on
  latency with parallel hardware, quality gains ground,
  so AR's serial bottleneck dooms it. Counter: tooling,
  coherence, and ecosystem maturity favor AR, and most
  "wins" are on narrow benchmarks. Experiment: matched
  training budgets, blind human eval on open generation,
  latency measured on the same hardware. Falsification: if
  diffusion does not win quality-per-latency on open tasks
  at matched budgets, the claim fails for now. Red flag:
  citing benchmark scores alone. Rubric: claim (1),
  counter (2), design (2). Remediation: C01, C08.

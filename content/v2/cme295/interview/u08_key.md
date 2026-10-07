# U08 interview key

Minimum sufficient explanation, strong answer, common red flags,
scoring rubric, remediation per question. Keep it closed-book.

## Breadth

- B1. Strong: the scoring law the judge follows, criteria plus
  scale. Anchors buy inter-judge agreement (0.58 to 0.81 in
  the toy). Red flag: "anchors make it objective". Rubric:
  definition (1), anchor effect (1). Remediation: C01.
- B2. Strong: pairwise, n(n-1)/2 pairs. Red flag: "pointwise,
  because it scores each". Rubric: which (1), why (1).
  Remediation: C02.
- B3. Strong: (w1 + (1 - w2)) / 2 over the two orders. Red
  flag: averaging the raw rates. Rubric: formula (2).
  Remediation: C03.
- B4. Strong: the weighted mean gap between confidence and
  accuracy across bins, honesty about doubt. Red flag: "the
  error rate". Rubric: what (2). Remediation: C04.
- B5. Strong: perplexity gap vs paraphrases, n-gram overlap
  with training data, canary hits. Red flag: "high score".
  Rubric: three (2). Remediation: C08.
- B6. Strong: the uncertainty band, the contamination status,
  the sample size. Red flag: "the score". Rubric: three (2).
  Remediation: C12.

## Deep ladders

- L1. Strong path: position (first-answer favor) and
  verbosity (longer-answer favor) defined, debiased 0.625,
  algebra: obs1 = p + b, obs2 = (1 - p) + b, mean of (obs1,
  1 - obs2) = p, debiased with the large-gap check (gap
  above 0.1 flags strong bias), length caps handle what
  swapping cannot, a stubborn gap means asymmetric bias or
  length-order correlation, "fixes all" ignores
  self-preference, experiment = bias vs length-difference
  curve. Rubric: 2 per rung, 10 total.
- L2. Strong path: Wilson score interval, center =
  (p + z^2/2n)/(1 + z^2/n), half-width = z sqrt(p(1-p)/n
  + z^2/4n^2)/(1 + z^2/n), z = 1.96, band [0.522, 0.709]
  for n = 100 w = 62, width shrinks as 1/sqrt(n), wilson
  with the shrink check on subsamples, bootstrap is flexible
  but needs care, non-shrinking bands mean clustered items
  (effective n small), overlap means "not sure" not "equal",
  experiment = cluster-aware intervals. Rubric: 2 per rung,
  10 total.

## Analytical exercises

- E1. Strong: gaps 0.02, 0.04, 0.09, 0.18, ECE 0.0825,
  worst bin 0.9. After the map: confidences [0.6, 0.7,
  0.72, 0.82], gaps 0.02, 0.04, 0.01, 0.10, ECE 0.0425.
  The map helped, nearly halved the error. Red flag:
  "0.08 off everything is fine". Rubric: first ECE (2),
  second (2), verdict (1). Remediation: C04.
- E2. Strong: 28 pairs x 300 x 2 = 16,800 calls, bill
  $50.40. Sampled: 4,200 calls, $12.60, bands 2x wider.
  Red flag: forgetting swapped orders. Rubric: full bill
  (2), sampled (2). Remediation: C11.

## Implementation/debug task

- D1. Strong: checks in order: (1) Wilson bands for 0.58
  vs 0.52 on n = 200 (bands [0.511, 0.646] and [0.451,
  0.588], overlap: not significant), (2) judge config (temp, seed,
  version pinned?), (3) rubric anchors and the raw
  disagreement sample. Culprit if fake: noise plus
  position bias on unswapped pairs. Decision rule: ship
  only if the lower band clears 0.5 after debiasing.
  Reporting lines: win rate, band, n, swapped or not,
  judge version, seed. Red flag: "0.58 > 0.52, ship".
  Rubric: ordered checks (2), culprit (1), rule (1),
  reporting (1). Remediation: C03, C10, C12.

## Changed-constraint scenarios

- S1. Strong: do not trust the numbers, re-run the
  decision-critical pairs pinned at temp 0 and measure
  agreement first. Trusting risks a noise-driven release,
  full delay wastes the window. First check: run-to-run
  agreement on 100 fixed items with the pinned config.
  Rubric: choice (1), rejected modes (2), check (1).
  Remediation: C05.
- S2. Strong: report the bands, the unknown contamination
  status, and the saturation warning. Refuse to claim a
  ranking: with 6 models above 0.90 and no quarantine, any
  order is noise plus leakage. Offer a fresh paraphrase set
  as the next step. Red flag: ranking anyway "with
  caveats". Rubric: report (2), refuse (2). Remediation:
  C08, C12.

## Research-critique question

- R1. Strong: strongest claim: judges are cheap, fast, and
  correlate highly with humans on many tasks, so they can
  carry routine eval. Counter: systematic biases (position,
  verbosity, self-preference) do not average out and can
  flip close calls. Experiment: stratified human sample of
  200+ items, measure agreement and the bias gaps, compare
  model-vs-model rankings under judge and human. Falsification:
  if agreement is below 0.75 or any bias gap exceeds 0.1,
  replacement is refused for this task. Red flag: "high
  correlation is enough". Rubric: claim (1), counter (2),
  design (2). Remediation: C01, C03, C06.

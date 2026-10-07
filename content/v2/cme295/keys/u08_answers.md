# U08 answer key

Answers to the lesson exercises. Do not open before attempting.

## C01

- R1. Correctness (solves the task), completeness (no gaps),
  clarity (followable). One anchor example per level.
- R2. Pilot an anchored rubric on 50 items. The anchors are
  the usual cure for 0.58.
- R3. So every past score stays comparable. A silent rubric
  change rewrites history.

## C02

- R4. 5 x 4 / 2 = 10 pairs.
- R5. Ties = 1 - 0.62 - 0.30 = 0.08. Report the tie rate
  with the win rate.
- R6. When absolute quality matters (pass/fail gates) or n
  is large enough that O(n^2) hurts.

## C03

- R7. (0.70 + (1 - 0.45)) / 2 = 0.625.
- R8. Cap length in the prompt or penalize verbosity in the
  rubric. Do not let tokens buy wins.
- R9. Because one order carries the position bias. Two
  orders let the algebra cancel it.

## C04

- R10. 0.2 x |0.75 - 0.85| = 0.2 x 0.10 = 0.02.
- R11. Mediocre. 7 points of average dishonesty is enough
  to mislead decisions, recalibrate.
- R12. Calibration is task-specific. One global map lies
  about every domain but one.

## C05

- R13. Noise. At temp 0.7 the run-to-run jitter (0.12
  disagreement) swamps a 0.03 delta.
- R14. Temperature 0, fixed seed, pinned model version.
- R15. Backends change under you. Monthly re-measurement
  catches silent drift.

## C06

- R16. Yes, with the caveat logged: 0.04 is inside the
  Wilson band for n = 100, and agreement 0.78 clears the
  bar.
- R17. Taxonomize them. If one class dominates, the rubric
  or the judge has a systematic blind spot, fix that.
- R18. Judge quality varies by task type. Unstratified
  samples hide the worst slice.

## C07

- R19. Self-preference bias: the judge likes its own house
  model by 0.09.
- R20. Signature phrases ("as an AI") and formatting quirks
  (headers, bullet style).
- R21. Position bias stacks with identity bias. Shuffle
  removes the order signal.

## C08

- R22. Contaminated until proven otherwise. A 3x perplexity
  gap is the memorization signature.
- R23. Perplexity gap, n-gram overlap with training data,
  canary string hits.
- R24. Canaries only work if they predate training. Plant
  them before the data is collected.

## C09

- R25. Metric A (0.91). The correlation gap justifies the
  judge-call cost.
- R26. It is decoration. A metric that never moves cannot
  guide decisions, replace it.
- R27. Enough for a stable rank correlation estimate. Fewer
  and the correlation itself is noise.

## C10

- R28. Wilson score: center = (0.62 + 1.96^2/200) /
  (1 + 1.96^2/100) = 0.616, half-width = 0.093.
  Interval [0.522, 0.709].
- R29. "Not sure". Overlapping bands mean the data cannot
  separate the models, say so.
- R30. Near 5. Clustered items share information, the
  binomial math overstates precision.

## C11

- R31. 15 pairs x 200 items x 2 orders = 6000 calls. At
  $0.002: $12.
- R32. Bands widen by sqrt(500/50) = 3.16x versus the full
  run. Say it up front.
- R33. Retries, price changes, or unswapped orders billed
  as swapped. Audit the call log.

## C12

- R34. Tie. The bands overlap, the 1-point gap is noise.
- R35. Retire or extend it. A saturated benchmark ranks
  noise, build harder items.
- R36. The uncertainty band, the contamination status, and
  the sample size. No naked scores.

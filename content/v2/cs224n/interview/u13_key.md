# Interview key , U13

## Breadth

A1. A direction in activation space separating labeled examples.
A probe proves correlation, nothing more.
A2. Behavior: what the model does (scores). Mechanism: how
(Circuits, interventions). Each needs its own test.
A3. Remove or patch a component, measure the change. One-sided:
an effect proves contribution, no effect proves little
(backups exist).
A4. Add alpha times a direction to steer behavior. Report the
side-effect metric (perplexity) with every claim.
A5. Min group rate over max: 0.667 on the toy. It does not prove
discrimination: base rates can explain it.
A6. Data provenance, bias test, misuse eval, privacy review,
uncertainty report, rollback plan. Missing items are blockers.

## Deep ladders

L1. (1) Project out the direction, remeasure. (2) 2.1 - 0.6 =
1.5. (3) Same input, component removed: the difference is its
contribution. (4) Backup circuits, or the intervention was not
surgical. (5) Ablate vs patch clean activations, compare
effects.

L2. (1) 0.28/0.42 = 0.667. (2) Far from 1.0 demands an
explanation. (3) A perfect predictor on different base rates
gives a low ratio. (4) The ratio is the alarm, the base rates
are the context. (5) Balance the data, re-measure, expect the
ratio to rise.

## Analytical

A7. "A steerable correlate of truth-like text", or "confidence-
like direction". Not "truth direction": the 20 mismatches are
the honest boundary.
A8. Report the hard-slice ECE 0.60 as the deployment number,
with the slice table. The 0.425 overall is irrelevant to this
deployment.

## Implementation/debugging

A9. (1) Lower alpha into the working range (cheapest). (2)
Sweep alpha, plot effect vs perplexity, pick the knee. (3)
Check the direction is causal first (C03). (4) Abandon
steering for fine-tuning if no clean range exists.

## Changed-constraint

A10. Deduplication (limit: rare singletons still extract) and
differential privacy (limit: utility cost). Neither fully
solves. Measure the rate after each.
A11. No. Coverage is a lower bound, and a 2-day weak team
proves little. Ship needs the checklist (C12), not the
absence of findings.

## Research critique

A12. Steelman: the probe separates true from false at 0.83 and
steering moves behavior, a real signal. Counterexample: the
confident lies fire it too (20/200). The honest label is a
correlate of truth-like text, not truth.

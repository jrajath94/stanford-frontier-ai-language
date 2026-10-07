# Answer key , U13 Interpretability and social impacts

Attempt the exercises before reading. Ladders are oral: answer aloud,
then check.

## Remediation

R1. Disparity ratio = 0.28/0.42 = 0.667.
R2. Intervention effect = 2.1 - 0.6 = 1.5.
R3. ECE 0.425: overconfident, report before deploying.

## Breadth

A1. A direction in activation space that separates labeled
examples. Toy probe 0.83. Correlation only.
A2. Behavior is the scoreboard, mechanism is the replay. Each
needs its own test.
A3. Remove the component, watch the behavior. Effect 1.5 on the
toy. One-sided: effect proves contribution, no effect proves
little.
A4. Our words are boxes, model concepts are clouds. The
mismatch test bounds the label: 20/200 on the toy.
A5. Add alpha times the direction. Toy: alpha 2.0 moves the gap
1.5, perplexity +0.3. Report side effects always.
A6. Ladder: correlation, causation, control. Honest summary:
"a steerable correlate of truth-like text".
A7. False-claim rate 0.18, 0.07 with grounding. Mitigations:
retrieval, citations, uncertainty flags.
A8. Ratio 0.667. The alarm, not the verdict. Fixes: data,
model, thresholds, each with trade-offs.
A9. Extraction rate 0.185 on the toy. Defenses: dedup and
differential privacy. Neither fully solves.
A10. Four harm classes, coverage 6/8, 4/8, 2/8, 3/8. Coverage
is a lower bound, not a certificate.
A11. ECE 0.425. Calibrate on your distribution, report per
slice.
A12. Six items: provenance, bias, misuse, privacy, uncertainty,
rollback. Toy 4/6, the 2 missing are blockers.

## Oral ladders

L1 (concepts). Define the direction. Run the probe. Derive the
correlation limit. Diagnose the length proxy. Design the
control.

L3 (interventions). Define ablation. Compute 1.5. Derive the
counterfactual. Diagnose the backup circuit. Design the patch
comparison.

L6 (attribution). Name the rungs. Run the toy. Derive the
gating. Diagnose the backup case. Design the next rung.

L8 (bias). Define the ratio. Compute 0.667. Derive the
alarm-not-verdict. Diagnose the base-rate case. Design the
balancing test.

L10 (misuse). Name the classes. Build the matrix. Derive the
lower bound. Diagnose the weak team. Design the cadence.

L12 (reporting). Define the report. Score the toy. Derive the
blocker rule. Diagnose the "N/A" dodge. Design the revisit
cadence.

## Exercises

E1. `find_direction` matches 0.83 on the toy.
E2. The length proxy confounds the probe.
E3. 6 claims classified correctly.
E4. The shortcut found in the toy.
E5. `ablate` matches effect 1.5.
E6. Random direction: near-zero effect (control).
E7. `mismatch` finds the 20.
E8. Honest relabel: "confidence-like direction".
E9. `steer` matches the toy numbers.
E10. Alpha sweep: the model breaks past alpha 10.
E11. The ladder check passes the toy.
E12. Honest summary written.
E13. `fact_rate` matches 0.18 and 0.07.
E14. Grounding drops the rate.
E15. `disparity` matches 0.667.
E16. Balanced data moves the ratio.
E17. Canary test matches 0.185.
E18. Dedup drops the rate.
E19. The matrix built for the toy.
E20. Privacy is thinnest at 2/8.
E21. `ece` matches 0.425.
E22. ECE sliced: worse on hard slices.
E23. `impact_report` matches 4/6.
E24. Blocker paragraph names misuse eval and uncertainty.

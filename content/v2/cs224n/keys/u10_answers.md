# Answer key , U10 Benchmarking and research methodology

Attempt the exercises before reading. Ladders are oral: answer aloud,
then check.

## Remediation

R1. Wilson: 78/100 gives [0.689, 0.850], 780/1000 gives [0.753,
0.805].
R2. Contamination: reported 0.75, clean-only 0.70, inflation 0.05.
R3. Margin 0.02 at 95 percent needs n = 2401.

## Breadth

A1. MMLU-style: many subjects, multiple choice. HELM-style: many
scenarios, many metrics. Both aggregate, both hide slices.
A2. The coverage matrix: benchmark tasks vs deployment
capabilities. An empty column is an untested capability.
A3. Kappa 0.551 on the toy, Brier 0.17. The judge is an
instrument: calibrate agreement and confidence.
A4. Leak share times memorization gap: 0.20 times 0.25 = 0.05
inflation. Quarantine and rescore.
A5. Report the interval, never the bare number. n = 2401 for
margin 0.02. Templated items break independence.
A6. Exact, normalized, judge. Exact rewards formatting, the
judge rewards its own biases. State the normalization.
A7. Replicate the baseline first. Verdicts: replicated, differs,
underpowered. The toy: replicated within noise.
A8. Delta = slice minus mean. Toy: -0.26 and +0.08. The mean
hides the failing slice.
A9. Length, position, style, self-preference. Control by
blinding. Toy bias: 0.09.
A10. Write the contract before measuring. Criterion vs metric.
The toy fails on slice A despite the 0.78 mean.
A11. Five criteria: feasible, data, metric, compute, novelty.
Below 3/5: rescope or drop.
A12. Two columns: borrowed (cited) and built (yours). One
falsifiable claim is the bar. Negative results count.

## Oral ladders

L1 (families). Describe MMLU/HELM. Compute the two intervals.
Derive why n shrinks the claim. Diagnose the 0.78 vs 0.76
"win". Design the benchmark-to-deployment rank test.

L3 (judges). Define kappa. Compute 0.551. Derive the chance
correction (0.51). Diagnose the length bias. Design the
blinding test.

L4 (contamination). Define the leak. Compute the 0.05.
Derive the mixture formula. Diagnose the n-gram false
positive. Design the canary test.

L8 (slices). Define a slice. Compute the deltas. Derive the
weighted-zero check. Diagnose the hidden -0.26. Design the
worst-slice tracking.

L12 (accounting). Define the ledger. Build the toy. Derive
the novelty bar. Diagnose the empty-built report. Design the
negative-result writeup.

## Exercises

E1. `report` matches [0.689, 0.850] and [0.753, 0.805].
E2. n = 2401 for margin 0.02.
E3. The matrix finds the empty "permission errors" column.
E4. "Covers 5 of 6 capabilities, tool-use untested."
E5. `judge_stats` matches kappa 0.551, Brier 0.17.
E6. Reliability table: stated 0.8, observed 0.67 in bin 1.
E7. `decontaminate` flags the 40, rescores 0.70.
E8. Common phrases overlap without leaking the answer.
E9. `margin` matches 0.069 at n = 200, 0.02 at n = 2401.
E10. 200 items from 10 templates: effective n near 10.
E11. `score` matches 0.78 exact, 0.90 normalized.
E12. Lowercasing moves the score by 0.12.
E13. `replicate` returns "replicated" for the toy.
E14. At n = 40 the verdict is "underpowered".
E15. `slice_report` matches -0.26 and +0.08.
E16. Without slice labels the mean hides slice A.
E17. `bias_check` matches the 0.09 length bias.
E18. Position blinding moves less than length blinding.
E19. `decide` fails the toy on slice A.
E20. A support-bot criterion: worst slice, n per slice, the
binding clause named.
E21. Idea A 4/5, idea B 2/5: kill B on paper.
E22. Define the metric: 3/5 becomes 4/5.
E23. The U09 lab accounting: borrowed vs built listed.
E24. One borrowed item labeled, e.g. the Wilson formula.

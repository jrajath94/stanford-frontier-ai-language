# Interview key , U10

## Breadth

A1. MMLU-style aggregates subjects, HELM-style aggregates
scenarios and metrics. The mean hides the slices.
A2. Test items in training. Correct by overlap probe, quarantine,
rescore on the clean set. Toy inflation: 0.05.
A3. Chance-corrected agreement: (p_o - p_e) / (1 - p_e). Raw
agreement ignores the chance baseline.
A4. A subgroup score with its delta from the mean. The mean can
hide a -0.26 slice.
A5. The pre-registered pass/fail contract, written before
measuring. It blocks negotiating with yourself after.
A6. Borrowed vs built, every component labeled, one falsifiable
claim as the bar.

## Deep ladders

L1. (1) [0.689, 0.850]. (2) [0.753, 0.805]. (3) SE falls as
1/sqrt(n). (4) Effective n near 10, the interval lies. (5)
n = 2401.

L2. (1) Reported = L s_m + (1 - L) s_c. (2) 0.75 vs 0.70.
(3) Flag by overlap, drop, rescore. (4) Common phrases overlap
without leaking answers: probe answer-bearing spans. (5)
Canary strings in training, extraction rate tracks inflation.

## Analytical

A7. p_e = 0.81 + 0.01 = 0.82. Kappa = (0.90 - 0.82) / 0.18 =
0.444, below 0.551. High agreement on a skewed task is mostly
chance.
A8. Wilson on 162/200: [0.750, 0.858]. On 156/200: [0.718,
0.832]. The intervals overlap heavily: no win declared. The
honest statement: underpowered, collect more items.

## Implementation/debugging

A9. n = 12 is too small for a claim (margin about 0.25). Collect
more slice items or merge the slice, and report the interval
with the delta.

## Changed-constraint

A10. The coverage matrix shows an empty column. Report: "5 of 6
capabilities covered, the language is untested." Build the
missing eval or state the gap.
A11. Underpowered by construction: no rerun, no interval. The
verdict is "cannot replicate", not "replicated". Cite the
paper number as reported, not verified.

## Research critique

A12. Steelman: the judge read both answers blind and preferred
one, at scale. Counterexample: the judge prefers longer,
self-styled answers (C09), so the win rate measures style
match, not quality. Calibrate and blind before any claim.

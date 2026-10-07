# Interview bank , U10 Benchmarking and research methodology

## Breadth (6)

Q1. What do the MMLU and HELM families aggregate, and what does the
aggregation hide?
Q2. What is contamination, and how do you correct for it?
Q3. What is Cohen's kappa, and why is raw agreement not enough?
Q4. What is a slice, and why can the mean mislead?
Q5. What is an acceptance criterion, and when do you write it?
Q6. What is contribution accounting?

## Deep ladders (2 x 5)

L1 (intervals). (1) Compute the Wilson interval for 78/100. (2)
Compute it for 780/1000. (3) Derive why the claim shrinks with n.
(4) 200 items from 10 templates: diagnose. (5) Design the n for a
0.02 margin.

L2 (contamination). (1) Write the mixture formula. (2) Compute the
0.05 inflation. (3) Derive quarantine and rescore. (4) Common
phrases overlap: diagnose the false positive. (5) Design the canary
test.

## Analytical (2)

Q7. Two judges agree on 78 of 100 items. Kappa is 0.551. A third
judge agrees on 90 of 100 with yes-rates 0.9 and 0.9. Compute its
kappa and explain why 90 percent agreement can be worse.
Q8. Your method beats the baseline 0.81 to 0.78 on 200 items. Is
that a win? Show the work.

## Implementation/debugging (1)

Q9. Your slice report shows delta -0.26 on slice A but n = 12 for
that slice. What do you do before any claim?

## Changed-constraint (2)

Q10. The deployment needs a language the benchmark does not cover.
What is the honest report?
Q11. The baseline code is absent but the paper reports 0.78. What
is the replication verdict?

## Research critique (1)

Q12. "Our judge-evaluated win rate proves the model is better."
Steelman, then give the strongest counterexample from this unit.

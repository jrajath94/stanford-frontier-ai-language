# Interview bank , U13 Interpretability and social impacts

## Breadth (6)

Q1. What is a concept direction, and what does a probe prove?
Q2. What is the difference between behavioral and mechanistic
evidence?
Q3. What is a causal intervention, and what is one-sided about it?
Q4. What does steering do, and what must you report with it?
Q5. What is the disparity ratio, and what does it not prove?
Q6. What goes in an impact report?

## Deep ladders (2 x 5)

L1 (interventions). (1) Define ablation. (2) Compute the 1.5
effect. (3) Derive the counterfactual logic. (4) No effect after
ablation: diagnose. (5) Design the patch-vs-ablate comparison.

L2 (bias). (1) Compute the 0.667 ratio. (2) State the alarm rule.
(3) Derive why it is not a verdict. (4) Different base rates with
a perfect predictor: diagnose. (5) Design the data-balancing test.

## Analytical (2)

Q7. Probe accuracy 0.83 on "truth", but it also fires on confident
lies (20/200 mismatches). What is the honest label for the
direction?
Q8. ECE is 0.425 overall but 0.15 on easy slices and 0.60 on hard
slices. The deployment is all hard slices. What do you report?

## Implementation/debugging (1)

Q9. Your steering at alpha = 10 breaks the model (perplexity +5).
List the fixes in order, cheapest first.

## Changed-constraint (2)

Q10. The canary test shows 0.185 extraction on a model trained on
user data. What are the two defenses, and what is each one's
limit?
Q11. The red team found nothing in 2 days. Can you ship? Explain.

## Research critique (1)

Q12. "We found the truth direction." Steelman, then give the
strongest counterexample from this unit.

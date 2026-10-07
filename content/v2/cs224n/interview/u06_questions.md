# Interview bank , U06 Pretraining, scaling, systems, data

## Breadth (6)

Q1. How does masked language modeling work, and why the 80/10/10
recipe?
Q2. What is the autoregressive objective, and what does it trade
against MLM?
Q3. What is a contextual embedding, and how does it differ from a
word2vec vector?
Q4. What are the four steps of corpus curation, and what does each
remove?
Q5. What is C = 6ND, and what does the 20:1 ratio mean?
Q6. What is test-set leakage, and how is it detected?

## Deep ladders (2 x 5)

L1 (MLM). (1) Write the masking procedure. (2) Compute the toy loss
0.487. (3) Derive why 100 percent [MASK] causes train/test skew.
(4) Fine-tuning underperforms: name the masking-related cause.
(5) Design the recipe ablation.

L2 (scaling). (1) Write C = 6ND and the power-law form. (2) Compute
the toy budget and the doubling ratio. (3) Derive the N/D tradeoff
at fixed C. (4) A 10x extrapolation misses badly: diagnose. (5)
Design the ratio experiment at fixed C.

## Analytical (2)

Q7. Two runs at identical C: one trains N = 4e8 on D = 5e8, the
other N = 1e8 on D = 2e9. The bigger model wins on every benchmark.
Does this contradict the 20:1 finding? Explain.
Q8. Your held-out perplexity is 22.2 but your train perplexity is
30.0. What are the two possible explanations, and which check
distinguishes them?

## Implementation/debugging (1)

Q9. Mid-training, the loss spikes and never recovers. List the
checks in order: data, batch, precision, hardware.

## Changed-constraint (2)

Q10. The training budget is cut 4x mid-project. What are the three
moves, in order?
Q11. The deployment target is a phone (no server GPU). How does the
pretraining plan change?

## Research critique (1)

Q12. "Scaling laws prove bigger is always better." Steelman, then
give the strongest counterexample from this unit.

# Interview bank , U05 Attention and transformers

## Breadth (6)

Q1. What are the query, key, and value, and why three projections?
Q2. Why divide attention scores by sqrt(d_k)?
Q3. What does multi-head attention add over one wide head?
Q4. Where does the causal mask go, and why not after softmax?
Q5. What do the residual connection and layer norm each fix?
Q6. Why does attention need position encodings at all?

## Deep ladders (2 x 5)

L1 (attention). (1) Write the single-head equations with shapes.
(2) Compute the toy: scores, weights, output. (3) Derive why the
weights sum to 1 and the output is a convex blend. (4) Two models
have identical attention weights but different outputs: explain.
(5) Design the test of whether weights explain predictions.

L2 (scale and mask). (1) Derive Var(q.k) = d_k. (2) Compute the
unscaled vs scaled entropy pair. (3) Prove exp(-1e9) = 0 makes the
mask exact. (4) Validation perplexity drops after you "fix" the
mask: diagnose. (5) Design the experiment that catches a leaking
mask.

## Analytical (2)

Q7. A 24-layer post-norm transformer will not train without warmup,
but the pre-norm twin trains fine. Which mechanism explains the
difference, and what is the cheapest fix for the post-norm run?
Q8. You double the context from 2k to 8k and step time goes up 14x,
not 4x. Which term explains it, and what two changes cut it?

## Implementation/debugging (1)

Q9. Your decoder generates well for 50 tokens then drifts into
repetition. List the checks in order, with the one-line fix each
check points to.

## Changed-constraint (2)

Q10. Inference must fit a 4GB GPU and the KV cache alone is 6GB.
What are the three moves, in order of desperation?
Q11. The task changes from generation to whole-document
classification. What changes in the mask, the readout, and the loss?

## Research critique (1)

Q12. "Attention weights show what the model looks at." Steelman the
claim, then give the strongest counterexample from this unit.

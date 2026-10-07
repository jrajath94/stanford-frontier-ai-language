# Answer key , U02 Word vectors and cooccurrence

Attempt the exercises before reading. Ladders are oral: answer aloud,
then check.

## Remediation

R1. cos((1,0), (1,1)) = 1 / (1 x sqrt(2)) = 0.7071. The angle is 45
degrees.

## Breadth

A1. One-hot: V-dim, one 1, rest 0, says which word, nothing else.
Distributed: d-dim learned vector, similarity means similarity.
One-hot cosine between distinct words is always 0, learned cosine
separates related (cat/dog 0.816) from unrelated (cat/mat 0.679).

A2. Skip-gram: center predicts each context word, many updates per
center, good for rare words. CBOW: averaged context predicts the
center, faster, smoother, worse for rare words. Both maximize log
P(context | center) over window pairs (66 pairs on the training
tokens).

A3. Negative sampling replaces the V-way softmax with k+1 binary
decisions: J = -log sigma(u_o . v_c) - sum log sigma(-u_i . v_c).
k controls the noise/speed tradeoff (k = 2 here). It is not a true
distribution.

A4. Hierarchical softmax: P(w | c) = product of binary decisions
along the root-to-leaf path. Cost O(depth x d) per pair vs O(V d).
Tradeoff: true probabilities, but rare words sit deep and get noisy
gradients.

A5. GloVe: weighted regression on log cooccurrence counts: minimize
sum f(X_ij)(w_i . w~_j + b_i + b~_j - log X_ij)^2. The weighting
f downweights rare noisy pairs. Counts beat sampling when the corpus
is fixed and you want every pair used once.

A6. PMI(w, c) = log(P(w,c)/(P(w)P(c))), in nats. Levy-Goldberg:
skip-gram with k negatives factorizes PMI(w,c) - log k. Toy:
PMI(cat, sat) = 1.569, target dot product 0.876.

A7. Analogy query: argmax cosine of (a - b + c), excluding a, b, c.
Toy: king - man + woman = (0.20, 1.40), cosine with queen 0.9992.
Two breaks: polysemy (offsets differ per sense), and the exclusion
rule (without it the answer is often b or c).

A8. One word form, one vector, several senses: the vector is a
frequency-weighted mixture. Toy: senses (1,0) and (0,1) at 4:1 give
(0.8, 0.2), cosines 0.970 and 0.243. The dominant sense wins.

A9. Intrinsic: judge vectors directly (similarity order, analogies),
fast, a proxy. Extrinsic: judge inside a task, slow, honest. They
disagree when the proxy property is not what the task needs.

A10. dJ/dv_c = -(1 - sigma(u_o . v_c)) u_o + sum (1 - sigma(-u_i .
v_c)) u_i: error times the other vector, pull for positive, push for
negatives. Finite-difference check: max error 2.76e-10.

A11. Face 1: frequent words dominate pairs (subsampling fixes it).
Face 2: frequent words dominate negatives (0.75 power fixes it).
Noise probs on training tokens: cat 0.228, sat 0.155, dog 0.115.

A12. Three lies of embedding plots: crowding (distant points
collide), t-SNE perplexity dependence (clusters reshape), and
cherry-picking (ugly regions cropped). Rule: the table wins.

## Oral ladders

L1 (representations). Define one-hot and distributed. Toy: the
14-dim name tag vs the 8-dim profile. Derive lookup as E^T e_w.
Explain the cosine separation (0.816 vs 0.679 vs 0.0). Critique:
one vector per form forces polysemy into one address.

L2 (word2vec). Define both arrows. Write the softmax loss and the
push-pull reading. Derive the two gradient terms. Predict the
window-10 collapse (context becomes the document). Design the window
sweep {1,2,5,10} on the cosine gap.

L3 (negative sampling). Define the real/fake game. Write J.
Derive dJ/dv_c term by term. Predict the k = 0 collapse (no push,
all cosines near 1). Design the noise-power test {0, 0.75, 1.0}.

L4 (hierarchical softmax). Define the path product. Compute 0.236
from the toy dots and turns. Derive the v_c gradient. Diagnose the
chain tree (O(V) cost, vanishing through L sigmoids). Design the
Huffman vs random tree test.

L5 (GloVe). Define the regression target. Compute the toy residual
(-0.043) and weight (0.053). Derive ratios-to-differences via the
log. Diagnose count-1 noise without f. Design the alpha sweep.

L6 (PMI). Define PMI. Compute 1.569 and the shifted target 0.876.
Sketch the optimum derivation (derivative zero, solve for the
score). Explain the three finite breaks (data, dimension, steps).
Design the d-vs-correlation test.

L7 (analogy). Define the query and the exclusion. Compute the toy
(0.9992). Derive the PMI-to-offset link. Explain the polysemy break
and the no-exclusion failure. Design the morphological/semantic/
polysemous split test.

L8 (polysemy). Define the mixture. Compute (0.8, 0.2) and the two
cosines. Derive the weighted-pull mechanism. Diagnose the mixed
neighbor list. Design the probe-vector correlation study.

L9 (evaluation). Define both. Run the toy pair (1.0). Derive the
Goodhart link. Explain the re-ranking finding. Design the
rank-correlation study over 5 embedding sets.

L10 (derivatives). State the gradient shape rule. Derive the positive
term by the chain rule. Explain the eps U-shape (truncation vs
rounding). Diagnose lr = 10 divergence (check validates the
derivative, not the optimization). Design the error-injection test.

L11 (frequency). Define both faces. Compute the noise table. Derive
the 0.75 compromise (raw vs uniform). Diagnose the collapsed run
("the" at noise prob 0.232, all cosines ~0.91). Design the t sweep
on negation items.

## Exercises

E1. `nearest` on the trained Win returns "dog" for "cat" (0.816).
One-hot `nearest` must return None or raise: all cosines are 0.

E2. One-hot vectors e_i, e_j: dot is 1 if i = j else 0, norms are 1,
cosine equals the dot. QED.

E3. `skipgram_pairs` on the 18 training tokens with window 2 yields
66 pairs.

E4. 50 epochs: loss falls from 0.6940 toward ~0.66 (exact value
depends on the learner's loop, the key property is monotone decrease
early and the final cosine order cat/dog > cat/mat).

E5. `neg_sample_step` reproduces v_c = (0.386, 0.043) from (0.30,
-0.20) at lr 0.5.

E6. Finite-difference max error 2.76e-10 < 1e-6: pass.

E7. `path_prob` returns 0.236 on the toy dots and turns.

E8. Huffman on counts {a:5, b:3, c:1, d:1}: merge c+d (2), merge
b+(cd) (5), merge a+(bcd) (10). Depths: a = 1, b = 2, c = 3, d = 3.
Shortest path goes to "a".

E9. f(0) = 0, f(2) = (2/100)^0.75 = 0.053, f(100) = 1.0.

E10. Residual -0.043, weight 0.053, contribution 0.053 x 0.043^2 =
9.8e-5.

E11. `pmi_matrix` reproduces PMI(cat, sat) = 1.569.

E12. Doubling k shifts every implicit target by -log 2 = -0.693.
The optimum dot products fall by 0.693 across the board, with fixed
dimension the model realizes this by shrinking norms and rotating.
Measurable prediction: the mean dot product over all pairs drops.

E13. `analogy` returns "queen", cosine 0.9992.

E14. Without the exclusion, q = (0.20, 1.40) = 2 x woman exactly, so
cosine(q, woman) = 1.0 and the answer is "woman". The exclusion
exists because the query is algebraically closest to its own parts.

E15. `sense_mixture` returns (0.8, 0.2), cosines 0.970 and 0.243.

E16. cos with the rare sense = 1/sqrt(r^2 + 1) for ratio r:1. Below
0.1 when r > 9.95, i.e. above roughly a 10:1 ratio.

E17. `intrinsic_pairs`: 1.0 on the toy pair, 0.5 on random vectors
(chance level), ties scored 0.5.

E18. Protocol: freeze vectors, train the U01 NB-equivalent linear
classifier on the same splits, compare accuracy. Inconclusive on the
toy because 4 documents cannot separate methods, the confidence
interval covers both.

E19. dJ/du_o = -(1 - sigma(u_o . v_c)) v_c. Finite-difference check
passes below 1e-6.

E20. eps = 1e-4: error ~1e-8 (truncation dominates). eps = 1e-6:
~1e-10 (sweet spot). eps = 1e-8: ~1e-8 (cancellation dominates).
U-shape: truncation falls with eps, rounding rises.

E21. `noise_dist` reproduces cat 0.228, sat 0.155, dog 0.115.

E22. Without stopword removal: the cat/dog vs cat/mat gap collapses
(all pair cosines near 0.9, gap ~0). With removal: 0.816 vs 0.679,
gap 0.137. Frequency bias is visible in the gap.

E23. `pca2` on the trained matrix: variance explained by 2
components is well below 1.0 (8 dims to 2 loses most of the
structure).

E24. Accept any pair the learner finds with 2-D cosine exceeding 8-D
cosine by more than 0.1, named with the mechanism (crowding: distant
in 8-D, collided in 2-D).

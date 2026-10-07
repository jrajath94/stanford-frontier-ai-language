# Interview key , U02

## B1

Minimum: one-hot stores identity only (V-dim, one 1), distributed
stores learned features (d-dim). Similarity breaks for one-hot (all
distinct pairs at cosine 0) and works for distributed (cat/dog 0.816
vs cat/mat 0.679). Strong: names the parameter difference (E is
trained, one-hot is fixed). Red flag: "distributed is just smaller".
Rubric: 2 for store plus the cosine contrast. Remediation: C01.

## B2

Minimum: skip-gram predicts context words from the center,
CBOW predicts the center from averaged context. Strong: states the
rare-word asymmetry (skip-gram updates rare words per occurrence,
CBOW averages them away). Red flag: "they are the same model".
Rubric: 2 for both arrows plus the asymmetry. Remediation: C02.

## B3

Minimum: J = -log sigma(u_o . v_c) - sum_{i=1..k} log sigma(-u_i .
v_c). Avoids the O(V d) softmax normalization per pair. Strong:
reads the push-pull off the formula. Red flag: "it computes
probabilities" (it does not). Rubric: 2 for the formula plus the
cost. Remediation: C03.

## B4

Minimum: P(w | c) as the product of binary decisions down the
root-to-leaf path, cost O(depth x d) per pair. Strong: adds the
Huffman detail (frequent words get short paths) and the true-
distribution property. Red flag: "it is faster than negative
sampling, always". Rubric: 2. Remediation: C04.

## B5

Minimum: weighted least squares on log cooccurrence counts, f
downweights rare noisy pairs (f(0) = 0, f caps at 1). Strong:
derives ratios-to-differences via the log. Red flag: "GloVe counts,
word2vec predicts, so GloVe is better". Rubric: 2. Remediation: C05.

## B6

Minimum: skip-gram with k negative samples implicitly factorizes the
matrix M_{w,c} = PMI(w, c) - log k. Strong: states the three
asymptote conditions (infinite data, enough dimensions, convergence).
Red flag: "word2vec computes PMI". Rubric: 2. Remediation: C06.

## D1 ladder

D1.1 Words in similar contexts have similar meanings (operationally:
similar cooccurrence distributions).
D1.2 18 tokens, window 2: each token pairs with up to 4 neighbors,
66 total after edge clipping. Count, do not guess: the script
enumerates them.
D1.3 dJ/dv_c = -(1 - sigma(u_o . v_c)) u_o + sum_i (1 - sigma(-u_i .
v_c)) u_i, by the chain rule on each log-sigma term.
D1.4 The noise distribution on raw tokens gives "the" probability
0.232, every word is pushed away from "the" far more than from
anything else, so all vectors align and pair cosines sit near 0.9.
Fix: remove or subsample function words.
D1.5 Skip-gram fits: it processes pairs online, no count table.
GloVe breaks: it needs the full cooccurrence matrix X first, which
never finishes on an endless stream (in practice: rebuild X
periodically, or switch to skip-gram).

## D2 ladder

D2.1 Intrinsic judges vectors directly (fast proxy), extrinsic judges
them inside a task (slow, honest).
D2.2 q = king - man + woman = (0.20, 1.40) = 2 x woman exactly in
this toy, so cosine(q, woman) = 1.0. Without the exclusion the query
is algebraically nearest to its own parts.
D2.3 Analogy accuracy is a 0/1 hit rate over questions with different
difficulties, it hides the linearity assumption and the exclusion
dependence. Harmonic thinking does not apply because there is no
precision/recall tradeoff being balanced, just hits.
D2.4 Training sums pulls over all occurrences, the gradient is the
frequency-weighted sum of sense pulls, so the vector lands between
sense clusters near the dominant one. The neighbor list mixes senses
because the single address serves both.
D2.5 Assumption: relations are constant vector offsets. Counter 1:
polysemy breaks the offset. Counter 2: analogy sets have artifacts
(the answer is often the nearest neighbor of c). The honest claim is
"linear structure on this test set", not "meaning".

## Q1

Softmax per pair: V x d = 14 x 8 = 112 multiply-adds for the scores
alone (plus the normalization). Negative sampling: (k+1) x d = 3 x 8
= 24. Ratio about 4.7x on the toy, at V = 50,000 the ratio is ~2000x.
Strong answer computes both numbers and scales V.

## Q2

Target = 1.569 - log 2 = 0.876. Observed 0.30. Gap 0.576: not near
the optimum. Reasons: (1) 400 epochs on 66 pairs is far from
convergence, (2) d = 8 cannot hold a rank-14 factorization, so even
asymptotically the dot products cannot all match. Strong answer
names both the optimization gap and the capacity gap.

## I1

Bug 1: `uo` is updated with the already-updated `vc`. All gradients
in one step must be computed from pre-step values, otherwise the
update uses a mixed state and the math no longer matches the derived
gradient. Fix: snapshot vc_old = vc.copy() (and uo_old) first, or
accumulate gradients then apply.
Bug 2: sign error on the noise terms. The noise gradient wrt vc is
+(1 - sn) un, so the descent update is `vc = vc - lr * (1 - sn) *
un`, the code has `+`, which pushes the center vector toward noise
words instead of away. Same for `un`. That sign flip collapses all
vectors in one direction and the loss never falls. Red flag: finding
only one bug. Rubric: 2 for both with fixes. Remediation: C03, C10.

## S1

Memory breaks first: 2M x 8 x 4 bytes x 2 matrices = 128 MB, fine,
but the optimizer state and the softmax are worse. Softmax: O(V d)
per pair is dead, use negative sampling or hierarchical softmax.
Analogy test: O(V d) per query over 2M types is slow but
precomputable, the real break is that most types are rare and their
vectors are noise. Fixes: negative sampling for training, quantized
or sharded storage, and frequency floors for the eval vocab.

## S2

Hand-built features. 500 sentences cannot train skip-gram (rare words
get ~1 update each, the geometry is noise) and cannot fill GloVe's
count table. A small feature set (or cross-lingual transfer from a
related language, stated as a follow-up) is the honest choice. Reason
from sample efficiency: 500 sentences is below the threshold where
distributional methods beat a lexicon.

## R1

Confound 1: 5x more epochs , fix by matched compute budgets (same
total pair-updates). Confound 2: larger window , fix by sweeping
window for both methods. Confound 3: single evaluation run , fix by
multiple seeds and a significance test. Without these, the 3% is
unattributed.

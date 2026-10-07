# Interview bank , U02 Word vectors and cooccurrence

Test mode: answer closed-book before opening `interview/u02_key.md`.

## Breadth (B1-B6)

B1. Contrast one-hot and distributed representations: what each
stores, and what breaks when you compute similarity with each.
B2. State the skip-gram and CBOW objectives and the direction each
predicts.
B3. Write the negative-sampling objective for one (center, context)
pair with k negatives, and name the cost it avoids.
B4. What does hierarchical softmax compute, and what does it cost per
pair?
B5. State the GloVe objective in words and the role of the weighting
function f.
B6. State the Levy-Goldberg result in one sentence: what matrix does
skip-gram factorize?

## Deep ladder 1 , from counts to vectors (D1.1-D1.5)

D1.1 Define the distributional hypothesis.
D1.2 Toy: 18 training tokens, window 2. How many (center, context)
pairs, and why 66?
D1.3 Derive the negative-sampling gradient wrt the center vector.
D1.4 Debug: after training on the raw 35 tokens, all pair cosines sit
near 0.9. Diagnose from the noise distribution.
D1.5 Compare skip-gram and GloVe on a streaming corpus that never ends.
Which fits, and what breaks for the other?

## Deep ladder 2 , evaluation and limits (D2.1-D2.5)

D2.1 Define intrinsic vs extrinsic evaluation.
D2.2 Toy: the analogy query returns "woman" at cosine 1.0 without the
exclusion rule. Explain exactly why.
D2.3 Derive why F1-style harmonic thinking does not apply to analogy
accuracy, and state what the metric hides.
D2.4 Debug: a polysemous word's neighbor list mixes two senses. A
learner reads it as one definition. Explain the mixture mechanism.
D2.5 Critique: "our vectors score 80% on analogies, so they capture
meaning." State the assumption and the two counterarguments.

## Analytical / quantitative (Q1-Q2)

Q1. Vocabulary 14, window 2, 18 training tokens, 66 pairs. The full
softmax costs O(V d) per pair, negative sampling with k = 2 costs
O(k d) per pair. With d = 8, how many multiply-adds does each cost
per pair, and what is the ratio?
Q2. PMI(cat, sat) = 1.569 nats, k = 2. A trained model reports
u_sat . v_cat = 0.30. Is the model near the Levy-Goldberg optimum?
Compute the target and the gap, and name two reasons for the gap.

## Implementation / debug (I1)

I1. This negative-sampling step trains, but the loss never falls and
all vectors drift in one direction. Find the two bugs and fix them.

```python
def step(vc, uo, negs, lr):
    import numpy as np
    s = 1 / (1 + np.exp(-uo @ vc))
    vc = vc + lr * (1 - s) * uo
    uo = uo + lr * (1 - s) * vc
    for un in negs:
        sn = 1 / (1 + np.exp(un @ vc))
        vc = vc + lr * (1 - sn) * un
        un = un + lr * (1 - sn) * vc
    return vc, uo
```

## Changed-constraint scenarios (S1-S2)

S1. The vocabulary grows to 2 million types but d stays 8. Which part
of the pipeline breaks first: memory, the softmax, or the analogy
test? Name the fix for each.
S2. You must ship embeddings for a language with 500 sentences of
text and no pretrained vectors. Skip-gram, GloVe, or hand-built
features? Decide with reasons.

## Research critique (R1)

R1. A preprint claims a new embedding method beats skip-gram on
analogy accuracy by 3%, trained with 5x more epochs and a larger
window, evaluated once. List three confounds and the controlled
experiment that removes each.

# Interview bank , U03 Neural fundamentals and dependency parsing

Test mode: answer closed-book before opening `interview/u03_key.md`.

## Breadth (B1-B6)

B1. Define a computational graph and state what topological order
guarantees for the forward and backward passes.
B2. Contrast sum and mean loss reduction: what changes in the
gradient, and what breaks if you switch mid-training?
B3. Why does naive softmax return nan on large logits, and what is
the one-line fix?
B4. Embeddings are parameters: what does the backward pass do for a
batch with a repeated word id?
B5. Write the vector chain rule and explain why reverse mode never
forms the full Jacobian.
B6. Why can a 2-layer ReLU net solve the toy task while a linear
model sits at 0.5 accuracy?

## Deep ladder 1 , from graph to gradients (D1.1-D1.5)

D1.1 Define forward and backward on f = (x + y) x z.
D1.2 Toy: compute df/dx, df/dy, df/dz at (2, 3, 4).
D1.3 Derive the vector-Jacobian product for y = Wx and its memory
saving over the full Jacobian.
D1.4 Debug: a hand-written backward pass gives wrong gradients only
when B > 1. Diagnose.
D1.5 Compare reverse-mode, forward-mode, and finite differences on a
model with 10M parameters and 1 output. Which do you use for
training, and which for a one-time check?

## Deep ladder 2 , parsing as classification (D2.1-D2.5)

D2.1 Define the parser state and the three actions.
D2.2 Toy: replay the 6 actions on "the cat sat" and list the arcs.
D2.3 Derive why arc-standard parsing is O(n) time.
D2.4 Debug: action accuracy is 95% but attachment score is far lower,
with errors clustered on long sentences. Explain.
D2.5 Critique: "greedy transition parsing is optimal because each
action is optimal." State the flaw and the experiment that shows it.

## Analytical / quantitative (Q1-Q2)

Q1. Softmax-cross-entropy on 3 classes: logits (1, 2, 3), true class
2 (0-indexed). Compute the probabilities with the stable softmax and
the gradient dL/dz.
Q2. A linear layer: X (32, 128), W (128, 64). Your backward pass
allocates dL/dW by an explicit loop over the 8192 entries with one
finite-difference each. How many forward passes does one gradient
check cost, and what do you do instead in practice?

## Implementation / debug (I1)

I1. This embedding backward is wrong on some batches but right on
others. Find the bug.

```python
def embed_backward(grad_out, ids, V, d):
    import numpy as np
    dE = np.zeros((V, d))
    dE[ids] = grad_out
    return dE
```

## Changed-constraint scenarios (S1-S2)

S1. The parser must now handle non-projective trees (crossing arcs).
Arc-standard cannot produce them. What is the smallest change to the
system: new actions, new objective, or a different parser?
S2. Training moves from full-batch to minibatches of 32 with mean
reduction, but the tuned learning rate came from sum reduction at
B = 256. What happens on the first run, and what is the fix?

## Research critique (R1)

R1. A preprint reports a new parser with 2% higher attachment score,
trained with gold POS tags at test time while the baseline uses
predicted tags. Name the confound and the fair comparison.

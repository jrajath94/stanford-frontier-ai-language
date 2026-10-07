# Answer key , U03 Neural fundamentals and dependency parsing

Attempt the exercises before reading. Ladders are oral: answer aloud,
then check.

## Remediation

R1. a = x + 2y = 8, f = 32. df/da = z = 4, da/dx = 1, da/dy = 2.
df/dx = 4, df/dy = 8, df/dz = a = 8.

## Breadth

A1. Nodes hold values, edges carry them, forward runs in topological
order, backward multiplies by local derivatives in reverse. The order
guarantees each node is computed after its parents (forward) and
before its parents need its gradient (backward).

A2. Sum: gradient scales with B. Mean: gradient is B-independent in
expectation. Toy: sum 0.6, mean 0.15, B = 4. Switching reduction
without touching lr rescales steps by B.

A3. exp overflows (exp(1000) is inf), inf/inf = nan. Shift: subtract
the max, the e^{-m} cancels in the fraction, largest exponent
becomes 0. Toy: nan vs (0.090, 0.245, 0.665).

A4. Lookup returns rows, backward scatter-adds incoming gradients to
the looked-up rows only. Repeated ids accumulate (np.add.at). Toy: 3
nonzero rows out of 14.

A5. dz/dx = dz/dy @ dy/dx, shapes (1,m) x (m,n). Reverse mode uses
vector-Jacobian products and never forms the Jacobian. Toy: (48, 68).

A6. Linear models draw one straight boundary, XOR-like labels are not
separable (linear stuck at 0.5 accuracy). A ReLU hidden layer folds
the space, the toy MLP reaches loss 0.0008 and accuracy 1.0.

A7. State = (stack, buffer, arcs). SHIFT moves buffer to stack,
LEFT-ARC wires top as head of second-top and pops the dependent,
RIGHT-ARC wires second-top as head of top and pops. Toy: 6 actions,
arcs {(1,0),(2,1),(-1,2)}.

A8. The parser is an action classifier: cross-entropy against the
oracle's action per state. Toy: scores (0.2, 0.5, 2.1), oracle
RIGHT-ARC, loss -log 0.725 = 0.322. Gradients flow into the
embeddings.

A9. SGD: theta - lr g. Momentum: velocity smooths the path. Adam:
per-parameter step ~lr x m/sqrt(v) with bias correction. Toy
quadratic: lr 0.5 converges, 1.5 oscillates, 2.5 diverges.

A10. A gradient check is a contract test between math and code:
central differences vs analytic, tolerance 1e-6 (float64). Toy:
max error 1e-9 on 2 X^T X W. It tests code, not math.

A11. Five contracts: lookup (B,n)->(B,n,d), linear (B,in)->(B,out),
batched scores (B,n,n), softmax keeps shape, CE -> scalar. The toy
bug: (B,V) x (V,) broadcasts to (B,V) instead of contracting to (B,).
Assert shapes, do not hope.

A12. Six stages with shapes: state -> features (24,) -> scores (3,)
-> loss scalar -> gradients (reverse shapes) -> update. Regression
anchors: 6 parser actions, softmax (0.090, 0.245, 0.665), graph
gradients (4, 4, 5), MLP loss 0.7228 -> 0.0008.

## Oral ladders

L1 (graph). Define nodes/edges/order. Run the toy both ways (a = 5,
f = 20, grads 4, 4, 5). Derive the O(edges) reverse-mode cost.
Diagnose the Python-`if` failure (gradient cannot flow through the
untaken branch). Design the 20-random-graph fuzz test.

L2 (batch). Define both reductions. Compute 0.6 vs 0.15. Derive the
1/B factor. Diagnose the copied lr at B = 1024 (noise drops, optimum
lr rises). Design the B sweep {32, 128, 512} with per-B lr search.

L3 (softmax). Define the overflow (exp(1000) = inf). Compute both
versions. Prove the shift identity (e^{-m} cancels). Diagnose
nan-first (check numerics before theory). Design the 100x-logit
scale test.

L4 (embeddings). Define lookup and scatter-add. Write the repeated-id
sum. Derive dL/dE[i] as the sum over positions. Diagnose rare-row
starvation (one update total). Design the update-count correlation
study.

L5 (chain rule). State the matrix form. Compute (48, 68). Derive the
VJP memory saving (O(m+n) vs O(mn)). Diagnose the missing batch axis
in hand-written backward. Design the 100-shape fuzz.

L6 (nonlinear). Define the separability gap. Run the toy numbers.
Derive the hinge tiling (ReLU features compose into XOR). Diagnose
memorization at h = 100. Design the width sweep {1,2,4,16} with a
noisy test set.

L7 (parsing). Define the state and three actions. Replay the 6
actions. Derive the 2n bound (each word shifted once, popped once).
Diagnose the garden-path (greedy cannot backtrack). Design the
greedy vs beam test on garden-path vs normal.

L8 (objective). Define the action-classification reduction. Compute
0.322. Derive the gradient path into embeddings. Diagnose the
95%-actions trap (early errors derail the tree, evaluate trees).
Design the action-accuracy vs attachment-score correlation study.

L9 (optimization). Define the three updates. Compute the quadratic
trio. Derive bias correction (divide out the zero-init pull).
Diagnose lr 0.1 divergence (adaptive is not automatic). Design the
lr grid x 5 seeds for SGD and Adam.

L10 (checks). State the contract. Compute the toy (max error 1e-9).
Derive 2 X^T X W. Diagnose the untested path (forward uses W.T
elsewhere). Design the 10-shape fuzz over three ops.

L11 (shapes). State the five contracts. Compute the (2,3)-vs-(2,)
toy. Derive the broadcast rule. Diagnose the (B,d,n) transpose.
Design the assert-ablation timing study.

## Exercises

E1. Scalar engine matches a = 5.0, f = 20.0, grads (4, 4, 5). `max`
backward routes the full gradient to the argmax input, zero to the
other.

E2. See E1, the max rule is the subgradient at ties (split or pick
first, document the choice).

E3. `batch_grad` on the toy: sum 0.6, mean 0.15. On the 1-D
quadratic, 10 SGD steps from x = 1 at lr 0.1: sum reduction ends 4x
farther than mean (steps scale by B = 4).

E4. E[mean gradient] = (1/B) sum E[g_i] = E[g_1] under IID,
independent of B. QED.

E5. Both softmaxes reproduce nan vs (0.090, 0.245, 0.665).

E6. softmax(z + c) = softmax(z): e^c factors out of numerator and
denominator. Numerical test passes to 1e-12.

E7. `embed_forward`/`embed_backward` with np.add.at: row 1 gets
g1a + g1b, row 5 gets g5, rest zero.

E8. Fancy indexing assignment E[[1,1,5]] = [g1a,g1b,g5] writes g1b
over g1a (last wins), the correct backward needs the sum. This is
why scatter-add exists.

E9. `vjp` for matmul and relu reproduces (48, 68).

E10. (100, 1000) map: full Jacobian 100k entries, VJP holds two
vectors (1100 entries). Ratio ~91x memory.

E11. MLP loop matches loss 0.0008 and accuracy 1.0.

E12. Identity activation: the network collapses to a linear map,
accuracy stays 0.5. Depth without nonlinearity adds nothing.

E13. Actions and oracle reproduce the 6-action list and the gold arc
set.

E14. "a dog barked": SHIFT, SHIFT, LEFT-ARC(det), SHIFT,
LEFT-ARC(nsubj), RIGHT-ARC(root). Arcs: (1,0), (2,1), (-1,2).

E15. `action_loss` returns 0.322 on the toy scores.

E16. Impossible under a correct oracle: the oracle's action sequence
derives the gold tree by construction, so 100% action accuracy
implies the gold tree. What this proves: the 95%-actions failure
comes from classifier errors at test time, where no oracle exists
to keep the parser on the gold path. Training is well-defined,
test-time search is where it breaks.

E17. Both optimizers reproduce the quadratic trio (converge /
oscillate / diverge).

E18. Accept the learner's measured lr (expect Adam to match near a
higher lr than SGD's 0.5, flatter U-curve). The point is the
comparison protocol, not a universal constant.

E19. Max error below 1e-9 on the toy.

E20. dL/dz = p - onehot(y): from L = -z_y + logsumexp(z). Check on a
3-class toy passes below 1e-9.

E21. Contracts asserted through the MLP forward, all pass.

E22. Snippet:

```python
# contract: Q (B,n,d), K (B,n,d) -> S (B,n,n)
S = Q.transpose(0, 2, 1) @ K   # suspect line
```

Diagnosis: (B,d,n) @ (B,n,d) gives (B,d,d), violating the (B,n,n)
contract. If d == n it runs silently and computes attention over the
wrong axis. Fix: `S = Q @ K.transpose(0, 2, 1)`.

E23. Scaffold completed, all four anchors reproduce, full-step
gradient check passes below 1e-6, shape asserts hold.

E24. Accept a complete chain naming every intermediate: e.g.
dL/dE = sum over looked-up rows of (dL/dscores @ dscores/dh @
dh/dfeatures @ dfeatures/dE_row), with shapes (3,) <- (3,4) <-
(4,24) <- (24,8) per row.

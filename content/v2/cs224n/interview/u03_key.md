# Interview key , U03

## B1

Minimum: nodes hold values, edges carry flow, forward visits parents
before children, backward visits children before parents. Strong:
states the O(edges) cost of each pass. Red flag: "topological order
is just sorting". Rubric: 2. Remediation: C01.

## B2

Minimum: sum scales with B, mean does not, switching mid-training
rescales steps by B. Strong: connects to lr scaling rules. Red flag:
"they are the same". Rubric: 2. Remediation: C02.

## B3

Minimum: exp overflows to inf, inf/inf = nan, subtract the max, the
shift cancels in the fraction. Strong: proves the identity. Red flag:
"use float64 instead" as the only fix. Rubric: 2. Remediation: C03.

## B4

Minimum: scatter-add: the repeated row gets the sum of its incoming
gradients, other rows get zero. Strong: names np.add.at and the
last-wins trap. Red flag: "each occurrence updates separately".
Rubric: 2. Remediation: C04.

## B5

Minimum: dz/dx = dz/dy @ dy/dx with shapes (1,m)x(m,n), reverse mode
multiplies vector-Jacobian products and never materializes the (m,n)
matrix. Strong: quantifies the memory saving. Red flag: shape
mismatch in the product. Rubric: 2. Remediation: C05.

## B6

Minimum: the labels are not linearly separable, the ReLU hidden layer
folds the space so the output layer can separate. Strong: cites the
computed gap (1.0 vs 0.5 accuracy). Red flag: "more parameters".
Rubric: 2. Remediation: C06.

## D1 ladder

D1.1 Forward: a = x + y, f = a z. Backward: df = 1, multiply by
local derivatives in reverse.
D1.2 df/dx = 4, df/dy = 4, df/dz = 5.
D1.3 VJP: v^T W for y = Wx, cost O(m + n) memory vs O(mn) for the
Jacobian. Derivation: never form dy/dx, multiply the vector through.
D1.4 The batch axis: the hand-written backward used plain matmul
where batched matmul was needed (or dropped the axis in a sum). At
B = 1 the shapes coincide and the bug hides.
D1.5 Training: reverse-mode (one backward pass for 10M params).
One-time check: finite differences on a tiny slice, or forward-mode
if the tooling exists. Never finite-difference 10M params.

## D2 ladder

D2.1 State (stack, buffer, arcs), SHIFT, LEFT-ARC, RIGHT-ARC.
D2.2 SHIFT, SHIFT, LEFT-ARC, SHIFT, LEFT-ARC, RIGHT-ARC(root), arcs
(1,0), (2,1), (-1,2).
D2.3 Each word is shifted once and popped once: at most 2n actions,
each O(1). No search.
D2.4 Early errors derail the tree: one wrong first action on a long
sentence ruins all later attachments, so action accuracy
overstates tree quality. Evaluate trees (attachment score), not
actions.
D2.5 Flaw: locally optimal actions need not compose into the
globally optimal tree (garden-path). Experiment: greedy vs beam
decoding over the same classifier, the gap concentrates on
garden-path sentences.

## Q1

Shifted: (-2,-1,0), exp (0.1353, 0.3679, 1.0), sum 1.5032,
p = (0.090, 0.245, 0.665). dL/dz = p - onehot(2) = (0.090, 0.245,
-0.335). Strong answer shows the shift and the sum-to-zero check
(0.090 + 0.245 - 0.335 = 0).

## Q2

Central differences: 2 x 8192 = 16384 forward passes per check.
Instead: derive the analytic backward once, check it with finite
differences on a tiny shape (e.g. 3x2) a single time, then trust the
pattern. Never check at full size in the loop.

## I1

Bug: `dE[ids] = grad_out` uses fancy-index assignment, which
overwrites on repeated ids (last gradient wins) instead of
accumulating. On batches without repeats it looks right, with a
repeat it silently drops all but one gradient. Fix: `np.add.at(dE,
ids, grad_out)`. Red flag: "it works, the shapes match". Rubric: 2.
Remediation: C04, C11.

## S1

Arc-standard's projectivity is structural, not a parameter: no
retraining fixes it. Smallest change: add a SWAP action (swap-based
transition system handles crossing arcs) or switch to graph-based
parsing with a maximum spanning tree decoder. The objective and
features can stay.

## S2

The first run takes 8x smaller steps than tuned (sum over 256 vs
mean over 32 differ by 256/32 = 8), so training crawls. Fix: scale
the learning rate by 8 (linear scaling as the starting point) or
keep sum reduction and divide explicitly. Verify by matching the
first-epoch loss curve.

## R1

Confound: the new parser sees gold POS tags at test time, the
baseline sees predicted tags, the comparison measures tag quality,
not parsing. Fair comparison: both use predicted tags (the realistic
setting), or both use gold tags (to isolate the parser). Report both.

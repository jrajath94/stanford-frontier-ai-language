# Interview key , U15

## Breadth

A1. Vectorize the big loops (numpy, C speed). Not for tiny
arrays: call overhead dominates.
A2. Records the forward graph, replays the chain rule backward.
Grad check: autograd vs finite differences to 1e-5.
A3. Tokenize, batch, forward, decode. Pin the model revision or
the hub moves under you.
A4. Embeddings, L transformer blocks (attention + MLP), output
head. Next-token cross-entropy.
A5. Proposal (the claim), milestone (first numbers), poster
(the story), report (the proof). Each needs a falsifiable
claim.
A6. Claim stated, evidence shown, limit named, transfer
answered. Four points.

## Deep ladders

L1. (1) Embeddings, attention, MLP, head. (2) 44,928 at V =
1000, d = 8, L = 4. (3) Vd + nd + L(4d^2 + d + 2dffn + ffn +
d). (4) Capabilities emerge with scale (U06). The miniature
has the plumbing only. (5) Sweep d in {8, 32, 128}, compare
loss curve shapes.

L2. (1) Seeds, versions, data, commands. (2) Mean 0.8045, std
0.0203. (3) Differences inside the std are noise. (4) GPU
nondeterminism or library drift: log those too. (5) The U09
lab receipt: seed 11, numpy 1.26.4, the six commands.

## Analytical

A7. z = (0.8045 - 0.79) / 0.0203 = 0.71. Inside one std:
consistent, noise not a bug.
A8. Define the metric (+1, now 4/5). Cut novelty: ship the
measurement study. Never cut the control group.

## Implementation/debugging

A9. (1) You returned the score matrix, not the output: check
which tensor you returned (cheapest). (2) Missing the final
matmul with V. (3) Wrong projection dim on the output layer.

## Changed-constraint

A10. Cut novelty and the third downstream task. Keep: the
falsifiable claim, the control, one task with honest SEs.
The gates compress, they do not vanish.
A11. SRC-04 lists a custom project option alongside the
default. The honest rule: read the official project
description for the custom track's requirements (proposal
approval) and follow them, do not assume.

## Research critique

A12. Steelman: the miniature has every part, and the loss
curve shape may generalize. Counterexample: capabilities
emerge with scale (U06). A 44k-param model cannot show what
a 7B model does. Mechanics, not emergence.

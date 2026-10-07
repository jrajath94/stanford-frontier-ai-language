# Interview key , U04

## Breadth

A1. The chain rule factors any joint distribution left to right, the
cost is serial generation and no right-context conditioning.
A2. Training feeds gold prefixes, inference feeds model outputs. The
mismatch is exposure bias: errors compound off the training
distribution.
A3. Perplexity = exp(mean NLL): effective choices per word. It
misleads under memorization (train ppl 1.326 says nothing about
generalization) and across tokenizers.
A4. A fixed d-vector summarizing the prefix, weights tie so the model
handles any length with fixed parameters and shares statistics across
positions.
A5. Backprop through the unrolled graph, the gradient sums per-step
contributions. Truncation cuts chains at k: dependencies beyond k are
unlearnable.
A6. Jacobian products grow or shrink as rho^T. Fixes: clip (exploding),
gates/orthogonal init (vanishing), attention (skip the product).

## Deep ladders

L1. (1) P(w_1..n) = prod P(w_t | w_<t), loss = -sum log P. (2)
-4.430, exp(4.430/6) = 4.38. (3) The model definition conditions only
on the prefix, right context has no path into the prediction. (4)
Perplexity averages over positions, direction affects which
dependencies are easy. (5) Controlled completion study, same data,
both directions, human or task metric.

L2. (1) dL/dW_h = sum_t sum_{k<=t} (dL_t/dh_t)(prod_{i} J_i)(dh_k/dW_h).
(2) 0.0115 and 38.34. (3) ||prod J|| <= prod ||J|| <= gamma^T. (4)
Vanishing: short range learns, long range gets no signal. (5) Sweep
spectral radius {0.5, 1.0, 2.0} at fixed budget, 1.0 should win on
long range.

## Analytical

A7. Training perplexity lies (memorization). Experiment: held-out
perplexity plus free-run generation quality from held-out seeds, a
memorizer shows low train ppl, high test ppl, poor generation.
A8. (i) Vanishing: more capacity, same rho^T decay, longer effective
paths get no signal, test: gradient norms by time lag. (ii)
Optimization: bigger model, same data, overfits the six sentences,
test: train/test ppl gap. Cheapest: measure both.

## Implementation/debugging

A9. (1) Check loss before nan: sudden spike -> gradient explosion ->
add global norm clipping. (2) Check lr: too high -> lower it. (3)
Check data: nan/inf in inputs -> sanitize. (4) Check softmax: large
logits -> subtract max. (5) Check init: spectral radius >> 1 ->
orthogonal init.

## Changed-constraint

A10. Survives: the recurrence itself (O(d^2) per step, tiny state),
greedy decoding. Breaks: batching/padding machinery, BPTT training
(no memory for the unroll), train offline, ship only the step.
A11. Training: sequence-level loss (e.g. binary CE on a pooled score)
replaces next-word CE. Decoding: no generation needed, just scoring.
Evaluation: accuracy/AUC replaces perplexity.

## Research critique

A12. Steelman: the forget gate near 1 gives dc_t/dc_{t-1} = 1, an
explicit constant-error path across arbitrary time. Counterexample:
the gates are learned through the same products, a forget gate
initialized near 0 wipes memory every step and gradients still
vanish. The fix needs the forget-bias trick to work at all.

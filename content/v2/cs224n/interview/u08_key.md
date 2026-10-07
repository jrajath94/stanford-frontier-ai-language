# Interview key , U08

## Breadth

A1. Zero-shot describes, few-shot shows. Both only steer the model
without weight updates, both are bounded by the window.
A2. Frozen weights, pattern pickup in the forward pass. Open: the
true mechanism at scale.
A3. Correct, diverse, format-matched. Pick by retrieval or
curation, balance classes.
A4. Extra serial compute on decomposable tasks. Does not prove the
steps caused the answer (rationalization).
A5. dW = BA, rank r, train A, B only. Merge into W once: one
matrix, zero extra latency.
A6. New training erases old tasks on shared weights. Mitigations:
replay, freeze (PEFT), multitask.

## Deep ladders

L1. (1) Instruction + 2 examples + query. (2) k examples eat the
window (n^2 attention). (3) No gradient step: the weights are
unchanged. (4) Format and label priors help regardless. (5)
Shuffled vs true labels: the gap is the task knowledge.

L2. (1) W (d,d) frozen, B (d,r), A (r,d). (2) 2dr = 65536,
4.19M/7B = 0.060%. (3) rank(BA) <= min(rank B, rank A) <= r. (4)
The true update is high-rank, r = 8 cannot express it. (5) r in
{2, 8, 32} on a near task, expect diminishing returns.

## Analytical

A7. (i) Distraction: more shots add noise past the useful pattern
(C02). (ii) Wrong-pattern reinforcement or order effects (C04).
Cheapest tests: the ICL curve per task (find the peak), and
permutation of example order (measure the spread).
A8. No conclusion about the methods: the comparison varied tuning
effort (errors.md U08 T2). The honest statement: under these
unequal budgets they tied, rerun at equal effort.

## Implementation/debugging

A9. (1) Retry with the parse error appended (cheapest). (2) Add
one format example to the prompt. (3) Constrain the schema
(fewer optional fields). (4) Constrained decoding (strongest,
most work).

## Changed-constraint

A10. One base model + 50 LoRA deltas (8 MB each), swap or batch
deltas per request. Breaks: batching across different deltas needs
per-request adapter logic, latency if swapped naively.
A11. No. A new language needs high-rank changes across the model,
r = 8 cannot express them (C07's far-task case). Use full
fine-tuning or continued pretraining.

## Research critique

A12. Steelman: the steps decompose the problem and often check
out, they expose the work for audit. Counterexample: intervention
studies show models produce plausible steps for answers reached
otherwise, the steps are optimized to look right, not to be the
causal path. Useful, not faithful.

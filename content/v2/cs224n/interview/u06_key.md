# Interview key , U06

## Breadth

A1. Mask 15 percent, predict from bidirectional context, loss on
masked only. 80/10/10 avoids the train/test skew on [MASK].
A2. Next-token loss on every position. Trades signal volume (6.74x)
against bidirectional context.
A3. The layer output at a position: word mixed with context.
word2vec is one static vector, contextual disambiguates (0.998 vs
0.424).
A4. Filter (boilerplate/toxicity), dedup (repeats), mix (domain
ratios), decontaminate (test overlap).
A5. 6 FLOPs per param per token (2 forward, 4 backward). 20:1 is the
compute-optimal tokens-per-param ratio.
A6. Test text in training inflates scores. Detect by n-gram overlap,
drop or flag, disclose.

## Deep ladders

L1. (1) Sample 15 percent, apply 80/10/10, masked CE. (2) -log
0.614 = 0.487. (3) Fine-tuning has no [MASK], the model learned a
mask-detector. (4) The recipe was skipped (100 percent [MASK]).
(5) 100 percent vs 80/10/10, no-mask downstream probe.

L2. (1) C = 6ND, L = aN^{-b}. (2) 1.20e18, 0.9659. (3) N x D fixed:
bigger N means less D. (4) Extrapolation past the fit range, or the
data/architecture changed. (5) Ratios 10:1 vs 40:1 at fixed C,
interpolate to 20:1.

## Analytical

A7. No. The ratio is about training efficiency (loss per FLOP), not
benchmark supremacy. The bigger model may win benchmarks while
wasting FLOPs, also, inference favors big models, and the eval may
reward size. Check: compare losses at fixed C, not just benchmark
wins.
A8. (i) The held-out set is easier than train (distribution shift).
(ii) Leakage: the held-out set is in training (memorization). Check:
n-gram overlap between held-out and train, also re-evaluate on a
fresh clean set.

## Implementation/debugging

A9. (1) Data: a corrupt batch (NaN scan the inputs). (2) Batch: loss
spike at a step boundary (check LR schedule). (3) Precision: fp16
overflow (check loss scaling). (4) Hardware: a sick GPU (check
per-rank losses in distributed training).

## Changed-constraint

A10. (1) Cut to the 20:1 ratio (stop overtraining N). (2) Shorter
schedule with the same ratio (fit a new scaling curve). (3) Smaller
model, more data per param is already optimal, so cut N first.
A11. Pretrain smaller (distillation target), prioritize data quality
over size, and plan quantization-aware training, the phone runs the
artifact, so optimize the artifact (size, latency), not the
training loss alone.

## Research critique

A12. Steelman: smooth power laws in loss vs N, D, C across orders of
magnitude let us predict the next scale's loss. Counterexample: the
laws are about loss, not capabilities, data exhaustion, architecture
changes, and emergent task jumps all break them. Bigger is
predictably better at loss within range, nothing more.

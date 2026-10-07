# Answer key , U06 Pretraining, scaling, systems, data

Attempt the exercises before reading. Ladders are oral: answer aloud,
then check.

## Remediation

R1. -log(0.614) = 0.487. Every pretraining loss is a mean of such
numbers.
R2. 0.15 x 512 = 76.8: about 77 training tokens per sequence.
R3. 1e9 x 2 bytes = 2 GB. All memory claims are this multiplication.

## Breadth

A1. Mask 15 percent (80/10/10 recipe), predict from bidirectional
context, loss on masked positions only. Toy: loss 0.487 on "sat".
A2. Next-token loss on every position: n predictions per sequence.
Signal ratio vs MLM at n = 512: 6.74x. Chosen for generation and
scale.
A3. Static (U02): one vector per word. Contextual: the layer output
at the position, a function of the sequence. Toy: same-sense cos
0.998, cross-sense 0.424.
A4. Filter, dedup, mix, decontaminate. Toy: 6 docs -> 3 unique.
Duplicates waste capacity on memorization.
A5. C = 6ND: 2 forward + 4 backward per param per token. Toy: 1e8 x
2e9 -> 1.20e18 FLOPs, 20 tokens/param (the compute-optimal ratio).
A6. L = aN^{-b} (illustrative a=10, b=0.05): doubling N multiplies
loss by 0.9659. Fits predict within range, extrapolation,
new architectures, and data exhaustion break them.
A7. Data-parallel: split the batch, allreduce the gradients. Tax:
2(p-1)/p x S = 1.75 GB/step at p = 8, S = 1 GB. Noise variance falls
as 1/batch (1.026 -> 0.0299 at batch 32).
A8. Test text in training: the benchmark measures recall. Detect by
n-gram overlap, drop or flag, disclose the rule. Toy: 50/1000 leaked
bought 2.9 points (45.0 vs 42.1 percent).
A9. Probe: linear head on frozen features (feature quality).
Fine-tune: train all (adaptability). Gap diagnoses where the work
happened. Toy: 92 vs 55 percent.
A10. Six sections: identity, data, training, evaluation,
limitations, intended use. Facts, not adjectives.
A11. Three levels: held-out perplexity (fit), probes
(representations), benchmarks (ability). Toy: 30.0 train, 36.2
clean held-out, 22.2 leaked (flagged).
A12. Three axes: data (split batch), tensor (split layers), pipeline
(split depth), FSDP shards weights/grads/states. 100B params = 1.2
TB -> 37.5 GB/GPU over 32.

## Oral ladders

L1 (MLM). Write the 80/10/10 recipe. Compute loss 0.487. Derive why
the recipe exists (train/test skew on [MASK]). Diagnose the
fine-tune gap. Design the recipe ablation.

L2 (AR). Write the loss. Compute 6.74x. Derive the signal-vs-context
tradeoff. Diagnose the small-scale MLM win. Design the fixed-FLOP
race.

L3 (contextual). Define both embedding kinds. Compute the cosines.
Derive the mixing from attention. Diagnose the tiny-data static win.
Design the neighbor probe.

L5 (compute). Write C = 6ND. Compute 1.20e18. Derive the N/D
tradeoff at fixed C. Diagnose the deployment-sized model. Design the
ratio test.

L6 (scaling). Write the form. Compute the four losses. Derive
2^{-b}. Diagnose the extrapolation miss. Design the fit-predict test.

L7 (distribution). Write the allreduce tax. Compute 1.75 GB. Derive
the 2S limit. Diagnose the sublinear speedup. Design the scaling
measurement.

L11 (eval). Name the three levels. Compute the triplet. Derive why
the leak flatters. Diagnose a 22.2 held-out. Design the
perplexity-probe correlation test.

## Exercises

E1. `mask_batch`: masked fraction within 0.15 +/- 0.02 on long runs.
E2. Loss 0.487 reproduced.
E3. `ar_loss`: n predictions for length n.
E4. n = 2048: MLM ~307 tokens, ratio 6.67x.
E5. Cosine check reproduces 0.998/0.424.
E6. Static embedding: cos = 1.0 across senses (no disambiguation).
E7. `dedup`: 6 -> 3 unique, first occurrences kept.
E8. Accept: duplicates get k-times the gradient, capacity goes to
recall instead of generalization.
E9. `budget` reproduces 1.20e18 and 20.0.
E10. N = sqrt(1e21/120) = 2.9e9 params, D = 5.8e10 tokens.
E11. `power_law` reproduces all four losses.
E12. b = 0.0500 from the first two points, predicts 3.5880 at 8e8
(matches 3.5879).
E13. `allreduce_bytes` reproduces 1.75 GB.
E14. 2(p-1)/p = 1.9 gives p = 20.
E15. `contaminated` flags the planted leak, nothing else.
E16. Reported 45.0 percent, clean 42.1 percent.
E17. `linear_probe` reproduces the 92 vs 55 gap.
E18. Accept: probe the pretrained body on the new domain first, if
the probe is at chance, expect negative transfer.
E19. Toy card filled, all six sections, numbers traceable.
E20. Accept a cited public card with the missing section named.
E21. `eval_report` reproduces 30.0/36.2/22.2.
E22. Accept: the leaked number measures recall, not ability,
without the flag it will be misread as a win.
E23. `shard_memory` reproduces 37.5 GB.
E24. Accept a paragraph: training is memory-per-GPU bound (weights,
grads, states, activations) solved by sharding, inference is
latency/throughput bound solved by caching and batching, the
bottlenecks differ, so the tricks differ.

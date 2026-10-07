# U02 interview bank , questions

Closed-book. Answer keys are in `u02_key.md`. Do not open the key before
attempting. Quotas per major lesson: 6 breadth, 2 deep ladders of 5
follow-ups, 2 analytical exercises, 1 implementation/debug task, 2
changed-constraint scenarios, 1 research-critique question.

## Breadth (6)

B1. What is shared across heads in MQA, and what stays private?
B2. Write the RoPE rotation for one pair and state the relative
property in one line.
B3. Why does the attention score divide by sqrt(d_k)?
B4. What is the KV cache, and which term in its byte formula does GQA
change?
B5. Contrast learned absolute positions with sinusoidal positions on
extrapolation.
B6. Name the three transformer layouts and the mask each uses.

## Deep ladders (2 x 5)

L1. RoPE mechanics.
- L1.1 Define the per-pair rotation angle.
- L1.2 Toy: rotate (1, 0) by 1.0 rad.
- L1.3 Prove (R(mt)q) . (R(nt)k) depends only on n - m.
- L1.4 Implement rope(x, m), state the checks (identity at m = 0,
  norm preservation).
- L1.5 Compare RoPE with ALiBi, debug a model that collapses past
  trained length, critique "RoPE extrapolates", propose the
  base-size sweep experiment.

L2. KV cache economics.
- L2.1 Define the per-layer cache content and shape.
- L2.2 Toy: cache bytes for 32 layers, 1 KV head, d_k = 128,
  T = 2048, fp16.
- L2.3 Justify prefill/decode split from the recompute cost.
- L2.4 Implement decode_step with append, state the consistency
  check vs uncached forward.
- L2.5 Compare MHA/GQA/MQA caches, debug a serving OOM at long
  context, critique "more heads is always better", propose the
  cache-vs-recompute crossover measurement.

## Analytical exercises (2)

E1. A model has 24 layers, d = 1024, h = 16, T = 4096. Compute the
attention score FLOPs per layer (QK^T + AV) and the fp16 memory of
the scores for B = 1. At what T do the score FLOPs equal the
projection FLOPs (T d^2 per layer, counting QKVO as 4 projections)?
E2. GQA with G = 4 on a 32-layer model, d_k = 128, T = 8192, fp16.
Compute the cache in GiB. If the GPU has 40 GB and weights take
14 GB, does the cache fit for B = 1? For B = 4?

## Implementation/debug task (1)

D1. A GQA conversion (mean-pool KV heads, no continued training)
shows a 5-point benchmark drop. You may inspect the conversion code,
the cache math, and eval logs. List the ordered checks, the most
likely culprit, and the fix. Then write the two asserts that guard
the conversion.

## Changed-constraint scenarios (2)

S1. Context must reach 128k tokens but the RoPE base was tuned for
4k. Name two fixes, the assumption each breaks, and how you detect
success (one metric each).
S2. Training budget allows MHA but serving budget allows only
32 MiB of KV cache at T = 2048 (32 layers, d_k = 128, fp16).
Which attention variant fits? Show the arithmetic. What quality
risk do you accept, and how do you monitor it?

## Research-critique question (1)

R1. "Linear attention is strictly better than softmax attention
because it is O(T d^2)." Present the strongest version of this
claim, then the variance counterexample, then design an experiment
that finds the feature count m where quality matches exact
attention on one task. State the falsification condition.

# Transfer sets key

Full keys for `transfer-sets.md`. Closed-book.

## T1

1. Strong: rare words shatter into many byte tokens, fertility
   explodes, and the model loses subword morphology. The merge
   table was the compression.
2. Strong: train a smaller BPE (2k merges) that fits, or use a
   byte-fallback model with a short context. Both trade quality
   for bytes.
3. Strong: fertility rises, so the same text costs more tokens,
   latency rises linearly with token count.

## T2

1. Strong: RoPE rotation angles beyond the trained range are
   unseen, attention scores go off-distribution, the model
   babbles past 4k.
2. Strong: position interpolation (scale angles down, cheap,
   blurs positions) or NTK-aware scaling (keeps high-frequency
   detail, needs tuning). Both trade fidelity for reach.
3. Strong: perplexity on 16k documents vs 4k slices of the same
   documents, plus a needle-in-haystack retrieval probe you
   build yourself.

## T3

1. Strong: they hit a dead expert, outputs degrade or the
   router must redistribute. Naive fallback drops them.
2. Strong: redistribute to the next-best expert per token, or
   run the dead expert's share on CPU at lower priority.
3. Strong: balanced routing spreads the damage evenly (small
   everywhere), imbalanced routing concentrates it (some
   tokens destroyed, most fine).

## T4

1. Strong: per rank: 32 layers x 2 adapted matrices x
   r(4096+4096) params x 2 bytes = 64 x 8192 x 2 =
   1,048,576 bytes. 20 MiB / 1,048,576 = 20. Max rank 20
   (19 in decimal MB). The dropped factor was the second
   LoRA matrix: each adapted weight needs both B (d x r)
   and A (r x k), r(d+k) params per the lesson's formula,
   not r(d).
2. Strong: fine-grained style and rare-task adaptation go
   first. High-rank directions carry the subtle stuff.
3. Strong: when rank 20 cannot move the needle on the eval,
   or when the task needs full-rank changes (new domain).

## T5

1. Strong: mostly noise. The BT loss fits the 5% signal and
   95% label noise, the reward becomes a noise amplifier.
2. Strong: pilot a clearer rubric on 200 pairs and re-label,
   or drop low-agreement pairs and label more. Never scale
   55%.
3. Strong: neither trains well, but DPO fails cheaper and
   faster. Fix the data first, the algorithm second.

## T6

1. Strong: 64 x 30 s = 32 minutes of verifier time per step,
   serial. Parallel workers divide it, but the bill stands.
2. Strong: cache verifier results per (prompt, program hash),
   and verify a subsample per group with the slow verifier
   while a fast proxy scores the rest.
3. Strong: when the proxy's ranking agrees with the slow
   verifier above 0.9 and the speed ratio exceeds 10x. Then
   audit with the slow one weekly.

## T7

1. Strong: at retrieval, the hostile page enters the context
   as a trusted-looking document. The model reads it as data
   and obeys it as instruction.
2. Strong: (1) mark retrieved text as data with delimiters
   (cheap), (2) scan retrieved docs for instruction patterns
   (medium), (3) sandbox the agent so injections cannot reach
   side effects (pricey, best).
3. Strong: stop, do not act on the injected instruction, log
   the attempt, and either ask the user or skip the document.

## T8

1. Strong: position bias. Every close call may be an order
   artifact.
2. Strong: randomize which model goes first per pair (halves
   the systematic bias), and spend the saved budget on more
   items for tighter bands.
3. Strong: report "unswapped pairs, position bias
   uncorrected", widen the tie band, and refuse to rank
   models within the bias margin.

## T9

1. Strong: confidence-thresholded unmasking (fewer, safer
   steps) and a final autoregressive polish pass on the
   uncertain tokens.
2. Strong: coherence rate vs step count on a fixed set. The
   knee is where the curve flattens.
3. Strong: when the knee sits above your latency budget.
   Then serial is the product answer.

## T10

1. Strong: model calls (the trace is long), tool latency
   (search is slow), and the observation round trips.
2. Strong: cap the trace (fewer steps, same tools) and
   parallelize independent tool calls. Both cut wall time,
   not smarts.
3. Strong: single-shot RAG with no loop. It costs
   multi-step tasks, which then fail or escalate.

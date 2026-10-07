# U03 answer key

Answers to the lesson exercises. Do not open before attempting.

## C01

- R1. 12 * 4096^2 * 32 ~= 6.44e9 params, fp16 -> ~12.9 GB.
- R2. Token ids shift, the embedding rows misalign with the text.
  First visible break: detokenize(tokenize(s)) != s, then garbage
  generations.
- R3. The sampler decides greedy vs stochastic, temperature,
  truncation. The same triple with different samplers is a
  different product.

## C02

- R4. 2/64 = 3.125% of experts active per token.
- R5. f = [1, 0, ...], P ~= [1, 0, ...], L_aux = E * 1 * 1 = E,
  the maximum. The loss screams, the coefficient decides whether
  the model listens.
- R6. The kept top-k gates sum to less than 1. Renormalizing
  keeps the output a convex combination of experts, without it
  the output scale drifts with the dropped mass.

## C03

- R7. 8 * 32768^2 * 2 bytes = 17,179,869,184 = 16 GiB per layer.
- R8. s = 65536 / 2048 = 32.
- R9. Interpolation changes what the position codes mean, the
  model must learn the new scale. Zero-shot, the compressed
  positions mismatch the trained patterns and quality drops.

## C04

- R10. [2, 0] / 0.25 = [8, 0], softmax = [0.9997, 0.0003].
- R11. Uniform over 4: entropy = log 4 = 1.386 nats.
- R12. Division by zero. T = 0 means argmax, implement it as a
  separate branch.

## C05

- R13. Keep [0.5, 0.3], renormalize -> [0.625, 0.375].
- R14. Accumulate 0.5 + 0.3 = 0.8 >= 0.8, keep 2 ->
  [0.625, 0.375].
- R15. When the top-k set is the minimal set with mass >= p:
  top-k mass >= p and top-(k-1) mass < p.

## C06

- R16. The model enters a high-probability cycle ("the" predicts
  "the"): each greedy step continues it because no randomness
  breaks the loop.
- R17. Entropy 0: one token has probability 1, sampling is
  deterministic. Entropy 2 nats: real choice among several
  tokens, samples vary.
- R18. Beam search keeps the top-k hypotheses by score with no
  randomness, it is deterministic search, not sampling from p.

## C07

- R19. System: "You are a precise summarizer. Output 3 bullets."
  User: "<article>". Assistant: (to generate).
- R20. The model misparses roles: the system instruction may be
  treated as user text, or role markers leak into the output.
- R21. Every request pays prefill over 2000 tokens plus the cache
  they occupy, it is a standing per-request tax.

## C08

- R22. "Review: loved it -> positive, Review: dull -> negative,
  Review: <new> ->".
- R23. Recency bias: models weight later examples more, the last
  example's label and format dominate.
- R24. ICL: task varies per request, examples cheap, no training
  infra. Fine-tuning: task fixed and frequent, latency/cost
  matters, behavior must be permanent.

## C09

- R25. Pick 0.9 (A) and 0.2 (B): similarity plus label coverage.
  Two A's teach the same thing twice.
- R26. Recency bias: the last example disproportionately shapes
  the output format and label.
- R27. When the pool is homogeneous and the task is simple, random
  examples already cover the pattern, selection adds nothing.

## C10

- R28. "<problem> Let's think step by step." Generate, then parse
  the final line as the answer.
- R29. 50 decode steps plus the cache they fill, roughly 50x the
  per-token cost of a direct answer, plus prefill of the prompt.
- R30. When each step is error-prone and the task needs no
  decomposition, CoT compounds errors and wastes tokens.

## C11

- R31. A (3/5).
- R32. C(3,2) * 0.7^2 * 0.3 + 0.7^3 = 0.441 + 0.343 = 0.784.
- R33. T = 0 makes every chain identical, voting over identical
  chains is theater. Diversity needs T > 0.

## C12

- R34. 20 memorized are removed: 60/80 = 75%.
- R35. Perplexity measures surprise under one tokenizer, a fluent
  falsehood has low perplexity. Truth needs grounding, not
  fluency.
- R36. Optimizing BLEU yields high-BLEU translations humans rate
  poorly, the metric became the target and stopped measuring
  quality.

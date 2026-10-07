# U01 answer key

Answers to the lesson exercises. Do not open before attempting.

## C01

- R1. Table: w_love = +2, w_hate = -2. "I love this movie": +2 -> pos.
  "I hate this movie": -2 -> neg. "This movie is fine": 0 -> tie, the
  rule must define a tie-break (default neg, or abstain).
- R2. Extractive QA. Input: (question string, context string). Output:
  (start index, end index) into the context. Closed output space over
  spans.
- R3. Open-domain dialogue. The output space is unbounded text, so no
  fixed label set exists, cross-entropy over K classes cannot even be
  written down.

## C02

- R4. After merge 1, splits: low -> l ow </w> (x2), lower ->
  l ow e r </w>, lowest -> l ow e s t </w>, new -> n e w </w>,
  newer -> n e w e r </w>. Pair counts: (l, ow): 2 + 1 + 1 = 4,
  (ow, e): 1 + 1 + 1 = 3, (e, r): 2, (n, e): 2. Merge 2 joins
  (l, ow).
- R5. Before: 25 chars / 6 words = 4.167 tokens per word (script
  counts characters, no end-of-word token). After merge 1
  (o -> ow): low x2 -> 2 tokens each = 4, lower -> l ow e r = 4,
  lowest -> l ow e s t = 5, new -> 3, newer -> 5. Total 21 / 6 =
  3.5 tokens per word. Fertility falls 4.167 -> 3.5, the merge
  helps on this corpus because (o, w) is frequent.
- R6. Invariant: decode(encode(s)) == s for all strings s. Test:
  roundtrip a corpus with tricky bytes (accents, emoji, trailing
  spaces) and assert equality.

## C03

- R7. 32000 * 4096 * 2 bytes = 262,144,000 bytes ~= 250 MiB.
- R8. Rows 3 and 7 update (row 3 gets two gradient contributions).
  All other rows are untouched.
- R9. The gradient only touches rows seen in the batch (dL/dE[id] +=
  dL/de). Unseen rows get zero gradient and keep their init values.

## C04

- R10. Scores: s_true = 1.5, s_neg = [-0.5, 0.2]. Loss =
  -log sigmoid(1.5) - log sigmoid(0.5) - log sigmoid(-0.2) =
  0.201 + 0.474 + 0.799 = 1.474.
- R11. W_out exists only to define the prediction task. The word
  vectors used downstream are the input rows W_in, the output table
  is discarded because its rows are context-role vectors, not word
  vectors.
- R12. On a correct prediction the loss gradient still pushes h
  toward the true context row (the pull is small but nonzero) and
  pushes negatives away. Only at probability 1 is the gradient zero.

## C05

- R13. W = [[1, 0], [0, 1]], U = [[0.5, 0], [0, 0.5]], b = 0,
  x_1 = [1, 0], x_2 = [0, 1]. h_1 = tanh([1, 0]) = [0.762, 0].
  h_2 = tanh([0.5*0.762, 1]) = tanh([0.381, 1]) = [0.364, 0.762].
- R14. 0.8^20 ~= 0.0115. The signal keeps ~1% of its strength.
- R15. Clipping caps large gradients (explosion) but cannot create
  signal where repeated multiplication by < 1 destroyed it. Vanishing
  is a missing-signal problem, not an oversized-signal problem.

## C06

- R16. c_0 = 1.0. Step 1: f = 1, i*g = 0 -> c_1 = 1.0. Step 2:
  f = 0.5, i*g = 0.6 -> c_2 = 0.5*1.0 + 0.6 = 1.1.
- R17. dc_5/dc_0 = 0.9^5 ~= 0.590. The gate value, not a matrix
  product, sets the decay.
- R18. The output gate scales h_t = o_t * tanh(c_t), the cell c_t
  itself evolves independently. The carousel is the c path, the
  output gate only controls what leaves the cell.

## C07

- R19. a = [1/3, 1/3, 1/3]. The context is the mean of the states.
- R20. c = s_2 exactly. Hard selection is the limit of sharp
  attention.
- R21. Without the softmax the weights are unnormalized scores, the
  context scale drifts with the score magnitudes and the "weights
  sum to 1" reading breaks. Gradients also lose the competition
  effect between positions.

## C08

- R22. x: (1, 4, 32). LN(x): (1, 4, 32). attention: (1, 4, 32).
  y = x + a: (1, 4, 32). LN(y): (1, 4, 32). FFN: (1, 4, 32).
  z = y + f: (1, 4, 32). Every step preserves (B, T, d).
- R23. The FFN applies the same MLP independently at each position,
  it has no cross-position weights. Mixing happens only in
  attention.
- R24. Attention: 4 * 32 * 32 = 4096 (Q, K, V, out). FFN with
  hidden 128: 2 * 32 * 128 = 8192. LayerNorms: 4 * 32 = 128. Total
  12416 per block.

## C09

- R25. [[0, -inf, -inf], [0, 0, -inf], [0, 0, 0]].
- R26. exp: [2.718, 0, 7.389], sum 10.107, softmax [0.269, 0, 0.731].
- R27. exp of all -inf is all 0, 0/0 = NaN. Guard: never mask the
  diagonal (each row keeps at least its own position), or add a
  fallback uniform row.

## C10

- R28. d_k = 512 / 8 = 64.
- R29. (4, 8, 16, 16).
- R30. Q, K, V, output: 4 * 128 * 128 = 65536.

## C11

- R31. Mean loss = -(log 0.9 + log 0.9)/2 = 0.105 nats. Perplexity =
  exp(0.105) ~= 1.11.
- R32. -log 0 -> +infinity. In practice the loss explodes, label
  smoothing and careful init keep probabilities off exact 0.
- R33. Teacher forcing feeds all true previous tokens at once, so
  every position computes in parallel. Generation feeds its own
  outputs back, so position t+1 waits for position t.

## C12

- R34. Relative error = |analytic - numeric| / (|analytic| +
  |numeric|). Threshold: < 1e-5 in fp64.
- R35. S = randn(4, 4), A = softmax(S + causal_mask(4)),
  assert allclose(A.sum(-1), 1), assert (A * triu(ones, 1) == 0).all().
- R36. Nondeterministic GPU kernels (e.g. some scatter/add ops) and
  different reduction orders change the last bits. Seeded runs match
  only with deterministic algorithms enabled.

# Notation and shapes , cs224n U01-U08

Symbols keep one meaning across all eight units. Shapes are given as
(batch, sequence, feature) unless stated. All tensors are float32 on CPU
in the executed scripts unless a script states otherwise.

## Scalars

| Symbol | Meaning | Unit |
|--------|---------|------|
| V | vocabulary size | count |
| d | embedding or model dimension | count |
| d_k, d_h | attention head dimension | count |
| h | number of attention heads | count |
| T, n | sequence length | tokens |
| B | batch size | count |
| L | number of layers | count |
| P | parameter count | count |
| D | training token count | tokens |
| C | training compute | FLOPs |
| eta, lr | learning rate | 1/step |
| beta | DPO or KL coefficient | scalar |
| gamma | discount factor (RL) | [0, 1] |
| ppl | perplexity | dimensionless |

## Vectors and matrices

| Symbol | Shape | Meaning |
|--------|-------|---------|
| x_i | (d,) | input embedding of token i |
| e(w) | (d,) | embedding row of word w |
| E | (V, d) | embedding matrix |
| W | (d_out, d_in) | weight matrix |
| Q, K, V | (B, n, d_k) per head | query, key, value |
| A | (n, n) per head | attention weights, rows sum to 1 |
| z | (V,) | logits |
| p, y-hat | (V,) | predicted distribution |
| y | () int or (V,) one-hot | target token or distribution |
| h_t | (d,) | recurrent hidden state at step t |
| s_t | (n,) | parser stack contents at step t |

## Operators

- softmax(z)_i = exp(z_i - max z) / sum_j exp(z_j - max z).
- cross-entropy H(p, q) = -sum_i p_i log q_i. Log is natural log, units
  are nats. Perplexity = exp(cross-entropy).
- KL(p || q) = sum_i p_i log(p_i / q_i), nats.
- cos(u, v) = u . v / (||u|| ||v||).
- sigma(x) = 1 / (1 + exp(-x)).

## Index conventions

- i, j index sequence positions, k indexes vocabulary or heads.
- w_t is token at position t, w_{<t} is the prefix before t.
- Superscripts in parentheses, e.g. x^(l), index layers.
- A colon slice a[i:j] follows Python rules: includes i, excludes j.

## Shape contracts (checked in every implementation)

- Embedding lookup: (B, n) ids -> (B, n, d).
- Attention scores: (B, h, n, d_h) x (B, h, d_h, n) -> (B, h, n, n).
- Next-token logits: (B, n, d) -> (B, n, V). Loss drops the last
  position: targets (B, n) use w_2..w_{n+1} against logits at 1..n.
- Masked LM: loss over masked positions only, shape (num_masked, V).

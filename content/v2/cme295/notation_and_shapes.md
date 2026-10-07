# Notation and shapes , cme295

One meaning per symbol across U01-U05. Shapes use B (batch), T
(sequence length), V (vocabulary), d (model width), h (heads),
d_k = d / h (head width), L (layers), E (experts).

## Core symbols

| Symbol | Meaning | Shape / unit |
|--------|---------|--------------|
| x | input token ids | (B, T), integers in [0, V) |
| E | token embedding table | (V, d) |
| X | embedded sequence | (B, T, d) |
| W_q, W_k, W_v | query/key/value projections | (d, d) each (MHA) |
| Q, K, V | projected queries, keys, values | (B, h, T, d_k) |
| S | attention scores | (B, h, T, T) |
| A | attention weights (softmax rows) | (B, h, T, T), rows sum to 1 |
| M | mask, additive (-inf where forbidden) | (T, T) broadcast |
| z | logits | (B, T, V) |
| p | next-token distribution | (B, T, V), rows sum to 1 |
| T (temp) | sampling temperature | scalar > 0 |
| theta | RoPE base angle per pair | scalar, radians |
| r | LoRA rank | integer, r << d |
| alpha | LoRA scale | scalar, effective scale alpha / r |
| s | quantization scale | scalar > 0 |
| pi_ref | frozen reference policy (DPO/PPO) | same shape as pi_theta |
| pi_old | rollout policy (PPO) | same shape as pi_theta |
| rho | policy ratio pi_theta / pi_old | scalar per token |
| A_hat | advantage estimate | scalar per token |
| eps | PPO clip range | scalar, often 0.2 |
| beta | DPO/KL temperature | scalar > 0 |
| r(x, y) | reward model score | scalar |

## Shape rules

- Attention is per head: scores are (T, T) per head, never (d, d).
- The embedding table is (V, d): row v is the vector for token v.
- Logits are (B, T,  V): one distribution per position.
- KV cache per layer holds K and V for past tokens: 2 x (B, h_kv, T, d_k).
- LoRA update: dW = B A with B (d, r), A (r, d).

## Units

- Probabilities are unitless in [0, 1]. Logits are unitless scores.
- Bits for surprisal (-log2 p). Nats for cross-entropy with natural log,
  state the base every time.
- Memory in bytes, 1 GiB = 2^30 bytes. FLOPs are counts, FLOP/s is rate.
- Angles in radians.

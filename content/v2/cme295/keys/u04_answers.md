# U04 answer key

Answers to the lesson exercises. Do not open before attempting.

## C01

- R1. 6 * 1e9 * 2e10 = 1.2e20 FLOPs.
- R2. 1.2e20 / 5e14 = 2.4e5 s ~= 2.8 GPU-days.
- R3. Input token t must predict token t+1, shifting the targets
  by one aligns each input with its label.

## C02

- R4. Used 600 of 3072 tokens, waste 2472 padding tokens (~80%).
- R5. [1, 1, 0, 0].
- R6. Near-duplicates (reworded copies) pass exact-match dedupe
  and still get memorized.

## C03

- R7. s = 2/127 ~= 0.01575, q = [-127, 0, 127].
- R8. s/2 ~= 0.00787.
- R9. Rows have different ranges, one tensor-wide scale is set by
  the largest row and crushes the small ones. Per-channel gives
  each row its own grid.

## C04

- R10. I = 1e12 / 5e11 = 2 FLOP/byte.
- R11. min(100, 1 * 2) = 2 TFLOP/s (memory-bound).
- R12. Batching reuses the same weight bytes across B tokens:
  bytes stay fixed while FLOPs scale with B, so intensity rises.

## C05

- R13. 50,000 * 300 = 15M tokens.
- R14. [0]*10 + [1]*20.
- R15. SFT data is small, many epochs memorize it and distort the
  base. Low LR and few epochs install the format without damage.

## C06

- R16. "<|user|>Hi<|end|><|assistant|>Hello<|end|><|user|>Thanks
  <|end|><|assistant|>" (marker names vary by model).
- R17. 32000 + 3 = 32003 rows.
- R18. Golden-string test: render a fixed conversation, assert
  byte equality, assert each marker tokenizes to one id.

## C07

- R19. 2 * 1024 * 16 = 32,768.
- R20. alpha/r = 16/8 = 2.0, the update is scaled by 2.
- R21. dW = B A = 0 at init: the adapter starts as the identity
  and the model behaves exactly as the base at step 0.

## C08

- R22. 1e9 * 16 bytes = 16 GB.
- R23. Per layer 2 * 2 * 4096 * 8 = 131,072, 32 layers =
  4,194,304, fraction 4.2e6 / 7e9 ~= 0.06%.
- R24. Activations, KV cache, CUDA context, and headroom. The
  three-term bill is necessary but not sufficient.

## C09

- R25. Max 65504, min normal ~6.1e-5.
- R26. bf16's exponent range matches fp32 (~1e-38..3e38), so
  gradients never underflow, only precision is reduced.
- R27. fp32 master weights accumulate tiny updates exactly,
  in bf16/fp16 they would round to zero and training would stall.

## C10

- R28. Deep capability change (new language, new domain) with
  enough data and budget, rank-r cannot hold it.
- R29. Merged: identical to dense speed. Unmerged: extra B(Ax)
  matmuls per token per adapted layer.
- R30. The base is frozen and shared, each adapter is an
  independent diff. Swap diffs per task without interference.

## C11

- R31. Example: ("Summarize: <short doc>", 3 bullets, no
  invented facts), ("Summarize in French: <doc>", French
  output), ("<doc>", length <= 50 words).
- R32. Safety: zero tolerance (any refusal break fails the
  gate). Capability: tolerance band (e.g. within 1 point of
  baseline).
- R33. When a real failure escapes the set: add it as a canary
  so it never escapes twice.

## C12

- R34. 8.4e16 / 1.5e14 = 560 s ~= 9.3 minutes.
- R35. Weights 26 GB (bf16) + adapter states ~2 GB +
  activations/cache ~30 GB at T = 2048 ~= 58 GB: needs 2x40 GB
  or 1x80 GB.
- R36. Owned: the price row is capex + power, and idle time is
  free. Cloud: per-hour billing, so time overruns cost directly.
  The MFU honesty requirement is the same.

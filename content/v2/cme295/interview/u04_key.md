# U04 interview key

Minimum sufficient explanation, strong answer, common red flags,
scoring rubric, remediation per question. Keep it closed-book.

## Breadth

- B1. Strong: pretraining = -sum log p over all tokens,
  SFT = same sum over response tokens only (prompt masked). Red
  flag: "SFT uses a different loss function". Rubric: both (1),
  difference (1). Remediation: C01, C05.
- B2. Strong: 7e9 * (4 weights + 4 grad + 8 m,v) = 112e9 bytes ~=
  104.3 GiB. Red flag: "weights alone". Rubric: three terms (2).
  Remediation: C08.
- B3. Strong: s = max|w|/127, q = round(w/s), w_hat = q*s. Red
  flag: missing s. Rubric: formulas (2). Remediation: C03.
- B4. Strong: FLOPs per byte, decode sits left of the ridge
  (memory-bound). Red flag: "decode is compute-bound". Rubric:
  definition (1), side (1). Remediation: C04.
- B5. Strong: h = W0 x + (alpha/r) B A x, alpha/r keeps the update
  scale stable across r. Red flag: "alpha is the learning rate".
  Rubric: equation (1), scale (1). Remediation: C07.
- B6. Strong: FLOPs, memory, time, money. Red flag: naming two.
  Rubric: four (2). Remediation: C12.

## Deep ladders

- L1. Strong path: B (d,r), A (r,d), rank <= r, 16.8M vs 65,536
  (256x), alpha/r stabilizes scale, B = 0 gives identity init,
  lora_linear + merge with the 1e-5 check, full wins on deep
  change, step-0 jump = nonzero init, "never forgets" needs
  evals, experiment = sweep r, find the knee. Rubric: 2 per rung,
  10 total.
- L2. Strong path: weights, grads, m+v states, 104.3 vs 13.1
  GiB, frozen params get no grads/states, memory_bill with the
  three terms, nvidia-smi within 20%, 8-bit optimizers as the
  next lever, underprediction at long context = activations,
  three-term bill misses activations/cache, experiment = measure
  peak vs bill over T, fit the activation term. Rubric: 2 per
  rung, 10 total.

## Analytical exercises

- E1. Strong: per matrix 2*5120*16 = 163,840, 80 matrices =
  13,107,200 (~0.10% of 13B), optimizer states 13.1M * 12 =
  ~157 MB. Red flag: forgetting grad bytes. Rubric: params (2),
  fraction (1), states (2). Remediation: C08.
- E2. Strong: I = 1.4e10/1.4e10 = 1 FLOP/byte, attainable 2
  TFLOP/s, 7 ms/token, compute-bound needs I >= 150, so B ~= 150.
  Red flag: "batching does not change intensity". Rubric:
  intensity (1), time (2), batch (2). Remediation: C04.

## Implementation/debug

- D1. Strong order: (1) render tokenized samples, check the mask
  (prompt 0, response 1), (2) check the template matches training,
  (3) inspect data format. Culprit: inverted or missing mask
  (prompt tokens trained) or template mismatch. Fix: correct the
  mask/template, retrain. Assert: loss on prompt positions == 0
  for a canary batch. Red flag: "train longer" as the fix.
  Rubric: checks (3), culprit (2), assert (2). Remediation: C05,
  C06.

## Changed-constraint scenarios

- S1. Strong: weights 14 GB + adapter ~1 GB + activations 15 GB =
  30 GB > 24 GB: does not fit. Changes: (1) shorter T (cuts
  activations), (2) gradient checkpointing (trades compute),
  (3) smaller r or 8-bit base. Red flag: no arithmetic. Rubric:
  rows + verdict (3), two changes (2).
- S2. Strong: full fine-tune (or continued pretraining): a new
  language needs deep change that rank 8 cannot hold. LoRA
  failure mode: saturation, poor fluency. Deciding eval: target-
  language perplexity + a downstream task, LoRA vs full at fixed
  budget. Red flag: "LoRA because it is cheaper" without the
  capacity argument. Rubric: choice + why (2), failure mode (2),
  eval (1).

## Research-critique

- R1. Strong: claim = SFT pairs teach facts the base lacks,
  counterexample = SFT on false answers teaches confident
  falsehoods (it elicits style, not knowledge), experiment = SFT
  on novel fictitious facts vs format-only data, then test
  fact recall vs format adherence separately, falsification =
  fact recall does not improve over the base (then SFT taught
  format, not facts). Red flag: "more SFT data = more knowledge".
  Rubric: steelman (2), counterexample (2), experiment +
  falsification (3). Remediation: C05 item 11.

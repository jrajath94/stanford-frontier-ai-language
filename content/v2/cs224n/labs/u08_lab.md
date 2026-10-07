# Lab U08 , adaptation arithmetic in numpy

Six tasks. Run `python3 labs/u08_lab_run.py` from the `cs224n/`
directory. Numpy only. Record the numbers, then read
`labs/u08_lab_key.md` to verify.

T1. LoRA: d = 4096, r = 8, 32 layers, Q+V. Report per-matrix
params, total, and share of 7B.
T2. Self-consistency: p = 0.6, n = 5. Report the majority
probability. Also report it at p = 0.4.
T3. Adapters: d = 4096, b = 64, 2 per layer, 32 layers. Report per
adapter, per layer, and total.
T4. Forgetting: linear model, task A y = 2x then task B y = -x.
Report task A loss before and after B, and the final w.
T5. Output validation: 5 toy outputs (2 valid JSON). Report the
valid fraction.
T6. Prompt sensitivity: 3 paraphrase scores from the toy scorer.
Report the scores and their std.

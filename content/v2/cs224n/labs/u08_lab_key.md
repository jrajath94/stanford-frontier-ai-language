# Lab key , U08

Execution-verified outputs from `labs/u08_lab_run.py`, run 2026-10-06
(CPython, numpy 1.26.4, CPU). Re-run the script to confirm.

## T1 , LoRA

65536 per matrix, 4.19e6 total (Q+V, 32 layers), 0.060% of 7B.

## T2 , self-consistency

p = 0.6: 0.6826. p = 0.4: 0.3174 (voting hurts below 0.5).

## T3 , adapters

524288 per adapter, 1048576 per layer, 3.36e7 total.

## T4 , forgetting

Task A loss 0.0 before B, 10.1885 after. Final w = -1.000: the
weight serves B now.

## T5 , validation

2/5 valid. The failures are truncation, non-JSON, and empty value.

## T6 , sensitivity

Scores (0.994, 0.919, 0.999), std 0.036. The protocol, not the
numbers, is the deliverable.

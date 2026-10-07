# U04 lab , training budgets in code

Prerequisites: the U04 lesson. Runner: `u04_lab_run.py` (numpy, CPU,
deterministic). Work each task by hand first, then verify with the
runner. Answers and verified outputs: `u04_lab_key.md`.

## Task 1 , LoRA parameter counts

1. Count full vs LoRA parameters for (d = 4096, r = 8) and
   (d = 1024, r = 16). State the ratios.
2. Explain why the ratio is d / (2r).

## Task 2 , quantization by hand

w = [-1.2, -0.3, 0.1, 0.7, 1.5].

1. Compute the symmetric int8 scale, the codes, and the max
   reconstruction error. Verify error <= s/2.
2. Quantize [-2, 0, 2].

## Task 3 , Adam memory bill

1. Compute the full fp32 Adam bill for 7B and 1B parameters.
2. Compute the frozen-bf16-base + LoRA bill (0.06% trainable).
3. State the ratio and the binding term.

## Task 4 , roofline

1. Intensity and attainable throughput for 1e12 FLOPs, 5e11 bytes
   (bw 1 TB/s, peak 100 TFLOP/s).
2. Attainable throughput and % of peak for the decode toy
   (I = 0.07, bw 2 TB/s, peak 300 TFLOP/s).

## Task 5 , SFT budget

N = 7B, D = 2M tokens.

1. Compute training FLOPs and wall time at 150 TFLOP/s effective.
2. State whether this fits a 1-hour maintenance window.

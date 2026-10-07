# U04 lab key , execution-verified outputs

Runner: `u04_lab_run.py`, executed 2026-10-06, CPython, numpy 1.26.4,
CPU. Outputs below are observed, not predicted.

## Task 1

Observed:

```
lab1 d=4096 r=8 full=16777216 lora=65536 ratio=256
lab1 d=1024 r=16 full=1048576 lora=32768 ratio=32
```

Ratio = d^2 / (2dr) = d / (2r): 4096/16 = 256, 1024/32 = 32.

## Task 2

Observed:

```
lab2 scale=0.01181 q=[-102, -25, 8, 59, 127] maxerr=0.00551 bound=0.00591
lab2 toy q=[-127, 0, 127]
```

0.00551 <= 0.00591: the s/2 bound holds.

## Task 3

Observed:

```
lab3 full fp32 7B GiB: 104.3
lab3 full fp32 1B GiB: 14.9
lab3 lora bf16 base + adapter GiB: 13.1
```

Ratio 104.3 / 13.1 ~= 8.0. The binding term in full training is the
optimizer states (8 of 16 bytes/param).

## Task 4

Observed:

```
lab4 intensity=2.0 attainable=2.0 TFLOP/s
lab4 decode attainable=0.140 TFLOP/s (0.05% of peak)
```

Decode at 0.05% of peak: memory-bound, exactly as the lesson claims.

## Task 5

Observed:

```
lab5 SFT FLOPs=8.40e+16
lab5 time at 150 TFLOP/s: 560 s
```

560 s < 1 hour: fits the window with room for eval.

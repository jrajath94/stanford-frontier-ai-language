# U01 lab key , execution-verified outputs

Runner: `u01_lab_run.py`, executed 2026-10-06, CPython, numpy 1.26.4,
CPU. Outputs below are observed, not predicted.

## Task 1

Observed:

```
lab1 merge 1: ('o', 'w') count 4 fertility 3.500
lab1 merge 2: ('l', 'ow') count 4 fertility 2.833
```

Fertility before any merge: 4.167 (25 chars / 6 words). Merge 1 joins
(o, w), merge 2 joins (l, ow). Each merge shortens the frequent words.

## Task 2

Observed:

```
lab2 row sums: [1. 1. 1. 1.]
lab2 future triangle max: 0.0
lab2 row0: [1. 0. 0. 0.]
```

Row 0 attends only to itself: every other cell in its row is masked,
so the softmax puts all mass on position 0.

## Task 3

Observed:

```
lab3 rho=0.5 steps=10 scale=0.000977
lab3 rho=0.8 steps=20 scale=0.011529
lab3 rho=1.2 steps=10 scale=6.191736
```

(0.5, 10): vanishing. (0.8, 20): weak but nonzero. (1.2, 10):
exploding. Clipping caps the 6.19x growth, it cannot restore the
0.000977 signal because the information was multiplied away.

## Task 4

Observed:

```
lab4 cell trace: [1.0, 1.0, 1.1]
lab4 dc2/dc0: 0.5
```

0.5 equals the product of forget gates (1.0 * 0.5), not 0.5^2 from a
matrix power. The gate values set the decay directly: this is the
carousel.

## Task 5

Observed:

```
lab5 per-layer params: 49408
lab5 embedding params: 64000
lab5 scores shape: (2, 4, 8, 8)
lab5 logits shape: (2, 8, 1000)
```

As T grows, the (B, h, T, T) scores dominate memory, the d^2 weights
are fixed. The T^2 term is the wall.

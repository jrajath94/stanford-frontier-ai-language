# U09 lab key , execution-verified outputs

Runner: `u09_lab_run.py`, executed 2026-10-07, CPython 3.12.3,
numpy 1.26.4, CPU. Outputs below are observed, not predicted.

## Task 1

Observed:

```
lab1 step1 unmasked=4 masked=4
lab1 remasked=0 step2 unmasked=6 masked left=2
```

Two parallel steps cover 6 of 8 tokens. The low-confidence
tail (0.4, 0.3) waits for later steps, that is the refinement.

## Task 2

Observed:

```
lab2 attention rows:
[[0.6698 0.3302]
 [0.3302 0.6698]]
lab2 row sums: [1. 1.]
lab2 logits:
[[ 0.6698  0.3302 -0.6698 -0.3302]
 [ 0.3302  0.6698 -0.3302 -0.6698]]
lab2 argmax per position: [0, 1]
```

Rows sum to 1, shapes check out. Position 2 picks token 1.
The whole transformer in sixteen multiplies.

## Task 3

Observed:

```
lab3 objectives=24 mapped=21 orphans=3 coverage=0.8750
```

87.5% mapped. The 3 orphans become study priorities or
explicit scope cuts, never silent gaps.

## Task 4

Observed:

```
lab4 attention downstream=15
lab4 policygrad downstream=9
lab4 retrieval downstream=7
```

Attention is the keystone. Study it first, test it hardest,
everything else leans on it.

## Task 5

Observed:

```
lab5 equal weights: AR=8.00 diffusion=6.00
lab5 latency weights: AR=2.40 diffusion=2.40
```

Equal weights: AR wins on maturity. Latency-heavy weights:
a tie. The weights are the decision, the matrix just does
the arithmetic.

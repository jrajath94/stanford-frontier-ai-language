# U03 lab key , execution-verified outputs

Runner: `u03_lab_run.py`, executed 2026-10-06, CPython, numpy 1.26.4,
CPU. Outputs below are observed, not predicted.

## Task 1

Observed:

```
lab1 T=0.5 probs=[0.979 0.018 0.002 0.   ] entropy=0.110
lab1 T=1.0 probs=[0.831 0.112 0.041 0.015] entropy=0.595
lab1 T=2.0 probs=[0.579 0.213 0.129 0.078] entropy=1.110
```

T -> 0: one-hot on the argmax. T -> infinity: uniform.

## Task 2

Observed:

```
lab2 top-k=2: [0.625 0.375 0.    0.   ]
lab2 top-p=0.8 keep=2: [0.625 0.375 0.    0.   ]
```

They coincide here because the top-2 set is the minimal set with
mass >= 0.8.

## Task 3

Observed:

```
lab3 chosen experts: [4, 3] weights: [0.681 0.319]
lab3 active fraction: 0.2500
```

2 of 8 experts active: 25% of the expert compute per token.

## Task 4

Observed:

```
lab4 votes: {'382': 1, '390': 1, '391': 3} winner: 391
lab4 majority accuracy p=0.6 N=5: 0.683
```

Majority 391 (3/5). The Condorcet math gives 0.683 under
independence, real chains are correlated, so treat it as an upper
bound.

## Task 5

Observed:

```
lab5 T=32768 h=8 scores fp16 GiB: 16.0
lab5 T=131072 h=32 scores fp16 GiB: 1024.0
```

At 131072 the scores alone need 1 TiB per layer: memory is the
wall, long before position codes matter.

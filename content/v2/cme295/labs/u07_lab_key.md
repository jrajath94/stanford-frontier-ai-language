# U07 lab key , execution-verified outputs

Runner: `u07_lab_run.py`, executed 2026-10-07, CPython 3.12.3,
numpy 1.26.4, CPU. Outputs below are observed, not predicted.

## Task 1

Observed:

```
lab1 R=8 K=20 H=6 recall=0.7500 precision=0.3000
lab1 R=8 K=40 H=7 recall=0.8750 precision=0.1750
```

Doubling k buys 0.125 of recall and costs 0.125 of precision.
The funnel widens both ways.

## Task 2

Observed:

```
lab2 alpha=0.5 fused=[0.65 0.55 0.5 ] order=[0, 1, 2]
lab2 alpha=0.8 fused=[0.8 0.4 0.5] order=[0, 2, 1]
lab2 RRF X=0.0320 Y=0.0323 winner=Y
```

Alpha 0.8 trusts BM25 more and flips docs 2 and 3. RRF needs
no normalization: Y's steady (2, 2) beats X's spiky (1, 4).

## Task 3

Observed:

```
lab3 budget=6400 total5=5100 fits=True
lab3 total6=7100 cut=700 tokens from last doc
```

Five docs fit with 1300 tokens to spare. The sixth doc
overflows by 700, which the packer cuts from its tail.

## Task 4

Observed:

```
lab4 guard fires at step 2 (repeat rule)
```

Two identical actions in a row trip the repeat guard before
the cap matters. The runaway signature is repetition, not
length.

## Task 5

Observed:

```
lab5 retrieval 47 (0.47)
lab5 model 31 (0.31)
lab5 tool 22 (0.22)
lab5 fix first: retrieval
```

Nearly half the errors are upstream. Fix retrieval first,
the histogram says so.

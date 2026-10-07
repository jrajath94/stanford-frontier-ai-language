# U08 lab key , execution-verified outputs

Runner: `u08_lab_run.py`, executed 2026-10-07, CPython 3.12.3,
numpy 1.26.4, CPU. Outputs below are observed, not predicted.

## Task 1

Observed:

```
lab1 debiased=0.6250 order gap=0.25
```

The 0.25 order gap is pure position bias. The debiased 0.625
is the number you quote.

## Task 2

Observed:

```
lab2 ECE=0.0740 worst bin conf=0.95 gap=0.17
```

Seven points of average dishonesty, and the most confident
bin is the worst liar. Recalibrate before trusting scores.

## Task 3

Observed:

```
lab3 n=100 w=62 interval=[0.522, 0.709] half-width=0.0935
lab3 n=400 w=248 interval=[0.572, 0.666] half-width=0.0474
```

Four times the items, half the band. The 1/sqrt(n) law in
one line of output.

## Task 4

Observed:

```
lab4 models=10 items=500 calls=45000 bill=$90.00
lab4 models=6 items=200 calls=6000 bill=$12.00
```

Price the eval before you run it. The bill function is the
cheapest line in the whole course.

## Task 5

Observed:

```
lab5 score=0.92 band=[0.893, 0.941]
lab5 score=0.91 band=[0.882, 0.932]
```

The bands overlap by a mile. Verdict: tie, not a win.

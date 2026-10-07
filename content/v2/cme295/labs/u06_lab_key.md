# U06 lab key , execution-verified outputs

Runner: `u06_lab_run.py`, executed 2026-10-07, CPython 3.12.3,
numpy 1.26.4, CPU. Outputs below are observed, not predicted.

## Task 1

Observed:

```
lab1 rewards=[1, 1, 0, 0] mu=0.5000 sigma=0.5000 A=[ 1., 1.,-1.,-1.]
lab1 rewards=[3, 1, 1, 1] mu=1.5000 sigma=0.8660 A=[ 1.7321,-0.5774,-0.5774,-0.5774]
```

The lone winner in case 2 gets +1.73, the strongest push in the
group. Scale-free: the ranking drives the update, not the raw
reward units.

## Task 2

Observed:

```
lab2 p=0.2 k= 1 pass@k=0.2000
lab2 p=0.2 k= 2 pass@k=0.3600
lab2 p=0.2 k= 4 pass@k=0.5904
lab2 p=0.2 k= 8 pass@k=0.8322
lab2 p=0.2 k=16 pass@k=0.9719
```

Doubling from 8 to 16 buys 0.14, doubling from 1 to 2 buys 0.16.
Diminishing returns set in fast.

## Task 3

Observed:

```
lab3 T=20 p=0.2835 k=8.0 score=0.9305
lab3 T=40 p=0.4866 k=6.0 score=0.9817
lab3 T=60 p=0.6321 k=4.0 score=0.9817
lab3 T=80 p=0.7364 k=2.0 score=0.9305
```

The flat-top zone is T = 40-60, both ends of the budget hurt.
Curves are hypothetical, the shape (interior optimum) is the
lesson, not the numbers.

## Task 4

Observed:

```
lab4 hack gap=0.52
lab4 false passes=36 of 200
```

A 52-point gap between verifier pass and strict grade is a
compromised verifier. Decision: stop, harden, resume.

## Task 5

Observed:

```
lab5 groupnorm delta=-0.17 z=-8.5
lab5 kl delta=-0.04 z=-2.0
lab5 smallG delta=-0.09 z=-4.5
```

Group norm and small G are real (beyond 4 sigma). The KL delta
at 2 sigma is suggestive, rerun with more seeds before
claiming it.

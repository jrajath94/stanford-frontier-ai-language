# U05 lab key , execution-verified outputs

Runner: `u05_lab_run.py`, executed 2026-10-06, CPython, numpy 1.26.4,
CPU. Outputs below are observed, not predicted.

## Task 1

Observed:

```
lab1 r_w=1.2 r_l=0.4 P=0.6900 loss=0.3711
lab1 r_w=0.5 r_l=0.5 P=0.5000 loss=0.6931
```

The 0.8 gap buys 0.19 of probability and cuts the loss nearly in
half.

## Task 2

Observed:

```
lab2 rho=2.0 A=1.0 obj=1.200
lab2 rho=0.5 A=-2.0 obj=-1.600
lab2 rho=1.0 A=1.5 obj=1.500
```

Clip binds in cases 1 (cap at 1.2) and 2 (floor at 0.8 x -2).
Case 3 is inside the trust zone.

## Task 3

Observed:

```
lab3 margin=0.1400 P=0.5349 loss=0.6256
lab3 margin=0.3000 P=0.5744 loss=0.5544
```

Case 2 has a wider implicit-reward gap (0.3 vs 0.14): the policy
beats the reference harder on the chosen response.

## Task 4

Observed:

```
lab4 KL=0.0823 nats
lab4 KL one-hot=0.6931 nats
```

The one-hot policy pays 0.693 nats (log 2): full commitment costs
exactly one bit against a uniform reference.

## Task 5

Observed:

```
lab5 W62 L28 T10: excl=0.689 half=0.670
lab5 W70 L20 T10: excl=0.778 half=0.750
```

Two conventions, two numbers: report both or the comparison is
meaningless.

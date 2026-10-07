# U02 lab key , execution-verified outputs

Runner: `u02_lab_run.py`, executed 2026-10-06, CPython, numpy 1.26.4,
CPU. Outputs below are observed, not predicted.

## Task 1

Observed:

```
lab1 T=4096 d=512 exact=8589934592 approx(d^2)=1073741824 ratio=8.0
lab1 T=1024 d=512 m=256 exact=536870912 approx=134217728 ratio=4.0
```

The ratio grows linearly with T (8.0 at T = 4096 vs 4.0 at T = 1024).

## Task 2

Observed:

```
lab2 rope(1,0) m=1 theta=1: [0.5403 0.8415]
lab2 relative max err: 1.11e-16
lab2 norm preserved: True
```

(cos 1, sin 1). The relative property holds to machine precision, the
rotation preserves length.

## Task 3

Observed:

```
lab3 MHA cache MiB: 1024.0
lab3 GQA-8 cache MiB: 256.0
lab3 MQA cache MiB: 32.0
```

MQA/MHA ratio: 32.0.

## Task 4

Observed:

```
lab4 alibi weights: [0.992  0.0041 0.0025 0.0015]
lab4 distance-3 penalty factor: 0.2231
```

The factor exp(-1.5) = 0.2231 matches the hand computation in the
lesson.

## Task 5

Observed:

```
lab5 sink weights: [0.909 0.045 0.045]
lab5 sink mass: 0.909
```

With no good match, the softmax dumps 90.9% of the mass on position
0. The sink is the softmax's need to put mass somewhere.

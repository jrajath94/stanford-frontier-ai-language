# Lab key , U03

Execution-verified outputs from `labs/u03_lab_run.py`, run 2026-10-06
(CPython, numpy 1.26.4, CPU). Re-run the script to confirm.

## T1 , softmax

Naive: (nan, nan, nan). Stable: (0.090, 0.245, 0.665), sum 1.0. The
shift changes the math not at all and the floats completely.

## T2 , graph

a = 5.0, f = 20.0, df/dx = 4.0, df/dy = 4.0, df/dz = 5.0. Chain rule
at every node, no exceptions.

## T3 , MLP vs linear

MLP: loss 0.7228 -> 0.0008, accuracy 0.375 -> 1.000. Linear: loss
0.6931, accuracy 0.500, stuck at the majority class. The task is not
linearly separable, the hidden ReLU layer is the difference.

## T4 , parser

Actions: SHIFT, SHIFT, LEFT-ARC, SHIFT, LEFT-ARC, RIGHT-ARC(root).
Arcs: {(1,0), (2,1), (-1,2)}, matching gold. Six actions, linear
time, exact recovery on the toy.

## T5 , batch gradient

Sum 0.6, mean 0.15, B = 4. Mean = sum / B exactly.

## T6 , embedding scatter

Nonzero rows: 1 and 5. Row 1 holds 0.75 in every entry (0.5 + 0.25,
the scatter-add sum). Plain fancy-index assignment gives 0.25 (last
write wins). The scatter-add is not optional with repeated ids.

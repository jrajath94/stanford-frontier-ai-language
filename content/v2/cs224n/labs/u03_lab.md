# Lab U03 , graphs, gradients, and parsing

Six tasks. Run `python3 labs/u03_lab_run.py` from the `cs224n/`
directory. Numpy only. Record the numbers, then read
`labs/u03_lab_key.md` to verify.

T1. Softmax of (1000, 1001, 1002): report the naive result and the
shifted result, plus the sum of the shifted result.
T2. Forward and backward on f = (x + y) x z at (2, 3, 4): report a,
f, and the three gradients.
T3. Train the 2-layer MLP on the 8-point toy task (2000 steps, lr
0.5). Report loss and accuracy before and after. Train the linear
baseline and report its final loss and accuracy.
T4. Run the arc-standard oracle on "the cat sat". Report the action
list and the arc set, and whether it matches gold.
T5. Per-example errors (0.5, -0.25, 0.25, 0.1): report the sum
gradient, the mean gradient, and B.
T6. Embedding table (14, 8), batch ids [[1, 1, 5]]: scatter-add three
gradient rows and report which table rows are nonzero and what row 1
holds. Show that plain fancy-index assignment gives the wrong answer
on the repeated id.

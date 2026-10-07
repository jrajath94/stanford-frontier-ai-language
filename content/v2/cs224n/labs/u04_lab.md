# Lab U04 , recurrence in numpy

Six tasks. Run `python3 labs/u04_lab_run.py` from the `cs224n/`
directory. Numpy only. Record the numbers, then read
`labs/u04_lab_key.md` to verify.

T1. Hand-unroll h_t = tanh(x_t + 0.5 h_{t-1}) for x_1 = (1,0),
x_2 = (0,1), h_0 = (0,0). Report h_1 and h_2.
T2. Train the toy word RNN (300 epochs, lr 0.1, clip 5.0). Report
loss, next-word accuracy, and training perplexity before and after.
T3. Linear RNN gradient norms: report rho^T for rho in {0.8, 1.2}
at T in {1, 5, 10, 20}.
T4. At t = 2 of sentence 1, report the forced input word, the free
input word, and KL(forced || free).
T5. Clip a random gradient of norm ~29 at cap 1.0: report both
norms and whether the direction survived.
T6. Logits (2.0, 1.0, 0.5, 0.1): report the greedy pick, the top
probability at T = 0.5, and at T = 2.0.

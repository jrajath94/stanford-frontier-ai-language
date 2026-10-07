# Lab U07 , preference arithmetic in numpy

Six tasks. Run `python3 labs/u07_lab_run.py` from the `cs224n/`
directory. Numpy only. Record the numbers, then read
`labs/u07_lab_key.md` to verify.

T1. Bradley-Terry: r_chosen = 1.2, r_rejected = 0.3. Report P and
the loss.
T2. DPO: beta = 0.1, chosen log-ratio 0.5, rejected -0.3. Report
the loss.
T3. KL: pi = (0.5, 0.3, 0.2), ref = (0.4, 0.4, 0.2). Report KL.
T4. PPO: ratio 1.3, advantage 0.5, eps 0.2. Report unclipped and
clipped.
T5. Reward model: pairs (1.2, 0.3), (0.8, 0.9), (2.0, 0.5). Report
each loss and the mean.
T6. Reward hacking: proxy (0.9, 0.4), true (0.3, 0.5). Report both
argmaxes.

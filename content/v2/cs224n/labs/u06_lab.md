# Lab U06 , pretraining arithmetic in numpy

Six tasks. Run `python3 labs/u06_lab_run.py` from the `cs224n/`
directory. Numpy only. Record the numbers, then read
`labs/u06_lab_key.md` to verify.

T1. Masked-LM toy: vocab of 8, sentence "the cat sat on mat", mask
"sat". Report the loss and p(true).
T2. n = 512: report AR training tokens, MLM training tokens, and the
ratio.
T3. N = 1e8, D = 2e9: report C = 6ND and tokens per param. Report N
and D at C = 1e21 with a 20:1 ratio.
T4. Ring allreduce: report bytes per step for p = 8, S = 1 GB, and
the p where the tax passes 1.9 GB.
T5. Training memory for 1B params (fp16 weights/grads, fp32 Adam):
report the three parts and the total.
T6. Power law L = 10 N^-0.05: report L at N = 1e8, 2e8, 4e8, 8e8 and
the doubling ratio.

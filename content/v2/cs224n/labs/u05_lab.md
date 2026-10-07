# Lab U05 , attention in numpy

Six tasks. Run `python3 labs/u05_lab_run.py` from the `cs224n/`
directory. Numpy only. Record the numbers, then read
`labs/u05_lab_key.md` to verify.

T1. Hand-compute one-head attention: q = (1,0), K rows (1,0),
(0,1), (-1,0), V rows (2,0), (0,2), (-2,0), d_k = 2. Report scores,
weights, output.
T2. d_k = 64, 8 random keys: report max weight and entropy with and
without the 1/sqrt(d_k) scale.
T3. 4x4 random scores with causal mask: report row sums and the
upper-triangle max.
T4. MHA shape trace for n = 8, d = 32, h = 4: report per-head
scores, concat, and W_o shapes.
T5. Layer norm of (1,2,3,4,5): report mean and variance. Report the
block output when the sublayer is zero.
T6. KV cache at n = 16, d = 8 per head per layer: report total work
with and without the cache, and the ratio.

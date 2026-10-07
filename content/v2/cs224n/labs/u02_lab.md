# Lab U02 , word vectors from scratch

Six tasks. Run `python3 labs/u02_lab_run.py` from the `cs224n/`
directory. Numpy only. The script prints every number the tasks ask
for. Record them, then read `labs/u02_lab_key.md` to verify.

T1. Build the (center, context) pairs on the 18 training tokens
(stopwords removed, window 2). Report the pair count.
T2. Take v_c = (0.30, -0.20), u_o = (0.10, 0.40), two fixed noise
vectors, lr = 0.5. Apply one negative-sampling gradient step and
report the new v_c. State whether it moved toward u_o.
T3. Finite-difference check of your T2 gradient: central differences,
eps = 1e-6. Report the max component error and the verdict at
tolerance 1e-6.
T4. Compute PMI(cat, sat) on the toy bigram counts and the shifted
target with k = 2.
T5. Run the analogy king - man + woman on the toy 2-D vectors.
Report the answer and cosine with the a/b/c exclusion, then without
it, and explain the difference.
T6. Compute the noise distribution (power 0.75) on the training
tokens and report the top three. Then train 200 epochs twice: once on
the raw 35 tokens, once on the 18 training tokens. Report the
cat/dog vs cat/mat cosine gap in both runs.

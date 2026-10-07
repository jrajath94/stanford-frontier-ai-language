# Lab key , U02

Execution-verified outputs from `labs/u02_lab_run.py`, run 2026-10-06
(CPython, numpy 1.26.4, CPU). Re-run the script to confirm.

## T1 , pairs

18 training tokens (stopwords removed), 66 (center, context) pairs at
window 2.

## T2 , one gradient step

v_c moves from (0.30, -0.20) to (0.386, 0.043) at lr 0.5. Distance to
u_o shrinks: the vector moves toward the positive context vector.

## T3 , finite differences

Max component error 2.76e-10 at eps 1e-6, below the 1e-6 tolerance:
pass. The analytic negative-sampling gradient is correct.

## T4 , PMI

PMI(cat, sat) = 1.569 nats. Shifted target with k = 2: 0.876. This is
the dot product the vectors approach at the Levy-Goldberg optimum.

## T5 , analogy

With the a/b/c exclusion: "queen", cosine 0.9992. Without it:
"woman", cosine 1.0, because the query (0.20, 1.40) is exactly twice
the woman vector in this toy. The exclusion rule exists for this
reason.

## T6 , frequency bias

Noise top three, raw tokens: the 0.232, cat 0.129, sat 0.088.
Filtered tokens: cat 0.228, sat 0.155, rug 0.115. After 200 epochs,
the cosine gap (cat/dog minus cat/mat) is -0.126 on raw tokens
(collapsed: "the" dominates negatives) and +0.358 on filtered tokens
(separated). Frequency bias is visible in the gap.

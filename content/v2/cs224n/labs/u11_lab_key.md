# Lab key , U11

Execution-verified outputs from `labs/u11_lab_run.py`, run 2026-10-07
(CPython, numpy 1.26.4, CPU). Re-run the script to confirm.

## T1 , majority at p = 0.6

0.6000, 0.6826, 0.7535, 0.8256. Diminishing gains.

## T2 , reversal at p = 0.4

0.3174 (n = 5), 0.2465 (n = 11). Voting hurts below 0.5.

## T3 , speculative decode

E = 2.941 accepted of 5 drafts at a = 0.7.

## T4 , allocation

10x10: 3.935. 1x100: 0.993. Spread the budget.

## T5 , cost-quality

Knee near n = 16: +0.13, +0.07, +0.04 per 4x cost step.

## T6 , failure counts

Wrong tool 14, bad args 11, no stop 9, retrieval miss 12,
rationalized 8, other 6. Total 60.

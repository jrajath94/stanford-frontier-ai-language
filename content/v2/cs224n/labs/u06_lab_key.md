# Lab key , U06

Execution-verified outputs from `labs/u06_lab_run.py`, run 2026-10-06
(CPython, numpy 1.26.4, CPU). Re-run the script to confirm.

## T1 , MLM toy

Loss 0.487, p(true) 0.614. One masked token, one cross-entropy term.

## T2 , signal ratio

AR 512 tokens, MLM 76 tokens, ratio 6.74x at n = 512.

## T3 , compute budget

C = 1.20e18 FLOPs, 20.0 tokens/param. At C = 1e21 and 20:1: N =
2.89e9 params, D = 5.77e10 tokens.

## T4 , allreduce

1.75 GB per step at p = 8, S = 1 GB. The tax passes 1.9 GB at p =
20 (and approaches 2.0 GB as p grows).

## T5 , training memory

Weights 2 GB, grads 2 GB, Adam m+v 8 GB, total 12 GB for 1B params.
Activations are extra.

## T6 , power law

3.9811, 3.8455, 3.7145, 3.5879. Doubling ratio 0.9659: each doubling
cuts 3.4 percent.

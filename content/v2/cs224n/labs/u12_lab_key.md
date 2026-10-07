# Lab key , U12

Execution-verified outputs from `labs/u12_lab_run.py`, run 2026-10-07
(CPython, numpy 1.26.4, CPU). Re-run the script to confirm.

## T1 , BPE merges

6 merges. Final pieces: "aaabdaaaba c". Frequent pairs fuse
first, the tail merges fuse noise.

## T2 , fertility

English 1.3, German 1.6, Turkish 2.1, Amharic 2.8 tokens per
word.

## T3 , rare word coverage

Top 1000 cover 0.765. The tail 0.235 shatters into pieces.

## T4 , cost inequality

Ratios: x1.00, x1.25, x1.71, x2.42. Per-token pricing taxes by
script.

## T5 , morphology

Agreement 1/3. BPE cuts by frequency, not by meaning.

## T6 , code switch

en en en en es es en. Two switch points, one Spanish run.

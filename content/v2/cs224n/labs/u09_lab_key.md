# Lab key , U09

Execution-verified outputs from `labs/u09_lab_run.py`, run 2026-10-07
(CPython, numpy 1.26.4, CPU). Re-run the script to confirm.

## T1 , ReAct trace

7 steps, 92 tokens. The two Observations carry the new facts.

## T2 , tool validation

4/6 valid. Failures: missing argument, unknown tool name.

## T3 , RAG recall at k

k = 1: 0.000. k = 3: 0.333. k = 5: 0.667. k = 10: 1.000. k = 1
finds nothing: the first relevant chunk is at rank 2.

## T4 , chunking

Overlap 0: 5 chunks, split rate 0.15. Overlap 50: 7 chunks,
split rate 0.00. The insurance costs 2 chunks.

## T5 , memory vs context

12000 / 900 = 13.3. The summary fits the budget, the raw log
does not.

## T6 , error attribution

R 4, T 3, P 3. Fix retrieval first, then re-measure.

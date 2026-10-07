# Lab U09 , agent and retrieval arithmetic in numpy

Six tasks. Run `python3 labs/u09_lab_run.py` from the `cs224n/`
directory. Numpy only. Record the numbers, then read
`labs/u09_lab_key.md` to verify.

T1. ReAct: 7-step toy trace. Report steps and total tokens.
T2. Tool validation: 6 calls, 2 invalid. Report the valid fraction.
T3. RAG: 20 chunks, relevant at ranks 2, 5, 9. Report recall at
k = 1, 3, 5, 10.
T4. Chunking: doc 1000 tokens, chunk 200, span 30. Report chunks
and split rate at overlap 0 and 50.
T5. Memory: raw log 12000 tokens, summary 900. Report the ratio.
T6. Attribution: 10 failures in R/T/P. Report the counts.

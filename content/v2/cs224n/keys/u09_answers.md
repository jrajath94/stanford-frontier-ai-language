# Answer key , U09 Tools, agents, and retrieval

Attempt the exercises before reading. Ladders are oral: answer aloud,
then check.

## Remediation

R1. Three checks: name exists, args match the schema, types hold.
Toy: 4/6 valid.
R2. Recall at k: 0.000, 0.333, 0.667, 1.000 at k = 1, 3, 5, 10.
R3. Split rate = max(0, s - o) / (c - o): 0.15 at o = 0, 0.00 at
o = 50.

## Breadth

A1. Thought plans, Action calls a tool, Observation returns the
result. The loop ends on a valid Answer or at the step cap.
A2. Name, argument schema, return schema, side-effect class. The
validator checks name, args, types before any execution.
A3. Retry with the error text in context, at most 2 retries, then
escalate. Never retry a call verbatim.
A4. Retrieve top-k, prepend, generate. Recall at k: 0.000, 0.333,
0.667, 1.000 on the toy. k trades precision for recall.
A5. Retriever's contract: the right chunk is in the top-k.
Generator's contract: the right answer given the right chunk.
A6. Chunk count: 1 + ceil((doc - c) / (c - o)). Toy: 5 chunks at
o = 0, 7 at o = 50. Split rate 0.15 vs 0.00.
A7. Fused = w times dense + (1 - w) times sparse. Rerank the top-m
pool with the heavy model, keep top-k.
A8. Dangling pointers (span not in the retrieved chunks) and
unsupported claims (span present, claim not entailed).
A9. Context is the working desk, memory is the store. Classes:
verbatim recent, summary, pinned anchors. Ratio 13.3 on the toy.
A10. required(tool) must be a subset of granted(run). Default
deny. Destructive tools route through approval.
A11. Success: the Answer validates. Fire escape: the step cap.
Both exits, always.
A12. R (retriever), T (tool), P (policy). Toy: 4/3/3. Fix the
biggest class first, then re-measure.

## Oral ladders

L1 (ReAct). Define T/A/O. Run the 7-step toy (92 tokens).
Derive why Observations are the only new information. Diagnose
the 50-step "no results" loop. Design the interleave vs
plan-then-execute test at equal tool budget.

L2 (tools). Write a schema. Run the validator on the toy.
Derive the three checks. Diagnose the invented `search_v2`.
Design the schema-reminder test.

L4 (RAG). Define the pipeline. Compute the recall values.
Derive the top-k factorization. Diagnose the rank-1 keyword
trap. Design the k sweep, predict rise then fall.

L6 (chunking). Define c and o. Compute 5 vs 7 chunks. Derive
the split rate. Diagnose the c = 50 collapse. Design the chunk
size sweep.

L12 (attribution). Define R/T/P. Classify the toy 4/3/3.
Derive the fix order. Diagnose the too-tight cap mislabeled as
policy. Design the fix-top-class re-measure.

## Exercises

E1. `react_step` replays the 7-step trace, 92 tokens.
E2. Step cap 10 stops the runaway with reason "cap".
E3. `validate` returns 4/6, failure kinds: missing arg, unknown
name.
E4. Closed schema rejects the extra argument.
E5. `call_with_retry` converges in 2 calls on the syntax toy.
E6. cap = 0 returns the error immediately, no retry.
E7. `rag_answer` reproduces recall 0.000/0.333/0.667/1.000.
E8. The poisoned chunk changes the answer: stale index, stale
truth.
E9. Swap test: retriever B raises recall at 3 from 0.333 to
1.000, generator frozen.
E10. Generator weights byte-identical across the two runs.
E11. `chunk` gives 5 and 7 chunks, split rates 0.15 and 0.00.
E12. At o = 50 no 30-token span is split: the R3 guarantee.
E13. `hybrid` matches fused scores 0.720, 0.565, 0.410 at w =
0, 0.5, 1.
E14. Rank-200 answer with m = 50: rerank never sees it.
E15. `check_cites` gives precision 0.75.
E16. The near-miss citation fails: right document, wrong
paragraph.
E17. `compact` reaches 900 tokens, ratio 13.3.
E18. The pinned anchor survives byte-identical.
E19. `authorize` passes `search`, blocks `delete`.
E20. `delete` marked destructive triggers the approval gate.
E21. `run_agent` returns "valid answer" on the toy, "cap" on
the broken tool.
E22. Without the cap the broken tool loops past 50 steps.
E23. `attribute` gives R 4, T 3, P 3.
E24. Sprint plan: fix retrieval first (C07), re-measure on 40
new tasks.

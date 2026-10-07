# Interview key , U09

## Breadth

A1. Thought, Action, Observation. The loop ends on a valid Answer
or at the step cap.
A2. Name, argument schema, return schema, side-effect class. Checks:
name exists, args match, types hold.
A3. Retrieve top-k chunks, prepend, generate. k trades precision
for recall.
A4. Split rate = max(0, s - o) / (c - o). Overlap at least the
span length removes splits. The toy goes 0.15 to 0.00.
A5. Context is the working window, memory is the store across
turns (verbatim recent, summary, pinned anchors).
A6. R (retriever), T (tool), P (policy). Fix the biggest class
first.

## Deep ladders

L1. (1) T/A/O x2 + Answer, 7 steps. (2) 14+8+22+10+8+18+12 =
92. (3) The model only reads the trace. Observations
are the only tokens it did not author. (4) No progress signal:
add a cap and a "no new info" detector. (5) 40 two-hop tasks,
equal tool budget, interleave wins when the first search misses.

L2. (1) Embed, top-k, format, generate. (2) 0.000, 0.333,
0.667, 1.000. (3) Sum over chunks approximated by top-k. (4) Rank
1 is a keyword match, not the answer. (5) k in {1, 5, 20}:
predict rise then fall from distraction.

## Analytical

A7. The retriever owned it (generator frozen, recall moved).
Next: rerank the pool (C07) or raise k, then re-run the swap
test.
A8. Overlap 50 adds 40 percent more chunks (7 vs 5). On 10M
tokens: roughly 4M extra index tokens plus embeddings. Price
it against the split-rate failures it removes.

## Implementation/debugging

A9. (1) Validator rejects unknown names (cheapest, already in
code). (2) Show only real tool names in the prompt. (3) Log the
hallucinated name as a prompt bug. (4) Constrain decoding to the
name set (strongest).

## Changed-constraint

A10. Hourly updates need incremental indexing and a freshness
timestamp per chunk, plus a stale-chunk quarantine in the
retriever. The generator is unchanged.
A11. Mark the tool destructive: require the approval gate
before execution, cap retries at 0 for it, log every call.
The loop shape is the same, the permissions change.

## Research critique

A12. Steelman: rerank reads carefully what retrieval skimmed,
and often promotes the right chunk. Counterexample: the answer
at rank 200 with m = 50, rerank never sees it. Rerank polishes
the pool, it does not rescue a missed retrieval.

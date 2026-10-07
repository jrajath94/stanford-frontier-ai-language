# U07 interview bank , questions

Closed-book. Answer keys are in `u07_key.md`. Do not open the key before
attempting. Quotas per major lesson: 6 breadth, 2 deep ladders of 5
follow-ups, 2 analytical exercises, 1 implementation/debug task, 2
changed-constraint scenarios, 1 research-critique question.

## Breadth (6)

B1. Define recall and precision for retrieval.
B2. What is hybrid search fusing, and why normalize first?
B3. What does the reranker add that the retriever cannot?
B4. Write the three ReAct roles in order.
B5. Name the three agent memory stores and their lifetimes.
B6. What are the three stopping guards?

## Deep ladders (2 x 5)

L1. Hybrid search.
- L1.1 Define BM25 and dense scores.
- L1.2 Toy: BM25 [0.9, 0.3], dense [0.4, 0.8], alpha 0.5,
  fuse and rank.
- L1.3 Justify normalization before the weighted sum.
- L1.4 Implement hybrid, state the rare-term check.
- L1.5 Compare weighted sum with RRF, debug a fusion where
  BM25 always wins, critique "hybrid always helps",
  propose the alpha-sweep experiment.

L2. ReAct loop.
- L2.1 Define Thought, Action, Observation.
- L2.2 Toy: write a 2-tool trace for "refund status".
- L2.3 Justify the trigger format (why parse at all).
- L2.4 Implement react_step, state the no-action check.
- L2.5 Compare ReAct with plan-then-act, debug 6 Thoughts
  and 0 Actions, critique "thoughts guide actions",
  propose the thought-ablation experiment.

## Analytical exercises (2)

E1. Corpus 200 docs, 10 relevant. System A: k = 30, H = 8.
System B: k = 60, H = 9. Compute recall and precision for
both. A reranker takes the top 50 to top 5 with precision@5
0.80 for both. Compute the end-to-end relevant docs in the
top 5 for each system, and state which system you ship.
E2. Window 6000, system 800, reserve 800. Docs: [2000, 1800,
1500, 1200]. How many fit fully, and what happens to the
rest? Then a summarizer compresses each doc to 60%: recompute.

## Implementation/debug task (1)

D1. A support agent loops: search, search, search, read, search,
search. It never answers. You may inspect the trace, the tool
schemas, and the stopping config. List the ordered checks, the
most likely culprit, and the fix. Then write the config lines
that stop it at step 4.

## Changed-constraint scenarios (2)

S1. The corpus is 90% part numbers and SKUs, 10% prose. Dense
retrieval alone buries exact matches. Do you still build
hybrid, or tune the dense embedder? Defend the choice, state
the failure mode of the rejected option, and name the first
metric you track.
S2. A red-tier tool (refund issuance) must stay human-approved,
but approvals take 4 hours and tickets pile up. How do you
keep the gate without killing throughput? Name two mechanisms
and how each works.

## Research-critique question (1)

R1. "Bigger k always improves RAG answers." Present the
strongest version of this claim, then the attention-dilution
counterexample, then design an experiment that finds the
optimal k for a fixed window. State the falsification
condition.

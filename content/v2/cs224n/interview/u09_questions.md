# Interview bank , U09 Tools, agents, and retrieval

## Breadth (6)

Q1. What are the three ReAct step types, and what ends the loop?
Q2. What does a tool schema declare, and what are the three
validation checks?
Q3. What is RAG, and what does the k in top-k control?
Q4. What is the chunking split rate, and how does overlap change it?
Q5. What is the difference between memory and context?
Q6. What are the three error attribution classes, and which do you
fix first?

## Deep ladders (2 x 5)

L1 (ReAct). (1) Write the 7-step toy trace. (2) Count the 92
tokens. (3) Derive why observations are the only new information.
(4) A tool returns "no results" for 50 steps: diagnose. (5) Design
the interleave vs plan-then-execute test.

L2 (RAG). (1) Write the pipeline. (2) Compute recall at k = 1, 3,
5, 10. (3) Derive the top-k factorization. (4) k = 1 finds
nothing: explain. (5) Design the k sweep and predict the shape.

## Analytical (2)

Q7. Recall at k rises from 0.333 to 1.000 when you swap the
retriever, with the generator frozen. Who owned the failure, and
what is the next experiment?
Q8. Chunk overlap 50 removes splits but adds 2 chunks. The corpus
is 10M tokens. Quantify the index cost of the insurance.

## Implementation/debugging (1)

Q9. Your agent emits `search_v2`, which is not a declared tool.
List the fixes in order, cheapest first.

## Changed-constraint (2)

Q10. The corpus updates hourly. Does your RAG design still work?
What changes?
Q11. The tool is destructive (sends email). Redesign the loop.

## Research critique (1)

Q12. "Reranking fixes retrieval." Steelman, then give the
strongest counterexample from this unit.

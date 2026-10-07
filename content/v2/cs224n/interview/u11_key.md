# Interview key , U11

## Breadth

A1. Step labels buy auditability, answer labels buy accuracy.
Steps can rationalize: supervised, not faithful.
A2. A reward a program checks. No learned middle layer. Main
failure: a buggy verifier teaches exploits.
A3. Majority vote over reasoning paths. Hurts when p < 0.5
(0.3174 at n = 5) or paths are identical.
A4. Draft with a small model, verify with the big one. Guarantees
the big model's output distribution exactly.
A5. Split a fixed sample budget across prompts vs depth. Rule:
concave gains favor breadth (3.935 vs 0.993).
A6. Traces are stories, not proofs. The corruption test decides
whether a step caused the answer.

## Deep ladders

L1. (1) Sum C(n,k) p^k (1-p)^(n-k) over k > n/2. (2) 0.6000,
0.6826, 0.7535, 0.8256. (3) The tail shrinks below 0.5. (4)
Correlated errors: the committee shares the bias. (5) Measure
pairwise wrong-answer agreement vs independence.

L2. (1) Accept the draft prefix while it matches the target
distribution, resample at the first reject. (2) (1 - 0.7^6) /
0.3 = 2.941. (3) Geometric run with the bonus token. (4) The
overhead exceeds the gain: measure a and draft cost first.
(5) Sweep g, measure wall-clock speedup, expect an interior
optimum.

## Analytical

A7. 10x10: 3.935. 5x20: 5 x 0.632 = 3.161. 1x100: 0.993.
10x10 wins: concave gains favor breadth.
A8. Outcome: 4 kept per 20 cost units. Process: 2 kept per 60.
Per dollar, outcome wins 6x. Choose process only when the
steps are the product.

## Implementation/debugging

A9. (1) Check temperature: 0 gives identical paths. (2) Check
path diversity (distinct answers). (3) Check p: below 0.5 the
vote is harmful anyway. (4) Check the answer extractor.

## Changed-constraint

A10. Yes. Two passes verify 2 x 5 drafts: about 5.9 tokens vs 2
greedy, if the draft is cheap and a is near 0.7. The guarantee
holds regardless.
A11. No. Threshold gains are not concave: 10x10 gives 0, 1x100
clicks. Know the gain shape before spreading.

## Research critique

A12. Steelman: the steps decompose the problem and often check
out, they expose the work for audit. Counterexample: the
corruption test, 14 of 20 answers unchanged after breaking a
step. Useful, not faithful.

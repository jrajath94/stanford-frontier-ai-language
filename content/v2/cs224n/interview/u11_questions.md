# Interview bank , U11 Reasoning and test-time compute

## Breadth (6)

Q1. What is the difference between supervising steps and
supervising answers?
Q2. What is a verifiable reward, and what is its main failure mode?
Q3. What is self-consistency, and when does it hurt?
Q4. What is speculative decoding, and what does it guarantee?
Q5. What is the compute allocation problem, and what is the rule?
Q6. What are the limits of trace evidence?

## Deep ladders (2 x 5)

L1 (sampling). (1) Write the binomial majority sum. (2) Compute
the curve at p = 0.6. (3) Derive the p < 0.5 reversal. (4) The
model repeats one wrong answer 21 times: diagnose. (5) Design
the pairwise-agreement test.

L2 (spec decode). (1) Define the accept rule. (2) Compute E =
2.941. (3) Derive the expectation formula. (4) Draft costs half
the target at a = 0.3: diagnose. (5) Design the draft-size sweep.

## Analytical (2)

Q7. Budget 100 samples, gain 1 - exp(-n/20). Compare 10x10,
5x20, 1x100. Which wins and why?
Q8. Outcome verification keeps 4 of 10 traces, process keeps 2,
at 5x the check cost. With a fixed dollar budget, which do you
choose? Show the reasoning.

## Implementation/debugging (1)

Q9. Your self-consistency vote never changes the answer across 5
paths. List the checks in order, cheapest first.

## Changed-constraint (2)

Q10. Latency budget allows exactly 2 sequential target passes.
Can speculative decoding help? Explain with the numbers.
Q11. The task has a threshold: it needs 50 samples to click, 10
samples give nothing. Does the spread rule still hold?

## Research critique (1)

Q12. "The trace shows the model's reasoning." Steelman, then
give the strongest counterexample from this unit.

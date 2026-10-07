# Interview bank , U08 Prompting and efficient adaptation

## Breadth (6)

Q1. What is the difference between zero-shot and few-shot, and what
limits both?
Q2. What is in-context learning, and what is still open about it?
Q3. What makes a good demonstration, and how do you pick k of them?
Q4. Why does chain-of-thought help, and what does it not prove?
Q5. What is LoRA, and why does it cost nothing at inference?
Q6. What is catastrophic forgetting, and what are the three
mitigations?

## Deep ladders (2 x 5)

L1 (prompting). (1) Build a 2-shot prompt. (2) State the window
limit. (3) Derive why conditioning is not learning. (4) Shuffled
labels still help: explain. (5) Design the test that separates
format from task knowledge.

L2 (LoRA). (1) Write dW = BA with shapes. (2) Compute 65536 and
0.060%. (3) Derive rank(BA) <= r. (4) r = 8 underfits a far task:
explain. (5) Design the rank sweep.

## Analytical (2)

Q7. Few-shot accuracy rises from 0 to 4 shots, then falls at 16
shots. Name two mechanisms from this unit, and the cheapest test
for each.
Q8. LoRA and full fine-tuning tie on your benchmark, but LoRA used
3x the tuning budget. What is the honest conclusion?

## Implementation/debugging (1)

Q9. Your JSON outputs parse only 40% of the time. List the fixes in
order, cheapest first.

## Changed-constraint (2)

Q10. You must serve 50 task-specific variants on one GPU. What is
the plan, and what breaks?
Q11. The task needs a new language the base model barely saw. Does
PEFT suffice? Explain with the rank argument.

## Research critique (1)

Q12. "Chain-of-thought shows the model's reasoning." Steelman, then
give the strongest counterexample from this unit.

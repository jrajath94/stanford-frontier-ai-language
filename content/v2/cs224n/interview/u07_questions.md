# Interview bank , U07 Post-training and preferences

## Breadth (6)

Q1. What does SFT teach that pretraining does not?
Q2. What makes a good instruction dataset?
Q3. How does a reward model turn pairs into a scalar?
Q4. What are the three stages of RLHF, and what does each add?
Q5. What does PPO's clipping do?
Q6. How does DPO avoid the RL loop?

## Deep ladders (2 x 5)

L1 (reward). (1) Write the Bradley-Terry loss. (2) Compute P and
loss for the toy pair. (3) Derive the gradient's (1-P) weighting.
(4) The reward's argmax differs from human preference: explain.
(5) Design the held-out agreement test.

L2 (DPO vs PPO). (1) Write both losses. (2) Compute the toy numbers.
(3) Derive DPO's inversion (reward from optimal policy). (4) DPO
collapses without the reference term: explain. (5) Design the
experiment that picks between them for a new task.

## Analytical (2)

Q7. The reward climbs for 2000 steps while human raters say quality
peaks at step 800. Name the two mechanisms from this unit, and the
cheapest check that distinguishes them.
Q8. DPO training loss falls to near zero but the policy's outputs
barely change. What happened, and what single number confirms it?

## Implementation/debugging (1)

Q9. PPO training diverges at step 300. List the checks in order,
with the one-line fix each check points to.

## Changed-constraint (2)

Q10. You have pairs but no budget for human eval. What is the
least-bad evaluation plan?
Q11. The deployment needs the model to refuse certain requests.
Where in the pipeline does refusal get taught, and what breaks if
it is only prompted?

## Research critique (1)

Q12. "RLHF aligns the model with human values." Steelman, then give
the strongest counterexample from this unit.

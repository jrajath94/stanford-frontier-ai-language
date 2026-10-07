# U05 interview bank: questions

## Breadth (6)

B1. Write the RLHF preference record schema.
B2. Write the reward model training loss.
B3. What does the KL penalty do, and what is beta?
B4. Write the closed-form KL-constrained optimum.
B5. Write the DPO loss in words or symbols.
B6. Define reward hacking in one sentence.

## Deep ladders (2 x 5)

### Ladder 1: from reward to DPO

L1.1 Define: write the reward model loss on one pair.
L1.2 Toy: margin 1.5. Compute the loss.
L1.3 Derive: invert the KL-constrained optimum to get r from pi.
L1.4 Implement and complexity: write the DPO loss code and state its cost.
L1.5 Compare: what disappears from the pipeline versus PPO?
L1.6 Debug: the DPO loss falls but human eval worsens. Diagnose.
L1.7 Critique: which assumption of the derivation breaks off-policy?
L1.8 Design: design the matched-compute DPO versus PPO experiment.

### Ladder 2: KL control

L2.1 Define: write the KL-penalized objective.
L2.2 Toy: rewards [2,1,0], uniform reference. Compute the optimum at beta 0.5.
L2.3 Derive: the Lagrange step that gives the exponential tilt.
L2.4 Implement and complexity: write the closed form and state its cost.
L2.5 Compare: penalty versus hard trust-region constraint.
L2.6 Debug: KL explodes mid-PPO. Name three knobs.
L2.7 Critique: the reference is mediocre. What caps the policy?
L2.8 Design: design the beta-tuning protocol on held-out preference loss.

## Analytical exercises (2)

A1. Derive the KL-constrained optimum via Lagrange multipliers, then derive the DPO loss by substituting the inverted reward into the Bradley-Terry likelihood. Show the constant cancels.
A2. On the three-response toy, compute the reward-KL frontier for beta in [0.5, 2.0] and state what happens as beta -> 0 and beta -> infinity.

## Implementation / debug (1)

D1. The DPO training below collapses: the policy puts all mass on one response and human eval tanks, while the loss keeps falling. The code is correct. Is the failure in the loss, the data, or the optimization? Justify in two sentences and give the smallest principled fix.

```python
for x, yw, yl in pairs:
    m = beta*(logp(yw)-logp_ref(yw)-logp(yl)+logp_ref(yl))
    loss = -log_sigmoid(m)
    loss.backward()
```

## Changed-constraint scenarios (2)

S1. Preference pairs are now k-wise rankings (k = 4). How do the reward loss and the DPO loss change?
S2. The reward model must serve 10 locales with different tastes. What breaks in the single-reward pipeline, and what is the minimal change?

## Research critique (1)

R1. A paper claims DPO beats PPO because its training loss is lower after the same number of steps. List three objections and design the evaluation that would convince you.

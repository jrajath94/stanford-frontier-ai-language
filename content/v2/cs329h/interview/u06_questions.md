# U06 interview bank: questions

## Breadth (6)

B1. Write the value-of-information formula.
B2. How do you rank candidate queries under Gaussian noise?
B3. What does the posterior over preferences represent, and what drives further elicitation?
B4. State the assistance-game information asymmetry in one sentence.
B5. Define the contextual bandit: what is observed, chosen, and earned?
B6. What is elicitation burden, and what stops the querying?

## Deep ladders (2 x 8)

### Ladder 1: value of information

L1.1 Define: write VoI(q).
L1.2 Toy: gap in {-0.5, 1.5}, prior 0.5 each. Compute VoI of a revealing query.
L1.3 Derive: show VoI >= 0 for a Bayesian decider.
L1.4 Implement and complexity: code the enumeration over answers, state its cost.
L1.5 Compare: VoI versus expected information gain.
L1.6 Debug: computed VoI 0.25, realized gain 0. Diagnose.
L1.7 Critique: which assumption fails when humans answer noisily?
L1.8 Design: design the experiment comparing VoI-ranked against random queries.

### Ladder 2: assisted bandit

L2.1 Define: who proposes and who disposes?
L2.2 Toy: compute the team reward 0.775 on the morning context.
L2.3 Derive: show team >= solo when overrides are informative.
L2.4 Implement and complexity: simulate the propose-override loop, state its cost.
L2.5 Compare: assisted versus full autonomy versus human alone.
L2.6 Debug: the team underperforms solo. Diagnose.
L2.7 Critique: the override model assumes vigilance. What breaks under fatigue?
L2.8 Design: design the deployment test for the break-even override rate.

## Analytical exercises (2)

A1. Derive the posterior variance reduction 1/(1+s^2) for the Gaussian query toy, then rank the three queries with noise sd 0.7, 1.0, 1.5.
A2. Prove VoI >= 0 for a Bayesian decider, then construct the zero-VoI case on the gap toy.

## Implementation / debug (1)

D1. The belief update below concentrates on the wrong theta though the human acted clearly. The code is correct. Is the failure in the update, the observation model, or the prior? Justify in two sentences and give the smallest principled fix.

```python
b = prior.copy()
for a in human_actions:
    b = b * obs_model[a]   # P(a | theta)
    b = b / b.sum()
```

## Changed-constraint scenarios (2)

S1. Queries now cost human time at $50/hour and each query takes 2 minutes. How does the query-choice rule change?
S2. The human answers must stay on their device (privacy). What changes in the elicitation architecture?

## Research critique (1)

R1. A paper claims its elicitation method 'asks the most informative questions' because posterior entropy falls fastest. List three objections and design the evaluation that would convince you.

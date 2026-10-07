# UU10 interview bank: questions

## Breadth (6)

B1. Name the five pre-analysis plan elements.
B2. State the literature-review contract.
B3. Write H2 in I/O/D/K form.
B4. Name the five data-consent elements.
B5. State the H1 and H2 verdicts with numbers.
B6. Name the four open-source gaps.

## Deep ladders (2 x 8)

### Ladder 1: replication

L1.1 Define: write the H1 hypothesis in I/O/D/K form.
L1.2 Toy: describe the 4-arm BT bandit setup.
L1.3 Derive: derive the verdict rule from the CI definition.
L1.4 Implement and complexity: sketch the TS loop, state the cost.
L1.5 Compare: why does TS beat uniform here?
L1.6 Debug: the seeds differ across methods. Diagnose.
L1.7 Critique: simulated BT vs real human duels. What is absent?
L1.8 Design: design the human-subject replication.

### Ladder 2: negative results

L2.1 Define: what are the four parts of the report?
L2.2 Toy: walk the H2 numbers (8.36 vs 3.11).
L2.3 Derive: derive the H2 verdict from the CIs.
L2.4 Implement and complexity: what is the cost of the follow-up ablation?
L2.5 Compare: honest report vs spin. What differs?
L2.6 Debug: the diagnosis names the beta prior with no test. Diagnose.
L2.7 Critique: when is a negative result not publishable?
L2.8 Design: design the ablation that tests the challenge-cost diagnosis.

## Analytical exercises (2)

A1. From the CIs (TS [2.23, 3.99], leader [6.86, 9.87]), derive the H2 verdict step by step and state what the numbers rule out.
A2. Design a consent form for a new preference-data collection (five elements with terms) and justify the withdrawal mechanism.

## Implementation / debug (1)

D1. The replication below compares two methods, but the verdict flips under a new seed set. The code runs. Find the bug.

```python
regrets = {}
for name, fn in methods.items():
    rng = np.random.default_rng(seed_by_method[name])
    regrets[name] = [fn(np.random.default_rng(s)) for s in range(20)]
```

## Changed-constraint scenarios (2)

S1. The replication fails on real human duel data (TS loses to uniform). What do you check first, second, third?
S2. A reviewer says the H2 negative result 'adds nothing'. Write the three-sentence reply.

## Research critique (1)

R1. A paper proposes a new dueling acquisition function, shows one seed, no baseline, no plan. List three objections and state what you would check first.

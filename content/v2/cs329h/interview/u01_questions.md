# U01 interview bank: questions

## Breadth (6)

B1. Define preference and choice in one sentence each, and name one context factor that can separate them.
B2. Write the Bradley-Terry model equation and name every symbol.
B3. What does it mean for Bradley-Terry scores to be identified only up to an additive constant?
B4. State the log-likelihood for n independent pairwise comparisons.
B5. How does label noise change the estimated score gap: shrink, grow, or leave it?
B6. Name two fairness questions to ask before deploying a preference model.

## Deep ladders (2 x 5)

### Ladder 1: the (y - p) update

L1.1 Define: write the Bradley-Terry probability for a beats b.
L1.2 Toy: A beat B 5 times, B beat A once. Write the log-likelihood as a function of the gap.
L1.3 Derive: differentiate and show the gradient step moves each score by (y - p).
L1.4 Implement and complexity: write the update loop and state its cost per step.
L1.5 Compare: how does the update change under Thurstone's probit link?
L1.6 Debug: all comparisons are A wins. What happens to the gap, and what is the fix?
L1.7 Critique: which assumption fails when one annotator judges the same pair fifty times?
L1.8 Design: propose an experiment that tests whether the logistic link fits your annotation task.

### Ladder 2: context versus preference

L2.1 Define: what is a context effect in the choice model?
L2.2 Toy: win rate 0.69 when A is shown first, 0.54 when B is shown first. What is the naive conclusion?
L2.3 Derive: write the two-parameter model and show the naive estimate confounds s and b.
L2.4 Implement and complexity: how do you fit (s, b), and what does each extra context feature cost?
L2.5 Compare: when do you model context versus randomize it away?
L2.6 Debug: the b estimate is 5.0 with a huge standard error. Diagnose.
L2.7 Critique: an unrecorded context correlates with item difficulty. What breaks?
L2.8 Design: design the order-randomization study that estimates the position effect cleanly.

## Analytical exercises (2)

A1. For one pair with w_a wins for A and w_b wins for B, derive the MLE gap in closed form and its approximate standard error. Evaluate for w_a = 5, w_b = 1.
A2. With flip probability q, derive the observed win rate p_obs in terms of the true win rate p. For true gap 1.5 and q = 0.1, compute the estimated gap from the formula.

## Implementation / debug (1)

D1. The fitter below diverges on a dataset where every comparison is A beats B. Find the bug class (not a typo): is it a data problem, a model problem, or an optimizer problem? Propose the smallest principled fix and justify it in two sentences.

```python
def fit(pairs, m):
    s = np.zeros(m)
    for _ in range(10000):
        g = np.zeros(m)
        for a, b, y in pairs:
            p = 1 / (1 + np.exp(-(s[a] - s[b])))
            g[a] += y - p
            g[b] -= y - p
        s += 0.1 * g
    return s
```

## Changed-constraint scenarios (2)

S1. Ties are now allowed and recorded as y = 0.5. How do the likelihood, the MLE for the 5-1-2 (wins-losses-ties) toy, and the identifiability argument change?
S2. Each annotator now judges 200 pairs instead of 2, and annotators disagree systematically. Which U01 assumption breaks first, and what is the minimal model change?

## Research critique (1)

R1. A paper claims a new pairwise model beats Bradley-Terry because its train log-loss is lower on three annotation datasets. List three methodological objections and design the experiment that would convince you.

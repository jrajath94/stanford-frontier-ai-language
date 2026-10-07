# U02 interview bank: questions

## Breadth (6)

B1. Write the random utility model and the choice rule in symbols.
B2. State the Gumbel-max trick in one sentence.
B3. What is IIA, and which model implies it?
B4. In a factor model, what are the shapes of R and V for 100 respondents, 50 items, d = 5?
B5. Why are utility numbers meaningless without a scale convention?
B6. Name two counterexamples from this unit and the repair for each.

## Deep ladders (2 x 5)

### Ladder 1: Gumbel to softmax

L1.1 Define: write U_i and the argmax choice rule.
L1.2 Toy: utilities [1.0, 0.5, 0.0]. Compute the three choice shares.
L1.3 Derive: sketch the integral that turns Gumbel argmax into softmax.
L1.4 Implement and complexity: write the sampler and state its cost.
L1.5 Compare: what changes with Gaussian noise instead of Gumbel?
L1.6 Debug: simulated shares are [0.9, 0.05, 0.05] but the formula says [0.5, 0.3, 0.2]. Name two causes.
L1.7 Critique: which assumption fails for the red bus / blue bus case?
L1.8 Design: design a simulation study that tests whether IIA holds on a dataset.

### Ladder 2: factorization

L2.1 Define: write M = R V' with shapes for the 4x5 toy.
L2.2 Toy: how many free numbers does rank 2 use versus the full 20?
L2.3 Derive: show the SVD recovers factors up to rotation.
L2.4 Implement and complexity: write the truncated SVD code and state its cost.
L2.5 Compare: factorization versus clustering respondents into types.
L2.6 Debug: reconstruction error stays high at d = 2. Diagnose.
L2.7 Critique: what breaks if a new respondent's tastes lie outside the span of V?
L2.8 Design: design the held-out experiment that selects the rank d.

## Analytical exercises (2)

A1. For utilities [2.0, 1.0, 0.0, -1.0, -2.0], compute all five choice shares and prove the top utility always gets the top share.
A2. Prove the scale invariance: show (u, sigma_e) and (2u, 2 sigma_e) give identical choice probabilities under argmax with scaled Gumbel noise.

## Implementation / debug (1)

D1. The fold-in below returns wild predictions for a new respondent with two ratings. The code runs without error. Is the bug in the data, the model, or the solver? Justify in two sentences and give the smallest fix.

```python
r_new, *_ = np.linalg.lstsq(V_known, m_obs, rcond=None)
pred = V_all @ r_new
```

## Changed-constraint scenarios (2)

S1. Shocks are now correlated within known groups of similar options. Which derivations in this unit break, and what is the minimal model change?
S2. Respondents arrive over time and their tastes drift. Which assumption breaks first, and how do you adapt the factor model?

## Research critique (1)

R1. A paper reports that a 50-factor model beats a 5-factor model on training log-likelihood for a preference dataset. List three objections and design the experiment that would convince you the extra factors help.

# U04 interview bank: questions

## Breadth (6)

B1. Define the score function in one sentence.
B2. State two definitions of Fisher information.
B3. What does a singular information matrix tell you?
B4. Write the asymptotic variance of the MLE.
B5. Which candidate pair is most informative for a Bradley-Terry gap, and why?
B6. Define A-optimality and D-optimality in one sentence each.

## Deep ladders (2 x 5)

### Ladder 1: information to variance

L1.1 Define: write the score and the information for Bernoulli(p).
L1.2 Toy: 7 wins in 10 trials. Compute the score at p = 0.5 and the information at p = 0.7.
L1.3 Derive: show the information equals the curvature, E[score] = 0 via the integral swap.
L1.4 Implement and complexity: write the SE computation and state its cost.
L1.5 Compare: expected versus observed information.
L1.6 Debug: the quadratic approximation does not match the likelihood. Name two causes.
L1.7 Critique: which regularity condition fails for Uniform(0, theta)?
L1.8 Design: size a study for SE below 0.05 when p = 0.7.

### Ladder 2: design

L2.1 Define: what is a design criterion?
L2.2 Toy: two designs for two gaps. Write both information matrices.
L2.3 Derive: compute A and D for the split design.
L2.4 Implement and complexity: write the design-scoring code and state its cost.
L2.5 Compare: A versus D versus E-optimality.
L2.6 Debug: the criterion ranks a design first that then fails in practice. Diagnose.
L2.7 Critique: the criteria depend on guessed gaps. What breaks when the guesses are wrong?
L2.8 Design: design the pilot study that validates the information guesses.

## Analytical exercises (2)

A1. Derive I(p) = n/(p(1-p)) for the Bernoulli model from both definitions, and compute the n needed for SE <= 0.05 at p = 0.7.
A2. Derive the per-label information under flip probability q for a Bradley-Terry gap, and compute the label multiplier at q = 0.3 versus q = 0.1.

## Implementation / debug (1)

D1. The active-selection loop below keeps asking the same pair forever. The scoring code is correct. Is the problem in the selection rule, the model, or the data? Justify in two sentences and give the fix.

```python
for round in range(50):
    scores = [info(pair, theta_hat) for pair in candidates]
    ask(candidates[argmax(scores)])
    theta_hat = refit(all_labels)
```

## Changed-constraint scenarios (2)

S1. Labels now arrive in batches of 100 with a week of latency. How does the active-selection loop change?
S2. Two annotator pools have different noise rates q1 and q2 and different per-label costs. How do you allocate the budget?

## Research critique (1)

R1. A paper claims its active-selection method beats random because the final model has higher train accuracy. List three objections and design the evaluation that would convince you.

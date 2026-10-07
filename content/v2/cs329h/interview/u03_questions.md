# U03 interview bank: questions

## Breadth (6)

B1. Define the MLE in one sentence.
B2. Write Bayes rule and name the four terms.
B3. What is the E-step and what is the M-step?
B4. How does MAP differ from MLE?
B5. What is the posterior predictive distribution?
B6. Define calibration and name one way to fix miscalibration.

## Deep ladders (2 x 5)

### Ladder 1: EM

L1.1 Define: write the marginal likelihood with a latent variable.
L1.2 Toy: 400 scores from two Gaussians, labels hidden. What is observed and what is latent?
L1.3 Derive: state the EM lower bound and show the E-step makes it touch the likelihood.
L1.4 Implement and complexity: write the E and M steps and state the per-iteration cost.
L1.5 Compare: EM versus direct gradient ascent on the marginal likelihood.
L1.6 Debug: the log-likelihood decreases between iterations. What is broken?
L1.7 Critique: two different starts give two different answers. What does that tell you?
L1.8 Design: design the restart protocol that defends against local optima.

### Ladder 2: Bayes update

L2.1 Define: prior, likelihood, posterior in words.
L2.2 Toy: Beta(2,2) prior, 7 wins in 10. Compute the posterior.
L2.3 Derive: show the exponents add in the Beta-Binomial update.
L2.4 Implement and complexity: write the update and state its cost.
L2.5 Compare: MAP versus posterior mean versus full posterior.
L2.6 Debug: the posterior equals the prior after seeing data. Diagnose.
L2.7 Critique: a Beta(1000,1000) prior with n = 10. What is wrong?
L2.8 Design: design the experiment that tests whether an elicited prior helps prediction.

## Analytical exercises (2)

A1. For the Bernoulli toy (7 wins, 10 trials), derive the MLE, its standard error, the Beta(2,2) posterior, and the posterior mean. Show the posterior mean sits between the prior mean and the MLE.
A2. Decompose the predictive variance for Beta(9,5) into aleatoric and epistemic terms with numbers. Which dominates, and what action does that suggest?

## Implementation / debug (1)

D1. The EM loop below runs but the likelihood never improves after the first iteration on a Bernoulli mixture. The code is correct. Explain in two sentences why, and name the smallest change to the model or data that would make EM informative.

```python
for _ in range(100):
    r = alpha * p1**y * (1-p1)**(1-y)
    r = r / (r + (1-alpha) * p2**y * (1-p2)**(1-y))
    alpha, p1, p2 = r.mean(), (r*y).sum()/r.sum(), ((1-r)*y).sum()/(1-r).sum()
```

## Changed-constraint scenarios (2)

S1. The prior is now a mixture of two Betas (skeptical and enthusiastic experts). How do the posterior computation and the MAP change?
S2. Data arrives in a stream and the true bias drifts slowly. Which U03 method breaks, and what is the minimal adaptation?

## Research critique (1)

R1. A paper claims a new regularizer beats L2 because its training log-likelihood is higher at the same lambda. List three objections and design the comparison that would convince you.

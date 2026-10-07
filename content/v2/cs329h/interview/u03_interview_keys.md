# U03 interview keys

## B1

Sufficient: the parameter value that maximizes the likelihood of the observed data. Strong answer adds "a peak-finder, no uncertainty attached." Red flags: confusing with the posterior. Rubric: 2 points. Remediation: U03-C01.

## B2

Sufficient: p(theta|D) = p(D|theta) p(theta) / p(D): posterior, likelihood, prior, evidence. Strong answer states the evidence normalizes. Red flags: dropping the evidence without comment. Rubric: 2 points. Remediation: U03-C04.

## B3

Sufficient: E-step computes posterior responsibilities for the latent variables. M-step maximizes the expected complete-data log-likelihood. Strong answer notes the bound. Red flags: "E estimates, M maximizes" with no objects. Rubric: 2 points. Remediation: U03-C03.

## B4

Sufficient: MAP maximizes likelihood times prior (penalized likelihood). MLE uses the likelihood alone. Strong answer gives the Beta shrinkage numbers. Red flags: calling MAP "Bayesian." Rubric: 2 points. Remediation: U03-C05.

## B5

Sufficient: the distribution of future data averaged over the posterior. Strong answer writes the integral. Red flags: confusing with the posterior itself. Rubric: 2 points. Remediation: U03-C06.

## B6

Sufficient: predicted probabilities match observed frequencies. Fix with temperature scaling or isotonic regression. Strong answer mentions the reliability diagram. Red flags: "calibration means accuracy." Rubric: 2 points. Remediation: U03-C09.

## Ladder 1

- L1.1: p(x|theta) = sum_z p(z|theta) p(x|z,theta).
- L1.2: observed: 400 scores. Latent: component labels.
- L1.3: bound = E_q[log p(x,z|theta)] - E_q[log q(z)]. Q = posterior makes KL zero so the bound touches.
- L1.4: responsibilities then weighted means/variances. O(n|z|) per iteration.
- L1.5: EM is stable with closed-form M-steps. Gradients handle non-closed-form M-steps but need tuning.
- L1.6: the M-step is wrong or the likelihood is miscomputed. EM theory forbids decreases.
- L1.7: local optima. The likelihood surface is non-convex.
- L1.8: multiple random restarts, keep the best final likelihood, report the spread.
- Red flags: asserting EM finds the global optimum. Rubric: 2 points per rung for mechanism. Remediation: U03-C02/C03 with the mixture rerun.

## Ladder 2

- L2.1: prior = belief before. Likelihood = data's voice. Posterior = updated belief.
- L2.2: Beta(9,5).
- L2.3: p^{a-1}(1-p)^{b-1} times p^w(1-p)^l gives exponents (a+w-1, b+l-1).
- L2.4: a += wins. B += losses. O(1).
- L2.5: MAP = mode (one regularized number). Posterior mean = squared-error optimal. Full posterior = keeps uncertainty.
- L2.6: dogmatic prior (point mass) or a bug that drops the likelihood term.
- L2.7: the prior overwhelms n = 10. The data moves the mean by ~0.005. The prior is too strong to be honest.
- L2.8: elicit the prior, compare posterior predictions against a flat prior on held-out data.
- Red flags: treating the prior as a tuning knob with no meaning. Rubric and remediation as ladder 1, using U03-C04/C05.

## A1

Sufficient: MLE 0.70. SE 0.145. Posterior Beta(9,5). Posterior mean 0.643. Prior mean 0.50 < 0.643 < 0.70. Strong answer derives each. Red flags: SE from the posterior instead of the sampling distribution. Rubric: 4 points. Remediation: U03-C01/C04.

## A2

Sufficient: Var(p) = 0.0153 (epistemic). E[p(1-p)] = 0.214 (aleatoric). Aleatoric dominates. Action: improve the model or accept noise, more data barely helps. Strong answer shows the arithmetic. Red flags: recommending more data. Rubric: 3 points numbers, 2 points action. Remediation: U03-C11.

## D1

Sufficient: the Bernoulli mixture is non-identifiable from binary outcomes alone: the likelihood depends only on the marginal mean m = alpha p1 + (1-alpha) p2, so EM converges in one step to a flat ridge (verified: -277.0138 from every start). Smallest change: use a richer observation (e.g., continuous scores, Gaussian mixture) or fix one parameter. Strong answer names the flat-ridge mechanism. Red flags: blaming the code. Rubric: 3 points diagnosis, 2 points fix. Remediation: U03-C02/C03.

## S1

Sufficient: the posterior becomes a mixture of Betas (one per component, weighted by marginal likelihoods). The MAP is the mode of the mixture, found by comparing component modes and checking boundaries. Strong answer notes multimodality. Red flags: averaging the priors first. Rubric: 3 points. Remediation: U03-C04.

## S2

Sufficient: the fixed-parameter assumption breaks (MLE, MAP, and the static posterior all assume one truth). Minimal adaptation: exponential weighting of recent data, or a state-space model with drift. Strong answer notes evaluation must also become temporal. Red flags: refitting from scratch each day without comment. Rubric: 2 points diagnosis, 2 points adaptation. Remediation: U03-C01/C11.

## R1

Sufficient objections: (1) training likelihood is not the metric. (2) same lambda means different effective strength across regularizers. (3) no held-out comparison, no error bars. Convincing: held-out log-loss with lambda tuned per method on validation, paired test. Strong answer adds multiple seeds. Red flags: accepting the claim. Rubric: 2 points per objection, 2 points design. Remediation: U03-C08 and P22.

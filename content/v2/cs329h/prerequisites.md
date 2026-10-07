# Prerequisites: cs329h

Shared bridges live at `~/workspace/stanford-frontier-ai/v2-pack/shared/prerequisites/` and are linked, not rebuilt. Each unit below carries its own local remediation.

## Prerequisite graph for owned units

| Unit | Bridges | Why |
| --- | --- | --- |
| U01 | [P06](../shared/prerequisites/p06_probability.md), [P07](../shared/prerequisites/p07_estimation.md), [P10](../shared/prerequisites/p10_ml_foundations.md) | Bernoulli outcomes, likelihood, train/test split |
| U02 | [P03](../shared/prerequisites/p03_vectors.md), [P06](../shared/prerequisites/p06_probability.md), [P07](../shared/prerequisites/p07_estimation.md) | Embeddings, Gumbel/softmax, axioms as constraints |
| U03 | [P07](../shared/prerequisites/p07_estimation.md), [P09](../shared/prerequisites/p09_optimization.md), [P18](../shared/prerequisites/p18_bayesian.md) | MLE, gradient descent, priors/posteriors, EM |
| U04 | [P05](../shared/prerequisites/p05_calculus.md), [P07](../shared/prerequisites/p07_estimation.md), [P18](../shared/prerequisites/p18_bayesian.md), [P22](../shared/prerequisites/p22_experiments.md) | Score as gradient, curvature as Hessian, design as experiment |
| U05 | [P08](../shared/prerequisites/p08_information.md), [P17](../shared/prerequisites/p17_rl.md), [P18](../shared/prerequisites/p18_bayesian.md) | KL, policy gradients, reward as latent variable |

## Local remediation per unit (self-contained, inside the lessons)

- U01 lesson opens with a Bernoulli refresher: events, PMF, expectation, variance, IID. No step in U01 uses a symbol before the refresher defines it.
- U02 lesson opens with a dot-product and norm refresher plus a Gumbel CDF derivation from scratch.
- U03 lesson opens with the Bayes rule and log-likelihood refresher and derives the EM lower bound inline.
- U04 lesson opens with gradient/Hessian refresher and derives the score identity inline.
- U05 lesson opens with a KL and policy-gradient refresher and derives the DPO reparameterization inline.

## Diagnostic (closed book, 20 minutes)

Answer in `prerequisites.md` only after attempting. Keys at the end of this file.

1. A coin lands heads with probability 0.6. Write its PMF, expectation, and variance.
2. State Bayes rule in words and in symbols.
3. Define the log-likelihood of n IID Bernoulli trials.
4. What does it mean for a parameter to be non-identifiable? Give a one-line example.
5. Compute the gradient of f(x) = x^2 + 3x with respect to x.
6. Define KL divergence between two discrete distributions in one line.
7. What is the difference between a training objective and an evaluation metric?
8. Write the shape of a matrix that maps a 4-vector to a 3-vector.

### Diagnostic keys

1. PMF: P(X=1)=0.6, P(X=0)=0.4. E[X]=0.6. Var=0.6*0.4=0.24.
2. Posterior equals likelihood times prior divided by evidence: P(A|B) = P(B|A)P(A)/P(B).
3. l(p) = sum_i [ y_i log p + (1-y_i) log(1-p) ].
4. Two different parameter values give the same data distribution, so data cannot tell them apart. Example: Bradley-Terry scores shifted by a constant.
5. 2x + 3.
6. KL(p||q) = sum_x p(x) log(p(x)/q(x)).
7. The objective is what the optimizer minimizes. The metric is what the stakeholder uses to judge success. They can disagree.
8. 3 by 4.

## Not-yet-understood dependency list (course level)

1. Measure-theoretic probability, not needed. Finite discrete cases suffice here.
2. Variational inference beyond the ELBO statement, introduced locally in U03.
3. PPO internals, compared at the objective level in U05, not re-derived.

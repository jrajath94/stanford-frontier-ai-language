---
page_id: cs329h-l06
course_slug: cs329h
course_name: "CS329H: Machine Learning from Human Preferences"
course_order: 6
order: 6
nav: "L06 · Active Learning"
title: "Lesson 6: Active Learning for Preferences"
summary: "Fisher information, optimal query selection, and Active DPO. When human feedback is expensive, which comparisons to request matters as much as how to learn from them."
instructor: "Sanmi Koyejo"
offering: "Autumn 2025"
concepts: [active learning, Fisher information, optimal experimental design, Gaussian processes, ADPO]
sources:
  - tag: paper
    label: "Machine Learning from Human Preferences, Ch. 2 (Truong, Haupt, Koyejo, 2025)"
    url: https://mlhp.stanford.edu/Machine-Learning-from-Human-Preferences.pdf
---

> [!KEY] When human feedback is expensive, which comparisons to request matters as much as how to learn from them. Model uncertainty should guide query selection, focusing annotation effort where it reduces uncertainty most.

## Fisher information

For a parametric model \(p(Y \mid \theta)\), the Fisher information about \(\theta\) is:

\[ \mathcal{I}(\theta) = \mathbb{E}\left[-\frac{\partial^2}{\partial \theta^2} \log p(Y \mid \theta)\right] = \mathbb{E}\left[\left(\frac{\partial}{\partial \theta} \log p(Y \mid \theta)\right)^2\right] \]

It quantifies how much a single observation reveals about the parameter. Higher Fisher information means more precise estimation per observation. The Cramér-Rao bound states that any unbiased estimator's variance is at least \(\mathcal{I}(\theta)^{-1}\).

## Optimal item difficulty

The Rasch model makes this concrete. For a candidate item \(j\), the success probability is \(p_j(U) = \sigma(U + V_j)\), and the Fisher information about user ability \(U\) from one response is:

\[ \mathcal{I}_j(U) = p_j(U)(1 - p_j(U)) \]

This is a downward parabola in \(p\), maximized at \(p = 0.5\). The optimal item has difficulty \(-V_j\) equal to the user's ability \(U\). Items too easy (\(p \approx 1\)) or too hard (\(p \approx 0\)) give near-deterministic responses that teach almost nothing.

Worked example from the textbook: a user with estimated ability \(\hat{U} = 1.0\) and three items with appeals \(V_1 = -2.0\) (hard), \(V_2 = -1.0\) (matched), \(V_3 = 1.0\) (easy).

| Item | \(p = \sigma(\hat{U} + V)\) | \(\mathcal{I} = p(1-p)\) |
|------|---------------------------|--------------------------|
| Hard | 0.269 | 0.197 |
| Matched | 0.500 | 0.250 |
| Easy | 0.881 | 0.105 |

The matched item wins. This is the principle behind computerized adaptive testing: the test follows the student, always selecting questions near their ability level. Active elicitation policies hover around the user's current location, rapidly shrinking uncertainty.

```mermaid
flowchart LR
    A[Current ability estimate] --> B[Score each candidate by p(1-p)]
    B --> C[Ask the item nearest p=0.5]
    C --> D[Update estimate via Newton step]
    D --> A
```

## A, D, and E optimality

Three classical criteria from optimal experimental design guide query selection:

- **A-optimal** minimizes average variance (the trace of the inverse information matrix).
- **D-optimal** maximizes the determinant of the information matrix, shrinking the volume of the posterior confidence ellipsoid.
- **E-optimal** minimizes the worst-case variance (the largest eigenvalue).

For multi-dimensional preferences, the Sherman-Morrison update gives efficient \(O(K^2)\) covariance updates after each query, making real-time active selection practical.

## Active learning with Gaussian processes

For nonparametric GP preference models, uncertainty varies across the input space: higher far from observed data, lower near it. The acquisition function measures expected information gain about the reward function:

\[ a(Q) = I(r; y \mid \mathcal{D}, Q) = H(y \mid \mathcal{D}, Q) - \mathbb{E}_{r \sim p(r \mid \mathcal{D})}[H(y \mid r, Q)] \]

Two terms trade off. The first is high when the model is uncertain (\(p(A \succ B) \approx 0.5\)). The second is low when a human could answer clearly. We want queries where the model is uncertain but a human would not be: genuine knowledge gaps, not inherently noisy comparisons.

A clean property: comparing an item to itself, \(Q = (x, x)\), is a global minimizer of this acquisition function, not a maximizer. The method cannot get stuck on uninformative queries.

## Active DPO

Applying Fisher information to billion-parameter language models faces a scale problem: exact Fisher computation is intractable. The textbook's key approximation treats the policy as log-linear in last-layer features:

\[ \pi(y \mid x; \theta) \propto \exp(\phi(x, y)^\top \theta) \]

Under this assumption, the DPO loss Hessian takes the form:

\[ H(\theta) = \sum_{i=1}^n p_i(1 - p_i) \cdot \Delta\phi_i \Delta\phi_i^\top \]

where \(\Delta\phi_i\) is the feature difference between winning and losing responses. This Hessian is exactly the Fisher information matrix for the DPO model. The D-optimal criterion selects the comparison set maximizing \(\log\det\) of this matrix.

The Active DPO (ADPO) algorithm: at each step, pick the candidate pair maximizing the D-optimal gain given the current Fisher matrix, then update. Under the log-linear assumption and standard regularity conditions, ADPO achieves \(\|\hat{\theta}_n - \theta^*\| = O(d/\sqrt{n})\), the minimax-optimal rate for \(d\)-dimensional estimation.

> [!CAVEAT] ADPO's guarantees rely on the log-linear policy assumption, which holds in the neural tangent kernel regime near the reference policy. It degrades when the policy is fine-tuned far from the reference, when the model is small relative to data complexity, or when the loss surface has multiple basins. Monitor whether predicted D-optimal gains correlate with actual policy improvements.

## Historical roots

Active learning for preferences did not start with LLMs. Computer-adaptive testing in psychometrics (1960s) used Fisher information to select test items matching student ability. Fedorov and Kiefer developed A/D/E-optimality for linear models in the 1970s. Chaloner and Verdinelli unified information-theoretic Bayesian experimental design in the 1990s. RLHF and Active DPO apply these decades-old ideas to LLM alignment at scale. As Fedorov wrote in 1972: the purpose of optimal design is to achieve the desired precision with minimum cost.

## The Bayesian update rule

Place a Gaussian prior \(U \sim \mathcal{N}(0, \sigma^2)\) on user ability. After observing responses, a Laplace update uses the score and the observed information:

\[ S(U) = \sum_{j \in \mathcal{J}} (y_j - p_j(U)), \qquad \mathcal{I}(U) = \sum_{j \in \mathcal{J}} p_j(U)(1 - p_j(U)) + \tau_0 \]

where \(\tau_0 = \sigma^{-2}\) is the prior precision. A Newton step moves the estimate:

\[ \hat{U} \leftarrow \hat{U} + S(\hat{U}) \, \mathcal{I}(\hat{U})^{-1} \]

The approximate posterior variance is \(\widehat{\text{Var}}(U) \approx \mathcal{I}(\hat{U})^{-1}\). Each new response adds its \(p(1-p)\) to the information total, shrinking the variance. An item-selection rule maximizing expected incremental information picks \(j_t = \arg\max_j \, p_j(\hat{U}_{t-1})(1 - p_j(\hat{U}_{t-1}))\).

```mermaid
flowchart TB
    A[Prior precision tau0] --> B[Observe response y]
    B --> C[Score S = y minus p]
    C --> D[Information I = sum p(1-p) + tau0]
    D --> E[Newton step: U += S/I]
    E --> F[Variance approx 1/I shrinks]
    F --> G[Select next item maximizing p(1-p)]
    G --> B
```

## Measuring progress: reliability

With population variance \(\text{Var}(U^*) = \sigma_U^2\), reliability is the fraction of total variance explained by the test rather than estimation noise:

\[ \text{Rel} \approx 1 - \frac{1}{N}\sum_{i=1}^N \frac{\widehat{\text{Var}}(U_i)}{\sigma_U^2} \]

The textbook's simulation compares Fisher-active item selection against random selection across many users. The active policy reaches a given reliability with far fewer queries. This is the quantitative payoff of optimal design.

## GP versus ADPO

| Aspect | GP active learning | ADPO |
|--------|-------------------|------|
| Model | Nonparametric (GP) | Parametric (neural network) |
| Scalability | \(O(n^3)\) | \(O(d^2 n)\) with approximations |
| Use case | Small datasets, need uncertainty | LLM alignment, large models |
| Acquisition | Information gain | D-optimal design |
| Theory | Mutual information | Fisher information |

Both share the core insight: use model uncertainty to guide query selection.

## When active learning helps most

| Factor | Active learning beneficial when |
|--------|-------------------------------|
| Annotation cost | High cost per comparison |
| Data heterogeneity | Diverse prompts, varied difficulty |
| Budget | Limited (< 10K comparisons) |
| Model capacity | Larger models, overfitting risk |
| Query pool | Large pool to select from |

GP active learning suits small datasets needing uncertainty quantification. ADPO suits LLM alignment at scale. Both share the core insight: spend the annotation budget where uncertainty is highest.

> [!INTERVIEW] "How would you cut your RLHF labeling budget in half?" The answer: active query selection. Score candidate pairs by Fisher information (or information gain for GPs), label the highest-scoring ones first, and stop when marginal information gain flattens. Interviewers want the p(1-p) intuition: close competitions reveal more than blowouts.

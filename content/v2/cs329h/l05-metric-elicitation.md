---
page_id: cs329h-l05
course_slug: cs329h
course_name: "CS329H: Machine Learning from Human Preferences"
course_order: 6
order: 5
nav: "L05 · Metric Elicitation"
title: "Lecture 5: Metric Elicitation, Asking the Best Questions"
summary: "Fisher information and the 50/50 rule, adaptive elicitation loops, D-optimal design for preference functions, GP active queries, and active DPO."
date: "[uncertain] Spring 2026"
instructor: "Sanmi Koyejo"
offering: "Spring 2026"
duration: "1:22:40"
video_id: 7i6WsIzZaeo
video_title: "Stanford CS329H Lecture 3: Metric Elicitation"
video_caption: "Original lecture. Sanmi Koyejo on metric elicitation and its tie to mechanism design."
concepts: [metric-elicitation, fisher-information, adaptive-testing, d-optimal, active-learning, gaussian-process, adpo]
sources:
  - tag: video
    label: "Lecture 3 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=7i6WsIzZaeo
  - tag: notes
    label: "Official subtitle transcript (en-US)"
  - tag: notes
    label: "Course textbook, chapters 2.17-2.20 (Truong, Haupt, Koyejo, 2025)"
---
## The problem: questions cost money

Every label in this course costs an annotator's time and
attention. A preference dataset of 100,000 pairs is 100,000 paid
judgments. Lectures 2 through 4 assumed the pairs arrive. This
chapter asks which pairs to buy. Random pairs waste the budget
on questions whose answers you can already predict. The goal:
pick the queries that shrink your uncertainty fastest.

The key tool is **Fisher information**. For a parameter theta, it
measures how much one observation reveals about theta. Higher
information means tighter estimates from fewer questions. The
Cramer-Rao bound makes this precise: no unbiased estimator beats
the inverse information. Information is the currency. Spend it
well.

## First attempt: ask random pairs

The naive pipeline samples pairs uniformly and labels them all.
It works. It is also wasteful in a way you can measure. Suppose
your current model says A beats B with probability 0.95. You pay
for the label. The annotator says A, as expected. What did you
learn? Almost nothing. The answer was predictable. The budget is
gone anyway.

The waste has a number. For a Bernoulli observation with success
probability p, the Fisher information about the underlying
parameter is:

I = p * (1 - p)

Work it at three values.

```ascii
p = 0.90:  I = 0.90 x 0.10 = 0.09
p = 0.50:  I = 0.50 x 0.50 = 0.25
p = 0.10:  I = 0.10 x 0.90 = 0.09
```

The 50/50 question carries 0.25 units of information. The
predictable question carries 0.09. One well-chosen question is
worth nearly three wasted ones. Random sampling spends most of
the budget near 0.09. That is the failure, demonstrated.

## The key question

If information peaks at 50/50, can we choose each question to
sit at the current edge of our knowledge?

## The 50/50 rule, worked

Take the Rasch model from Lecture 1. A candidate item j has
acceptance probability p_j(U) = sigma(U + V_j) for a user of
ability U. The Fisher information about U from one response is
p_j(1 - p_j): the downward parabola from above, maximized at
p = 0.5. The most informative item is the one the user gets right
half the time. Too easy teaches nothing. Too hard teaches
nothing.

![Fisher information picks the next question](assets/l05-fisher.svg "I = p(1-p) peaks at p = 0.5. Worked: matched item gives I = 0.250, hard gives 0.197, easy gives 0.105. Source: original figure for Stanford Frontier AI.")

Worked example from the textbook. Estimated ability U-hat = 1.0.
Three candidate items: hard (V = -2.0), matched (V = -1.0), easy
(V = 1.0).

```ascii
hard:     p = sigma(1.0 - 2.0) = sigma(-1.0) = 0.269,  I = 0.197
matched:  p = sigma(1.0 - 1.0) = sigma(0.0)  = 0.500,  I = 0.250
easy:     p = sigma(1.0 + 1.0) = sigma(2.0)   = 0.881,  I = 0.105
```

Ask the matched item. Its difficulty, -V_j = 1.0, equals the
ability estimate. This is the principle behind computerized
adaptive testing: the SAT and GRE pick each question to sit at
your estimated level. Every question is a 50/50 bet against your
current belief.

> [!QA]
> Q: Why is a 50/50 question the most informative?
> A: Fisher information for a Bernoulli response is p(1-p), which peaks at p = 0.5 with value 0.25. A question the user always gets right or always wrong has information near zero: the answer was predictable. The 50/50 question splits the posterior most evenly, so the answer moves your estimate the most.
> Follow-up: What goes wrong if your ability estimate is bad?
> A: You ask at the wrong difficulty. Overestimate the user and every question looks hard: answers are near-deterministic failures, information collapses. The fix is the adaptive loop below: re-estimate after each answer so the difficulty tracks the improving estimate. Early questions should hedge across difficulties.

## The adaptive loop

One good question is not enough. Chain them.

![The adaptive elicitation loop](assets/l05-adaptive.svg "Estimate, select the max-information query, ask, update. Repeat until the budget runs out. Source: original figure for Stanford Frontier AI.")

1. **Estimate.** Current best guess of the parameter.
2. **Select.** The candidate query with max Fisher information.
3. **Ask.** Get the human's answer.
4. **Update.** Refit the estimate with the new data.

Repeat until the budget runs out. Each answer sharpens the next
question. The loop is greedy: it maximizes immediate information,
not the optimal sequence. Greedy is near-optimal here and far
simpler than planning the whole sequence.

> [!QA]
> Q: Why not plan the whole question sequence up front?
> A: Because answers change the plan. A surprising answer moves the estimate, which changes which question is most informative next. Pre-planned sequences cannot adapt. The greedy loop re-optimizes after each answer, which is why adaptive testing beats fixed forms with the same number of questions.
> Follow-up: When does greedy fail?
> A: When information is complementary: two questions together reveal more than the sum of their separate gains. Greedy picks the best single question and can miss synergistic pairs. In practice the loss is small for smooth parametric models, and the textbook's D-optimal simulations show greedy active selection beating random selection clearly.

## From one parameter to many: D-optimality

So far the parameter was one user's ability. Now learn a shared
weight vector W over item features: V_j = W^T X_j. The pair
probability is sigma(W^T (X_j - X_k)): logistic regression on
feature differences.

The Fisher information becomes a matrix. Each query adds a
rank-one update scaled by p(1-p). The 50/50 rule survives inside
the matrix: uncertain pairs contribute more. The selection
criterion is **D-optimality**: pick the pair maximizing the
log-determinant gain of the precision matrix.

logdet(Lambda + w x_jk x_jk^T) - logdet(Lambda), w = p(1-p)

D-optimality maximizes the information volume: it shrinks the
posterior ellipsoid fastest in all directions at once. The
textbook simulates this: 50 items, 5 features, 100 queries.
Active D-optimal selection beats random pairs on estimating W.
The margin is the payoff of the whole chapter.

## Asking well for flexible models

Parametric models have Fisher information. **Gaussian processes**
have posterior variance, which varies across the input space:
high far from data, low near it. The acquisition function for a
candidate query Q = (x_A, x_B) is the expected information gain
about the reward function:

a(Q) = H(y | D, Q) - E_r[H(y | r, Q)]

Two terms. Predictive entropy: how uncertain is the outcome under
the current model? High when p(A beats B) is near 0.5. Expected
conditional entropy: how uncertain would the answer be even
knowing the true reward? Low for clean comparisons, high for
inherently noisy ones.

The rule: ask where the model is uncertain but a human could
answer clearly. Skip queries where the model is certain, waste,
and queries where even the truth is noisy, unanswerable.

![GP preference models](assets/l05-gp.svg "Prior, pairwise data, posterior. Query where the posterior is wide but the human is reliable. Source: original figure for Stanford Frontier AI.")

The textbook also sketches **active DPO**: DPO trains
billion-parameter policies, so exact Fisher information is
intractable. The ADPO algorithm approximates the policy locally
and selects pairs by approximate information gain. The principle
is unchanged. The computation is approximate.

## Mapping back: what active selection buys

| Random-sampling pain | Active answer | How |
|---|---|---|
| Budget spent on predictable answers | Ask at 50/50 | I = p(1-p) peaks at 0.25; predictable pairs sit near 0.09 |
| Fixed forms cannot react | Adaptive loop | Re-estimate after each answer; difficulty tracks belief |
| Many parameters, one rule | D-optimality | Max logdet gain shrinks the ellipsoid in all directions |
| Flexible models, no Fisher matrix | Information gain | Ask where the model is uncertain but the human is clear |

## The honest price: selection can bias

Active selection is not free. The queries it picks are not a
random sample, so naive estimators on actively collected data
can be biased. Greedy selection can miss complementary question
pairs. And the whole scheme assumes the model is right: if the
BT model is misspecified, the "most informative" query under a
wrong model teaches the wrong thing efficiently. Active learning
amplifies model errors as well as model gains. The textbook's
remedy: keep a stream of random queries as a reality check, and
re-examine the model when the active stream disagrees with it.

## Recap: the whole lesson on one screen

The story in eight steps. Each step answers the one before it.

1. **Questions cost.** Each label is an annotator's time.
   Random pairs spend it on predictable answers.
2. **Information is measurable.** I = p(1-p). The 0.9
   question carries 0.09. The 0.5 question carries 0.25.
   Nearly 3x the value.
3. **Ask at 50/50.** Match difficulty to the current
   estimate. The toy: hard 0.197, matched 0.250, easy 0.105.
   The SAT and GRE run this rule.
4. **Loop it.** Estimate, select, ask, update. Greedy is
   near-optimal and far simpler than planning ahead.
5. **Many parameters: D-optimality.** Max the logdet gain.
   Shrink the ellipsoid in all directions. 50 items, 5
   features, 100 queries: active beats random.
6. **GPs use information gain.** Predictive entropy minus
   expected conditional entropy. Uncertain model, clear
   human.
7. **Active DPO approximates.** Billion parameters break
   exact Fisher. Approximate locally, select greedily.
8. **The price is bias.** Selected queries are not random.
   Keep a random stream. A wrong model teaches the wrong
   thing fast.

## Official sources and further reading

**Official:**
- Lecture 5 video (metric elicitation): video id 7i6WsIzZaeo.
  The lecture ties elicitation to the earlier choice models and
  to future applications.
- Course textbook, chapters 5.x: Fisher information, the 50/50
  rule, the adaptive loop, D-optimality, GP acquisition, ADPO.

**Further reading:**
- Chaloner and Verdinelli (1995): the Bayesian experimental
  design reference behind the information-gain rule.

**Caveats from these sources.** The 50-item, 5-feature,
100-query simulation is the textbook's illustration. The
Cramer-Rao bound applies to unbiased estimators; regularized
estimators trade bias for variance. ADPO is sketched in the
textbook, not derived in full.

## Connections to the other courses

- **CS329H L01:** the Rasch model whose p(1-p) starts the
  chapter.
- **CS329H L04:** the posterior variance from Bayes is what
  active selection spends.
- **CS329H L06:** the same information rule picks duels in
  preferential Bayesian optimization.
- **CS329H L07:** active DPO selects pairs for the alignment
  loop.
- **CS329H L09:** annotation budgets are mechanism design;
  elicitation and incentives are one story.

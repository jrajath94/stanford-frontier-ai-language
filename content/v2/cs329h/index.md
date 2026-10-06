---
page_id: cs329h-index
course_slug: cs329h
course_name: "CS329H: Machine Learning from Human Preferences"
course_order: 6
order: 0
nav: "CS329H · Overview"
title: "CS329H: Machine Learning from Human Preferences"
summary: "Choice theory, preference learning, RLHF, and alignment."
instructor: "Sanmi Koyejo"
offering: "Spring 2026"
---
## The arc

One question runs this course: how do you turn human comparisons
into trained models that behave well? The answer chains ten
chapters.

Comparisons beat scores because scales drift (L01). Random
utility turns comparisons into probabilities: Gumbel noise gives
the softmax, two items give Bradley-Terry, rankings stage it
into Plackett-Luce (L02). The IIA axiom buys tractability and
breaks on clones and mixtures; only differences are identified
(L03). Fitting is logistic regression with specific traps:
undefeated items explode, priors cap them, Elo learns online
(L04). Questions cost money: ask at 50/50 Fisher information
(L05). Acting prices exploration: Thompson sampling, duels,
preferential optimization (L06). RLHF applies it all; DPO skips
the reward model and inherits every assumption (L07). Many
voices cannot aggregate neutrally: Arrow, cycles, Borda (L08).
Strategists need mechanism design: second-price, VCG,
revelation (L09). Deployment is values all the way down (L10).

## Lessons

1. [L01 · Preference Foundations](l01-preference-foundations.html) — why comparisons, the Rasch model, the LLM running example
2. [L02 · Choice Models](l02-choice-models.html) — random utility, Bradley-Terry, Plackett-Luce, the preference pair
3. [L03 · IIA and Identification](l03-iia-identification.html) — red-bus/blue-bus, mixtures, anchors, Rashomon
4. [L04 · Learning Rewards](l04-learning-rewards.html) — MLE, Bayes, Elo, noise, overfitting
5. [L05 · Metric Elicitation](l05-metric-elicitation.html) — Fisher information, the 50/50 rule, adaptive asking
6. [L06 · Bandits](l06-bandits-exploration.html) — Thompson sampling, dueling, preferential BO, CIRL
7. [L07 · RLHF](l07-rlhf.html) — the loop, reward hacking, DPO, the assumption checklist
8. [L08 · Social Choice](l08-social-choice.html) — voting rules, Arrow, Borda, the DPO connection
9. [L09 · Mechanism Design](l09-mechanism-design.html) — auctions, incentive compatibility, VCG, revelation
10. [L10 · Alignment in Practice](l10-alignment-practice.html) — inversion, fairness, Polis, human-centered design

Also: [Cheatsheet](cheatsheet.html) for formulas and interview one-liners,
[Crash course](crash-course.html) for the 30-minute review.

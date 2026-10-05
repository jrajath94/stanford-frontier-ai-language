---
title: "CS329H: Machine Learning from Human Preferences"
course: cs329h
type: course-index
instructor: Sanmi Koyejo
term: Autumn 2025 / Autumn 2026
---

How to optimize systems that are used by humans. Choice modeling, preference learning, RLHF, bandits, and social choice, assembled from six fields that rarely talk to each other.

CS329H is unique in this system. No other course teaches preference modeling, choice theory, or social choice. The RLHF/DPO lessons bridge to CS336; everything else is taught here in full.

> [!CAVEAT] Primary sources: the course textbook "Machine Learning from Human Preferences" (Truong, Haupt, Koyejo, 2025) and the Autumn 2024 lecture videos. Autumn 2026 is running now; its schedule informs the lesson plan.

## Lessons

### Foundations
1. [Introduction to preference learning](l01-introduction.html) — why preferences, the running example
2. [Models of preferences](l02-preference-models.html) — Bradley-Terry, Plackett-Luce, Luce's axiom
3. [Utility models and identification](l03-utility-models.html) — stochastic utility, IIA, the Rashomon effect

### Learning
4. [Learning from preference data](l04-learning-preferences.html) — MLE, DPO from first principles
5. [RLHF: PPO and GRPO](l05-rlhf.html) — bridge to CS336, the preference-learning view
6. [Active learning](l06-active-learning.html) — Gaussian processes, optimal experimental design

### Action and inversion
7. [Metric elicitation and inverse RL](l07-elicitation-irl.html) — learning the reward function
8. [Assistance games and bandits](l08-games-bandits.html) — acting under preference uncertainty

### Aggregation
9. [Social choice and aggregation](l09-social-choice.html) — voting, fairness, the DPO-Borda connection
10. [Mechanism design and ethics](l10-mechanism-design-ethics.html) — incentives, privacy, human-centered design

## Sources

- Textbook: "Machine Learning from Human Preferences" (Truong, Haupt, Koyejo, 2025), mlhp.stanford.edu
- Videos: Stanford Online YouTube, Autumn 2024
- Schedules: Autumn 2025 and Autumn 2026 course sites

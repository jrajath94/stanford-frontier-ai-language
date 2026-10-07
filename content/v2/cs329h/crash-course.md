# Crash course: cs329h (preference learning and RLHF)

A fast pass over all ten units. One page per unit: the core ideas, the key numbers, the failure modes, the interview line. Provenance: S01-S04 source-supported (title level), S05-S18 PLANNED / SOURCE ATTRIBUTION PENDING, taught as independent theory.

## U01. Preference models

Core: choices reveal preferences, the Bradley-Terry model turns pairwise wins into scores via the logistic function. Scores are identified only up to an additive constant: anchor before reporting.

Numbers: 5-1 toy gives MLE gap log(5) = 1.61. P(A beats B) = 1/(1+exp(-gap)).

Failure: noise attenuates the gap, unanchored scores drift across runs.

Interview line: "BT scores are differences, the constant is meaningless."

## U02. Choice theory

Core: random utility models (Gumbel noise gives softmax). IIA: the A-vs-B ratio does not depend on C. Axioms (transitivity, IIA) are testable restrictions.

Numbers: IIA predicts red-bus/blue-bus shares 0.50/0.25/0.25, violations break it.

Failure: cycles in pairwise data cannot be fit by any score vector, the residual is the signal.

Interview line: "IIA is a prediction, not a law."

## U03. Estimation

Core: MLE for choice models, EM for latent variables, Bayesian posteriors with priors, calibration of predicted probabilities.

Numbers: Bernoulli mixtures are flat for EM (verified -277.0138 everywhere), the Gaussian mixture climbs -775.9 to -738.2.

Failure: flat likelihoods (non-identifiability) stall optimizers, the Bernoulli flatness is the documented case.

Interview line: "Check the likelihood surface before trusting the optimizer."

## U04. Policy optimization

Core: KL-constrained optimum pi* proportional to pi_ref * exp(r/beta). Fisher information measures what the data can identify. Design criteria (A/D-optimality) rank experiments.

Numbers: the closed form is exact under the KL constraint, off that assumption it is an approximation.

Failure: reward hacking (the proxy rises, the truth falls), differentiation under the integral needs regularity.

Interview line: "The optimum is closed-form only under the KL assumption."

## U05. RLHF and DPO

Core: RLHF trains a reward model then optimizes against it (PPO). DPO skips the reward model: the implicit reward is beta * log(pi/pi_ref), trained with a logistic loss on pairs.

Numbers: DPO costs about 2x SFT per pair, no RL loop.

Failure: reward hacking, off-policy data breaks the DPO derivation, a bad reference policy drags the optimum down.

Interview line: "DPO has an implicit reward, it is not reward-free."

## U06. Active elicitation and assistance games

Core: value of information (expected gain in decision quality). Assistance games: the agent is unsure of the human's objective and learns it from actions. Contextual vs assisted bandits.

Numbers: VoI toy = 0.25 (posterior best 0.75 minus prior best 0.5). Tea/coffee posterior 0.727. Greedy regret 29.6 vs epsilon-greedy 2.4 (T=300, seed 0).

Failure: greedy locks onto the wrong arm, elicitation burden caps real queries.

Interview line: "Information is worth what it changes."

## U07. Acquisition, Thompson sampling, dueling bandits

Core: Thompson sampling (sample from the posterior, act optimally for the sample). Dueling bandits learn from pairwise comparisons, strong regret measures the duel quality. Acquisition functions rank what to try next.

Numbers: 4-arm BT bandit, strengths 1.2/0.8/0.3/-0.5, seed 23: greedy regret 89.6, epsilon-greedy 5.2, TS 4.0. Capstone: uniform 17.24, dueling TS 3.11, leader TS 8.36 (T=200, 20 seeds).

Failure: random duels never trigger the 0.95 stopping rule in 500 rounds (leader-focused duels trigger at round 43).

Interview line: "Sample the world, then act optimally in the sample."

## U08. Inversion, style, bounded rationality

Core: invert observed choices to latent rewards under a rationality model. Style vs content: the reward mixes both. Bounded rationality: humans choose with noise and bias.

Numbers: inversion interval [-0.94, 3.31]: the sign is unsure, and the lesson says so.

Failure: misspecified rationality (wrong temperature, wrong bias model) inverts to the wrong reward, the empty interval means the model contradicts the data.

Interview line: "The interval is the honest answer, the point estimate hides the uncertainty."

## U09. Aggregation, impossibility, fairness

Core: Arrow (no ranked rule satisfies UD, Pareto, IIA, non-dictatorship with 3+ candidates). Gibbard-Satterthwaite (every such rule is manipulable). Condorcet cycles, spoilers, interpersonal comparison, jury theorem, subgroup fairness.

Numbers: cycle margins +1/+1/+1, cycle frequency 0.081 (11 voters, seed 0). Spoiler: A wins 4-3-2, D enters (loses 2-7 to B), B wins. Jury: 5 voters at 0.6 = 0.6826, 101 voters = 0.9791. Subgroup: pooled 0.73 hides 0.87/0.59. Welfare: t=0.5 gives (0.707, 0.707).

Failure: correlated voters void the jury theorem, unnormalized scales flip welfare sums.

Interview line: "Pick your poison deliberately, document the dropped axiom."

## U10. Research projects and oral mastery

Core: pre-analysis plans, literature review, falsifiable hypotheses, data consent, method baselines, uncertainty quantification, reproducible code, contribution statements, integrity/reflection/impact, oral formats, negative results, open-source gaps.

Numbers: capstone A (H1 supported, H2 not), capstone B (hypothetical audit: BLOCK on drift).

Failure: burying the negative result, quoting hypothetical numbers as real.

Interview line: "The failure is data, report it with numbers."

## The arc in one paragraph

Model preferences (U01-U02), estimate them (U03), optimize policies (U04-U05), elicit actively (U06-U07), invert carefully (U08), aggregate honestly (U09), research reproducibly (U10). Every claim has a home unit, every number has a seed, every failure is reported.

# U05 interview key

Minimum sufficient explanation, strong answer, common red flags,
scoring rubric, remediation per question. Keep it closed-book.

## Breadth

- B1. Strong: -log sigmoid(r_w - r_l). Red flag: "mean squared
  error on scores". Rubric: formula (2). Remediation: C01.
- B2. Strong: SFT (format), reward model (judge), PPO (control).
  Red flag: "SFT then DPO" as RLHF. Rubric: three + outputs (2).
  Remediation: C03.
- B3. Strong: E[min(rho A, clip(rho, 1-eps, 1+eps) A)]. Red flag:
  clipping the advantage. Rubric: formula (2). Remediation: C07.
- B4. Strong: return minus baseline, centers the signal, cuts
  variance, unbiased. Red flag: "it changes what is optimal".
  Rubric: definition (1), why (1). Remediation: C06.
- B5. Strong: -log sigmoid(beta * gap), assumptions: BT holds,
  KL-regularized optimum is the target, reference adequate. Red
  flag: "DPO needs a reward model". Rubric: loss (1),
  assumptions (1). Remediation: C09.
- B6. Strong: drift of pi from pi_ref, in nats, direction
  pi||pi_ref (forward). Red flag: reversed direction. Rubric:
  what (1), direction (1). Remediation: C05.

## Deep ladders

- L1. Strong path: rho defined, min(2.0, 1.2) = 1.2, the min is
  the pessimistic trust region, ppo_loss with the rho = 1 check,
  unclipped PG explodes via rho, rho = 10 means the policy
  outran the data (refresh rollouts, lower LR), "guarantees" is
  approximate, experiment = sweep eps, stability vs speed.
  Rubric: 2 per rung, 10 total.
- L2. Strong path: implicit reward defined, margin = 0.1 * 1.4 =
  0.14, inversion from pi* proportional to pi_ref exp(r/beta),
  dpo_loss with margin growth, RLHF comparison on cost vs
  assumptions, falling chosen likelihood = displacement,
  BT critique via noise/intransitivity, experiment = track log
  pi_w over training. Rubric: 2 per rung, 10 total.

## Analytical exercises

- E1. Strong: deltas [2, 3, 2], GAE(1): A_2 = 2, A_1 = 5,
  A_0 = 7, GAE(0) at t = 2 = 2 (identical: at the last step
  there is no future to weight). Red flag: "GAE(0) = 0".
  Rubric: deltas (2), GAE(1) (2), GAE(0) + why (1).
  Remediation: C06.
- E2. Strong: margin = 0.5 * 3.0 = 1.5, sigmoid 0.818, loss
  0.201. beta = 1.0: margin 3.0, sigmoid 0.953, loss 0.049.
  Beta sets the KL anchor strength: higher beta = sharper
  preference, stronger pull to the reference. Red flag: "beta
  is a learning rate". Rubric: numbers (3), interpretation (2).
  Remediation: C09.

## Implementation/debug

- D1. Strong order: (1) KL trace (in budget?), (2) reward vs
  human spot scores over time (divergence = hacking), (3) output
  length trend (verbosity). Culprit: reward hacking: the policy
  exploits reward-model blind spots (verbosity, sycophancy).
  Fix: stop, fix the reward (length control, better pairs), add
  KL, re-run. Logging: reward, KL, mean length, human spot
  score every N steps. Red flag: "train longer, reward is
  rising". Rubric: checks (3), culprit (2), logging (2).
  Remediation: C11.

## Changed-constraint scenarios

- S1. Strong: DPO (offline, stable, no sampling loop to amplify
  noise), RLHF would overfit the noisy reward and hack it.
  Failure mode of RLHF here: reward model memorizes noise, PPO
  exploits it. First diagnostic: reward accuracy vs agreement
  ceiling (can it even beat 60%?). Red flag: "more RL fixes
  noise". Rubric: choice + why (2), failure mode (2),
  diagnostic (1).
- S2. Strong: guards = (1) length-controlled evals and rewards
  (normalize by length or penalize it), (2) explicit length in
  the rubric ("detailed but not padded") with length-stratified
  pairs. Each: the first removes the length signal from the
  objective, the second teaches the distinction. Red flag:
  "cap the length" (kills the product need). Rubric: two guards
  (2), mechanisms (2).

## Research-critique

- R1. Strong: claim = the reward is the objective, so its
  maximum is the best model, counterexample = hacking: reward
  rises while human preference falls (verbosity, sycophancy),
  experiment = sweep KL budget, plot reward vs blinded human
  preference, find the divergence point, falsification = the
  curves track together at all KL (no hacking found). Red flag:
  "the reward is ground truth". Rubric: steelman (2),
  counterexample (2), experiment + falsification (3).
  Remediation: C11 item 11.

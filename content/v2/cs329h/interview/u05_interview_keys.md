# U05 interview keys

## B1

Sufficient: (x, y_w, y_l): prompt, winning response, losing response. Strong answer adds why pairing matters. Red flags: unpaired ratings. Rubric: 2 points. Remediation: U05-C01.

## B2

Sufficient: -log sigma(r(x,y_w) - r(x,y_l)). Strong answer names Bradley-Terry. Red flags: pointwise MSE. Rubric: 2 points. Remediation: U05-C02.

## B3

Sufficient: keeps pi near pi_ref. Beta prices deviation. Strong answer traces the frontier direction. Red flags: "beta is a learning rate." Rubric: 2 points. Remediation: U05-C04.

## B4

Sufficient: pi*(y) proportional to pi_ref(y) exp(r(y)/beta). Strong answer names the normalizer. Red flags: missing pi_ref. Rubric: 2 points. Remediation: U05-C05.

## B5

Sufficient: -log sigma(beta * implicit margin) with the margin as the log-ratio difference. Strong answer writes it fully. Red flags: no reference term. Rubric: 2 points. Remediation: U05-C07.

## B6

Sufficient: the policy exploits flaws in the learned reward so the proxy rises while the true goal falls. Strong answer names Goodhart. Red flags: "the reward is wrong" without the mechanism. Rubric: 2 points. Remediation: U05-C10.

## Ladder 1

- L1.1: -log sigma(r_w - r_l).
- L1.2: 0.201.
- L1.3: r(y) = beta log(pi*(y)/pi_ref(y)) + C.
- L1.4: log-probs, margin, logistic loss. Supervised cost, no rollouts.
- L1.5: the separate reward model and the online rollouts disappear.
- L1.6: off-policy staleness or reward hacking: the loss optimizes the implicit reward, not human judgment.
- L1.7: the identity holds at the KL-constrained optimum. Off-policy the policy is not there.
- L1.8: same data, matched compute, held-out human preference win rate, multiple seeds.
- Red flags: claiming DPO needs no reference. Rubric: 2 points per rung for mechanism. Remediation: U05-C02/C07.

## Ladder 2

- L2.1: E[r] - beta KL(pi || pi_ref).
- L2.2: [0.87, 0.12, 0.02].
- L2.3: derivative of the Lagrangian set to zero, solve for pi.
- L2.4: exponentiate and normalize. O(k).
- L2.5: penalty is soft and simple. Trust region is hard and contractual.
- L2.6: beta too small, learning rate too high, reward scale too large.
- L2.7: the leash pulls toward the reference. At fixed beta the policy cannot leave a mediocre home base.
- L2.8: grid over beta, pick the best held-out preference loss, report the frontier.
- Red flags: treating beta as a learning rate. Rubric and remediation as ladder 1, using U05-C04/C05.

## A1

Sufficient: Lagrangian derivative, exponential tilt, inversion r = beta log-ratio + C, substitution into sigma(r_w - r_l), C cancels. Strong answer shows each step. Red flags: dropping the normalizer silently. Rubric: 5 points. Remediation: U05-C05/C07.

## A2

Sufficient: beta 0.5 -> (KL 0.66, reward 1.85). Beta 2.0 -> (KL 0.08, reward 1.32). Beta -> 0: greedy point mass on best response. Beta -> infinity: the reference. Strong answer shows the arithmetic. Red flags: reversed limits. Rubric: 3 points numbers, 2 points limits. Remediation: U05-C04.

## D1

Sufficient: data/optimization failure, not the loss formula. The pairs are stale or narrow: the policy already decides them, so the loss pushes an already-collapsed margin while the true objective is unmeasured. Smallest principled fix: recollect fresh pairs from the current policy (iterative DPO) or add a human-eval monitor with a stopping rule. Strong answer names the staleness mechanism. Red flags: blaming the logistic loss. Rubric: 2 points diagnosis, 2 points fix. Remediation: U05-C07/C10/C11.

## S1

Sufficient: reward loss becomes the Plackett-Luce likelihood over the ranking. DPO generalizes to the sum of pairwise logistic terms (or the Plackett-Luce form on implicit rewards). Strong answer writes one form. Red flags: keeping the pairwise loss unchanged. Rubric: 3 points. Remediation: U05-C02/C07.

## S2

Sufficient: one reward averages conflicting tastes (the U02-C09 problem at scale). Minimal change: per-locale reward heads or mixture, with locale-aware evaluation. Strong answer notes the majority-vote trap. Red flags: more data without a model change. Rubric: 2 points diagnosis, 2 points change. Remediation: U05-C12 and U02-C09.

## R1

Sufficient objections: (1) training loss is not the goal. (2) steps are not matched compute (PPO steps cost rollouts). (3) no human or held-out preference evaluation. Convincing: held-out human preference win rate at matched compute, multiple seeds. Strong answer adds preregistration. Red flags: accepting train loss. Rubric: 2 points per objection, 2 points design. Remediation: U05-C08 and P22.

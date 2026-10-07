# U07 interview keys

## B1

Sufficient: sample arm means from the posterior, pull the argmax, update. Strong answer adds probability matching. Red flags: 'epsilon-greedy with priors'. Rubric: 2 points. Remediation: U07-C01.

## B2

Sufficient: UCB = mu + k sd, EI = E[max(0, f - f_best)], PI = P(f > f_best). Strong answer writes the EI closed form. Red flags: confusing EI with PI. Rubric: 2 points. Remediation: U07-C03.

## B3

Sufficient: observe only which of two arms wins, never a number. Strong answer names BT. Red flags: 'a duel is a reward'. Rubric: 2 points. Remediation: U07-C05.

## B4

Sufficient: the arm with the highest average win probability against the others. Strong answer writes the formula. Red flags: 'the arm with most duels won'. Rubric: 2 points. Remediation: U07-C07.

## B5

Sufficient: regret counted when neither duelist is the winner. Strong answer contrasts with simple regret. Red flags: standard bandit regret. Rubric: 2 points. Remediation: U07-C07.

## B6

Sufficient: stationarity and a well-specified model (plus bounded noise). Strong answer explains the void when broken. Red flags: 'no assumptions'. Rubric: 2 points. Remediation: U07-C04.

## Ladder 1

- L1.1: Sample means from the posterior, pull the argmax, update with the reward.
- L1.2: Arm 1: P(best) = 0.850, so TS pulls it about that often.
- L1.3: The sampled world is a posterior draw, the argmax action is optimal in exactly the drawn worlds, hence the equality.
- L1.4: Beta updates per arm. O(arms) per round.
- L1.5: TS: no tuning, needs a sampler. Eps-greedy: tuned, simple. UCB: deterministic, needs bounds.
- L1.6: Posterior misspecification or collapsed uncertainty: the posterior concentrated on the wrong arm.
- L1.7: Stale history dominates, the changed arm is under-explored.
- L1.8: Matched bandit and T, 20 seeds, cumulative regret with CIs.
- Red flags: claiming TS needs no posterior
- Rubric: 2 points per rung for mechanism. Remediation: U07-C01/C02.

## Ladder 2

- L2.1: sigma(s_i - s_j).
- L2.2: 0.846.
- L2.3: Average row i of the win matrix excluding the diagonal.
- L2.4: O(k^2).
- L2.5: Copeland: pairwise averages. Borda: rank sums. Condorcet: beats-all. They agree here, they can disagree in general.
- L2.6: Break ties by fewest duels, or report the tie honestly.
- L2.7: BT cannot represent cycles, the fit distorts.
- L2.8: Held-out duel log-loss of the BT fit versus a saturated model.
- Red flags: treating the win matrix as observed rewards
- Rubric: 2 points per rung for mechanism. Remediation: U07-C05/C07.

## A1

Sufficient: EI = sd(z Phi(z) + phi(z)) with the derivation via the Gaussian tail integral, values EI1, EI2, EI3. Strong answer shows the integration step. Red flags: EI negative. Rubric: 5 points. Remediation: U07-C03.

## A2

Sufficient: sum_{k=3..5} C(5,k) 0.6^k 0.4^{5-k} = 0.683, the tail sum rises with n for p > 0.5 (monotone). Strong answer shows the n=101 value 0.994. Red flags: 'more voters always help' without p > 0.5. Rubric: 5 points. Remediation: U09-C07.

## D1

Sufficient: the surrogate, not the acquisition or the feedback. A quadratic cannot represent the bimodal truth, so EI keeps dueling near the wrong peak. Smallest principled fix: replace the quadratic with a flexible surrogate (or validate the surrogate on held-out duels) before trusting the acquisition. Strong answer names the misspecification check. Red flags: blaming EI. Rubric: 2 points diagnosis, 2 points fix. Remediation: U07-C08.

## S1

Sufficient: BT gains a tie model (e.g. Rao-Kupper) or ties split, Copeland uses P(win) + 0.5 P(tie). Strong answer writes one form. Red flags: dropping ties. Rubric: 3 points. Remediation: U07-C05/C07.

## S2

Sufficient: Beta-Bernoulli TS or epsilon-greedy survive, GP-based preferential BO does not. Lost: sample efficiency and uncertainty quality. Strong answer quantifies O(k) vs O(n^3). Red flags: 'just use a smaller GP'. Rubric: 3 points. Remediation: U07-C11.

## R1

Sufficient objections: (1) one seed is one draw. (2) one synthetic bandit is not a benchmark. (3) no baselines, no uncertainty. Convincing: multiple bandits, multiple seeds, baselines, regret with CIs, preregistered metric. Strong answer adds the protocol from U07-C12. Red flags: accepting the single run. Rubric: 2 points per objection, 2 points design. Remediation: U07-C12 and P22.

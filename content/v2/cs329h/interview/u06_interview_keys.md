# U06 interview keys

## B1

Sufficient: VoI(q) = E_y[max_a E[u(a)|y]] - max_a E[u(a)]. Strong answer adds the units and the re-optimization point. Red flags: confusing VoI with information gain. Rubric: 2 points. Remediation: U06-C01.

## B2

Sufficient: expected variance reduction 1/(1+s^2), rank q1 > q2 > q3. Strong answer derives it. Red flags: ranking by noise sd without the formula. Rubric: 2 points. Remediation: U06-C02.

## B3

Sufficient: the current belief about the preference, its spread drives elicitation. Strong answer splits noise from parameter uncertainty. Red flags: 'uncertainty means the model is bad'. Rubric: 2 points. Remediation: U06-C03.

## B4

Sufficient: the human knows the reward theta, the agent does not, both share the payoff. Strong answer names the POMDP view. Red flags: 'the agent has a different goal'. Rubric: 2 points. Remediation: U06-C04.

## B5

Sufficient: observe context x, choose arm a, earn r(x,a). Strong answer states the feedback is only for the chosen arm. Red flags: confusing with full RL. Rubric: 2 points. Remediation: U06-C07.

## B6

Sufficient: human time/attention cost, stop when marginal gain falls below cost per query. Strong answer cites the diminishing sequence. Red flags: 'ask until the posterior is exact'. Rubric: 2 points. Remediation: U06-C10.

## Ladder 1

- L1.1: E_y[max_a E[u(a)|y]] - max_a E[u(a)].
- L1.2: 0.25.
- L1.3: The posterior best is at least the prior best in expectation, because ignoring the answer recovers the prior policy.
- L1.4: Enumerate answers and actions. O(answers x actions).
- L1.5: VoI is decision-relevant, info gain measures uncertainty without a decision.
- L1.6: The answer model was overconfident: humans answered near-random, so true VoI was near 0.
- L1.7: The answer likelihood: noisy answers carry less information than modeled.
- L1.8: VoI-ranked vs random queries, matched budgets, held-out preference accuracy per query, multiple runs.
- Red flags: treating VoI as a property of the query alone, forgetting the decision
- Rubric: 2 points per rung for mechanism. Remediation: U06-C01/C02.

## Ladder 2

- L2.1: The agent proposes an arm, the human accepts or overrides.
- L2.2: 0.775.
- L2.3: Informative overrides replace some wrong suggestions with right ones, the team cannot do worse than the solo proposal policy.
- L2.4: Loop over rounds. O(rounds).
- L2.5: Assisted bounds downside via veto, autonomy is cheaper, human alone lacks the agent's search.
- L2.6: Rubber-stamping: override rate collapsed, so the team pays attention cost for no correction.
- L2.7: Fatigue: the override probability falls and the informativeness assumption fails.
- L2.8: Measure override rates in deployment, compare against the break-even rate from the toy arithmetic.
- Red flags: crediting the agent for the human's corrections
- Rubric: 2 points per rung for mechanism. Remediation: U06-C08.

## A1

Sufficient: posterior var = s^2/(1+s^2), gain = 1/(1+s^2), ranking 0.671 > 0.500 > 0.308. Strong answer shows the completing-the-square step. Red flags: gain increasing in noise. Rubric: 5 points. Remediation: U06-C02.

## A2

Sufficient: ignoring the answer is feasible, so the posterior best weakly dominates, zero-VoI case: gap in {0.5, 1.5} where the query never flips the pick. Strong answer states both directions. Red flags: 'VoI is always positive'. Rubric: 5 points. Remediation: U06-C01.

## D1

Sufficient: the observation model, not the update or the prior. The code implements Bayes correctly, the likelihood P(a | theta) mismatches the human's actual behavior (e.g. habit). Smallest principled fix: validate or learn the observation model on held-out human actions before trusting the belief. Strong answer names a calibration check. Red flags: blaming the normalization. Rubric: 2 points diagnosis, 2 points fix. Remediation: U06-C04/C06.

## S1

Sufficient: subtract cost per query ($1.67) from VoI, ask only when VoI exceeds it. Strong answer notes the stopping rule becomes economic. Red flags: ignoring the cost. Rubric: 3 points. Remediation: U06-C01/C10.

## S2

Sufficient: the posterior lives on device, only aggregates or decision-relevant summaries leave. Strong answer names on-device inference or DP. Red flags: 'encrypt and send everything'. Rubric: 3 points. Remediation: U06-C11.

## R1

Sufficient objections: (1) entropy fall is not the goal, decision quality is. (2) no baseline (random queries). (3) no held-out evaluation. Convincing: held-out preference accuracy per query at matched budgets, multiple runs. Strong answer adds preregistration. Red flags: accepting entropy. Rubric: 2 points per objection, 2 points design. Remediation: U06-C01/C02 and P22.

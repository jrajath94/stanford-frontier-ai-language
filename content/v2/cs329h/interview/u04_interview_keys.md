# U04 interview keys

## B1

Sufficient: the gradient of the log-likelihood with respect to the parameter. Strong answer adds it has mean zero at the truth. Red flags: confusing with the loss gradient sign. Rubric: 2 points. Remediation: U04-C01.

## B2

Sufficient: variance of the score. Negative expected Hessian. Strong answer notes they agree under regularity. Red flags: stating only one. Rubric: 2 points. Remediation: U04-C02.

## B3

Sufficient: some direction has zero information: a parameter (combination) the data cannot identify. Strong answer names the all-ones example. Red flags: "more data fixes it" (it does not, for structural singularity). Rubric: 2 points. Remediation: U04-C05.

## B4

Sufficient: Var(mle) ≈ 1/(n I). Strong answer adds the conditions. Red flags: dropping the n. Rubric: 2 points. Remediation: U04-C06.

## B5

Sufficient: the pair with gap nearest 0, because sigma(d)(1-sigma(d)) peaks there. Strong answer gives 0.25 max. Red flags: "the pair with the biggest gap." Rubric: 2 points. Remediation: U04-C07.

## B6

Sufficient: A minimizes average variance (trace of inverse). D maximizes information volume (determinant). Strong answer notes both punish singular designs. Red flags: confusing the two. Rubric: 2 points. Remediation: U04-C08.

## Ladder 1

- L1.1: s = y/p - (1-y)/(1-p). I = n/(p(1-p)).
- L1.2: score 8.0 at 0.5. I = 47.6 at 0.7.
- L1.3: second derivative expectation. E[score] = d/dp integral p = 0.
- L1.4: SE = 1/sqrt(n I). O(n).
- L1.5: expected for design, observed for inference after data.
- L1.6: near a boundary, or too far from the peak, or a non-regular model.
- L1.7: support depends on theta (condition i).
- L1.8: n >= 84.
- Red flags: quoting 1/sqrt(n) without the information. Rubric: 2 points per rung for mechanism. Remediation: U04-C01/C02/C06.

## Ladder 2

- L2.1: a scalar summary of the information matrix used to rank designs.
- L2.2: I1 = [[2.35,0],[0,0]]. I2 = 1.175 * identity.
- L2.3: A = 1.70, D = 1.38.
- L2.4: build I, trace of inverse, determinant. O(d^3).
- L2.5: A average variance, D volume, E weakest direction.
- L2.6: gap guesses wrong, or costs ignored, or the model misspecified.
- L2.7: the ranking optimizes the wrong matrix. Validate with a pilot.
- L2.8: label a small pilot across diverse pairs, compare achieved versus predicted information.
- Red flags: treating the criterion as the goal rather than a proxy. Rubric and remediation as ladder 1, using U04-C05/C07/C08.

## A1

Sufficient: from Var(score): Var(y/p - (1-y)/(1-p)) = 1/(p(1-p)) per trial, times n. From -E[Hessian]: n/(p(1-p)). n >= 1/(0.0025 * 4.76) = 84. Strong answer shows both derivations. Red flags: one derivation only. Rubric: 4 points. Remediation: U04-C02/C06.

## A2

Sufficient: I(q) = (1-2q)^2 p^2(1-p)^2/(p_obs(1-p_obs)). At gap 1.0: rel 0.130 at q=0.3 (multiplier 7.7), rel 0.583 at q=0.1 (multiplier 1.7). Strong answer notes cleaning beats buying. Red flags: linear in q. Rubric: 3 points derivation, 2 points numbers. Remediation: U04-C11.

## D1

Sufficient: selection-rule problem (myopia plus no diminishing returns modeled). The scores do not account for labels already asked: the same pair stays top because its information is recomputed from the same gap estimate. Fix: remove asked pairs from candidates, or track cumulative information and stop re-asking saturated pairs. Strong answer notes the refit should change the scores. Red flags: blaming the model. Rubric: 2 points diagnosis, 2 points fix. Remediation: U04-C09.

## S1

Sufficient: the loop becomes batch selection: pick 100 diverse high-information queries per round (batch diversity matters, greedy top-100 may duplicate). The refit happens weekly. Strong answer mentions the exploration cost of stale estimates. Red flags: keeping the sequential loop. Rubric: 2 points changes, 2 points batch issue. Remediation: U04-C09.

## S2

Sufficient: allocate by information per dollar: score each pool's queries by I(q)/cost, fill the budget greedily, possibly mixing pools. Strong answer writes the ratio. Red flags: splitting 50/50. Rubric: 2 points rule, 2 points ratio. Remediation: U04-C07/C10/C11.

## R1

Sufficient objections: (1) train accuracy is optimistic. (2) no held-out comparison. (3) no accounting for selection cost or seeds. Convincing: held-out log-loss, fixed labeling budget, multiple seeds, random baseline. Strong answer adds the budget be matched exactly. Red flags: accepting train accuracy. Rubric: 2 points per objection, 2 points design. Remediation: U04-C09/C12 and P22.

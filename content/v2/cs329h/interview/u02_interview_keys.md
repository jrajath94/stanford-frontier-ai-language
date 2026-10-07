# U02 interview keys

## B1

Sufficient: U_i = u_i + e_i, choice = argmax_i U_i. Strong answer names the Gumbel law for e_i. Red flags: confusing u_i with U_i. Rubric: 2 points model, 1 point rule. Remediation: U02-C01/C02.

## B2

Sufficient: argmax of utilities plus independent Gumbel noise picks i with softmax probability. Strong answer adds the CDF. Red flags: "Gumbel is just noise." Rubric: 2 points. Remediation: U02-C02.

## B3

Sufficient: IIA means P(a)/P(b) is constant across choice sets. The multinomial logit implies it. Strong answer gives the ratio formula. Red flags: claiming IIA is a fact about the world. Rubric: 2 points. Remediation: U02-C05.

## B4

Sufficient: R is 100x5, V is 50x5. Strong answer counts free parameters (750 minus rotation). Red flags: transposed shapes. Rubric: 2 points. Remediation: U02-C07.

## B5

Sufficient: scaling utilities and noise together leaves choices unchanged. Only the ratio is identified. Strong answer names the fix (noise scale 1). Red flags: interpreting raw utility gaps across models. Rubric: 2 points. Remediation: U02-C10.

## B6

Sufficient: any two of cycles/nested-logit-or-pairwise, buses/nested logit, pooling/mixture, scale/scale-fix. Strong answer pairs each break with its repair. Red flags: naming breaks without repairs. Rubric: 1 point per pair. Remediation: U02-C12.

## Ladder 1

- L1.1: U_i = u_i + e_i, e_i iid Gumbel. Choose argmax U_i.
- L1.2: [0.506, 0.307, 0.186].
- L1.3: condition on e_1, multiply CDFs, integrate. The integral collapses to the softmax. Minimum: the conditioning step.
- L1.4: -log(-log(uniform)) draws, argmax, histogram. O(n*k).
- L1.5: probit: no closed form. Needs numerical integration per probability.
- L1.6: wrong sampler (e.g., Gaussian), or far too few draws, or a code bug in the argmax axis. Minimum: two plausible causes.
- L1.7: independence of shocks across options.
- L1.8: simulate from a fitted logit, drop an option, refit, compare ratios. Or the subset likelihood-ratio test on real data.
- Red flags: hand-waving the integral. Confusing IIA with transitivity. Rubric: 2 points per rung for mechanism. Remediation: U02-C02/C05 with the simulation rerun.

## Ladder 2

- L2.1: M 4x5 = R 4x2 times V' 2x5.
- L2.2: 18 numbers minus 3 for rotation = 15 free, versus 20.
- L2.3: M = U S V'. (U S^{1/2})(V S^{1/2})' recovers factors. Right-multiplying by orthogonal Q preserves the product.
- L2.4: numpy.linalg.svd, keep top d. O(mn min(m,n)).
- L2.5: factorization for smooth taste variation. Types for clustered tastes.
- L2.6: true rank above 2, or noise, or a bug in centering. Diagnose via the residual spectrum.
- L2.7: the new tastes are outside the model span. Fold-in extrapolates wildly. Fall back to averages.
- L2.8: split respondents, fit on train for each d, score held-out predictions, pick the best d.
- Red flags: claiming factors are unique. Ignoring rotation. Rubric and remediation as ladder 1, using U02-C07/C08.

## A1

Sufficient: shares [0.636, 0.234, 0.086, 0.032, 0.012]. Proof: p_i > p_j iff u_i > u_j via the monotone exp. Strong answer shows the arithmetic. Red flags: unsorted computation. Rubric: 3 points numbers, 2 points proof. Remediation: U02-C06.

## A2

Sufficient: argmax_i (2u_i + 2e_i). Factor 2 > 0 preserves order argument by argument. Identical argmax, identical probabilities. Strong answer notes c > 0 is the condition. Red flags: confusing with the additive shift. Rubric: 3 points. Remediation: U02-C10.

## D1

Sufficient: data/model problem, not the solver. The known item factors are nearly collinear, so two ratings leave r_new almost undetermined. Lstsq returns the minimum-norm solution, which is wild. Fix: require varied item factors (or regularize the least squares). Strong answer adds the |S| >= d rule is not enough. Red flags: blaming lstsq. Rubric: 2 points diagnosis, 2 points fix. Remediation: U02-C08 lab task 6.

## S1

Sufficient: the Gumbel derivation and IIA break. The ratio formula no longer holds. Minimal change: nested logit with the known groups as nests (or a covariance structure on the shocks). Strong answer notes the flat logit is the one-nest special case. Red flags: keeping the flat formula. Rubric: 2 points breaks, 2 points repair. Remediation: U02-C05/C12.

## S2

Sufficient: the fixed-utility assumption breaks first. Adapt with time-varying factors (e.g., fit per time window, or add a time embedding). Strong answer notes the evaluation must also become temporal. Red flags: ignoring the drift in evaluation. Rubric: 2 points diagnosis, 2 points adaptation. Remediation: U02-C01/C11.

## R1

Sufficient objections: (1) training likelihood always improves with more factors. (2) no held-out comparison. (3) no accounting for the extra parameters (overfitting). Convincing experiment: held-out prediction by rank, with the rank picked on validation. Strong answer adds multiple seeds. Red flags: accepting train likelihood. Rubric: 2 points per objection, 2 points design. Remediation: U02-C07 and P22.

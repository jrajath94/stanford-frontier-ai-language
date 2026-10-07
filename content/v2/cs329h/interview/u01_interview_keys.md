# U01 interview keys

Each key gives the minimum sufficient explanation, a strong answer, red flags, a rubric, and remediation.

## B1

Sufficient: preference is the latent ordering. Choice is the observed pick. Order-on-screen is a context factor. Strong answer adds that the choice equals preference plus context plus noise. Red flags: calling the choice the preference. No example. Rubric: 2 points definitions, 1 point factor. Remediation: reread U01-C01 and draw figure u01_f01 from memory.

## B2

Sufficient: P(a beats b) = sigma(s_a - s_b), sigma the logistic function, s latent scores. Strong answer adds that only the gap matters. Red flags: missing sigma. Claiming absolute scores matter. Rubric: 2 points equation, 1 point symbols. Remediation: U01-C05 derivation.

## B3

Sufficient: adding a constant to all scores leaves every probability unchanged, so the data cannot pick the level. Strong answer names an anchor (sum zero). Red flags: claiming the scores are fully identified. Rubric: 2 points invariance, 1 point anchor. Remediation: U01-C07 proof.

## B4

Sufficient: l = sum_i y_i log sigma(d_i) + (1-y_i) log(1 - sigma(d_i)). Strong answer notes independence across records. Red flags: product instead of sum of logs. Rubric: 2 points form, 1 point assumption. Remediation: U01-C06.

## B5

Sufficient: shrink toward zero. Strong answer gives the p_obs formula and the (1-2q) approximation. Red flags: "noise cancels out." Rubric: 2 points direction, 1 point formula. Remediation: U01-C08.

## B6

Sufficient: any two of representation (whose data), performance (per-group accuracy), target (whose preference). Strong answer adds the privacy cost of group labels. Red flags: only "bias is bad." Rubric: 1 point each, 1 point depth. Remediation: U01-C12.

## Ladder 1

- L1.1: P = 1/(1+exp(-(s_a - s_b))). Minimum: the equation.
- L1.2: l(d) = 5 log sigma(d) + log(1 - sigma(d)). Minimum: correct counts in the right terms.
- L1.3: dl/dd = 5(1-sigma(d)) - sigma(d). Per-record contribution (y - p). Minimum: the derivative step, not just the result.
- L1.4: loop over records, O(n) per step, convex so it converges. Minimum: code plus cost.
- L1.5: probit link replaces sigma with the Gaussian CDF. The update becomes (y - Phi) * phi/Phi(1-Phi), no longer the clean residual. Minimum: names the link change and one consequence.
- L1.6: the gap diverges to infinity. The data is separable. Fix: a prior or regularization. Minimum: names separability and one principled fix.
- L1.7: independence fails. Fifty repeats from one annotator are not fifty independent contests. Minimum: names the broken assumption.
- L1.8: hold out pairs, compare predicted versus empirical win rates in bins (calibration), or a likelihood-ratio test against a saturated per-pair model. Minimum: a concrete test with a decision rule.
- Red flags: memorizing "(y-p)" without the derivative. Confusing the link with the loss. Rubric: 2 points per rung for a correct mechanism, 1 for a correct statement. Remediation: redo U01-C05 with the derivation written by hand.

## Ladder 2

- L2.1: a term b*c added to the gap that moves choices without moving preference. Minimum: the equation or a clean example.
- L2.2: naive conclusion is that preference changed across contexts. Minimum: states the confound.
- L2.3: model P = sigma(s + b*c). The pooled estimate mixes s and b. Minimum: shows the mixing algebraically or numerically.
- L2.4: joint grid or gradient fit, O(n*k) per step for k context features. Minimum: method plus cost.
- L2.5: model when context varies beyond control. Randomize when you design the task. Minimum: the decision rule.
- L2.6: b is weakly identified: little context variation or collinearity with the intercept. Diagnose via the design matrix rank and the standard error. Minimum: names weak identification.
- L2.7: the coefficient absorbs item difficulty. The preference estimate is biased. Minimum: names the confounding path.
- L2.8: randomize order per judgment, fit with and without the order term, likelihood-ratio test. Minimum: randomization plus test.
- Red flags: treating context as pure noise to average away. Rubric and remediation as in ladder 1, using U01-C01/C02.

## A1

Sufficient: l(d) = w_a log sigma(d) + w_b log(1 - sigma(d)). Derivative zero gives sigma(d) = w_a/(w_a+w_b). D_hat = log(w_a/w_b) = log 5 = 1.609. SE ≈ sqrt(1/w_a + 1/w_b) = 1.095. Strong answer notes the curvature derivation of the SE. Red flags: forgetting the SE or misplacing the logs. Rubric: 3 points derivation, 2 points numbers. Remediation: U01-C06 exercises.

## A2

Sufficient: p_obs = (1-q)p + q(1-p). p = sigma(1.5) = 0.818. P_obs = 0.9(0.818) + 0.1(0.182) = 0.754. Gap_hat = log(0.754/0.246) = 1.12. Strong answer notes the estimate is below the true 1.5. Red flags: applying q to the gap instead of the outcome. Rubric: 3 points formula, 2 points numbers. Remediation: U01-C08.

## D1

Sufficient: model/data problem, not an optimizer bug. Separable data gives no finite MLE. The gradient keeps pushing the gap up forever. Smallest principled fix: add a zero-mean Gaussian prior on the scores (L2 penalty), which bounds the gap. Strong answer adds that early stopping would hide the problem without fixing it. Red flags: blaming the learning rate. Proposing to delete the data. Rubric: 2 points diagnosis, 2 points fix, 1 point justification. Remediation: U01-C06/C07 plus U03-C08.

## S1

Sufficient: likelihood term for a tie is 0.5 log p + 0.5 log(1-p). With 5-1-2 the MLE solves sigma(d) = (5 + 1)/(5+1+2) = 0.75, d = log 3 = 1.099. The additive-constant invariance is unchanged because ties also depend only on gaps. Strong answer notes ties add information about closeness. Red flags: treating ties as missing data without comment. Rubric: 2 points likelihood, 2 points MLE, 1 point identifiability. Remediation: re-derive C06 with the tie term.

## S2

Sufficient: the independence assumption breaks first, then the single-preference assumption: systematic disagreement means one score vector cannot fit all annotators. Minimal change: per-annotator scores or a hierarchical model (preview of U02-C09). Strong answer notes the likelihood is still fine as a starting point but the standard errors lie. Red flags: proposing more data without a model change. Rubric: 2 points diagnosis, 2 points minimal change. Remediation: U01-C08/C11 and U02-C09.

## R1

Sufficient objections: (1) train loss is optimistic. Report held-out loss. (2) Three datasets with no error bars or significance test. (3) No anchor/regularization matched between models. The win may come from tuning, not the model. Convincing experiment: fixed protocol, held-out log-loss, paired test across datasets, ablation of the novel component. Strong answer adds preregistration of the metric. Red flags: accepting train loss as evidence. Rubric: 2 points per objection, 2 points design. Remediation: U01-C09 and P22.

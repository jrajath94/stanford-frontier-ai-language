# U03 answer keys

## C01

- E1: l'(p) = 7/p - 3/(1-p) = 0 -> 7(1-p) = 3p -> p = 0.70.
- E2: SE = sqrt(0.7*0.3/10) = 0.145.
- E3: With 10 wins in 10 trials, l(p) = 10 log p, increasing on (0,1]. The MLE is the boundary point 1.0. The interior derivative never vanishes.

## C02

- E1: L = product_i [alpha p1^{y_i}(1-p1)^{1-y_i} + (1-alpha) p2^{y_i}(1-p2)^{1-y_i}].
- E2: The product depends only on the total count of ones k: L = m^k (1-m)^{n-k} with m = alpha p1 + (1-alpha) p2. One equation, three unknowns: flat ridge. Verified: every start gave -277.0138.
- E3: L = product_i [alpha N(x_i. Mu1, sd1) + (1-alpha) N(x_i. Mu2, sd2)]. The continuous x carries shape information, so the components are identified.

## C03

- E1: r_i = alpha N(x_i. Mu1, sd1) / [alpha N(x_i. Mu1, sd1) + (1-alpha) N(x_i. Mu2, sd2)].
- E2: After the E-step, q(z) = p(z | D, theta), so KL(q || p(z|D,theta)) = 0 and the bound equals the true log-likelihood at the current theta.
- E3: Start with mu1 = mu2 and alpha = 0.5: responsibilities are all 0.5 forever, the M-step keeps the means equal. Symmetry never breaks. Fix: random asymmetric starts.

## C04

- E1: Prior p^{2-1}(1-p)^{2-1}, likelihood p^7(1-p)^3. Product: p^8(1-p)^4, i.e., Beta(9,5).
- E2: 90% interval of Beta(9,5) is [0.42, 0.83] (computed from draws. Closed form via the incomplete beta).
- E3: Prior delta at 0.5 times any likelihood is delta at 0.5. The posterior never moves. Learning requires a prior with spread.

## C05

- E1: Log posterior = 7 log p + 3 log(1-p) + (a-1) log p + (a-1) log(1-p). Derivative zero: p = (6+a)/(8+2a). Values: a=1 -> 0.700 (uniform prior, equals MLE). A=2 -> 0.667. A=10 -> 0.571. A=50 -> 0.519. The posterior mean (9/14 = 0.643 at a=2) is close but not identical. Report which functional you use.
- E2: 0.700, 0.667, 0.571, 0.519, monotone toward the prior mode 0.5.
- E3: If phi = logit(p), maximizing over phi with a flat prior on phi differs from maximizing over p with a flat prior on p, because the priors transform with a Jacobian. MAP depends on the parameterization. The posterior itself does not.

## C06

- E1: Integral of p * Beta(p. 9,5) dp = 9/14 = 0.643.
- E2: 5th and 95th percentiles of Beta(9,5): [0.423, 0.830] from 5000 draws.
- E3: After the swap, future trials come from a new bias. The predictive distribution computed from old data is wrong with full confidence. Detect via a change-point check.

## C07

- E1: Posterior mean = (0/1 + 2.0/0.25)/(1/1 + 1/0.25) = 8/5 = 1.6.
- E2: max_theta log p(D|theta) - ||theta - theta_pre||^2/(2 tau^2).
- E3: The KL penalty keeps the output distribution near the reference. The quadratic penalty keeps weights near the pretrained point. Both encode "stay near what you knew," one in weight space, one in output space.

## C08

- E1: Gradient of l(s) - (lambda/2)||s||^2 is the data gradient minus lambda s.
- E2: The penalized objective goes to -infinity as ||s|| -> infinity (the quadratic dominates), so a finite maximizer exists even on separable data.
- E3: Split pairs into train/validation. Pick the lambda with the best validation log-loss.

## C09

- E1: Two bins: predicted 0.8 with 80 hits in 100 (gap 0.0), predicted 0.6 with 50 hits in 100 (gap 0.1). ECE = (0.0 + 0.1)/2 = 0.05.
- E2: Multiplying gaps by 1.3 pushes sigma outputs toward 0/1 faster than the true frequencies move. Predictions are sharper than reality.
- E3: Fit p_cal = sigma(logit(p)/T) on validation data. Choose T to minimize validation log-loss. T > 1 softens overconfidence.

## C10

- E1: Under MCAR the kept set is a random subset. The score equations on the subset have the same expectation as on the full data. Unbiased, only noisier.
- E2: Full 0.910, kept 1.077: close pairs (|gap|<0.5) went missing, so the survivors look more decisive.
- E3: Weight each kept record by 1/P(kept | gap). The weighted score equations recover the full-data expectation. Verified: IPW 0.899.

## C11

- E1: E[p(1-p)] with p ~ Beta(9,5) = (9*5)/((14)*(15))... E[p(1-p)] = ab/((a+b)(a+b+1)) = 45/(14*15) = 0.214. Var(p) = ab/((a+b)^2 (a+b+1)) = 45/(196*15) = 0.0153.
- E2: Var(y) = E[Var(y|p)] + Var(E[y|p]) by conditioning on p.
- E3: Epistemic 0.015 is small next to aleatoric 0.214: more data barely helps. Improve the model (better features) or accept the noise.

## C12

- E1: One fold's accuracy is noisy (sd 0.06 here). The mean over five folds averages out split luck.
- E2: Split by time or by annotator block, never mixing pairs across the boundary. Report the split rule.
- E3: Stratify: keep the win-rate constant across folds so no fold is trivially easy or hard.

# U04 answer keys

## C01

- E1: d/dp [7 log p + 3 log(1-p)] = 7/p - 3/(1-p).
- E2: At 0.5: 14 - 6 = 8.0. At 0.9: 7.78 - 30 = -22.2.
- E3: E[score] = integral (L'/L) L dx = d/dp integral L dx = d/dp 1 = 0, moving d/dp inside by regularity.

## C02

- E1: l''(p) = -7/p^2 - 3/(1-p)^2. At the truth with n trials: -E[l''] = n/(p(1-p)).
- E2: 10/(0.7*0.3) = 47.62.
- E3: Observed uses the data's second derivative at the MLE. Expected averages over the model. They agree on average. Observed is better conditioned on the actual data.

## C03

- E1: l(p) ≈ l(mle) - 0.5 I (p-mle)^2. Exponentiated and normalized: Normal(mle, 1/I).
- E2: 1/sqrt(47.62) = 0.145.
- E3: 9 wins in 10: the curve is skewed left. The symmetric quadratic interval extends too far down. Use profile likelihood.

## C04

- E1: The step E[score] = d/dp integral p dx needs the integral's limits (support) fixed in theta and dominated derivatives.
- E2: MLE = max(x). P(max < theta - e) = (1 - e/theta)^n ≈ exp(-ne/theta). Mean error ≈ theta/n: 1/n rate.
- E3: A mixture weight at 0: the parameter is on the boundary and the usual asymptotics fail.

## C05

- E1: (I 1)_a = sum_b I_ab. Shifting all scores by c leaves every pair probability unchanged, so the directional derivative along 1 is zero: I 1 = 0.
- E2: det = 1.175^2 = 1.381.
- E3: Two pairs sharing one item, one comparison each, plus a third pair barely observed: the smallest eigenvalue is tiny but nonzero.

## C06

- E1: n >= 1/(0.05^2 * 4.76) = 84.0.
- E2: Simulation at n = 84 gives sd 0.0495, matching 0.0500.
- E3: Worst case p = 0.5 gives I = 4 per sample: n >= 1/(0.0025*4) = 100.

## C07

- E1: For one comparison, l(d) = y log sigma(d) + (1-y) log(1-sigma(d)). l''(d) = -sigma(d)(1-sigma(d)). Fisher info = sigma(d)(1-sigma(d)).
- E2: 0.250, 0.197, 0.105.
- E3: Score queries by info/cost and pick the max ratio.

## C08

- E1: I2 = diag(1.175, 1.175). A = 2/1.175 = 1.702. D = 1.175^2 = 1.381.
- E2: det = 0 because the second row/column is zero: no information about d2.
- E3: A_cost = trace(C^{1/2} I^{-1} C^{1/2}) with C the cost matrix, or maximize information per dollar.

## C09

- E1: 0.248 and 0.105 from sigma(0.2)(1-sigma(0.2)) and sigma(2)(1-sigma(2)).
- E2: precision_k = 1 + k * w. Variance = 1/precision_k. Greedy uses w = 0.248, random uses mean 0.177.
- E3: Query A and query B are each weak alone but jointly identify an interaction. Greedy scores them low individually and never asks either.

## C10

- E1: The knee is near n = 50-100 where MSE falls below 0.1.
- E2: If a label costs c and a unit of MSE is worth v, stop when v*(e(n)-e(n+1)) < c.
- E3: Label 50, fit the error curve, extrapolate to the budget, decide.

## C11

- E1: p_obs = (1-q) sigma(d) + q (1-sigma(d)). dp_obs/dd = (1-2q) sigma(d)(1-sigma(d)). I = (dp_obs/dd)^2/(p_obs(1-p_obs)).
- E2: rel = 0.130, multiplier 1/0.130 = 7.7.
- E3: Cutting q from 0.3 to 0.1 moves rel from 0.13 to 0.58: same precision with 4.5x fewer labels. Protocol work beats buying labels.

## C12

- E1: Two columns: planned SE, achieved SE, per parameter.
- E2: Achieved variance is 4x planned: the information came in at one quarter the rate. Suspect noise (estimate q) or wrong gap guesses.
- E3: Re-estimate q from repeats, replan the remaining budget with the corrected information values.

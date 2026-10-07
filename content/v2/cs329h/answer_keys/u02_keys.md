# U02 answer keys

## C01

- E1: exp([1.0, 0.5, -0.5]) = [2.718, 1.649, 0.607]. Sum 4.974. Shares [0.547, 0.331, 0.122].
- E2: exp(u_i + c)/sum_j exp(u_j + c) = exp(c) exp(u_i) / (exp(c) sum_j exp(u_j)). The exp(c) cancels.
- E3: exp overflows past about 709 in float64. Max-subtraction keeps the largest argument at 0.

## C02

- E1: If U ~ Uniform(0,1), then -log(-log(U)) has CDF exp(-exp(-e)): P(-log(-log U) <= e) = P(U <= exp(-exp(-e))) = exp(-exp(-e)).
- E2: With u = [0,0,0] the formula gives [1/3, 1/3, 1/3]. The simulation approaches it.
- E3: The histogram is a mean of n Bernoulli draws. Its standard error is sqrt(p(1-p)/n), the 1/sqrt(n) rate.

## C03

- E1: If a > b and b > c under one utility, u_a > u_b > u_c, so u_a > u_c and a > c. A cycle contradicts the chain, so no single utility explains it.
- E2: Three pairs, each 1-0 by a fair coin: 8 equally likely outcomes, 2 cycle (clockwise and counter). Probability 2/8 = 1/4.
- E3: Permute outcomes within each pair many times, count cycles each time. If the observed cycle count sits inside the permutation range, call it noise.

## C04

- E1: d_ab = logit(0.7) = 0.847, same for bc. p_ac = sigma(1.694) = 0.845.
- E2: Fitted probabilities come from one score vector, and sigma of summed gaps rises monotonically, so the inequality holds by construction.
- E3: Fit Bradley-Terry, bootstrap datasets from the fitted model, count triple violations in each. Excess observed violations reject the model.

## C05

- E1: P(a)/P(b) = exp(u_a)/exp(u_b) = exp(u_a - u_b). The denominator (choice set) cancels.
- E2: Ratio car:red stays 2:1. Symmetry gives red:blue 1:1. Shares: car 1/2, red 1/4, blue 1/4.
- E3: Fit on the full set and on the set without the blue bus. Compare the car:red ratio. A likelihood-ratio test decides whether the movement exceeds noise.

## C06

- E1: exp([2,1,0,-1,-2]) = [7.389, 2.718, 1.0, 0.368, 0.135]. Sum 11.61. Shares [0.636, 0.234, 0.086, 0.032, 0.012].
- E2: p_i > p_j iff exp(u_i) > exp(u_j) iff u_i > u_j. The max utility gives the max share.
- E3: log sum_j exp(u_j) equals E[max_i (u_i + e_i)] for standard Gumbel e_i. Used in U05 for the KL-constrained optimum.

## C07

- E1: 4*2 + 5*2 = 18 numbers, minus 2*2 - 1 = 3 for rotation/degrees (orthogonal 2x2 has 1 free angle plus reflection), so 15 free. Fewer than 20.
- E2: (R Q)(V Q)' = R Q Q' V' = R V' for orthogonal Q. The predictions never see the rotation.
- E3: Split respondents into train/test, fit factors on train for each d, score held-out predictions. Pick the d with the best held-out error.

## C08

- E1: (V_S' V_S) r = V_S' m. Solve for r.
- E2: V_S' V_S must be invertible: need at least d linearly independent item factors among the rated items.
- E3: The prediction is a guess outside the model. Report it with a wide interval or fall back to the population average.

## C09

- E1: 0.5 * 0.90 + 0.5 * 0.10 = 0.50.
- E2: Component labels are arbitrary: swapping group 1 and 2 gives the same likelihood. Fix by ordering groups by their mean score.
- E3: Fit 1 vs 2 groups on training respondents, compare win-rate predictions on held-out respondents. Keep 2 only if it wins out of sample.

## C10

- E1: argmax_i (c u_i + c e_i): multiplying every argument by c > 0 preserves the argmax. Same choice, same probabilities.
- E2: "Group A has larger utilities" may mean group A has smaller noise. Without a shared scale convention the comparison is empty.
- E3: Fix the noise scale at 1 (standard Gumbel). Then utility gaps are measured in units of noise, comparable across fits.

## C11

- E1: (i) repeats -> U01-C08. (ii) link -> C05 buses. (iii) graph -> U01-C07. (iv) scale -> C10. (v) rank -> C07. (vi) heterogeneity -> C09.
- E2: With n = 10, every statistical check lacks power and passes vacuously. The checklist screens gross problems, not small-sample risk.
- E3: Fix what blocks the decision first: connectivity and scale (cheap), then heterogeneity (model change), then the link (hardest).

## C12

- E1: 0.30 (cycle), 0.17 (car under IIA), 0.40 (pooling gap), 0.000 (scale, identical).
- E2: A cycle inside one group needs the pairwise model. Opposing groups need the mixture. Fit the mixture of pairwise models, or model group-specific cycles.
- E3: Any clean break from your data: e.g., a position effect that moves win rates (U01-C01) combined with IIA. The battery grows with experience.

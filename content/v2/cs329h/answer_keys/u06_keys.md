# U06 answer keys

## C01

- E1: VoI(q) = E_y[max_a E[u(a)|y]] - max_a E[u(a)]. The expected posterior best minus the prior best.
- E2: Prior best 0.5 (pick A). Posterior best: 0.5*0 + 0.5*1.5 = 0.75. VoI = 0.25.
- E3: Gap in {0.5, 1.5}, prior 0.5 each. Prior mean 1.0, pick A. The query never flips the pick, so VoI = 0.

## C02

- E1: I(q) = Var_prior - E[Var_post]. For the Gaussian toy: 1/(1+s^2).
- E2: 0.671, 0.500, 0.308. Query 1 ranks first.
- E3: The s = 0.01 query assumes near-perfect answers. If humans answer at chance its true gain is near 0 and the ranking misleads.

## C03

- E1: 2000 draws from N(0.8, 0.16) with seed 0. Histogram and quantiles summarize the belief.
- E2: Fraction of samples above 0: 0.977.
- E3: Answer flips are noise: they persist with more data. The sd 0.4 from little data shrinks as queries arrive.

## C04

- E1: The human knows theta. The agent sees the human action and holds a belief. Both maximize the same expected reward.
- E2: 0.8*0.5 / (0.8*0.5 + 0.3*0.5) = 0.727.
- E3: The update trusts the observation model. Habit-driven actions feed the wrong likelihood and the belief concentrates wrongly.

## C05

- E1: J = E[R(theta)]. pi_H conditions on theta. pi_A conditions on the belief.
- E2: 0.727 - 0.5 = 0.227.
- E3: The agent maximizes clicks, not the human tea/coffee goal. The payoff is not common, so the assistance logic misfires.

## C06

- E1: b'(theta) proportional to P(a_H | theta) b(theta), normalized.
- E2: 0.727 for tea.
- E3: Habit makes the true likelihoods 0.5/0.5. The update with 0.8/0.3 concentrates on tea from noise.

## C07

- E1: pi*(morning) = tea, pi*(evening) = coffee.
- E2: 0.85 - 0.5 = 0.35.
- E3: Stationarity fails. The learned map reflects old tastes and the blind policy (or a re-learned map) wins.

## C08

- E1: Agent proposes an arm each round. Human accepts or overrides. Reward follows the final arm.
- E2: 0.5*0.8 + 0.5*(0.72 + 0.03) = 0.775.
- E3: 0.5*0.8 + 0.5*(0.1*0.8 + 0.9*0.3) = 0.575. Minus attention cost: the team loses.

## C09

- E1: Regret(T) = sum over rounds of (best mean - pulled arm mean).
- E2: Greedy commits to the arm with the best early average. An unlucky start locks it onto a suboptimal arm forever.
- E3: The best arm changes at t = 100. Regret against the old best rewards the policy for ignoring the change.

## C10

- E1: G(n) = n/(1+n). Marginal = G(n) - G(n-1).
- E2: 0.500, 0.167, 0.083, 0.050, 0.033.
- E3: Fatigue doubles the fifth query noise sd. Its true gain drops to about 0.008: the model overstates it fourfold.

## C11

- E1: The answers, the query sequence, and the inferred posterior.
- E2: Tag sensitivity per query. Drop high-sensitivity queries unless VoI justifies them. Expire raw answers.
- E3: Preference patterns identify users. Removing names leaves the pattern, which re-identifies.

## C12

- E1: Population, query policy, stopping rule, privacy, metric.
- E2: Marginal gain below 0.02 buys little per unit of human attention. The budget cap bounds total burden.
- E3: Employees as beta users: the persona optimizes for insiders and misses the deployment population.

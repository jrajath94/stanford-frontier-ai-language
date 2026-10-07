# U07 lab: Thompson sampling and dueling bandits in code

Environment: Python 3 with NumPy and matplotlib. Seed 1 for the bandit simulations, seed 0/2 where stated. Keep outputs. Keys show verified numbers.

## Task 1: Thompson sampling

3-arm Bernoulli [0.7, 0.5, 0.4], T = 300, seed 23. Implement Beta-Bernoulli TS: sample arm means from Beta posteriors, pull the argmax, update. Report cumulative regret.

## Task 2: policy comparison

Same bandit and seed. Run greedy and epsilon-greedy (0.1) alongside TS. Report all three cumulative regrets.

## Task 3: acquisition values

f_best = 1.0. Candidates (mu, sd): (1.2, 0.1), (0.9, 0.5), (0.5, 0.8). Compute UCB (k=2), EI, PI for each. Report the table and the top candidate per acquisition.

## Task 4: win matrix

Strengths [1.2, 0.8, 0.3, -0.5]. Build the 4x4 BT win matrix. Verify the diagonal is 0.5 and W_ji = 1 - W_ij. Report P(1 beats 2) and P(1 beats 4).

## Task 5: averaged win probabilities

Posterior: strengths ~ N(true, 0.3^2), 2000 samples, seed 0. Compute the posterior-averaged win matrix. Report averaged P(1 beats 2) and P(1 beats 4) next to the plug-in values.

## Task 6: Copeland winner

From the win matrix compute Copeland scores. Report all four and the winner.

## Task 7: preferential BO loop

11-point grid on [0,1], f(x) = -40(x-0.7)^2, 20 duel rounds, seed 0. Duel the Copeland leader against the least-tried challenger each round. Report best-so-far true f at round 1 and round 20.

## Task 8: stopping rule

4-arm strengths as above, leader-focused duels (C08 acquisition), seed 2. Track P(arm 1 is the Copeland winner) via 500 posterior draws per round. Report the first round it exceeds 0.95.

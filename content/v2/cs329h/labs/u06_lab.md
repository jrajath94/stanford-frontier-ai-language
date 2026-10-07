# U06 lab: active elicitation and assistance games in code

Environment: Python 3 with NumPy and matplotlib. Seed 0. Keep outputs. Keys show verified numbers.

## Task 1: value of information

Gap g in {-0.5, 1.5}, prior 0.5 each. A query reveals g. Compute the prior best expected gap, the posterior best expected gap, and VoI. Report all three.

## Task 2: query ranking

Prior gap N(0,1). Three queries with noise sd 0.7, 1.0, 1.5. Compute the expected variance reduction 1/(1+s^2) for each and report the ranking.

## Task 3: posterior sampling

Draw 2000 samples from N(0.8, 0.16) with seed 0. Report the sample mean, sample sd, and the fraction of samples above 0.

## Task 4: assistance belief update

Prior over {tea, coffee} 0.5/0.5. Likelihoods of observing 'reach left': 0.8 given tea, 0.3 given coffee. Compute the posterior after observing 'reach left'. Report it.

## Task 5: policy comparison

Morning context, arms tea 0.8 / coffee 0.3. Compare three agent policies: ignore the human (random guess, 0.5), imitate, and belief-optimal with belief 0.727. Report expected payoff each.

## Task 6: contextual bandit optimum

Rewards [[0.8, 0.3], [0.2, 0.9]] for (morning, evening) x (tea, coffee). Contexts equally likely. Compute the optimal policy, its expected reward, and the best context-blind reward. Report the value of context.

## Task 7: assisted bandit loop

Morning context. Agent suggests uniformly. Human overrides a wrong suggestion with prob 0.9. Simulate 5000 rounds, seed 0. Report the team mean reward and the solo uniform mean.

## Task 8: regret simulation

3-arm Bernoulli [0.6, 0.55, 0.5], T = 300, seed 0. Run greedy (one pull each first) and epsilon-greedy (eps 0.1). Report cumulative regret for both.

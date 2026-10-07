# U07 answer keys

## C01

- E1: Repeat: sample arm means from the posterior, pull the argmax, update the posterior with the reward.
- E2: 0.850 of pulls go to arm 1 in the long run: probability matching.
- E3: A wrong likelihood concentrates the posterior on a bad arm. TS then exploits the mistake instead of correcting it.

## C02

- E1: Each pull earns reward now and information for later. The tradeoff is how much reward to sacrifice for information.
- E2: Greedy commits to the early leader. One unlucky start locks it onto a suboptimal arm.
- E3: One seed is one draw. The ordering can flip across seeds, report the spread.

## C03

- E1: UCB = mu + k sd. EI = sd(z Phi(z) + phi(z)). PI = Phi(z), z = (mu - f_best)/sd.
- E2: See the table: 1.40, 1.90, 2.10 / 0.201, 0.153, 0.130 / 0.977, 0.421, 0.266.
- E3: The scores trust the Gaussian posterior. Heavy tails make the optimism of UCB miscalibrated.

## C04

- E1: Stationary rewards, bounded (sub-Gaussian) noise, well-specified model.
- E2: Average regret per round falls as 1/sqrt(T): the policy learns.
- E3: Best arm flips every 100 rounds. No fixed-arm policy learns, regret is linear.

## C05

- E1: P(i beats j) = sigma(s_i - s_j).
- E2: 0.599 = 0.599, 0.846 = 0.846.
- E3: BT implies transitivity of the ordering. Cyclic truth cannot be represented.

## C06

- E1: P(i beats j) = average over posterior samples of sigma(s_i - s_j).
- E2: 0.5930 and 0.8360 (0.5930, 0.8360).
- E3: Averaging a wrong posterior gives confident wrong probabilities. Calibration on held-out duels catches it.

## C07

- E1: C_i = average over j != i of P(i beats j). Winner = argmax.
- E2: 0.718, 0.603, 0.452, 0.226 (see keys).
- E3: With cycles no arm beats all others. Copeland ranks but the 'winner' concept weakens.

## C08

- E1: Update pair posteriors, duel leader vs least-tried challenger, observe winner, repeat.
- E2: Start -19.600, end -1.600.
- E3: Duels near one peak never reveal the other. The surrogate declares the local peak global.

## C09

- E1: Gumbel noise on utilities gives P(i beats j) = sigma(u_i - u_j).
- E2: 0.731.
- E3: Tired humans answer noisier late. Early and late win rates differ and the pooled majority misleads.

## C10

- E1: Stop when P(winner) >= 1 - delta, or at N_max duels.
- E2: Round 43.
- E3: An overconfident posterior crosses 0.95 early on the wrong arm. Calibration checks catch it.

## C11

- E1: Beta TS O(k) per round. GP refit O(n^3). Acquisition scan O(candidates).
- E2: 8,000,000 vs about 4: six orders of magnitude.
- E3: Sparse approximations can understate uncertainty. TS then under-explores and the regret theory voids.

## C12

- E1: Matched T and arms, shared seeds, uncertainty bars, preregistered metric, negative-result reporting.
- E2: se = 4.0/sqrt(20) = 0.89. CI = 12.3 +- 1.75 = [10.5, 14.1].
- E3: Different seeds mean different worlds. The policy effect confounds with luck.

# U05 answer key

Answers to the lesson exercises. Do not open before attempting.

## C01

- R1. P = sigmoid(0) = 0.5, loss = -log 0.5 = 0.693 nats.
- R2. 50%: random scores carry no signal.
- R3. The loss sees only r_w - r_l, adding a constant to all
  scores changes nothing. Gaps are the signal.

## C02

- R4. Ties 75, disagreements 50: usable 375 (ties dropped or
  downweighted).
- R5. Annotators favor the first (or second) position, blinding
  the order removes the bias.
- R6. Stop and revise: pilot a clearer rubric on 200 pairs. Never
  scale 55% agreement, the reward would learn noise.

## C03

- R7. RL from a raw base is unstable and formatless, SFT installs
  the behavior format and a sane starting policy.
- R8. Humans cannot score at RL speed (thousands of samples per
  step). The reward model distills judgment into a fast scorer.
- R9. Policy, reference, reward, value.

## C04

- R10. rho = 0.9 / 0.3 = 3.0.
- R11. Probabilities underflow in fp32/fp16, log space keeps the
  ratio exact: exp(new_logp - old_logp).
- R12. The policy moved far from the rollout data (or the data is
  stale). Slow down or refresh rollouts.

## C05

- R13. 1 * log(1/0.5) + 0 = 0.693 nats.
- R14. Forward KL(pi||ref) penalizes pi for mass where ref is
  thin: mode-covering, keeps the policy near the base. Reverse
  would chase ref's modes instead.
- R15. No leash: pure reward maximization, reward hacking follows.

## C06

- R16. A = 0: the action exactly met expectations, no push either
  way.
- R17. GAE(0) = the one-step TD error delta_t.
- R18. Reward scales differ across batches, normalization keeps
  the update size stable.

## C07

- R19. Unclipped: 0.5 * -2 = -1.0. Clipped: 0.8 * -2 = -1.6.
  min(-1.0, -1.6) = -1.6.
- R20. For A < 0 the clip binds when rho < 1 - eps: the policy is
  fleeing a bad action too fast.
- R21. Pessimism: the min takes the worse of the two, bounding
  the incentive to move far in one step.

## C08

- R22. 0.3 > 1.5 * 0.1: beta *= 1.5.
- R23. Pretraining batches: their LM loss mixes into the RL
  objective to preserve base skills.
- R24. min((v - R)^2, (clip(v, v_old - eps, v_old + eps) - R)^2):
  the same pessimistic clip on the value update.

## C09

- R25. Margin = 0.5 * (0.2 + 1.2) = 0.7.
- R26. The additive constant in r = beta log(pi/pi_ref) + C:
  it cancels in r_w - r_l.
- R27. log pi(chosen) falls while the margin (gap) rises: the
  model gets relatively better and absolutely worse.

## C10

- R28. 2 * 7e9 * 2 bytes = 28 GB.
- R29. At init pi_theta = pi_ref, so log pi - log pi_ref = 0
  everywhere.
- R30. Almost never: dropping it means unanchored drift. Only if
  you replace it with another anchor (e.g. a hard constraint).

## C11

- R31. Mean output length over training steps.
- R32. Human spot evals: reward up + human flat or down =
  hacking. Reward up + human up = genuine.
- R33. Chosen likelihood falls while the margin rises.

## C12

- R34. Excluding ties: 70/90 = 77.8%. Ties as half: 75%.
- R35. Finite samples: the bootstrap turns 100 matchups into a
  confidence interval instead of a point illusion.
- R36. Verbosity bias: without it, longer answers win on length,
  not quality.

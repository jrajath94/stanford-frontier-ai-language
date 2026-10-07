# U06 answer key

Answers to the lesson exercises. Do not open before attempting.

## C01

- R1. mu = 0.5, sigma = 0.5, A = [1, 1, -1, -1]. Same as the
  C01 toy, symmetry of the group.
- R2. The gradient acts on log pi of sampled tokens. If chain
  tokens are excluded, the update cannot reinforce the
  reasoning steps, only the final answer tokens.
- R3. Unit tests for code, a proof checker for formal math, an
  exact-match key for short answers.

## C02

- R4. Strip leading and trailing whitespace, collapse inner
  whitespace, lowercase if the key allows it, compare to the
  key string.
- R5. One sample gives one binary verdict, at most one bit of
  information about the policy.
- R6. The tests check input-output behavior on edge cases:
  empty input, single element, duplicates, large input, plus
  a time limit.

## C03

- R7. mu = 0.25, sigma = sqrt(0.1875) = 0.433. A_winner =
  (1 - 0.25) / 0.433 = 1.73. Each loser: -0.58.
- R8. All advantages are 0, so the gradient is 0. No update,
  correctly: the group carries no contrast.
- R9. Without it the policy drifts far from the reference
  during long RL runs. The KL term is the anchor, the critic
  was the baseline, they serve different roles.

## C04

- R10. mu = 0.25, sigma = 0.433, A = [-0.58, -0.58, -0.58,
  1.73]. Same shape as R7, the winner is whichever index
  holds the 1.
- R11. E[(r - b) d log pi] = E[r d log pi] - b E[d log pi],
  and E[d log pi] = 0 since probabilities sum to 1. One
  line, done.
- R12. When sigma is tiny but nonzero from noise, the
  division amplifies noise into huge advantages. The tie
  guard and a sigma floor handle it.

## C05

- R13. p(1 - p) = 0.09. Low contrast, weak signal.
- R14. All chains identical, sigma = 0, advantages 0, no
  update. 64 wasted rollouts.
- R15. Most chains break format (verifier scores 0 for
  format, not reasoning), and reward variance collapses
  because nearly everything fails.

## C06

- R16. Reward-verifier gap on a sample, growth of the gap
  while reward rises, and n-gram patterns in high-reward
  low-grade samples.
- R17. Stop training. The verifier is compromised, more
  steps only deepen the hack. Fix the verifier, then
  resume.
- R18. Optimization is argmax-seeking on the proxy. More
  steps mean a more thorough search for proxy maxima,
  including the false ones.

## C07

- R19. 1 - 0.7^8 = 1 - 0.0576 = 0.942.
- R20. When a verifier exists, latency allows k samples,
  and p is too low for training alone to fix cheaply.
- R21. Correlated failures across samples, and a selector
  weaker than the assumed perfect verifier.

## C08

- R22. p(1-p) = 0.21 per sample, times 8 samples the group
  contrast scales with 0.21. Lower than the 0.25 max but
  still rich.
- R23. The policy improves, so yesterday's frontier band is
  today's easy set. Re-filter to keep the mass near p =
  0.5.
- R24. The policy is below the terrain. Train on easier
  items first, or widen the band downward temporarily.

## C09

- R25. Precision = 164 / 200 = 0.82. Recall = 200 / 200 =
  1.0. FP rate 0.18.
- R26. Harden the verifier before any RL. An 18% leak will
  be found and farmed by the policy.
- R27. The policy invents new hack classes. Static audit
  items go stale, fresh adversaries keep the audit honest.

## C10

- R28. k = (100 - 20) / 10 = 8 samples per query.
- R29. Into test-time sampling, up to the latency limit.
  Training past saturation buys nothing.
- R30. Real p(T) is measured not assumed, verifier cost per
  sample is nonzero, and latency caps k in production.

## C11

- R31. Yes, 8.5 sigma is far beyond seed noise. Real.
- R32. Otherwise a cheaper config wins by spending less,
  not by being better. Budgets must match for a fair
  read.
- R33. Nothing clean. The two removals are confounded,
  rerun each alone.

## C12

- R34. The verifier-shift drop (-0.34). The policy tracked
  the referee, not the skill.
- R35. Surface brittleness. The skill does not survive
  rephrasing, the model memorized prompt shapes.
- R36. Built after training, suites leak training items
  and phrasings. Pre-built suites stay clean.

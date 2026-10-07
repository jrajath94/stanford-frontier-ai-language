# U05 answer keys

## C01

- E1: (x, y_w, y_l): prompt, winning response, losing response. y_w != y_l, x non-empty.
- E2: Prompt difficulty confounds: a great answer to a hard prompt can be rated below a mediocre answer to an easy prompt. Pairing on x removes the confound.
- E3: Pairs control the prompt and cost one judgment per bit of ranking info. Ratings need cross-prompt calibration. Choose pairs unless you need absolute quality levels.

## C02

- E1: log(1+e^-1.5) = 0.201. Log 2 = 0.693. Log(1+e^1.5) = 1.701.
- E2: d/dm [-log sigma(m)] = -(1 - sigma(m)). Chain rule through r gives the push on winner up and loser down.
- E3: The reward fits the training distribution. The policy later visits new regions where the reward extrapolates. The proxy is trusted where it was never trained.

## C03

- E1: J(pi) = E_{y ~ pi}[r(x, y)].
- E2: Put all mass on argmax r: any spread moves mass from the max to lower rewards. The optimum is degenerate.
- E3: Without a leash the policy collapses to one response and exploits reward errors. The KL term (C04) is the standard fix.

## C04

- E1: Beta 0.5: KL 0.658, reward 1.851. Beta 2.0: KL 0.078, reward 1.320.
- E2: exp(r/beta) -> 1 for all y, so pi* -> pi_ref. The penalty dominates.
- E3: Small beta lets the policy chase a flawed reward into regions where the reward is wrong. The hacking failure is a beta-too-small failure.

## C05

- E1: L = sum pi r - beta sum pi log(pi/pi_ref) + lambda (sum pi - 1). dL/dpi(y) = r(y) - beta(log pi(y) - log pi_ref(y) + 1) - lambda = 0. Solve for pi(y).
- E2: Beta 5: [0.40, 0.33, 0.27]. Beta 0.5: [0.87, 0.12, 0.02].
- E3: Beta -> infinity: pi* -> pi_ref. Beta -> 0: pi* -> point mass on argmax r.

## C06

- E1: KL of [0.37, 0.33, 0.30] against uniform: 0.001 (beta 20 row).
- E2: The KL leash pulls toward the reference. At fixed beta the policy cannot move far from a mediocre SFT no matter how good the reward is.
- E3: SFT default (capable, safe). Iterative (improving rounds). None (pure reward chasing, dangerous). Match the reference to the trust you have.

## C07

- E1: r(y) = beta log(pi*(y)/pi_ref(y)) + C. In sigma(r_w - r_l) the C cancels: loss = -log sigma(beta (log-ratio margin)).
- E2: -log sigma(0.5*2) = -log sigma(1.0) = 0.313. At margin 0: 0.693.
- E3: The pairs came from older policies. The identity holds at the optimum, not at the current pi. The loss optimizes a stale picture: it can fall while the true objective stalls.

## C08

- E1: From the table: objective, reward model, sampling, KL control, stability.
- E2: PPO explores with fresh rollouts and can find better responses. DPO only distills the given pairs. Exploration is the gap.
- E3: One GPU-week, offline pairs, no serving loop: DPO. Online product with exploration budget: PPO. Name the constraint first.

## C09

- E1: log(0.65/0.35) = 0.619.
- E2: Randomize order and length cues across pairs. Fit with and without the cue features. Test the cue coefficients.
- E3: Model the bias when deployment differs from annotation (debias travels). Accept it when deployment matches annotation (the bias is then part of the target).

## C10

- E1: d/dt [0.1t - 0.004t^2] = 0.1 - 0.008t = 0 -> t = 12.5, peak 0.625.
- E2: Track a human-eval probe on held-out prompts. Stop when the probe falls twice while the proxy rises.
- E3: KL keeps the policy near trusted regions where the reward was trained, shrinking the room for exploits. It does not fix a broken reward.

## C11

- E1: pi puts 0.9 on y_w already. The pair's gradient is near zero but DPO still spends compute pushing an already-decided margin. Wasted.
- E2: Keep pairs where the current policy's P(y_w) is in [0.3, 0.7]: informative and fresh.
- E3: Recollect pairs from the current policy every round. Each round's DPO sees fresh contests. Middle path between one-shot DPO and PPO.

## C12

- E1: 0.86, 0.90, 0.86, 0.88, 0.81. Range matches the 15% noise prediction.
- E2: Simulate pure noise: if observed agreement matches the noise model, call it noise. Clustered disagreement (same annotators disagreeing together) needs the mixture.
- E3: Personalize when groups are stable, identifiable, and served separately. Keep the majority when one product serves everyone. Report the disagreement.

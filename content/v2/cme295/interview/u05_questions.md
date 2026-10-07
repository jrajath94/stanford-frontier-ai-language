# U05 interview bank , questions

Closed-book. Answer keys are in `u05_key.md`. Do not open the key before
attempting. Quotas per major lesson: 6 breadth, 2 deep ladders of 5
follow-ups, 2 analytical exercises, 1 implementation/debug task, 2
changed-constraint scenarios, 1 research-critique question.

## Breadth (6)

B1. Write the Bradley-Terry loss for one preference pair.
B2. What are the three RLHF stages and what does each produce?
B3. Write the PPO clipped objective.
B4. What is the advantage, and why subtract a baseline?
B5. Write the DPO loss and name its three assumptions.
B6. What does the KL term penalize, and in which direction?

## Deep ladders (2 x 5)

L1. PPO clipping.
- L1.1 Define rho = pi/pi_old.
- L1.2 Toy: objective for rho = 2.0, A = 1.0, eps = 0.2.
- L1.3 Justify the min from the trust-region idea.
- L1.4 Implement ppo_loss, state the rho = 1 check.
- L1.5 Compare PPO with unclipped policy gradient, debug a run
  where rho drifts to 10, critique "clipping guarantees
  improvement", propose the eps-sweep experiment.

L2. DPO derivation.
- L2.1 Define the implicit reward beta log(pi/pi_ref).
- L2.2 Toy: margin for beta = 0.1, gaps (0.2, -1.2).
- L2.3 Derive the inversion r = beta log(pi*/pi_ref) + C from the
  KL-regularized optimum.
- L2.4 Implement dpo_loss, state the margin-growth check.
- L2.5 Compare DPO with RLHF, debug falling chosen likelihood,
  critique the BT assumption, propose the likelihood-tracking
  experiment.

## Analytical exercises (2)

E1. Rewards [0, 0, 10], gamma = 1, values [3, 5, 8]. Compute TD
errors and GAE(1) advantages for t = 0, 1, 2. Then compute GAE(0)
for t = 2 and explain the difference.
E2. A DPO run uses beta = 0.5. Log-probs: chosen pi = -1.0, ref =
-2.0, rejected pi = -4.0, ref = -2.0. Compute the margin, the
sigmoid, and the loss. Then double beta and recompute, explain
what beta controls.

## Implementation/debug task (1)

D1. An RLHF run shows rising reward but human spot checks say the
model got worse (verbose, sycophantic). You may inspect the reward
model, the KL trace, and outputs. List the ordered checks, the most
likely culprit, and the fix. Then write the logging lines that
would have caught it early.

## Changed-constraint scenarios (2)

S1. Only 10k preference pairs exist and they are noisy (60%
agreement). DPO or RLHF? Defend the choice, state the failure mode
of the rejected option, and name the first diagnostic you run.
S2. The product needs verbose answers (users like detail) but the
reward model has length bias. How do you train without rewarding
verbosity itself? Name two guards and how each works.

## Research-critique question (1)

R1. "Higher reward means a better model." Present the strongest
version of this claim, then the reward-hacking counterexample, then
design an experiment that finds where reward and human preference
diverge as KL grows. State the falsification condition.

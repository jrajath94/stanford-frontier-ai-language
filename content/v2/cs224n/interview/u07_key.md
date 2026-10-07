# Interview key , U07

## Breadth

A1. Format and style (how to answer), not facts. Loss on response
tokens of demos.
A2. Quality, diversity, coverage across tasks. Diversity wins per
demo.
A3. Bradley-Terry: P = sigmoid(r_c - r_r), logistic loss on the gap.
A4. SFT (format), reward model (taste), RL with KL leash (policy).
A5. Bounds each policy update to [1-eps, 1+eps] ratio, takes the
pessimistic of clipped and unclipped.
A6. Inverts the KL-regularized optimum so the BT loss trains the
policy directly on pairs. No RL, no exploration.

## Deep ladders

L1. (1) -log sigmoid(r_c - r_r). (2) 0.7109, 0.3412. (3) dL/dgap =
-(1-P): uncertain pairs teach most. (4) The reward is a proxy
trained on stale/noisy pairs (C09-C11). (5) Held-out human pairs,
measure agreement (~0.7 expected).

L2. (1) PPO: E[min(rA, clip(r)A)], DPO: -log sigmoid(beta x
log-ratio gap). (2) 0.650/0.600, 0.6539. (3) Optimal policy gives r
= beta log pi/pi_ref + const, substitute into BT. (4) Without the
reference, chosen log-probs diverge unboundedly. (5) Fixed pairs:
DPO first (cheap), if exploration is needed, PPO.

## Analytical

A7. (i) Reward hacking: the proxy's argmax left true quality
(C10). (ii) Stale reward: the policy left the reward's training
distribution (C09). Cheapest check: human-rank outputs at steps
800 and 2000, if 800 wins, it is a hack or stale proxy, not progress.
A8. The pairs were too easy (large gaps, saturated sigmoid):
gradient ~0 despite falling loss. Confirm: mean |inside| >> 1, or
mean P near 1.

## Implementation/debugging

A9. (1) Check KL: exploded -> raise beta. (2) Check reward: spiked
-> the reward has a hack, stop and inspect outputs. (3) Check
eps: too large -> lower it. (4) Check advantages: value loss
diverged -> retune value coefficient. (5) Check data: a bad batch
-> inspect the rollout.

## Changed-constraint

A10. Model-judged eval with a different judge model, plus the
reward model's own held-out agreement, plus spot human audits on a
small sample. Report all three with the caveat that none is the
truth.
A11. Refusal is taught in SFT demos and reinforced in pairs/RL
(the policy must prefer refusals). Prompt-only refusal breaks
under paraphrase and pressure: the weights never learned it.

## Research critique

A12. Steelman: RLHF demonstrably shifts model behavior toward
judged preferences across many tasks, the pipeline is the best
working method we have. Counterexample: the reward is a proxy
(C10), the judges are biased (C11), the data goes stale (C09),
"aligned with human values" claims a generality the pipeline
cannot deliver. It aligns with the judges' votes on the training
distribution, nothing more.

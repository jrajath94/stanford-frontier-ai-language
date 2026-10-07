# Answer key , U07 Post-training and preferences

Attempt the exercises before reading. Ladders are oral: answer aloud,
then check.

## Remediation

R1. sigmoid(0.9) = 0.7109. Gaps become probabilities.
R2. KL = 0.0253. The leash, measured in nats.
R3. Upweight log-prob where advantage is positive, downweight where
negative. PPO and DPO are safe versions.

## Breadth

A1. Next-token loss on response tokens of demos. Teaches format and
style, not facts. Toy loss 2.797.
A2. Desiderata: quality, diversity, coverage. Diversity wins per
demo: 20 new tasks beat 1000 repeats of one.
A3. Scalar head on pairs, BT loss -log sigmoid(r_c - r_r). Toy:
P = 0.7109, loss 0.3412, 3-pair mean 0.429.
A4. Sample two responses, judge picks. Disagreement: majority vote
or soft labels (honest). Judging is the bottleneck.
A5. SFT -> reward model -> RL with KL leash: E[r] - beta KL.
Format, then taste, then exploration.
A6. E[min(rA, clip(r,1-eps,1+eps)A)]: bounds each update. Toy:
0.650 -> 0.600.
A7. BT loss on the policy's own log-ratios vs the reference: -log
sigmoid(beta x gap). Toy: 0.6539. No RL loop, no exploration.
A8. Penalty beta x KL(pi || pi_ref) per token. Without it the
policy exploits reward errors. Toy: 0.0253.
A9. The reward trained on pi_old's samples goes stale as pi moves.
Fixes: iterate (fresh pairs) or the KL leash. Toy: 0.85 -> 0.60.
A10. Optimizing the proxy picks argmax r != argmax q. Toy: proxy
picks 0, truth picks 1. Defenses: leash, iteration, human eval.
A11. Length, position, sycophancy, culture. Mitigate: balance
pairs, randomize order, instruct, diversify judges. Toy: 60 -> 51
percent after balancing.
A12. Automated (cheap, gameable), model-judged (cheap, biased),
human blinded pairwise (the standard). The reward curve is not an
eval. Toy: 60 percent, CI (53, 67).

## Oral ladders

L1 (SFT). Write the masked loss. Compute 2.797. Derive why prompts
are masked. Diagnose the confident hallucination. Design the
wrong-facts test.

L3 (reward). Write BT. Compute 0.7109/0.3412. Derive the gradient
(1-P) weighting. Diagnose the proxy (C10). Design the agreement
test.

L5 (RLHF). Name the stages. Write E[r] - beta KL. Derive the
tradeoff. Diagnose over-optimization. Design the beta sweep.

L6 (PPO). Write the clipped objective. Compute 0.650/0.600. Derive
the min (pessimism). Diagnose the divergence. Design the eps sweep.

L7 (DPO). Write the loss. Compute 0.6539. Derive the inversion
(r from the optimal policy). Diagnose the no-reference collapse.
Design the reference ablation.

L8 (KL). Write the penalty. Compute 0.0253. Derive reward-minus-
distance optimum. Diagnose the gamed KL. Design the beta sweep.

L10 (hacking). Define it. Compute the flip. Derive the off-
distribution mechanism. Diagnose the rising reward. Design the
human-rank test.

## Exercises

E1. `sft_loss` reproduces 2.797.
E2. Prompt masking lowers the loss (fewer targets), the delta is
the prompt tokens' contribution.
E3. `coverage` reproduces min 20, entropy 3.91.
E4. Accept: per-demo marginal value falls within a task and stays
high across tasks, measure by the probe delta.
E5. `bt_loss` reproduces 0.3412.
E6. 3-pair mean 0.429 reproduced.
E7. `soft_bt_loss` with p_target = 1 equals `bt_loss`.
E8. Soft loss at P_A = 0.711, target 0.667: 0.641.
E9. `rlhf_objective` with beta = 0 returns r.
E10. Accept an inverted-U sketch labeled with the C10 mechanism.
E11. `ppo_loss` reproduces 0.650/0.600.
E12. A = -0.5, r = 1.3: min(-0.650, -0.600) = -0.650, the floor
binds instead of the cap.
E13. `dpo_loss` reproduces 0.6539.
E14. No reference: loss = -log sigmoid(beta x (log pi_c - log
pi_r)), pushing log pi_c up unboundedly lowers it: divergence.
E15. `kl_penalty` reproduces 0.0253.
E16. pi = ref: KL = 0, penalty 0.
E17. `staleness_check` reproduces 0.85/0.60.
E18. Accept: the leash bounds KL(pi || pi_old), keeping pi inside
the reward's accurate region.
E19. `hack_gap` returns 1 on the toy.
E20. Accept: length (win rate vs length delta), confidence
(hedge-word rate), sycophancy (agreement rate with user errors).
E21. `bias_audit` reproduces 0.60, then 0.51.
E22. Accept: shuffle display order, measure first-shown win rate
at tied quality.
E23. `win_rate` reproduces 60 percent, CI (53, 67).
E24. Accept a paragraph: post-training fixed format and taste,
prompting changes behavior without touching weights (U08).

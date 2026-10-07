# Oral defenses , 10 deep ladders

Draw a ladder, climb all 8 rungs closed-book. Keys in
`keys-oral.md`. Rungs: define, intuitive toy, derive, implement
and complexity, compare, debug, critique assumptions, design
experiment or production transfer.

## D1 , attention

1. Define scaled dot-product attention.
2. Toy: 2x2 scores [[1, 0], [0, 1]], compute the weights.
3. Derive why the scale is sqrt(d).
4. Implement attention, state the complexity.
5. Compare with an RNN on long sequences.
6. Debug: all weights uniform, what broke?
7. Critique: what does attention assume about positions?
8. Design: an experiment proving the scale matters.

## D2 , RoPE

1. Define rotary position embeddings.
2. Toy: rotate the pair (1, 0) by 90 degrees.
3. Derive why relative distance survives the rotation.
4. Implement the rotation, state the cost.
5. Compare with learned absolute positions.
6. Debug: performance collapses past trained length, why?
7. Critique: what breaks if the base frequency is wrong?
8. Design: a length-generalization test.

## D3 , temperature sampling

1. Define temperature in softmax sampling.
2. Toy: logits [2, 1, 0], compute p at T = 0.5 and T = 2.
3. Derive the T -> 0 and T -> infinity limits.
4. Implement tempered sampling, state the cost.
5. Compare with top-k truncation.
6. Debug: outputs are repetitive at T = 1, what do you change?
7. Critique: what does temperature assume about the logits?
8. Design: a diversity-vs-quality sweep.

## D4 , LoRA

1. Define the low-rank update.
2. Toy: d = 4, k = 4, r = 1, count the parameters.
3. Derive the parameter savings vs full fine-tune.
4. Implement the forward pass, state the inference cost.
5. Compare with full fine-tuning.
6. Debug: no learning at r = 1, what next?
7. Critique: what task structure does LoRA assume?
8. Design: a rank sweep experiment.

## D5 , PPO clipping

1. Define the clipped surrogate objective.
2. Toy: rho = 2.0, A = 1.0, eps = 0.2, compute the objective.
3. Derive the pessimistic bound from the trust region.
4. Implement ppo_loss, state the rho = 1 check.
5. Compare with unclipped policy gradient.
6. Debug: rho drifts to 10, what happened?
7. Critique: does clipping guarantee improvement?
8. Design: an eps-sweep experiment.

## D6 , GRPO

1. Define the group-relative advantage.
2. Toy: rewards [1, 1, 0, 0], compute A.
3. Derive why the group mean is an unbiased baseline.
4. Implement grpo_update, state the all-tied check.
5. Compare with PPO plus a learned critic.
6. Debug: every group is tied, what broke?
7. Critique: what does within-group comparability assume?
8. Design: a group-size sweep experiment.

## D7 , RAG pipeline

1. Define recall and precision for retrieval.
2. Toy: R = 8, K = 20, H = 6, compute both.
3. Derive the two-stage bound (recall x precision).
4. Implement retrieve plus rerank, state the costs.
5. Compare with long-context stuffing.
6. Debug: the answer ignores a retrieved doc, which bin?
7. Critique: what does the chunker assume?
8. Design: a chunk-size sweep experiment.

## D8 , judge bias

1. Define position bias and verbosity bias.
2. Toy: orders give 0.70 and 0.45, debias.
3. Derive the swap cancellation.
4. Implement debiased, state the large-gap check.
5. Compare swapping with length caps.
6. Debug: the gap will not shrink, what is wrong?
7. Critique: does swapping fix self-preference?
8. Design: a bias-vs-length experiment.

## D9 , masked diffusion

1. Define the mask schedule.
2. Toy: 8 tokens, unmask the top 4 by confidence.
3. Derive the training objective (masked cross-entropy).
4. Implement diffuse_decode, state the per-step cost.
5. Compare with autoregressive generation.
6. Debug: incoherent outputs at few steps, why?
7. Critique: what does parallel unmasking assume?
8. Design: a steps-vs-quality experiment.

## D10 , DPO

1. Define the implicit reward.
2. Toy: beta = 0.1, log gaps (0.2, -1.2), compute the margin.
3. Derive the inversion from the KL-regularized optimum.
4. Implement dpo_loss, state the margin-growth check.
5. Compare with RLHF.
6. Debug: chosen likelihood falls during training, why?
7. Critique: what does the BT assumption hide?
8. Design: a likelihood-tracking experiment.

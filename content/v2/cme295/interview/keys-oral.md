# Oral defense keys

Full keys for `oral-defenses.md`. Two points per rung, 16 per ladder.

## D1 , attention

1. Strong: softmax(QK^T/sqrt(d))V, weights over values.
2. Strong: rows [0.731, 0.269] and [0.269, 0.731].
3. Strong: dot products grow as sqrt(d), unscaled softmax
   saturates, gradients die. The scale keeps variance at 1.
4. Strong: four lines, O(n^2 d) time, O(n^2) memory.
5. Strong: RNN is O(n) serial with fading memory, attention
   is parallel with direct links but quadratic cost.
6. Strong: scores are flat (bad QK init) or the scale is
   wrong. Check score variance first.
7. Strong: nothing, without positions it is permutation
   invariant. Positions are added, not assumed.
8. Strong: sweep the scale in {1, sqrt(d), d} on a toy copy
   task, measure convergence steps.

## D2 , RoPE

1. Strong: rotate each 2D pair of q/k by an angle
   proportional to position, different frequencies per pair.
2. Strong: (1, 0) rotated 90 degrees is (0, 1).
3. Strong: the dot product of rotated vectors depends on the
   angle difference, which is the relative distance.
4. Strong: pair loop with cos/sin, O(d) per position.
5. Strong: learned positions memorize absolute slots and die
   past trained length, RoPE generalizes relatively.
6. Strong: unseen rotation angles, the model never learned
   those frequencies. Interpolate or rescale.
7. Strong: too high loses long-range order, too low wastes
   dimensions. The base sets the wavelength ladder.
8. Strong: perplexity on 2x-length documents plus a
   distance-sensitive retrieval probe.

## D3 , temperature

1. Strong: divide logits by T before softmax, T reshapes
   sharpness.
2. Strong: T=0.5: [0.867, 0.117, 0.016]. T=2: [0.506,
   0.307, 0.186].
3. Strong: T->0 gives argmax (one-hot), T->infinity gives
   uniform. Limits bracket all behavior.
4. Strong: softmax(logits/T), one extra division, free.
5. Strong: top-k cuts the tail by rank, temperature reshapes
   the whole distribution. They compose.
6. Strong: lower T, or add top-p. Repetition means the
   distribution is too flat or too peaked on a loop.
7. Strong: that the logits are calibrated enough for
   reshaping to be meaningful. Garbage logits stay garbage.
8. Strong: sweep T in {0.3, 0.7, 1.0, 1.5}, measure quality
   (judge) vs diversity (distinct n-grams), find the knee.

## D4 , LoRA

1. Strong: W + BA with B in R^(dxr), A in R^(rxk), r small.
2. Strong: 1 x (4 + 4) = 8 parameters vs 16 full.
3. Strong: full needs dk, LoRA needs r(d + k), ratio about
   r(d+k)/dk. At r=8, d=k=4096: 64x fewer.
4. Strong: h = Wx + BAx, inference adds the BAx term, tiny
   cost, mergeable into W.
5. Strong: full moves every weight (expressive, expensive,
   forgetful), LoRA moves a subspace (cheap, stable).
6. Strong: rank too low for the task, or the LR is wrong.
   Raise r, then check the data.
7. Strong: that the task change lives in a low-rank
   subspace. New domains may need full rank.
8. Strong: train r in {1, 4, 16, 64} on the task, plot eval
   vs r, find the knee.

## D5 , PPO clipping

1. Strong: E[min(rho A, clip(rho, 1-eps, 1+eps) A)].
2. Strong: min(2.0 x 1.0, 1.2 x 1.0) = 1.2.
3. Strong: the trust region bounds the KL per step, the min
   takes the pessimistic side so bad steps cannot pay.
4. Strong: five lines, check that rho = 1 gives the plain
   advantage.
5. Strong: unclipped PG follows rho anywhere, one bad batch
   can destroy the policy. Clipping bounds the damage.
6. Strong: the policy outran the rollout data (stale
   batches) or the LR is too high. Refresh rollouts.
7. Strong: no, it is an approximate guardrail, not a proof.
   Pathological cases still slip through.
8. Strong: sweep eps in {0.1, 0.2, 0.3}, measure stability
   (KL per step) vs speed (reward per step).

## D6 , GRPO

1. Strong: A_i = (r_i - mu)/sigma over the prompt group.
2. Strong: [1, 1, -1, -1].
3. Strong: E[(r - b) d log pi] = E[r d log pi] since
   E[d log pi] = 0. The group mean is action-independent.
4. Strong: sample G, normalize, clipped loss, guards: eps
   on sigma and skip on all-tied.
5. Strong: PPO needs a critic the size of the policy, GRPO
   uses the group. Cheaper, prompt-relative.
6. Strong: no contrast: sampler too cold, task too easy, or
   the verifier is constant. Check temperature first.
7. Strong: that rewards in one group are comparable: same
   prompt, same verifier. Mixed prompts break it.
8. Strong: sweep G in {4, 16, 64} at fixed rollout budget,
   measure reward per step.

## D7 , RAG

1. Strong: recall = H/R, precision = H/K.
2. Strong: recall 0.75, precision 0.30.
3. Strong: end quality <= recall@N x precision@M, the stages
   multiply, the chain is only as strong as its weakest.
4. Strong: retrieve top-N, rerank to top-M. Costs: ANN
   search plus N cross-encoder calls.
5. Strong: stuffing is exact but bounded by the window and
   dilutes attention. RAG scales the corpus, not the
   window.
6. Strong: model bin (retrieval worked, the model failed).
   Fix the reader, not the index.
7. Strong: that facts are not split across chunk
   boundaries. Tables and lists break it.
8. Strong: vary chunk size on a fixed corpus, plot
   recall@20, find the peak.

## D8 , judge bias

1. Strong: position favors the first answer, verbosity
   favors the longer one.
2. Strong: (0.70 + 0.55)/2 = 0.625.
3. Strong: obs1 = p + b, obs2 = (1 - p) + b, the mean of
   (obs1, 1 - obs2) equals p. Bias cancels.
4. Strong: judge both orders, average. Large order gaps
   flag strong bias, investigate.
5. Strong: swapping cancels position, caps handle length.
   They address different biases, use both.
6. Strong: asymmetric bias or length correlating with
   order. Control length first, then re-swap.
7. Strong: no, self-preference needs blinding, swapping
   only fixes position.
8. Strong: vary the length difference, measure the win-rate
   shift, plot the bias curve.

## D9 , diffusion

1. Strong: how many tokens unmask per step, by confidence.
2. Strong: the 4 most confident tokens unmask first.
3. Strong: mask a random fraction, cross-entropy on masked
   positions only, learn to denoise.
4. Strong: loop over steps, full-sequence forward pass per
   step, O(steps x n^2 d).
5. Strong: AR is serial and coherent, diffusion is parallel
   and refining. Latency vs maturity.
6. Strong: miscalibrated confidence locks in wrong tokens
   early. More steps or better calibration.
7. Strong: that parallel predictions do not destroy
   coherence, refinement repairs it.
8. Strong: vary steps in {1, 2, 4, 8, 16}, measure quality,
   find the knee.

## D10 , DPO

1. Strong: r = beta log(pi/pi_ref), the policy as reward.
2. Strong: margin = 0.1 x 1.4 = 0.14.
3. Strong: the KL-regularized optimum is pi* proportional
   to pi_ref exp(r/beta), invert for r.
4. Strong: -log sigmoid(margin), check the margin grows
   during training.
5. Strong: RLHF trains 3 stages with a learned reward,
   DPO is one loss on pairs. Simpler, assumption-bound.
6. Strong: likelihood displacement: the loss widens the
   margin, both likelihoods can fall. Track log pi_w.
7. Strong: that preferences are transitive and noiseless
   enough for BT. Noise and cycles break it.
8. Strong: log the chosen log-prob over training, correlate
   with the margin, find displacement.

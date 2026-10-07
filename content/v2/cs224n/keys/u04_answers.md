# Answer key , U04 Language models and recurrence

Attempt the exercises before reading. Ladders are oral: answer aloud,
then check.

## Remediation

R1. Perplexity = exp(-log 0.5) = 2. A constant 0.5 model is a fair
coin per word.

## Breadth

A1. P(w_1..w_n) = product of P(w_t | prefix). Training is parallel
(the sentence is known), generation is serial (each word conditions
on the last). Toy: -4.430 log-likelihood, perplexity 4.38.

A2. Teacher forcing trains on gold prefixes, free running tests on
model prefixes. The gap is exposure bias: errors compound off the
training distribution. Toy: KL 2.9543 at one step.

A3. Perplexity = exp(mean NLL): effective choices per word. Toy
train: 1.326 (near-certain on memorized sentences). Three breaks:
memorization, tokenizer mismatch, task gap (C02).

A4. h_t = tanh(W_x x_t + W_h h_{t-1} + b): a fixed d-vector
summarizing the prefix, with tied weights over time. Toy unroll:
h_2 = (0.363, 0.762).

A5. Unroll into a tied-weight feedforward net, dL/dW_h sums
per-step contributions through Jacobian products. Truncated BPTT
cuts the chain every k steps: dependencies beyond k are unlearnable.

A6. ||dh_T/dh_0|| <= product of Jacobian norms: rho^T. rho = 0.8 ->
0.0115 at T = 20 (vanished), rho = 1.2 -> 38.34 (exploded). Fixes:
clip the explosions, gate the vanishing (LSTM), or skip the product
(attention).

A7. c_t = f_t c_{t-1} + i_t g_t, with f_t = 1 the gradient crosses
intact (constant error carousel). GRU merges cell and hidden (2
gates). Toy: c_4 = 0.2, gradient 0.1 with f = (1,1,1,0.1).

A8. If ||g|| > c, g <- g c/||g||: norm capped, direction kept. Toy:
29.025 -> 1.000, direction True. It contains explosions, it cannot
create vanished signal.

A9. Pad to (B, n_max), mask real tokens, loss = masked mean. Toy:
unmasked 1.983 vs masked 0.4, without the mask the model learns to
predict padding.

A10. Reset h_0 = 0 per sequence (default). Carry only deliberately
(stateful training), with detach so gradients do not cross batches.
The bug: order-dependent validation scores from leaked state.

A11. Greedy (argmax), temperature (scale logits: T->0 greedy, T->inf
uniform), top-k (cut the tail). Toy: greedy -> 0, T = 0.5 top 0.828,
T = 2.0 top 0.406.

A12. The fixed state is a hard ceiling: d numbers cannot hold
arbitrary lengths, and vanishing products kill long dependencies.
0.862 on six memorized sentences is not learning the language. The
successor is attention (U05).

## Oral ladders

L1 (factorization). Write the product. Compute -4.430/4.38. Derive
the sum-of-losses form. Explain the order failure (right context
unusable). Design the L-to-R vs R-to-L perplexity test.

L2 (teacher forcing). Define both modes. Compute KL 2.9543. Derive
the distribution mismatch (loss sees gold prefixes only). Diagnose
the 1.326-vs-2.95 paradox (training loss is forced, generation is
free). Design the scheduled-sampling test.

L3 (perplexity). Define it. Compute 1.326. Derive the uniform case.
Diagnose the three breaks. Design the train-vs-held-out test.

L4 (state). Write the update. Unroll the toy. Derive weight tying.
Diagnose the d=2/n=100 crush. Design the d sweep {2,8,32}.

L5 (BPTT). Define the unroll. Write the double sum. Derive the
Jacobian product. Diagnose the k = 5 failure on a 20-step task.
Design the k sweep {1,3,10}.

L6 (vanishing). Define the product. Compute both rows. Derive the
gamma^T bound. Diagnose the 2e-10 case (signal gone, not quiet).
Design the spectral-radius sweep {0.5,1.0,2.0}.

L7 (gates). Write the cell update. Compute the toy. Derive the
carousel (f=1 gives derivative 1). Diagnose the zero-forget init.
Design the LSTM/GRU/RNN race at fixed budget.

L8 (clipping). Write the rule. Compute the toy. Prove direction
kept (algebra + allclose). Diagnose the vanishing case (clipping
cannot create signal). Design the cap sweep {1,5,50}.

L9 (padding). Define pad and mask. Compute 0.4 vs 1.983. Derive the
masked mean. Diagnose the pad-prediction (most frequent "token").
Design the mask ablation.

L10 (state carry). State the reset rule. Compute the toy shift.
Derive the detach need (memory + shuffling). Diagnose
order-dependent validation. Design the reset ablation.

L11 (decoding). Define the knob. Compute the three settings. Derive
the T limits. Diagnose T = 2 incoherence. Design the human
coherence/interestingness study.

## Exercises

E1. `seq_logprob` returns -4.430 on the toy.

E2. Reversed "sat cat the": bigrams (sat,cat),(cat,the) have
different counts than (the,cat),(cat,sat), the score changes.

E3. `free_run` reproduces KL 2.9543 at t = 2.

E4. Accept the learner's per-sentence first-divergence steps
(depends on the trained model, the script's model diverges at t =
2 on sentence 1).

E5. `perplexity` returns 1.326 on the training run.

E6. P = 0.5 constant: H = -log 0.5, perplexity = 2 exactly.

E7. `rnn_step`/`rnn_forward` reproduce h_2 = (0.363, 0.762).

E8. With W_h = 0: h_2 = tanh(x_2) only, x_1 has no path to h_2.
Verified numerically.

E9. Reversed-time loop matches the script's first-epoch loss
(2.8011 start, first-epoch value depends on lr, report it).

E10. `bptt_k`: gradient norm falls as k shrinks (k = 1 uses only
the current step).

E11. `grad_norms` reproduces all eight numbers.

E12. Radius-2.0 run without clipping: expect nan within the first
few epochs (learner reports the epoch, the mechanism is the
exploding product).

E13. `lstm_step` reproduces c_4 = 0.2.

E14. f = 1, i = 0: c_t = c_{t-1} exactly for 10 steps, the cell is
a perfect conveyor.

E15. `clip_grad` reproduces 29.025 -> 1.000, direction True.

E16. Per-value clip on (3, 4) at cap 2: (2, 2), direction changed
from (0.6, 0.8) to (0.707, 0.707). Global norm clip keeps (0.6,
0.8).

E17. `pad_batch`/`masked_ce` reproduce 0.4 vs 1.983.

E18. Lengths (100, 3): padded cells 200, real cells 103, wasted
fraction 97/200 = 0.485. Nearly half the batch compute is pad cells.

E19. Reset: seq 2 first-step distribution equals the standalone
run. Carry: it differs (shifted toward seq-1 continuations).

E20. Without detach, the graph spans batches: memory grows with
the number of batches and shuffling breaks the independence
assumption. Detach keeps values, cuts gradients.

E21. `sample` reproduces greedy -> 0, T = 0.5 top 0.828, T = 2.0
top 0.406.

E22. As T -> 0, (z_i - z_max)/T -> -inf for i != argmax and 0 for
the argmax, softmax -> one-hot on the argmax. QED.

E23. `length_probe` smoke test: accuracy flat across lengths 4-6
(in-support).

E24. Accept a paragraph naming: the fixed-state bottleneck, the
vanishing product, the serial cost, and attention as the successor
(direct access, parallel training).

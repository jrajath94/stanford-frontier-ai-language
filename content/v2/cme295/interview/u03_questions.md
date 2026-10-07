# U03 interview bank , questions

Closed-book. Answer keys are in `u03_key.md`. Do not open the key before
attempting. Quotas per major lesson: 6 breadth, 2 deep ladders of 5
follow-ups, 2 analytical exercises, 1 implementation/debug task, 2
changed-constraint scenarios, 1 research-critique question.

## Breadth (6)

B1. State the temperature limits T -> 0 and T -> infinity.
B2. What does top-p keep, and how does it differ from top-k?
B3. In MoE, what does the router output and what does the auxiliary
loss prevent?
B4. Name the three costs of longer context.
B5. Why does greedy decoding loop, and what breaks the loop?
B6. What is ICL, and what does not change during ICL?

## Deep ladders (2 x 5)

L1. Temperature and truncation.
- L1.1 Define p_i = softmax(z_i / T).
- L1.2 Toy: distribution of [3, 1, 0, -1] at T = 0.5 and T = 2.
- L1.3 Justify both limits from the formula.
- L1.4 Implement sample with T then top-p, state the order and why.
- L1.5 Compare temperature vs truncation, debug high-T gibberish,
  critique "higher T is more creative", propose the T-sweep
  experiment on factual vs story tasks.

L2. MoE routing.
- L2.1 Define gates g = softmax(x W_g) and the top-k rule.
- L2.2 Toy: route with gates [0.03, 0.11, 0.09, 0.20, 0.43, 0.09,
  0.03, 0.02], k = 2.
- L2.3 Justify the aux loss from the collapse dynamics.
- L2.4 Implement moe(x), state the active-fraction check.
- L2.5 Compare MoE with dense, debug one expert taking 90% of
  tokens, critique "experts specialize", propose the aux-ablation
  experiment.

## Analytical exercises (2)

E1. Logits [4, 2, 0]. Compute the T = 1 distribution, then top-p =
0.75 (renormalized). Then compute the entropy before and after
truncation. What did truncation do to entropy?
E2. A 70B-class MoE has E = 8 experts, top-2 routing, d = 4096,
expert FFN hidden 14336. Compute active expert parameters per token
vs a dense model with the same total expert parameters. State the
active fraction.

## Implementation/debug task (1)

D1. A chat model ignores the system prompt but follows user
instructions. You may inspect the chat template, tokenization, and
training config. List the ordered checks, the most likely culprit,
and the fix. Then write the template assertion that prevents
regression.

## Changed-constraint scenarios (2)

S1. The task needs exact arithmetic on 6-digit numbers but tokens
are limited to 20 per query. Compare CoT, direct answer, and tool
use (calculator). Which do you choose and why? State the failure
mode of each.
S2. A story generator at T = 1.5 produces great variety but drifts
off-topic. Name two guards, the order to apply them, and how each
changes the distribution.

## Research-critique question (1)

R1. "Chain-of-thought shows the model's reasoning." Present the
strongest version of this claim, then the faithfulness
counterexample, then design an experiment that tests whether the
steps cause the answer. State the falsification condition.

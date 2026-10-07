# U04 interview bank , questions

Closed-book. Answer keys are in `u04_key.md`. Do not open the key before
attempting. Quotas per major lesson: 6 breadth, 2 deep ladders of 5
follow-ups, 2 analytical exercises, 1 implementation/debug task, 2
changed-constraint scenarios, 1 research-critique question.

## Breadth (6)

B1. Write the pretraining loss and the SFT loss, state the one
difference.
B2. Why does full fp32 Adam on 7B need ~104 GiB?
B3. Write the symmetric int8 quantization formulas.
B4. What is arithmetic intensity, and which side of the ridge does
decode sit on?
B5. Write the LoRA forward equation and explain alpha/r.
B6. Name the four rows of a training budget sheet.

## Deep ladders (2 x 5)

L1. LoRA mechanics.
- L1.1 Define B, A, r and the rank constraint.
- L1.2 Toy: parameter counts for d = 4096, r = 8.
- L1.3 Justify alpha/r and B = 0 init.
- L1.4 Implement lora_linear and merge, state the merge check.
- L1.5 Compare LoRA with full fine-tuning, debug a step-0
  performance jump, critique "LoRA never forgets", propose the
  rank-sweep experiment.

L2. Memory budgets.
- L2.1 Define the three Adam memory terms.
- L2.2 Toy: the 104.3 GiB vs 13.1 GiB bills.
- L2.3 Justify why freezing deletes two terms.
- L2.4 Implement memory_bill, state the nvidia-smi check.
- L2.5 Compare LoRA with 8-bit optimizers, debug a bill that
  underpredicts OOM at long context, critique the three-term bill,
  propose the activation-term measurement.

## Analytical exercises (2)

E1. A 13B model trains with LoRA (r = 16) on all query and value
matrices (2 per layer), d = 5120, L = 40, bf16. Compute trainable
parameters and their fraction of 13B. Compute the optimizer-state
bytes for the adapter in fp32.
E2. Weights 14 GB (fp16), bandwidth 2 TB/s. A decode step does
2 * 7e9 FLOPs per token. Compute intensity, attainable TFLOP/s,
and time per token. What batch size makes it compute-bound at peak
300 TFLOP/s?

## Implementation/debug task (1)

D1. An SFT run shows falling loss but the model answers in the
wrong format. You may inspect the data, the mask, and the template.
List the ordered checks, the most likely culprit, and the fix. Then
write the mask assertion that prevents regression.

## Changed-constraint scenarios (2)

S1. Only one 24 GB GPU is available for a 7B LoRA tune at T = 4096.
Build the memory rows (weights bf16, adapter states, activations
estimate 15 GB). Does it fit? Name two changes if not.
S2. The task needs a new language (not in pretraining) with 10M
tokens of data. LoRA r = 8 or full fine-tune? Defend the choice,
state the failure mode of the rejected option, and name the eval
that decides.

## Research-critique question (1)

R1. "SFT teaches the model new facts." Present the strongest version
of this claim, then the elicitation counterexample, then design an
experiment that separates format learning from knowledge gain. State
the falsification condition.

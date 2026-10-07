# Answer key , U08 Prompting and efficient adaptation

Attempt the exercises before reading. Ladders are oral: answer aloud,
then check.

## Remediation

R1. rank(BA) <= 8 for the LoRA factors. 65536 params vs 16.7M.
R2. P(majority of 5) = 0.6826 at p = 0.6.
R3. '{"ans": 4}' parses, '{"ans": 5' does not. Validation starts
with parsing.

## Breadth

A1. Zero-shot: describe the task. Few-shot: show k examples. No
weights change, the context is the program. Limit: the window, and
conditioning is not learning.
A2. Frozen weights, "learning" in the forward pass over the
context. Leading account: attention implements updates, the honest
status: open research. Curve rises then flattens.
A3. Correct, diverse, format-matched, near the boundary. Selectors:
random, retrieval, curation. Balance classes in the k.
A4. Wording moves scores, measure with the paraphrase protocol
(toy std 0.036). Mitigate: average, dev-tune, or fine-tune. Prompts
overfit dev sets too.
A5. Chain-of-thought and kin: intermediate tokens buy serial
compute. Useful, not faithful (rationalization is documented).
A6. Binomial majority: 0.6826 at p = 0.6, n = 5. Helps when errors
are independent, hurts below p = 0.5.
A7. Freeze the base, train a small delta: adapters, LoRA, prefixes.
Memory win (no optimizer states for W), one delta per task.
A8. x + W_up relu(W_down x), bottleneck b. 524288 per adapter at
d = 4096, b = 64. Inference cost: stays in the graph (latency).
A9. dW = BA, r = 8: 65536 per matrix, 256:1 vs full. Merge BA into
W at inference: zero extra cost. B zero-init keeps step 0 exact.
A10. Five axes: params, memory, storage, capacity, inference.
Fairness rule: equal data, equal tuning effort, vary one thing.
A11. Training B erases A on shared weights. Toy: 0.0 -> 10.1885,
w ends at -1.000. Mitigations: replay, freeze (PEFT), multitask.
A12. Three layers: parse, schema, semantics. Toy: 2/5 valid.
Validation checks the shape, not the truth.

## Oral ladders

L1 (few-shot). Define both. Build the toy prompt. Derive the
window limit. Diagnose the label-words collapse. Design the shuffle
test.

L2 (ICL). Describe the phenomenon. Sketch the GD account. State the
open question. Diagnose the flat curve. Design the measurement.

L4 (sensitivity). Define it. Run the protocol. Derive the
mitigations. Diagnose the dev overfit. Design the measurement.

L5 (reasoning). Describe the formats. Explain the serial compute.
Derive the faithfulness limit. Diagnose the rationalization. Design
the corruption test.

L6 (voting). Write the binomial sum. Compute 0.6826. Derive the p <
0.5 reversal. Diagnose correlated errors. Design the measurement.

L7 (PEFT). Define it. Compute both counts. Derive the optimizer-
state win. Diagnose the far-task gap. Design the comparison.

L9 (LoRA). Write dW = BA. Count 65536. Derive the merge. Diagnose
the r = 8 underfit. Design the rank sweep.

L11 (forgetting). Define it. Run the demo. Derive the ~9x loss.
Diagnose the specialist. Design the replay test.

## Exercises

E1. `few_shot_prompt` contains k examples, query last.
E2. Accept: the model conditions on the examples' surface, no
weight update means no learning in the optimization sense.
E3. Accept the learner's measured curve (rise then flatten
expected).
E4. Accept: order changes the numbers (document the spread).
E5. `retrieve` returns top-k by cosine, nearest first.
E6. Accept: near-duplicates teach copying, class balance in the k.
E7. `sensitivity` reproduces 0.994/0.919/0.999, std 0.036.
E8. Accept the re-measured spread (wider paraphrases, larger std
expected).
E9. `cot_prompt` requests steps before the answer marker.
E10. Accept: steps are generated text optimized to look right,
causality needs intervention.
E11. `majority_p` reproduces 0.6826.
E12. p = 0.4, n = 5: 0.3174, worse than one sample.
E13. `peft_params` reproduces 4.19M/0.060% and 33.6M.
E14. r = 16: 131072 per matrix, 8.39M total (Q+V, 32 layers).
E15. `adapter` with zero init returns the input exactly.
E16. b = 16: 131072 per adapter.
E17. `lora_forward` with B = 0 equals Wx exactly.
E18. Merged W' forward matches unmerged to float precision.
E19. Table filled with the unit's numbers on all five axes.
E20. Accept: the described comparison varies data or tuning
effort, name the confound.
E21. `forget_demo` reproduces 0.0/10.1885.
E22. With 10 percent replay, task A loss stays near 0 (report the
number).
E23. `validate` reproduces 2/5 with per-layer failure kinds.
E24. Accept a paragraph: words (U01) -> vectors (U02) -> nets (U03)
-> sequences (U04) -> attention (U05) -> scale (U06) -> taste (U07)
-> cheap control (U08).

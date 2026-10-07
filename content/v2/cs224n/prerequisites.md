# Prerequisites , cs224n U01-U15

Shared modules P01-P24 live at `../shared/prerequisites/` and are linked,
never rebuilt. Each unit lists its parent prerequisites from the course
prompt, each lesson carries a local remediation block for the misses that
matter most. Take the diagnostic before U01. Score each item 0 (miss),
1 (partial), or 2 (full). Key: `keys/diagnostic_key.md`.

## Unit prerequisite map

| Unit | Parent prereqs | Bridge content | Remediation pointer in lesson |
|------|----------------|----------------|-------------------------------|
| U01 | P02, P06, P10, P13 | Python and scientific software, probability, ML foundations and evaluation, language and sequence modelling | Lesson U01 block: n-gram counting and split discipline |
| U02 | P03, P05, P08 | Vectors and linear maps, calculus, information theory | Lesson U02 block: dot product, softmax, cross-entropy |
| U03 | P05, P11, P12 | Calculus, neural nets and autodiff, PyTorch and stability | Lesson U03 block: computational graph and chain rule |
| U04 | P11, P13 | Neural nets and autodiff, language and sequence modelling | Lesson U04 block: teacher forcing and perplexity |
| U05 | P12, P14 | PyTorch and stability, transformer mechanics | Lesson U05 block: Q/K/V shapes and causal mask |
| U06 | P10, P14, P15 | ML foundations, transformer mechanics, hardware | Lesson U06 block: FLOPs counting and tokens budget |
| U07 | P08, P14, P17 | Information theory, transformer mechanics, RL | Lesson U07 block: KL, Bradley-Terry, policy gradient |
| U08 | P09, P14 | Optimization, transformer mechanics | Lesson U08 block: LoRA rank and parameter counts |
| U09 | P19, P20, P21 | Retrieval and information access, tools, APIs, and agent state, security, privacy, and safety | Lesson U09 block: schema validity, recall at k, split rate |
| U10 | P07, P10, P22 | Statistical estimation and uncertainty, ML foundations and evaluation, experimental method and research literacy | Lesson U10 block: Wilson interval, contamination arithmetic, sample size |
| U11 | P14, P17, P22 | Transformer mechanics, reinforcement learning, experimental method and research literacy | Lesson U11 block: binomial majority, reversal, diminishing returns |
| U12 | P06, P13 | Probability from events to distributions, language and sequence modelling | Lesson U12 block: BPE merges, fertility, cost inequality |
| U13 | P10, P21, P22 | ML foundations and evaluation, security, privacy, and safety, experimental method and research literacy | Lesson U13 block: disparity ratio, intervention effect, ECE |
| U14 | P11, P14, P22 | Neural networks and autodiff, transformer mechanics, experimental method and research literacy | Lesson U14 block: fusion params, routing aux loss, token budget |
| U15 | P02, P12, P22, P24 | Python and scientific software, PyTorch, tensors, and numerical stability, experimental method and research literacy, production ML and stakeholder foundations | Lesson U15 block: tiny GPT-2 params, seed variance, project gates |

## Diagnostic (closed book, 20 minutes)

D1. Encode "€" (U+20AC) in UTF-8 by hand. State the byte count.
D2. With a fair coin, compute the cross-entropy between the true
distribution and the model prediction (0.5, 0.5). State units.
D3. Write the dot product of u=(1,2) and v=(3,-1). State the angle
sign: positive, zero, or negative.
D4. Differentiate f(x)=log(1+exp(x)) with respect to x. Name the result.
D5. Softmax of (1000, 1001, 1002). Compute without a calculator, by
shifting first. State the largest entry.
D6. A batch of 4 sequences, each length 7, vocab 50. State the shape of
the logit tensor and of the target tensor for next-token loss.
D7. Define perplexity in one sentence and give the perplexity of a
model that assigns probability 0.25 to every correct next token.
D8. Sketch the Q/K/V shapes for one attention head: sequence length 6,
head dim 8, batch 2. Include the score matrix shape.
D9. Estimate the FLOPs of one dense layer, input dim 1024, output dim
4096, batch 32. Count multiply-add as 2 FLOPs.
D10. A reward model prefers response A over B. Write the Bradley-Terry
probability that A wins, in terms of scalar scores rA and rB.

Scoring: 16-20 , proceed. 10-15 , read the flagged remediation blocks.
Below 10 , work P01-P13 first. The diagnostic is a placement tool, not a
mastery claim.

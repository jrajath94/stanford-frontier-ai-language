# U01 interview bank , questions

Closed-book. Answer keys are in `u01_key.md`. Do not open the key before
attempting. Quotas per major lesson: 6 breadth, 2 deep ladders of 5
follow-ups, 2 analytical exercises, 1 implementation/debug task, 2
changed-constraint scenarios, 1 research-critique question.

## Breadth (6)

B1. What is the difference between a token id and an embedding?
B2. Why does BPE merge the most frequent pair first?
B3. Write the Elman RNN recurrence and name each matrix.
B4. What does the forget gate multiply, and why does that matter for
gradients?
B5. Why must the causal mask apply before the softmax?
B6. State the shapes of the attention scores and the logits for
(B, T, d) inputs with h heads and vocabulary V.

## Deep ladders (2 x 5)

L1. BPE training mechanics.
- L1.1 Define the pair count on a weighted corpus.
- L1.2 Toy: compute merge 1 on {low x2, lower, lowest, new, newer}.
- L1.3 Justify why merges apply in rank order at encode time.
- L1.4 Implement one merge-rewrite pass, state its complexity.
- L1.5 Compare BPE with word-level tokenization, debug an encoder
  that merges across spaces, critique the fixed-vocabulary
  assumption, propose the code-versus-prose merge experiment.

L2. Masks and the softmax.
- L2.1 Define the additive causal mask.
- L2.2 Toy: masked softmax of [1, -inf, 2].
- L2.3 Justify -inf-before-softmax from the softmax formula.
- L2.4 Implement masked softmax, state the row-sum and triangle
  checks.
- L2.5 Compare additive -inf with post-softmax zeroing, debug a run
  whose loss looks great but generates garbage, critique "the mask
  ran" as an assumption, propose the mask-ablation experiment.

## Analytical exercises (2)

E1. Corpus {the: 100, then: 40, their: 10} with end-of-word markers
ignored for this exercise. Compute the first BPE merge and its count.
Then compute fertility (characters per word) before and after that
single merge.
E2. A model has d = 512, h = 8, V = 32000. Compute d_k, the attention
score shape for B = 4, T = 1024, and the embedding table size in fp16
bytes. Which term dominates memory at T = 8192: the scores or the
weights? Show the numbers.

## Implementation/debug task (1)

D1. A tiny attention net trains: loss decreases, but generated text is
incoherent and the eval loss is far above train loss. You may inspect
shapes, the mask, and the data pipeline. List the ordered checks you
run, what each rules out, and the most likely culprit. Then write the
three-line mask test that would have caught it on day one.

## Changed-constraint scenarios (2)

S1. The vocabulary must support a new language with no shared
subwords, but the embedding table cannot grow. Propose a tokenization
change, state what breaks (ids shift? fertility?), and how you detect
the breakage.
S2. Inference must run on a device with no fp32 and 256 MB RAM. The
model has V = 50000, d = 1024. Compute the embedding table alone in
fp16. Does it fit? Propose two changes and their costs.

## Research-critique question (1)

R1. "Attention weights explain model decisions." Present the strongest
version of this claim, then the counterexample from the lesson, then
design an experiment that tests whether weights are faithful
explanations on one task. State the falsification condition.

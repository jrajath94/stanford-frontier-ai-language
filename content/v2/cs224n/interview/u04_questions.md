# Interview bank , U04 Language models and recurrence

## Breadth (6)

Q1. Why does a language model factor left to right, and what does the
factorization cost?
Q2. What is teacher forcing, and where does it mismatch inference?
Q3. What does perplexity measure, and what are two ways it misleads?
Q4. What is the RNN hidden state, and why do the weights tie across time?
Q5. What is BPTT, and what does truncation sacrifice?
Q6. Why do RNN gradients vanish or explode, and what are the two
standard fixes?

## Deep ladders (2 x 5)

L1 (language modeling). (1) Write the chain-rule factorization and the
training loss. (2) Compute the toy log-likelihood -4.430 and perplexity
4.38. (3) Derive why P(w_n | w_1) cannot use w_{n+1}. (4) An L-to-R and
R-to-L model tie on perplexity but differ on completion quality: explain.
(5) Design the experiment that settles which direction fits the task.

L2 (vanishing gradients). (1) Write the Jacobian product in the BPTT
gradient. (2) Compute rho^T for rho = 0.8 and 1.2 at T = 20. (3) Derive
the gamma^T bound from submultiplicativity. (4) Training loss is flat
but short-range accuracy is fine: diagnose. (5) Design the
spectral-radius sweep that confirms the diagnosis.

## Analytical (2)

Q7. A char-RNN trains to perplexity 1.1 on the training set but
generates gibberish from a seed. Which metric lies, and what experiment
distinguishes memorization from modeling?
Q8. You double the hidden size and the long-range task gets worse, not
better. Name two mechanisms from this unit that explain it, and the
cheapest test for each.

## Implementation/debugging (1)

Q9. Your RNN loss is nan by epoch 3. List the checks in order, with the
one-line fix each check points to.

## Changed-constraint (2)

Q10. Inference must run on a device with 64KB of RAM and no batching.
Which parts of the RNN pipeline survive unchanged, and what breaks?
Q11. The task changes from next-word prediction to whole-sentence
scoring (accept/reject). What changes in training, decoding, and
evaluation?

## Research critique (1)

Q12. "LSTMs solve the vanishing gradient problem." Steelman the claim,
then give the strongest counterexample from this unit's mechanisms.

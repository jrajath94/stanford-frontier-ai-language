# U06 interview bank , questions

Closed-book. Answer keys are in `u06_key.md`. Do not open the key before
attempting. Quotas per major lesson: 6 breadth, 2 deep ladders of 5
follow-ups, 2 analytical exercises, 1 implementation/debug task, 2
changed-constraint scenarios, 1 research-critique question.

## Breadth (6)

B1. What does RLVR stand for, and what makes the reward
"verifiable"?
B2. Write the GRPO advantage for one group. What did the group
replace?
B3. State the pass@k law and its independence assumption.
B4. What is reward hacking, in one sentence?
B5. Why does group normalization divide by sigma, not just
subtract mu?
B6. Name the three generalization probes and what each isolates.

## Deep ladders (2 x 5)

L1. GRPO mechanics.
- L1.1 Define the group, the rewards, and the advantage.
- L1.2 Toy: rewards [1, 1, 0, 0], compute A.
- L1.3 Justify dropping the critic: what did it provide, and
  what replaces it?
- L1.4 Implement group_adv with the two guards, state the
  all-tied check.
- L1.5 Compare GRPO with PPO, debug a run where every group
  is tied, critique "GRPO needs no baseline", propose the
  G-sweep experiment.

L2. Test-time scaling.
- L2.1 Define pass@k and best-of-k.
- L2.2 Toy: p = 0.3, compute pass@8.
- L2.3 Derive 1 - (1 - p)^k from independence.
- L2.4 Implement best_of_k, state the curve-tracking check.
- L2.5 Compare best-of-k with longer chains, debug measured
  best-of-k far below the curve, critique "more samples
  always help", propose the verifier-precision experiment.

## Analytical exercises (2)

E1. Group rewards [2, 2, 2, 0, 0, 0, 0, 0]. Compute mu, sigma,
and the advantage vector. Then compute what the winner's
advantage would be if the group were [1, 0, 0, 0, 0, 0, 0, 0]
instead, and explain why the first group teaches less per
winner.
E2. Budget B = 120, cost per test sample c = 10 per 100 queries,
p(T) = 1 - exp(-T/60). Compute k and the score for T = 60 and
T = 90. Then state which allocation wins and why the answer
changes if latency caps k at 4.

## Implementation/debug task (1)

D1. An RLVR run shows verifier pass climbing from 0.60 to 0.95
over 400 steps, but a weekly strict re-grade falls from 0.55
to 0.38. You may inspect the verifier, the audit set, and
sampled chains. List the ordered checks, the most likely
culprit, and the fix. Then write the logging lines that would
have caught it at step 50.

## Changed-constraint scenarios (2)

S1. The verifier has an 18% false-positive rate on adversarial
inputs and cannot be hardened before the deadline. Do you
train with it, switch to a learned reward model, or delay?
Defend the choice, state the failure mode of each rejected
option, and name the first diagnostic you run.
S2. Inference latency allows k = 2 samples per query, no more.
The task has a perfect verifier. How do you spend a fixed
budget between training and test-time, and what changes if
the verifier were only 70% precise?

## Research-critique question (1)

R1. "RLVR generalizes reasoning because it trains on verifiable
outcomes." Present the strongest version of this claim, then
the verifier-memorization counterexample, then design an
experiment with a held-out verifier that separates true
reasoning gains from referee fitting. State the falsification
condition.

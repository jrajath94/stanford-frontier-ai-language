# U09 interview bank , questions

Closed-book. Answer keys are in `u09_key.md`. Do not open the key before
attempting. Quotas per major lesson: 6 breadth, 2 deep ladders of 5
follow-ups, 2 analytical exercises, 1 implementation/debug task, 2
changed-constraint scenarios, 1 research-critique question.

## Breadth (6)

B1. How does masked diffusion inference differ from
autoregressive generation?
B2. What is an objective map, and what is an orphan?
B3. Name the five benchmark caveats.
B4. What is a keystone concept?
B5. List the five production gates in order.
B6. What are the 8 rungs of the oral defense ladder?

## Deep ladders (2 x 5)

L1. Diffusion decoding.
- L1.1 Define the mask schedule.
- L1.2 Toy: 8 tokens, confidences given, unmask 4.
- L1.3 Justify re-masking low-confidence tokens.
- L1.4 Implement diffuse_decode, state the k = 1 check.
- L1.5 Compare diffusion with speculative decoding, debug
  quality collapse at few steps, critique "parallel is
  free", propose the steps-vs-quality experiment.

L2. End-to-end toy.
- L2.1 Define the six forward-pass lines.
- L2.2 Toy: compute the attention rows for the 2x2 case.
- L2.3 Justify dividing by sqrt(d).
- L2.4 Implement tiny_forward, state the row-sum check.
- L2.5 Compare the toy with a real model, debug logits
  with wrong shape, critique "the toy teaches scale",
  propose the add-the-mask experiment.

## Analytical exercises (2)

E1. 30 objectives, 24 mapped. Compute coverage. The 6 orphans
include 2 from L8 (diffusion). The final is rumored to weight
recent lectures. How do you allocate 10 study hours?
E2. Dependency edges: A -> {B, C, D}, B -> {D}, C -> {}. Count
transitive downstream dependents for A, B, C. Which is the
keystone?

## Implementation/debug task (1)

D1. A team reports "92 on the benchmark" for a model upgrade and
wants a launch announcement. You may inspect the eval config,
the contamination log, and the raw scores. List the ordered
checks, the most likely culprit if the number is hollow, and
what the announcement may honestly claim. Then write the
three lines every reported score must carry.

## Changed-constraint scenarios (2)

S1. The final exam is in 3 days and the L8/L9 artifacts never
published. Slides missing, videos missing. How do you study
diffusion and trending topics? Name the sources you trust, the
ones you do not, and the first thing you derive from scratch.
S2. Production shows p99 latency 3x the budget with k = 8
test-time samples. The accuracy target is met. Do you cut k,
distill, or renegotiate the budget? Defend the choice and name
the measurement that decides.

## Research-critique question (1)

R1. "Diffusion LMs will replace autoregressive models within a
year." Present the strongest version of this claim, then the
maturity counterexample, then design an experiment that would
change your mind. State the falsification condition.

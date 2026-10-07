# Answer key , U11 Reasoning and test-time compute

Attempt the exercises before reading. Ladders are oral: answer aloud,
then check.

## Remediation

R1. Majority at p = 0.6: 0.6000, 0.6826, 0.7535, 0.8256 at n =
1, 5, 11, 21.
R2. At p = 0.4: 0.3174 (n = 5), 0.2465 (n = 11). Voting hurts
below 0.5.
R3. Gain 1 - exp(-n/20): 10x10 gives 3.935, 1x100 gives 0.993.

## Breadth

A1. Supervise steps for auditability, answers for accuracy. Steps
can still rationalize: supervised, not faithful.
A2. A reward a program checks (tests, exact match). No reward
model, no hacking surface beyond the verifier. Verify the
verifier.
A3. The S15-S16 reading cards: title known, content not
inspected. No claim outruns its card.
A4. Outcome checks the answer (cheap), process checks every step
(5x). Toy: 4 kept vs 2 kept.
A5. Majority vote over samples: 0.6000 to 0.8256 at p = 0.6.
Needs independent errors.
A6. Vote over reasoning paths. Same binomial. Needs p > 0.5 and
diverse paths (temperature 0 is theater).
A7. Spread the budget: 3.935 vs 0.993 on the toy. Concave gains
favor breadth. Threshold tasks are the exception.
A8. Draft with the small model, verify with the big one. E =
2.941 at a = 0.7, g = 5. Output equals the big model's alone.
A9. Trace length looks like thought. Control by blinding: toy
bias 0.09.
A10. Six classes, 60 traces: wrong tool 14, bad args 11, no stop
9, retrieval miss 12, rationalized 8, other 6. Fix the top
classes first.
A11. The knee: spend to it, not past it. Toy knee at 16
samples: 4x cost for +0.04 after.
A12. Traces are stories, not proofs. The corruption test: 14 of
20 answers unchanged means decoration.

## Oral ladders

L1 (supervision). Define the targets. Run the toy. Derive the
spotlight. Diagnose rationalization. Design the corruption
test.

L2 (verifiable). Define verifiability. Run the toy. Derive the
honesty. Diagnose the buggy test. Design the verifier
strength test.

L4 (verification). Define both. Run the 4-vs-2 toy. Derive the
subset. Diagnose the non sequitur. Design the per-dollar
test.

L5 (sampling). Define the vote. Compute the curve. Derive the
binomial tail. Diagnose correlated errors. Design the
agreement test.

L7 (allocation). Define the problem. Compute 3.935 vs 0.993.
Derive the concave rule. Diagnose the threshold task. Design
the g measurement.

L8 (spec decode). Define the rule. Compute 2.941. Derive the
expectation. Diagnose the expensive draft. Design the g
sweep.

L10 (failures). Define the classes. Count the toy. Derive the
fix order. Diagnose the double label. Design the re-measure.

L12 (traces). Define faithfulness. Run the toy. Derive the
intervention logic. Diagnose redundant reasoning. Design the
supervision comparison.

## Exercises

E1. `step_loss` beats answer-only on step checks.
E2. Corrupted step moves the answer on the step-trained model.
E3. `rl_step` matches the 63/37 toy split.
E4. The buggy test teaches the exploit: score 1 on wrong code.
E5. Cards for DeepSeek-R1 and DAPO: title known, not read.
E6. One overclaim found and fixed in your notes.
E7. `select` keeps 4 (outcome) vs 2 (process).
E8. Process costs about 5x on the toy.
E9. `majority_vote` matches the curve.
E10. Correlated errors break the committee.
E11. `self_consistent` matches the reversal numbers.
E12. Temperature 0: 5 identical paths, theater vote.
E13. `allocate` picks 10x10 on the toy.
E14. Threshold gain breaks the spread rule.
E15. `spec_step` matches E = 2.941.
E16. Draft at half cost with a = 0.3: no speedup.
E17. `reason_bias` matches 0.09.
E18. Style blinding moves less than length blinding.
E19. `classify_trace` matches the six counts.
E20. Sprint: tools and retrieval first (26 of 60).
E21. `knee` finds 16 on the toy.
E22. Shifted curve moves the knee.
E23. `corrupt_test` matches 14/20 unchanged.
E24. Honest-use paragraph: traces as evidence, not proof.

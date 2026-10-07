# Capstone A , replication plus a falsifiable extension

## Question

Does the binomial majority-vote math (U08 C06, U11 C05) replicate in
simulation, and does a cheap diversity intervention rescue the vote
when errors are correlated?

## Replication

Target: at p = 0.6, n = 5, the majority-vote accuracy is 0.6826
(binomial). Method: 50000 simulated trials, seed 42, numpy only.
Result: 0.6821. |diff| = 0.0005 < 0.01. Verdict: PASS. The script is
`capstones/capstone_a_run.py`, the figure is
`capstones/fig_a1_replication.png`.

## Falsifiable extension

Hypothesis H: shuffling the answer order, a cheap diversity
intervention, recovers the independent-draw gain under correlated
errors. Correlated error model: with probability c, all 5 draws in a
trial copy a single draw. Otherwise independent. Falsifier: no gain from
shuffling at any c.

Result: at c = 0.00, 0.25, 0.50, 0.75 the vote accuracies are 0.6874,
0.6628, 0.6423, 0.6203, and shuffling changes them by -0.0039,
-0.0017, -0.0030, -0.0007. No rescue at any c. Verdict: NEGATIVE
RESULT, H rejected. The figure is `capstones/fig_a2_extension.png`.

## Honest reading

Order shuffling is not a diversity intervention: the copies are
identical in any order. Real diversity needs independent error
sources (different prompts, different models, different
temperatures), not permutations of one source. The negative result
is the finding: it rules out a tempting cheap fix.

## Limitations

The correlation model is stylized (perfect copies). Real
correlation is partial. The next experiment: graded correlation and
genuine diversity sources (prompt variants), with the same
falsifier.

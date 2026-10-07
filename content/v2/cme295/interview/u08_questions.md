# U08 interview bank , questions

Closed-book. Answer keys are in `u08_key.md`. Do not open the key before
attempting. Quotas per major lesson: 6 breadth, 2 deep ladders of 5
follow-ups, 2 analytical exercises, 1 implementation/debug task, 2
changed-constraint scenarios, 1 research-critique question.

## Breadth (6)

B1. What is a judge rubric, and what does an anchor buy?
B2. Pointwise or pairwise: which costs O(n^2), and why?
B3. Write the swap-corrected win rate formula.
B4. What is ECE measuring?
B5. Name three contamination signals.
B6. What must every reported benchmark score carry?

## Deep ladders (2 x 5)

L1. Judge bias.
- L1.1 Define position bias and verbosity bias.
- L1.2 Toy: orders give 0.70 and 0.45, debias.
- L1.3 Derive the swap cancellation (p + b algebra).
- L1.4 Implement debiased, state the large-gap check.
- L1.5 Compare swapping with length caps, debug a judge
  whose gap will not shrink, critique "swapping fixes
  all bias", propose the length-curve experiment.

L2. Uncertainty.
- L2.1 Define the Wilson interval.
- L2.2 Toy: n = 100, w = 62, compute the band.
- L2.3 Justify 1/sqrt(n) from the binomial variance.
- L2.4 Implement wilson, state the shrink check.
- L2.5 Compare Wilson with bootstrap, debug bands that do
  not shrink with n, critique "overlapping bands mean
  equal models", propose the clustered-items experiment.

## Analytical exercises (2)

E1. Judge confidences [0.6, 0.7, 0.8, 0.9], accuracies [0.58,
0.66, 0.71, 0.72], equal weights. Compute ECE and name the
worst bin. Then a recalibration map subtracts 0.08 from every
confidence above 0.75: recompute ECE and state whether the
map helped.
E2. 8 models, 300 items, swapped orders, $0.003 per call.
Compute the bill. The team samples 75 items instead: new bill
and band-widening factor versus the full run.

## Implementation/debug task (1)

D1. A model upgrade shows win rate 0.58 vs 0.52 on 200 items,
and the team wants to ship. You may inspect the judge config,
the rubric, and the raw pairs. List the ordered checks, the
most likely culprit if the win is fake, and the decision rule.
Then write the reporting lines every eval must print.

## Changed-constraint scenarios (2)

S1. The judge backend updated silently and run-to-run
agreement fell from 0.97 to 0.88. A release decision is due
tomorrow. Do you trust the current numbers, re-run, or delay?
Defend the choice, state the failure mode of each rejected
option, and name the first check you run.
S2. The benchmark is saturated (6 models above 0.90) and
contamination status is unknown. Leadership wants a ranking
tomorrow. What do you report, and what do you refuse to
claim?

## Research-critique question (1)

R1. "LLM judges can replace human eval." Present the
strongest version of this claim, then the systematic-bias
counterexample, then design an experiment that measures the
human-judge gap on your task. State the falsification
condition.

# UU10 lab keys: execution-verified outputs

Seeds 0-19 (capstone A). Consent terms hypothetical. No invented numbers.

## Task 1

The five-element plan: H (leader-vs-random challenger lowers mean strong regret), metric (mean cumulative strong regret, T=200, 20 seeds), success (lower mean, no CI overlap), failure (overlap or reversal).

## Task 2

Problem: learn the best arm from duels. Theorem: TS achieves low regret empirically. Proof sketch: posterior sampling concentrates duels on plausible leaders. Setup: Bradley-Terry simulator, 4 arms.

## Task 3

I: leader-vs-random challenger dueling. O: mean strong regret. D: lower. K: 4-arm BT bandit, T=200, 20 seeds. Consent: purpose (train a preference model), voluntariness, withdrawal (labels removed within 30 days), research-only use, 2-year retention. ALL TERMS HYPOTHETICAL.

## Task 4

Uniform mean 17.24. Dueling TS mean 3.11. Leader TS mean 8.36.

## Task 5

Uniform: SD 1.55, CI [16.56, 17.92]. TS: SD 2.01, CI [2.23, 3.99]. Leader: SD 3.43, CI [6.86, 9.87]. H1 supported. H2 not supported.

## Task 6

The two runs produce identical JSON. REPRODUCIBLE.

## Task 7

Contribution: baseline dueling TS, intervention leader focus, evidence 8.36 vs 3.11 with non-overlapping CIs over 20 seeds, scope simulation only. Reflection: H2 failure reported, limits (no human data), impact (elicitation concentrates influence).

## Task 8

Negative-result report: H2, numbers 8.36 vs 3.11 with CIs, diagnosis (challenges cost strong regret), rules out 'any leader focus helps'. Gap: an open reward-model eval rig with subgroup slicing. Done: per-group accuracy reports on a public preference dataset.

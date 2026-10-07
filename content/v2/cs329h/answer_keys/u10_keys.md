# UU10 answer keys

## C01

- E1: Hypotheses, metric, sample, success rule, failure rule.
- E2: Lower mean with non-overlapping 95 percent CIs.
- E3: 'Performs well' has no metric. Rewrite with the metric and the success rule.

## C02

- E1: Problem, theorem, proof sketch, experimental setup.
- E2: See the worked example: pairwise comparisons, TS, Bradley-Terry simulator.
- E3: Cumulative vs simple regret are different claims. Align the metric first.

## C03

- E1: Intervention, outcome, direction, condition.
- E2: I: leader-focused dueling TS. O: mean strong regret. D: lower. K: 4-arm BT bandit, T=200, 20 seeds.
- E3: True by definition. The outcome must be able to go either way.

## C04

- E1: Purpose, voluntariness, withdrawal, data-use limits, retention.
- E2: Labels removed within 30 days of request (hypothetical terms).
- E3: The promise has no mechanism. Add the removal procedure or drop the promise.

## C05

- E1: H1 supported: TS 3.11 vs uniform 17.24, no overlap. H2 not supported: leader 8.36 vs TS 3.11.
- E2: mean +/- 1.96 x sd / sqrt(20).
- E3: A 95 percent CI is a procedure property, not a bound on the truth.

## C06

- E1: mean +/- 1.96 x SD / sqrt(20).
- E2: 3.43 / sqrt(20) = 0.767. CI [6.86, 9.87].
- E3: Dropping seeds invalidates the CI. The wide SD is the finding.

## C07

- E1: Fixed seeds, named dependencies, one command, file outputs.
- E2: Two runs, diff of the JSON: empty.
- E3: Record the dependency versions. The drift is then diagnosable.

## C08

- E1: Baseline, intervention, evidence, scope.
- E2: Simulation only. No claim about human duelists.
- E3: The method exists in the literature. The contribution must be the delta.

## C09

- E1: Integrity, limits, impact.
- E2: Simulated BT only. Four arms. No human subjects.
- E3: The draft hides the failure. Restore it with numbers.

## C10

- E1: Define, toy, derive, implement and complexity, compare, debug, critique, design.
- E2: The seeds differ across methods. The comparison is void.
- E3: 'It is in the paper' is not a derivation. Derive it here.

## C11

- E1: Hypothesis, numbers, diagnosis, what it rules out.
- E2: Forced challenges spend rounds on weak arms, adding strong regret.
- E3: "Promising direction" without numbers is spin. The numbers say it lost.

## C12

- E1: Provenance-rich datasets, subgroup eval riges, replication code, audit tooling.
- E2: A rig that reports per-group accuracy on a public preference dataset.
- E3: Search first. The gap must be real.

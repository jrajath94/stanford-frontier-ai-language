# UU10 interview keys

## B1

Sufficient: hypotheses, metric, sample, success rule, failure rule. Strong answer states each. Red flags: 'just run it'. Rubric: 2 points. Remediation: U10-C01.

## B2

Sufficient: problem, theorem, proof sketch, experimental setup. Strong answer fills each. Red flags: 'I read the abstract'. Rubric: 2 points. Remediation: U10-C02.

## B3

Sufficient: I: leader-focused dueling TS. O: mean strong regret. D: lower. K: 4-arm BT bandit, T=200, 20 seeds. Strong answer states all four. Red flags: no direction. Rubric: 2 points. Remediation: U10-C03.

## B4

Sufficient: purpose, voluntariness, withdrawal, data-use limits, retention. Strong answer gives terms. Red flags: 'click-through is fine'. Rubric: 2 points. Remediation: U10-C04.

## B5

Sufficient: H1 supported (3.11 vs 17.24, no overlap), H2 not supported (8.36 vs 3.11). Strong answer gives CIs. Red flags: 'both worked'. Rubric: 2 points. Remediation: U10-C05/C06.

## B6

Sufficient: provenance-rich datasets, subgroup eval riges, replication code, audit tooling. Strong answer names all four. Red flags: 'more data'. Rubric: 2 points. Remediation: U10-C12.

## Ladder 1

- L1.1: I: dueling TS. O: mean strong regret. D: lower. K: 4-arm BT, T=200, 20 seeds.
- L1.2: Strengths 1.2, 0.8, 0.3, -0.5. Pairwise BT wins. T=200.
- L1.3: Non-overlapping 95 percent CIs plus lower mean = supported.
- L1.4: Beta posteriors per ordered pair, duel top-2 sampled Copeland. O(T x K^2).
- L1.5: TS concentrates duels on plausible leaders, uniform wastes duels.
- L1.6: Seeds differ: the comparison confounds method with RNG luck. Re-run paired.
- L1.7: No noise model of humans, no context, no fatigue. The BT model is a toy.
- L1.8: Human duel interface, paired seeds across methods, preregistered plan, consent (C04).
- Red flags: confounding method with RNG luck
- Rubric: 2 points per rung for mechanism. Remediation: U10-C04/C05.

## Ladder 2

- L2.1: Hypothesis, numbers, diagnosis, what it rules out.
- L2.2: Leader TS mean 8.36 vs TS 3.11, CIs [6.86, 9.87] vs [2.23, 3.99].
- L2.3: Leader mean above TS mean, CIs do not overlap, H2 predicted leader lower: not supported.
- L2.4: Vary the focus rule over 20 seeds: O(rules x seeds x T).
- L2.5: The honest report has numbers and a ruled-out claim, spin has neither.
- L2.6: The diagnosis is invented. Run the ablation before naming the cause.
- L2.7: When the experiment was unsound (no plan, no baseline). A sound negative result is publishable.
- L2.8: Ablate: random challenger, least-tried challenger, no focus. Compare means and CIs.
- Red flags: inventing a diagnosis without a test
- Rubric: 2 points per rung for mechanism. Remediation: U10-C11.

## A1

Sufficient: leader mean 8.36 above TS mean 3.11, CIs [6.86, 9.87] and [2.23, 3.99] do not overlap, H2 predicted leader lower, so H2 is not supported. Rules out 'any leader focus helps'. Strong answer shows the arithmetic. Red flags: 'overlapping means equal'. Rubric: 5 points. Remediation: U10-C05/C06/C11.

## A2

Sufficient: five elements with named terms (e.g. withdrawal: labels removed within 30 days) and a justification (withdrawal needs a removal mechanism or the promise is theater). Strong answer ties the mechanism to the promise. Red flags: impossible promises. Rubric: 5 points. Remediation: U10-C04.

## D1

Sufficient: each method uses a different seed stream (seed_by_method), so the comparison confounds the method with RNG luck, the flip under new seeds is expected. Minimal fix: one seed per trial, shared across methods (rng=np.random.default_rng(s)). Strong answer names confounding. Red flags: blaming the method. Rubric: 2 points diagnosis, 2 points fix. Remediation: U10-C04/C07.

## S1

Sufficient: first, check the human noise model (BT may not hold). Second, check the protocol (fatigue, interface). Third, check power (seeds, T). Strong answer orders by likelihood. Red flags: 'humans are just noisy'. Rubric: 3 points. Remediation: U10-C05/C11.

## S2

Sufficient reply: the negative result rules out a natural extension (leader focus), with numbers, it constrains the design space, dropping it is publication bias. Strong answer is three sentences. Red flags: defensiveness without numbers. Rubric: 3 points. Remediation: U10-C09/C11.

## R1

Sufficient objections: (1) one seed is not evidence. (2) no baseline: nothing to beat. (3) no plan: the claim can shift. Check first: ask for the per-seed numbers and the baseline. Strong answer names all three. Red flags: accepting the plot. Rubric: 2 points per objection, 2 points design. Remediation: U10-C01/C04.

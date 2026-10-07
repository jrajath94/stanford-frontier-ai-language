# U09 interview keys

## B1

Sufficient: a function from ballot profiles to outcomes, plurality, Borda, Condorcet. Strong answer defines each. Red flags: 'majority always'. Rubric: 2 points. Remediation: U09-C01.

## B2

Sufficient: UD, Pareto, IIA, non-dictatorship. Strong answer states each precisely. Red flags: confusing IIA with choice IIA. Rubric: 2 points. Remediation: U09-C02.

## B3

Sufficient: with 3+ outcomes every non-dictatorial deterministic rule is manipulable. Strong answer names strategyproofness. Red flags: 'some rules are strategyproof'. Rubric: 2 points. Remediation: U09-C03.

## B4

Sufficient: A beats B, B beats C, C beats A by majority, no Condorcet winner. Strong answer gives the margins. Red flags: 'a tie'. Rubric: 2 points. Remediation: U09-C04.

## B5

Sufficient: rescaling one voter changes the sum without changing preferences. Strong answer shows the flip. Red flags: 'average them anyway'. Rubric: 2 points. Remediation: U09-C06.

## B6

Sufficient: independence and competence (p > 0.5). Strong answer explains the reversal for p < 0.5. Red flags: 'more voters always help'. Rubric: 2 points. Remediation: U09-C07.

## Ladder 1

- L1.1: UD, Pareto, IIA, non-dictatorship.
- L1.2: A wins 4-3-2, D enters (loses 2-7 to B), B wins. IIA violated.
- L1.3: IIA + Pareto force a swing voter decisive on all pairs: a dictator.
- L1.4: Tallies per ballot. O(n m log m).
- L1.5: Plurality drops IIA (spoiler), Borda drops IIA, Condorcet may not exist (drops determinacy/resoluteness).
- L1.6: Arrow scopes ranked rules, it does not say all rules are rigged or that justification is pointless.
- L1.7: Two candidates, restricted domains, cardinal ballots, randomization.
- L1.8: Franchise, ballot, rule, tie-break, decider, justification, guardrails.
- Red flags: using Arrow to avoid justifying a rule
- Rubric: 2 points per rung for mechanism. Remediation: U09-C02/C10.

## Ladder 2

- L2.1: A misreport that yields a better outcome by the voter own ranking.
- L2.2: Sincere Borda A4 B5 C0 -> B. Misreport A>C>B -> A4 B4 C1 -> A. Voter 0 gains A over B.
- L2.3: The misreport moves one Borda point from B to C, tying A/B, the tie-break elects A.
- L2.4: Enumerate profiles and misreports. O(profiles x misreports).
- L2.5: Deterministic: manipulable (GS). Randomized (random dictator): strategyproof in expectation.
- L2.6: No: the search can miss cases (tie-breaks, ballot limits). Proof needs the theorem, not the search.
- L2.7: Without common knowledge the best-response uses wrong beliefs.
- L2.8: Random profiles, count profitable misreports per rule, report the rate.
- Red flags: assuming voters know each other ballots
- Rubric: 2 points per rung for mechanism. Remediation: U09-C03/C05.

## A1

Sufficient: 0.6826 and 0.9791 from the binomial tail, for p < 0.5 the tail sum falls with n (crowd converges to wrong). Strong answer shows monotonicity. Red flags: 'n always helps'. Rubric: 5 points. Remediation: U09-C07.

## A2

Sufficient: raw 100 vs 1 (A landslide), normalized 1 vs 1 (tie), the rescale is meaning-preserving per voter but not across voters. Strong answer names the invariance failure. Red flags: 'normalize then sum is fine'. Rubric: 5 points. Remediation: U09-C06.

## D1

Sufficient: IIA (independence of irrelevant alternatives): the A-vs-B order changed though no A-vs-B ballot changed. Smallest principled fix: adopt a less spoiler-prone rule (e.g. Condorcet check) and document the rule choice with its dropped axiom. Strong answer names the spoiler mechanism. Red flags: blaming the tie-break. Rubric: 2 points diagnosis, 2 points fix. Remediation: U09-C02/C11.

## S1

Sufficient: the profile is no longer fixed: voters arrive over time, early votes may influence later ones (bandwagon), and the electorate is endogenous. Minimal repair: time-stamped ballots, eligibility window, re-run rule. Strong answer names strategic timing. Red flags: 'just add them'. Rubric: 3 points. Remediation: U09-C01/C05.

## S2

Sufficient: plurality and simple majority survive (hand-countable), Borda is borderline, Condorcet pairwise matrices are explainable but heavier. Lost: expressiveness and some fairness properties. Strong answer trades auditability against axiom satisfaction. Red flags: 'use the fanciest rule'. Rubric: 3 points. Remediation: U09-C09/C11.

## R1

Sufficient objections: (1) Arrow forbids it for ranked deterministic rules: check which assumption was dropped. (2) the 'proof' may redefine an axiom. (3) no empirical test. Check first: is the rule ranked and deterministic with 3+ outcomes? Strong answer names the escape. Red flags: accepting the proof. Rubric: 2 points per objection, 2 points design. Remediation: U09-C02/C10.

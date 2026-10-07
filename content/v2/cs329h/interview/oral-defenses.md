# Oral defenses: cs329h

Ten deep ladders. Each ladder states one claim from the course, then eight follow-ups: define, toy, derive, implement and complexity, compare, debug, critique, design. Keys in keys-oral.md. Provenance: PLANNED / SOURCE ATTRIBUTION PENDING throughout.

## D1. Bradley-Terry MLE (U01)

Claim: on the 5-1 toy, the MLE score gap is 1.61.

1. Define: what does the score gap mean in the Bradley-Terry model?
2. Toy: write the 6 observed pairs.
3. Derive: derive the log-likelihood and the first-order condition for the gap.
4. Implement and complexity: sketch the optimizer loop. State the cost per iteration.
5. Compare: MLE vs MAP with a weak prior. When do they differ?
6. Debug: the optimizer returns a gap of 0. What went wrong?
7. Critique: what does the MLE assume about the annotators?
8. Design: design the data collection that halves the standard error of the gap.

## D2. DPO without a reward model (U05)

Claim: DPO trains the policy directly on preference pairs, with no separate reward model.

1. Define: what is the implicit reward in DPO?
2. Toy: write the DPO loss for one pair (winner yw, loser yl).
3. Derive: derive the DPO loss from the KL-constrained optimum.
4. Implement and complexity: sketch the training step. State the cost relative to SFT.
5. Compare: DPO vs RLHF with a learned reward model. When is each better?
6. Debug: the loss falls but the policy gets worse. Diagnose.
7. Critique: where does the derivation break?
8. Design: design the experiment that tests whether DPO needs on-policy data.

## D3. Value of information (U06)

Claim: the query in the U06 toy is worth 0.25.

1. Define: what is the value of information?
2. Toy: restate the gap toy (gap in {-0.5, 1.5}, prior 0.5 each).
3. Derive: derive 0.25 from the prior and posterior best actions.
4. Implement and complexity: sketch the VoI computation for n queries. State the cost.
5. Compare: VoI vs expected information gain. When do they rank queries differently?
6. Debug: VoI comes out negative. Diagnose.
7. Critique: what does VoI assume about the downstream decision?
8. Design: design the elicitation policy for a 3-query budget.

## D4. Dueling TS beats uniform (U07, U10)

Claim: H1 is supported (dueling TS mean 3.11 vs uniform 17.24, non-overlapping CIs).

1. Define: what is strong regret?
2. Toy: describe the 4-arm BT bandit.
3. Derive: derive the verdict rule from the CI definition.
4. Implement and complexity: sketch the TS loop. State the cost.
5. Compare: why does TS beat uniform here?
6. Debug: the verdict flips under a new seed set. Diagnose.
7. Critique: simulated BT vs real human duels. What is absent?
8. Design: design the human-subject replication.

## D5. Reward inversion interval (U08)

Claim: the inversion interval on the toy is [-0.94, 3.31], so the sign of the reward is unsure.

1. Define: what is the inverse problem here?
2. Toy: restate the observed choice and the rationality model.
3. Derive: derive the interval endpoints from the choice likelihood.
4. Implement and complexity: sketch the interval computation. State the cost.
5. Compare: point estimate vs interval. When does the point estimate mislead?
6. Debug: the interval is empty. Diagnose.
7. Critique: what does the interval assume about the human?
8. Design: design the experiment that shrinks the interval.

## D6. Arrow's impossibility (U09)

Claim: no ranked voting rule satisfies UD, Pareto, IIA, and non-dictatorship with 3 or more candidates.

1. Define: state the four conditions precisely.
2. Toy: show the spoiler flip (plurality, A wins 4-3-2, D enters, B wins).
3. Derive (intuition): sketch the swing-voter argument.
4. Implement and complexity: code the three tallies (plurality, Borda, Condorcet check). State the cost.
5. Compare: which axiom does each of plurality, Borda, Condorcet drop?
6. Debug: someone cites Arrow to claim "all voting is rigged". Diagnose the error.
7. Critique: what escapes the theorem's scope?
8. Design: design the rule-selection memo for a real election.

## D7. The jury theorem (U09)

Claim: 101 voters at competence 0.6 reach majority-correct probability 0.9791.

1. Define: state the jury theorem and its assumptions.
2. Toy: write the n=5 case and compute 0.6826.
3. Derive: derive the binomial tail formula.
4. Implement and complexity: sketch the computation. State the cost.
5. Compare: jury voting vs following the best expert. When is each better?
6. Debug: adding voters makes the majority worse. Diagnose.
7. Critique: what breaks when errors are correlated?
8. Design: design the panel (size and selection) for a medical diagnosis task.

## D8. The spoiler (U09)

Claim: D enters, loses 2-7 to B pairwise, yet flips the plurality winner from A to B.

1. Define: what is IIA?
2. Toy: restate the two profiles (without and with D).
3. Derive: show the A-vs-B pairwise order never changed across the two profiles.
4. Implement and complexity: code the spoiler check. State the cost.
5. Compare: plurality vs Borda vs Condorcet on spoiler resistance.
6. Debug: the spoiler check fires on every election. Diagnose.
7. Critique: is the spoiler a flaw in the rule or in the electorate?
8. Design: design the guardrail that flags spoiler-like flips.

## D9. The honest negative result (U10)

Claim: H2 is not supported (leader TS 8.36 vs dueling TS 3.11, non-overlapping CIs).

1. Define: what are the four parts of a negative-result report?
2. Toy: restate H2 in I/O/D/K form.
3. Derive: derive the H2 verdict from the CIs step by step.
4. Implement and complexity: what is the cost of the follow-up ablation?
5. Compare: honest report vs spin. What differs?
6. Debug: the diagnosis names the beta prior with no test. Diagnose.
7. Critique: when is a negative result not publishable?
8. Design: design the ablation that tests the challenge-cost diagnosis.

## D10. Identifiability (U01, U02)

Claim: Bradley-Terry scores are identified only up to an additive constant.

1. Define: what does identifiability mean here?
2. Toy: add 1 to all scores in the 5-1 toy. What changes?
3. Derive: show the likelihood depends only on score differences.
4. Implement and complexity: how do you anchor the scores in code? State the cost.
5. Compare: sum-zero vs first-score-zero anchoring. When does the choice matter?
6. Debug: the fitted scores drift across runs. Diagnose.
7. Critique: what does the anchoring choice hide?
8. Design: design the reporting standard for BT scores in a paper.

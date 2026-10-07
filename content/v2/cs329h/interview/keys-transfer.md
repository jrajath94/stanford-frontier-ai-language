# Transfer set keys: cs329h

Full keys for transfer-sets.md. Provenance: PLANNED / SOURCE ATTRIBUTION PENDING throughout.

## T1. Expensive duels

K1. Sufficient: maximize information per duel, not per round: acquire the pair with highest expected information gain per unit cost (all costs 1 here, so highest gain). Strong answer writes the objective as argmax E[information gain] subject to the 50-duel budget, and names the stopping rule (stop when the best gain falls below the cost). Red flags: "run the same loop for 50 rounds" with no change. Rubric: 3 points for the objective, 2 for the stopping rule. Remediation: U06-C01, U07-C03.

K2. Sufficient: the regret analysis counts rounds, with costly duels the right currency is regret per unit budget, and the horizon is the budget, not time. Strong answer names the change: the benchmark becomes the best fixed pair under the budget. Red flags: "regret is unchanged". Rubric: 3 points. Remediation: U07-C04.

## T2. Two candidates

K1. Sufficient: majority rule. With 2 candidates, majority satisfies UD, Pareto, IIA (no third candidate to be irrelevant), and non-dictatorship. Strong answer gives the one-paragraph proof: the pairwise majority is transitive with 2 outcomes, so no cycle and no dictator is needed. Red flags: "Arrow still applies". Rubric: 4 points. Remediation: U09-C02, U09-C10.

K2. Sufficient: minimal repair is to re-run the escape-map analysis: with 3 candidates Arrow binds again, so pick the rule by the documented tradeoff (e.g. Condorcet check with a tie-break, or Borda with a spoiler guardrail). Strong answer names the dropped axiom. Red flags: "keep majority" (undefined for 3). Rubric: 3 points. Remediation: U09-C10, U09-C11.

## T3. Biased annotators

K1. Sufficient: the bias inflates the win rate of whichever response is listed first, so the estimated gap shrinks toward zero or flips sign depending on the listing order. Strong answer derives it: observed P(A wins) = 0.6 when A is first regardless of the true gap, so the MLE gap is attenuated. Red flags: "the noise cancels out". Rubric: 4 points. Remediation: U01-C08, U08-C05.

K2. Sufficient: collection-time fix is to randomize the listing order (assumes the bias is positional, not content-based). Modeling-time fix is to add a position term to the BT model (assumes the bias is additive and constant). Strong answer states both assumptions. Red flags: "collect more data" (does not fix bias). Rubric: 2 points per fix. Remediation: U01-C10, U08-C09.

## T4. Adversarial human

K1. Sufficient: observing the human is now anti-informative about the agent's own objective: the human's action signals what the human wants, which is what the agent does not want. Strong answer works the toy: P(tea | reach left) is high for the human, so the agent should make coffee. Red flags: "ignore the human" (the observation still has value, reversed). Rubric: 4 points. Remediation: U06-C04, U06-C05.

K2. Sufficient: the minimal change is to flip the sign in the belief update (treat the human's action as evidence against the corresponding objective) or to model the human's objective explicitly with a sign parameter. Strong answer names the parameter. Red flags: "the framework cannot handle it". Rubric: 3 points. Remediation: U06-C04.

## T5. Off-policy reward data

K1. Sufficient: the gap enters at (1) reward training (the model is accurate only where the old policy went) and (2) policy optimization (the new policy exploits regions where the reward model is wrong). Strong answer names both. Red flags: "more data fixes it" (more off-policy data does not). Rubric: 3 points. Remediation: U05-C11, U08-C02.

K2. Sufficient: one guardrail is a held-out on-policy evaluation: collect fresh pairs under the new policy and measure reward-model accuracy there. Strong answer states what it measures (accuracy under the deployment distribution). Red flags: "check the training loss". Rubric: 3 points. Remediation: U05-C12.

## T6. Correlated jury

K1. Sufficient: the theorem's independence assumption fails, so the conclusion fails with it: the effective sample size collapses toward 1, and the majority amplifies the shared error. Strong answer argues from the assumption, not from new numbers. Red flags: "101 voters still help" (they do not, under perfect correlation). Rubric: 4 points. Remediation: U09-C07.

K2. Sufficient: the minimal repair is to model the correlation (e.g. a shared latent signal) and down-weight correlated votes, or to diversify the information sources. Strong answer names one concrete repair. Red flags: "add more voters" (more correlated voters do not help). Rubric: 3 points. Remediation: U09-C07.

## T7. Three-query budget

K1. Sufficient: choose the batch of 3 queries that maximizes expected information gain jointly (batch experimental design), e.g. greedy selection by marginal gain. Strong answer writes the objective (maximize joint EIG over 3-query batches) and names the algorithm (greedy batch selection). Red flags: "ask the 3 best single queries" (ignores redundancy). Rubric: 4 points. Remediation: U06-C02, U07-C03.

K2. Sufficient: what is lost is adaptivity: the second query cannot condition on the first answer, so the batch pays the price of committing without information. Strong answer names it as the adaptivity gap. Red flags: "nothing is lost". Rubric: 3 points. Remediation: U06-C01.

## T8. Bad reference policy

K1. Sufficient: the KL term anchors the policy to the bad reference, so the optimum stays near bad responses, the preference gradient must fight the KL penalty everywhere. Strong answer traces it: pi* is proportional to pi_ref * exp(r/beta), so a bad pi_ref drags the optimum down. Red flags: "the data overrides the reference" (only as beta goes to 0). Rubric: 4 points. Remediation: U05-C07.

K2. Sufficient: fix the reference (use a better supervised model as pi_ref) is minimal: the data is clean and the loss is sound, so the broken piece is the anchor. Strong answer justifies by elimination. Red flags: "collect more preferences" (the anchor is the problem). Rubric: 3 points. Remediation: U05-C07.

## T9. Unknown scales

K1. Sufficient: the toy is voter 1: A=10, B=0, voter 2: A=0, B=100 (same preference intensity, different scales). Raw sums give B=100 vs A=10: B wins. Rescale voter 2 to 0-10: tie. Strong answer shows the flip. Red flags: "average them anyway". Rubric: 4 points. Remediation: U09-C06.

K2. Sufficient: escape 1 is normalization per voter (drops interpersonal comparability of raw scores). Escape 2 is to use ranks only (drops cardinal information). Strong answer names the dropped assumption for each. Red flags: "there is no escape". Rubric: 2 points per escape. Remediation: U09-C06, U09-C10.

## T10. The extension wins

K1. Sufficient: baseline dueling TS, intervention leader-focused dueling, evidence mean 2.10 vs 3.11 with non-overlapping CIs over 20 seeds, scope 4-arm BT simulation. Strong answer gives all four elements. Red flags: claiming human generality. Rubric: 4 points. Remediation: U10-C08.

K2. Sufficient: the follow-up varies the challenge cost independently of the focus rule (e.g. change the strength gaps so weak arms are weaker, then re-run): if the win persists when challenges are expensive, the rule is genuinely better, if it vanishes, challenges were just cheap. Strong answer names the manipulated variable. Red flags: "run more seeds" (does not distinguish the mechanisms). Rubric: 4 points. Remediation: U10-C03, U10-C11.

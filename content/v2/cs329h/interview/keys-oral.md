# Oral defense keys: cs329h

Full keys for oral-defenses.md. Each rung: minimum sufficient, strong answer, red flags, rubric, remediation. Provenance: PLANNED / SOURCE ATTRIBUTION PENDING throughout.

## D1. Bradley-Terry MLE

1. Sufficient: the gap is the log-odds that A beats B. Strong answer writes P(A beats B) = 1/(1+exp(-gap)). Red flags: "the gap is a probability". Rubric: 2 points. Remediation: U01-C05.
2. Sufficient: 5 pairs A beats B, 1 pair B beats A. Strong answer lists them. Red flags: inventing scores. Rubric: 1 point. Remediation: U01-C05.
3. Sufficient: L = 5*log(p) + 1*log(1-p), p = sigmoid(gap), first-order condition 5*(1-p) - 1*p = 0 gives p = 5/6, gap = log(5) = 1.61. Strong answer shows each step. Red flags: "set the derivative to the data". Rubric: 3 points. Remediation: U01-C06.
4. Sufficient: Newton or gradient ascent on the one-parameter NLL, O(n) per iteration. Strong answer names Newton. Red flags: "grid search is fine at scale". Rubric: 2 points. Remediation: U01-C06.
5. Sufficient: they differ when data is scarce, the prior shrinks the gap toward 0. Strong answer gives the direction. Red flags: "MAP is always better". Rubric: 2 points. Remediation: U03-C05.
6. Sufficient: the optimizer failed (bad start, or the labels were flipped). Strong answer checks the NLL at gap 0 vs 1.61. Red flags: "the data has no signal". Rubric: 2 points. Remediation: U01-C06.
7. Sufficient: annotators are exchangeable, independent, and follow the logistic noise model. Strong answer names all three. Red flags: "it assumes honest annotators" (it assumes the noise model, which covers flips). Rubric: 2 points. Remediation: U01-C08.
8. Sufficient: the SE scales as 1/sqrt(n), quadrupling the pairs halves the SE. Strong answer gives the scaling. Red flags: "collect until it looks right". Rubric: 2 points. Remediation: U04-C02.

## D2. DPO without a reward model

1. Sufficient: the implicit reward is beta * log(pi(y|x)/pi_ref(y|x)). Strong answer writes it. Red flags: "DPO has no reward" (it has an implicit one). Rubric: 2 points. Remediation: U05-C07.
2. Sufficient: L = -log sigmoid(beta * (log(pi(yw)/pi_ref(yw)) - log(pi(yl)/pi_ref(yl)))). Strong answer writes it exactly. Red flags: dropping the reference terms. Rubric: 2 points. Remediation: U05-C07.
3. Sufficient: from pi*(y|x) proportional to pi_ref * exp(r/beta), solve for r, plug into the BT likelihood. Strong answer shows the substitution. Red flags: "DPO is just classification". Rubric: 3 points. Remediation: U05-C07.
4. Sufficient: one forward/backward pass on the pair, like SFT but with two sequences and the reference log-probs. Strong answer states the cost (about 2x SFT per pair). Red flags: "DPO needs RL". Rubric: 2 points. Remediation: U05-C07.
5. Sufficient: DPO is simpler and stable, RLHF with a reward model allows online data and explicit reward shaping. Strong answer names the tradeoff. Red flags: "DPO always wins". Rubric: 2 points. Remediation: U05-C08.
6. Sufficient: the policy exploits the implicit reward (off-policy drift) or the reference is bad. Strong answer names reward hacking. Red flags: "train longer". Rubric: 2 points. Remediation: U05-C09.
7. Sufficient: the derivation assumes the KL-constrained optimum form and on-policy data, off-policy data breaks it. Strong answer names both. Red flags: "the math is exact". Rubric: 2 points. Remediation: U05-C11.
8. Sufficient: train DPO on on-policy vs off-policy pairs, compare win rates, the gap measures the data dependence. Strong answer names the comparison. Red flags: "more data". Rubric: 2 points. Remediation: U05-C11.

## D3. Value of information

1. Sufficient: VoI is the expected gain in decision quality from the answer before asking. Strong answer writes E[max_a E[U|answer]] - max_a E[U]. Red flags: "information is always valuable" (it can be zero). Rubric: 2 points. Remediation: U06-C01.
2. Sufficient: gap in {-0.5, 1.5}, prior 0.5 each, pick A (the arm) with prior mean 0.5. Strong answer restates it. Red flags: confusing the gap with the reward. Rubric: 1 point. Remediation: U06-C01.
3. Sufficient: prior best gives 0.5, after the answer, if gap is -0.5 take 0, if 1.5 take 1.5, posterior best = 0.5*0 + 0.5*1.5 = 0.75, VoI = 0.75 - 0.5 = 0.25. Strong answer shows the arithmetic. Red flags: "VoI is the posterior mean". Rubric: 3 points. Remediation: U06-C01.
4. Sufficient: enumerate answer outcomes per query, recurse, O(branches^depth). Strong answer names the exponential cost. Red flags: "it is linear". Rubric: 2 points. Remediation: U06-C02.
5. Sufficient: VoI values the decision, EIG values the belief change. They differ when information does not change the action. Strong answer gives the case. Red flags: "they are the same". Rubric: 2 points. Remediation: U06-C01, U07-C03.
6. Sufficient: VoI cannot be negative for a free query, a negative value means a sign error or a cost was subtracted. Strong answer names the error. Red flags: "some queries hurt". Rubric: 2 points. Remediation: U06-C01.
7. Sufficient: it assumes the downstream decision is known and the utility is correct. Strong answer names both. Red flags: "VoI needs no model". Rubric: 2 points. Remediation: U06-C01.
8. Sufficient: batch VoI over 3-query sets (no adaptation), greedy selection. Strong answer states the objective. Red flags: "ask the 3 highest single VoI" (ignores redundancy). Rubric: 2 points. Remediation: U06-C02.

## D4. Dueling TS beats uniform

1. Sufficient: strong regret per round is C* - max(C_i, C_j) with C the true Copeland scores. Strong answer writes it. Red flags: "regret is 0/1". Rubric: 2 points. Remediation: U07-C07.
2. Sufficient: strengths 1.2, 0.8, 0.3, -0.5, pairwise BT win probabilities, T=200. Strong answer lists them. Red flags: inventing strengths. Rubric: 1 point. Remediation: U07-C07.
3. Sufficient: non-overlapping 95 percent CIs plus lower mean means the difference is not explained by seed noise. Strong answer states the rule. Red flags: "non-overlap proves the truth". Rubric: 2 points. Remediation: U10-C05.
4. Sufficient: Beta(1,1) posteriors per ordered pair, each round sample win probs, compute sampled Copeland, duel the top 2, O(T * K^2). Strong answer sketches it. Red flags: "TS needs the true model". Rubric: 2 points. Remediation: U07-C07.
5. Sufficient: TS concentrates duels on plausible leaders, uniform wastes duels on weak pairs. Strong answer names the mechanism. Red flags: "TS is optimal" (unproven here). Rubric: 2 points. Remediation: U07-C07.
6. Sufficient: the seeds differed across methods, confounding method with RNG luck. Strong answer names confounding. Red flags: blaming the method. Rubric: 2 points. Remediation: U10-C04.
7. Sufficient: no human noise model, no context, no fatigue, BT is a toy. Strong answer lists what is absent. Red flags: "simulations prove the method". Rubric: 2 points. Remediation: U10-C09.
8. Sufficient: human duel interface, paired seeds across methods, preregistered plan, consent. Strong answer names all four. Red flags: "just deploy it". Rubric: 2 points. Remediation: U10-C01, U10-C04.

## D5. Reward inversion interval

1. Sufficient: the inverse problem is to infer the latent reward from observed choices under a rationality model. Strong answer states it. Red flags: "read the reward off the choice". Rubric: 2 points. Remediation: U08-C07.
2. Sufficient: the observed choice plus the assumed choice rule (e.g. softmax with temperature). Strong answer restates both. Red flags: dropping the temperature. Rubric: 1 point. Remediation: U08-C04.
3. Sufficient: invert the choice likelihood: the set of rewards consistent with the observed choice at the stated confidence. Strong answer shows the inversion. Red flags: "the interval is the posterior" (without saying so). Rubric: 3 points. Remediation: U08-C11.
4. Sufficient: grid or root-find over the reward, O(grid). Strong answer names the method. Red flags: "closed form always". Rubric: 2 points. Remediation: U08-C11.
5. Sufficient: the point estimate hides the uncertainty, it misleads when the interval crosses zero (sign unsure). Strong answer names the crossing. Red flags: "the point estimate is enough". Rubric: 2 points. Remediation: U08-C11.
6. Sufficient: the rationality model contradicts the observation (no reward explains the choice). Strong answer names the contradiction. Red flags: "more data". Rubric: 2 points. Remediation: U08-C06.
7. Sufficient: it assumes the human follows the stated choice rule with the stated temperature. Strong answer names both. Red flags: "it assumes rationality" (too vague). Rubric: 2 points. Remediation: U08-C06.
8. Sufficient: elicit more choices that discriminate the reward sign (targeted queries). Strong answer names the design. Red flags: "repeat the same query". Rubric: 2 points. Remediation: U06-C02.

## D6. Arrow's impossibility

1. Sufficient: UD (any profile), Pareto (unanimous pairs respected), IIA (pairwise order independent of others), non-dictatorship. Strong answer states each precisely. Red flags: confusing IIA with choice IIA. Rubric: 2 points. Remediation: U09-C02.
2. Sufficient: A wins 4-3-2, D enters (loses 2-7 to B), B wins with A2 D2 B3 C2. Strong answer gives the tallies. Red flags: "D won". Rubric: 1 point. Remediation: U09-C02.
3. Sufficient: IIA plus Pareto force a swing voter who is decisive on all pairs: a dictator. Strong answer sketches it. Red flags: "the proof is just the example". Rubric: 3 points. Remediation: U09-C02.
4. Sufficient: tally per ballot: plurality O(n*m), Borda O(n*m log m), Condorcet pairwise O(n*m^2). Strong answer states each. Red flags: "all are linear". Rubric: 2 points. Remediation: U09-C01.
5. Sufficient: plurality drops IIA (spoiler), Borda drops IIA, Condorcet may not exist (drops determinacy). Strong answer names each. Red flags: "Borda satisfies IIA". Rubric: 2 points. Remediation: U09-C02, U09-C10.
6. Sufficient: Arrow scopes ranked deterministic rules, it does not say all rules are rigged or that justification is pointless. Strong answer names the scope error. Red flags: agreeing with the claim. Rubric: 2 points. Remediation: U09-C10.
7. Sufficient: two candidates, restricted domains, cardinal ballots, randomization. Strong answer names all four. Red flags: "nothing escapes". Rubric: 2 points. Remediation: U09-C10.
8. Sufficient: franchise, ballot, rule, tie-break, decider, justification, guardrails. Strong answer names each. Red flags: "let the data decide". Rubric: 2 points. Remediation: U09-C09, U09-C11.

## D7. The jury theorem

1. Sufficient: n odd voters, each correct with prob p > 0.5 independently, P(majority correct) rises with n. Strong answer states both assumptions. Red flags: "more voters always help" (needs p > 0.5). Rubric: 2 points. Remediation: U09-C07.
2. Sufficient: sum_{k=3..5} C(5,k) 0.6^k 0.4^{5-k} = 0.6826. Strong answer writes the sum. Red flags: "0.6^3". Rubric: 2 points. Remediation: U09-C07.
3. Sufficient: the majority is correct when more than n/2 voters are correct, sum the binomial terms. Strong answer derives it. Red flags: "it follows from the law of large numbers" (that is the limit argument, not the formula). Rubric: 2 points. Remediation: U09-C07.
4. Sufficient: sum the tail terms, O(n). Strong answer states it. Red flags: "O(2^n)". Rubric: 1 point. Remediation: U09-C07.
5. Sufficient: the jury wins when competence is symmetric and unknown, the expert wins when one p is much higher and known. Strong answer names the condition. Red flags: "the jury always wins". Rubric: 2 points. Remediation: U09-C07.
6. Sufficient: p < 0.5 (the crowd converges to wrong) or correlated errors. Strong answer names one. Red flags: "the theorem is wrong". Rubric: 2 points. Remediation: U09-C07.
7. Sufficient: the effective sample size collapses, the majority amplifies the shared error. Strong answer argues from independence. Red flags: "correlation helps". Rubric: 2 points. Remediation: U09-C07.
8. Sufficient: size from the target CI width, selection for competence and independence (diverse sources). Strong answer names both. Red flags: "as many as possible". Rubric: 2 points. Remediation: U09-C07.

## D8. The spoiler

1. Sufficient: IIA is the requirement that the social order of A vs B depends only on the A-vs-B ballots. Strong answer states it. Red flags: "irrelevant candidates do not matter" (too vague). Rubric: 2 points. Remediation: U09-C02.
2. Sufficient: without D: 4x A>B>C, 3x B>A>C, 2x C>B>A (A wins). With D: 2x D>A>B>C, 2x A>B>C>D, 3x B>A>C>D, 2x C>B>A>D (B wins). Strong answer gives both. Red flags: changing a ballot. Rubric: 2 points. Remediation: U09-C02.
3. Sufficient: every voter ranks A vs B the same in both profiles (check each ballot), only the winner changed. Strong answer checks it. Red flags: "D changed minds" (no ballot changed). Rubric: 2 points. Remediation: U09-C02.
4. Sufficient: re-run the tally with and without each loser, flag winner changes, O(losers * tally). Strong answer sketches it. Red flags: "check by hand". Rubric: 2 points. Remediation: U09-C11.
5. Sufficient: plurality is the most spoiler-prone, Borda less so but still vulnerable, Condorcet is spoiler-proof when a Condorcet winner exists. Strong answer orders them. Red flags: "Borda is safe". Rubric: 2 points. Remediation: U09-C02, U09-C10.
6. Sufficient: the threshold is too tight (flags near-ties) or the rule is plurality (spoilers are normal). Strong answer names one. Red flags: "the code is wrong". Rubric: 2 points. Remediation: U09-C11.
7. Sufficient: it is a flaw in the rule (IIA violation), not in the voters, the ballots are sincere. Strong answer takes the position. Red flags: "blame the voters". Rubric: 2 points. Remediation: U09-C02, U09-C09.
8. Sufficient: flag when the top-two gap is below epsilon or when removing a loser changes the winner, route to human review. Strong answer names the trigger. Red flags: "ban third candidates". Rubric: 2 points. Remediation: U09-C11.

## D9. The honest negative result

1. Sufficient: hypothesis, numbers, diagnosis, what it rules out. Strong answer names each. Red flags: burying the failure. Rubric: 2 points. Remediation: U10-C11.
2. Sufficient: I: leader-focused dueling TS. O: mean strong regret. D: lower. K: 4-arm BT bandit, T=200, 20 seeds. Strong answer states all four. Red flags: no direction. Rubric: 2 points. Remediation: U10-C03.
3. Sufficient: leader mean 8.36 above TS mean 3.11, CIs [6.86, 9.87] and [2.23, 3.99] do not overlap, H2 predicted leader lower: not supported. Strong answer shows the arithmetic. Red flags: "overlapping means equal". Rubric: 3 points. Remediation: U10-C05, U10-C11.
4. Sufficient: vary the focus rule over 20 seeds: O(rules * seeds * T). Strong answer states it. Red flags: "free". Rubric: 1 point. Remediation: U10-C04.
5. Sufficient: the honest report has numbers and a ruled-out claim, spin has neither. Strong answer names the difference. Red flags: "spin is fine for morale". Rubric: 2 points. Remediation: U10-C11.
6. Sufficient: the diagnosis is invented without a test, run the ablation first. Strong answer names the error. Red flags: "the prior is the obvious cause". Rubric: 2 points. Remediation: U10-C11.
7. Sufficient: when the experiment was unsound (no plan, no baseline), a sound negative result is publishable. Strong answer names the condition. Red flags: "negative results are unpublishable". Rubric: 2 points. Remediation: U10-C09.
8. Sufficient: ablate the focus rule (random challenger, least-tried challenger, no focus) and compare means and CIs. Strong answer names the variants. Red flags: "re-run the same thing". Rubric: 2 points. Remediation: U10-C11.

## D10. Identifiability

1. Sufficient: the parameters are identified if distinct values give distinct data distributions. Strong answer states it. Red flags: "the optimizer finds them". Rubric: 2 points. Remediation: U01-C07.
2. Sufficient: nothing changes: all pairwise probabilities depend on differences, which are unchanged. Strong answer states it. Red flags: "the probabilities shift". Rubric: 1 point. Remediation: U01-C07.
3. Sufficient: P(A beats B) = sigmoid(s_A - s_B), adding c to all scores leaves every difference unchanged. Strong answer shows it. Red flags: "the likelihood is flat" (it is flat only in the constant direction). Rubric: 2 points. Remediation: U01-C07.
4. Sufficient: fix one score to 0 or enforce sum zero after fitting, O(1). Strong answer names one. Red flags: "regularization fixes it" (it masks it). Rubric: 2 points. Remediation: U01-C07.
5. Sufficient: the choice matters for interpretation (which item is the reference) but not for predictions. Strong answer names the distinction. Red flags: "it never matters". Rubric: 2 points. Remediation: U01-C07.
6. Sufficient: the anchor was not fixed (or the optimizer starts differ and the constant drifts). Strong answer names the cause. Red flags: "the data changed". Rubric: 2 points. Remediation: U01-C07.
7. Sufficient: anchoring hides the unidentified degree of freedom, readers may mistake the level for signal. Strong answer names the risk. Red flags: "nothing is hidden". Rubric: 2 points. Remediation: U01-C07.
8. Sufficient: report the anchoring convention, the differences (not levels), and the standard errors of the differences. Strong answer names all three. Red flags: "report the scores". Rubric: 2 points. Remediation: U01-C07.

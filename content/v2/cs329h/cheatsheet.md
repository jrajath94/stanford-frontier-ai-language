---
page_id: cs329h-cheatsheet
course_slug: cs329h
course_name: "CS329H: Machine Learning from Human Preferences"
course_order: 6
order: 900
nav: "CS329H · Cheatsheet"
title: "CS329H Cheatsheet"
summary: "Dense reference: definitions, formulas, numbers, decisions, mistakes, interview one-liners."
date: "2026-10-05"
instructor: "Sanmi Koyejo"
offering: "Spring 2026"
---
## The story in one page

Human scores drift. Comparisons hold. Two annotators rate the
same responses 7/5 and 9/7: different scales, same verdict, X
beats Y by 2. That gap is the whole field. Preference learning
turns comparisons into trained models: comparisons in, utility
numbers out, behavior steered by the numbers.

The machinery runs one arc. Random utility adds noise to fixed
utilities. Gumbel noise gives the softmax. Two items give
Bradley-Terry: p(j beats k) = sigma(V_j - V_k). Rankings stage
it: Plackett-Luce. Pairs are fit by logistic regression.
Questions are bought with Fisher information. Actions balance
exploration and exploitation. Many voices aggregate by voting
rules that cannot all be fair. Strategists need mechanism
design. Deployment needs the values written down.

## Formulas you must be able to write

**Sigmoid.** sigma(z) = 1 / (1 + e^{-z}). Maps any real number
to (0, 1). Gap 0 gives 0.5. Gap 1 gives 0.731. Gap 2 gives
0.881.

**Random utility.** H_j = V_j + noise_j, noise i.i.d. Gumbel.
Choice goes to max H_j.

**Softmax.** p(j from S) = e^{V_j} / sum_{k in S} e^{V_k}.
Only differences matter: V and V + c predict identically.

**Bradley-Terry.** p(j beats k) = sigma(V_j - V_k).

**Plackett-Luce.** p(A>B>C) = [e^{V_A}/sum] x
[e^{V_B}/(e^{V_B}+e^{V_C})]. Worked: 0.665 x 0.731 = 0.486.

**Rasch.** p(Y_ij = 1) = sigma(U_i + V_j). U is appetite, V is
appeal. Pairwise, U cancels: p(j>k|i) = sigma(V_j - V_k).

**BT log-likelihood, one pair.** y*d - log(1 + e^d), d =
V_j - V_k. Concave. Anchor V_1 = 0.

**Elo.** V_new = V_old + K(score - expected). Worked: 1500 vs
1500, K = 32, win moves 1516/1484. Upset vs 1700 pays 24.

**Fisher, Bernoulli.** I = p(1-p). Max 0.25 at p = 0.5. Ask
at 50/50.

**DPO.** r*(x,y) = beta log[pi(y|x)/pi_ref(y|x)]. loss = -log
sigma(r*(x,y_w) - r*(x,y_l)). Worked, beta = 1: gap 0.8,
sigma 0.69, loss 0.37.

**PPO objective.** maximize E[r(x,y)] - beta KL(pi || pi_ref).

**GRPO advantage.** A_i = (r_i - mean(r)) / std(r) over the
group of G samples for one prompt. Worked: [1,0,0,0] gives
+1.73, -0.58 x3. [1,1,1,0] gives +0.58 x3, -1.73. KL 0.001 to
pi_ref. No critic.

**UCB score.** A: 1.2 + 0.305 = 1.505. B: 1.0 + 1.076 = 2.076.
Play B. Bonus shrinks with pulls: explore the uncertain.

**Thompson.** Sample once from each posterior, play the
winner. Probability matching, not maximization.

**Nested logit.** Nest the buses under one node with
dissimilarity lambda. Worked: lambda 1.0 -> train 0.155, each
bus 0.422. lambda 0.01 -> train 0.268, each bus 0.366. As
lambda falls, buses cannibalize each other, not the train.

**Elo to BT scale.** d' = d / 173.7. A 16-point Elo swing =
0.092 BT units. Elo is SGD: V_new = V_old + K(score -
expected), K is the learning rate.

**First-price shading.** b(v) = (n-1)/n * v. 3 bidders,
uniform: bid 2/3 of value. v = 10 -> bid 6.67. Revenue
equivalence: expected revenue 5.0 either format.

**VCG payment.** Pay your externality: welfare of others
without you minus welfare of others with you. Worked: bundle
bidder 10, splitters 6+6. Split wins. Each splitter pays
10 - 6 = 4.

**Myerson virtual value.** phi(v) = v - (1-F(v))/f(v).
Uniform [0,10]: phi = 2v - 10, zero at 5. Optimal reserve 5.

**Revelation principle.** Any equilibrium outcome of any
mechanism is a truthful equilibrium of a direct mechanism.
Search among truthful mechanisms.

**STV.** Rounds: no majority, eliminate weakest, transfer
votes to next choice. Worked 5-voter: round 1 A=2 B=2 C=1,
C out, transfers to B. Round 2: B wins 3-2.

**Bridging.** r = mu + alpha + beta + p^T q + noise. Select
on beta only: the note both sides rate helpful. Conservative
by construction.

**D-optimality.** Pick the query maximizing the
log-determinant gain. Worked: gain 0.223 vs 0.086, ask X.

**Polis.** u = mu + alpha_j + beta_j + p^T q_j + noise. Select
on beta after factoring out ideology.

## Never-confuse pairs

- Elo rating vs BT strength: same model, different scale.
  Divide by 173.7.
- DPO vs PPO: DPO is offline classification on pairs. PPO is
  online RL with rollouts. Different data regimes.
- GRPO vs PPO: GRPO drops the critic, uses group z-scores.
  Needs verifiable rewards, not human pairs.
- UCB vs Thompson: UCB plays the max upper bound. Thompson
  samples and plays the sample winner. Deterministic vs
  randomized exploration.
- Borda vs Condorcet: Borda counts points across ranks.
  Condorcet checks pairwise majorities. They can disagree.
- First-price vs second-price: first-price shades, revenue
  equivalence holds in expectation, not per auction.
- IIA vs IIA-prime: IIA ignores all other alternatives.
  IIA-prime lets the count between y and y' matter.
- Revealed vs informed: revealed is what behavior shows.
  Informed is what reflection would endorse. Never equate.

## Numbers that demonstrate

- Red bus splits: train share falls 0.269 to 0.155. Clones
  steal share under IIA.
- Mixture 50/50: A/B ratio moves 1.54 to 1.00 when C leaves.
  IIA breaks.
- Ten straight wins: likelihood climbs at gap 3, 5, 10. MLE
  wants infinity. Use a prior.
- Greedy bandit: A 8/10, B 1/2. P(B's true rate > 0.8) =
  0.104. Greedy locks in wrong 10% of the time.
- Condorcet cycle: A beats B, B beats C, C beats A, each 2-1.
  No group ranking is transitive.
- Borda toy: ballots A>B>C, A>B>C, B>C>A, C>B>A. Plurality
  crowns A. Borda crowns B 5-4-3.
- Second-price toy: values 10, 7, 5. Bid 10, pay 7. Truth
  dominates: 9 loses profitable wins, 11 buys losses.
- GRPO toy: group [1,0,0,0]. Winner advantage +1.73, losers
  -0.58 each. The lone success gets the whole push.
- UCB toy: A 1.505 vs B 2.076. B wins despite lower mean.
  Uncertainty beats point estimate.
- Nested logit: lambda 1.0 to 0.01. Train share 0.155 to
  0.268. The nest parameter moves 11 points of share.
- First-price: v = 10, 3 bidders. Equilibrium bid 6.67 vs
  second-price payment 7. Expected revenue both 5.0.
- STV 5-voter: round 1 A=2 B=2 C=1. C eliminated, transfers
  to B. Round 2 B wins 3-2. Plurality tied 2-2.
- D-optimal: gains 0.223 vs 0.086. Ask where uncertainty is
  largest, 2.6x the information.
- Elo swing: 16 points = 0.092 BT units. Scale matters for
  interpreting leaderboards.

## Decisions: which tool when

- Pairs only, anonymous users: Bradley-Terry. It is logistic
  regression on pair differences.
- Rankings from annotators: Plackett-Luce, staged softmax.
- Near-duplicate options: nested logit or probit. Plain BT
  spreads probability across clones.
- Disagreeing annotator groups: mixture or per-group models.
  One BT fit is a compromise nobody holds.
- Label budget: Fisher information, ask at 50/50, adaptive
  loop. Keep a random stream as a reality check.
- Acting under uncertainty: Thompson sampling. Pull each arm
  with P(it is best).
- Scores unanchored, comparisons easy: dueling bandits or
  preferential BO.
- Strategic labelers: mechanism thinking. Pay for the thing
  you measure.
- Many voices, one policy: name the voting rule and the
  relaxed axiom. DPO is a Borda election.

## Mistakes that cost interviews

- Fitting BT without an anchor. The likelihood has a flat
  direction. Always fix V_1 = 0 first.
- Treating fitted utilities as truth. Rashomon: BT, mixture,
  and nested logit can tie at 90%. Report the set.
- Ignoring annotator bias. Verbosity and position effects get
  learned as quality. Model the bias.
- Forgetting the KL leash. Unconstrained optimization hacks
  the proxy. Goodhart with a training loop.
- Assuming DPO fixes misspecification. DPO inherits BT, IIA,
  and homogeneity. Six assumptions, six failures.
- Calling aggregation neutral. Arrow: no perfect rule. Name
  the relaxed axiom.
- Treating humans as passive oracles. CIRL: feedback depends
  on beliefs about you. Mechanisms: pay for honesty.

## Interview one-liners

- "Scores drift, comparisons hold. That is why the field runs
  on pairs."
- "Gumbel noise in, softmax out. Two items give Bradley-Terry:
  sigma of the gap."
- "Only differences are identified. Anchor before fitting."
- "Clones break IIA: the train's share fell 0.269 to 0.155."
- "Undefeated items explode MLE to infinity. The prior caps
  it."
- "Ask at 50/50: Fisher information peaks at 0.25."
- "Thompson sampling explores by probability matching, no
  bonus knob."
- "DPO is BT with the reward parameterized through the
  policy. Loss 0.37 on the worked pair."
- "DPO is a Borda election over responses. Audit the
  electorate."
- "No neutral aggregation: Arrow. Name the relaxed axiom."
- "Pay for the thing you measure. The payment rule is the
  mechanism."

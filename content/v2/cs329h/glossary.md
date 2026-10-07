# Glossary: cs329h (U01-U05)

| Term | Definition |
| --- | --- |
| Preference | A latent ordering over items: what the person would want if fully informed. Not directly observed. |
| Choice | The observed pick from a set of options in a context. The data we fit. |
| Pairwise data | Records of the form (a, b, y): item a faced item b, outcome y says which was chosen. |
| Ordinal | Information about order only: A beats B. No statement about how much. |
| Cardinal | Information about strength: A beats B by a stated amount. |
| Bradley-Terry | Model: P(a beats b) = sigma(s_a - s_b). Scores s are latent, estimated from pairwise outcomes. |
| Logistic likelihood | Product over comparisons of sigma(y * gap) terms. Its log is the training objective. |
| Identifiability | Whether data can pin down a unique parameter value. Bradley-Terry scores are identified only up to an additive constant. |
| Random utility | U_i = u_i + epsilon_i: deterministic value plus a random shock. The choice is argmax of U. |
| Gumbel | Noise distribution whose argmax over independent draws gives the softmax (multinomial logit) choice rule. |
| IIA | Independence of irrelevant alternatives: the ratio P(a)/P(b) does not change when other options enter or leave. |
| Factor model | Choice probability from low-rank structure: score(a) = w . v_a, respondent embedding dot item embedding. |
| Heterogeneity | Different respondents follow different preference parameters. |
| MLE | Maximum likelihood estimator: the parameter value that makes the observed data most probable. |
| MAP | Maximum a posteriori: like MLE but multiplied by a prior. Equals penalized likelihood. |
| Posterior predictive | Distribution of a future outcome averaged over the posterior, not conditioned on one point estimate. |
| Calibration | Agreement between predicted probabilities and observed frequencies. |
| Score function | Gradient of the log-likelihood with respect to parameters. Its expectation at the truth is zero. |
| Fisher information | Expected squared score. Measures curvature of the log-likelihood and sets the asymptotic variance floor. |
| Active selection | Choosing which query to label next to maximize information per unit of budget. |
| Design criterion | A scalar summary of the information matrix used to rank designs: A-optimality (trace of inverse), D-optimality (determinant). |
| Reward model | A learned scalar r(x, y) trained on preference data to predict which response a human would prefer. |
| KL regularization | Penalty beta * KL(pi || pi_ref) that keeps the tuned policy near the reference policy. |
| Regularized optimum | Closed-form best policy under KL penalty: pi*(y|x) proportional to pi_ref(y|x) exp(r(x,y)/beta). |
| DPO | Direct preference optimization: trains the policy directly on preference pairs with a logistic loss on the implicit reward margin, no separate reward model. |
| PPO | Proximal policy optimization: on-policy RL algorithm used in classic RLHF against a learned reward model with a KL penalty. |
| Reward hacking | The policy exploits flaws in the learned reward to raise the proxy score while the true objective falls. |
| Off-policy | Data collected under a different policy than the one being evaluated or trained. |

| Value of information | Expected gain in decision quality from an answer before asking: E[max_a E[U\|answer]] - max_a E[U]. |
| Assistance game | The agent is unsure of the human's objective and learns it from the human's actions. Shared payoff, asymmetric knowledge. |
| Contextual bandit | The agent sees a context, then picks an arm. The best arm depends on the context. |
| Assisted bandit | The agent proposes. The human can veto or override. The team payoff exceeds either alone. |
| Thompson sampling | Sample a world from the posterior. Act optimally for the sample. P(pull a) = P(a optimal). |
| Acquisition function | A score over candidate queries or arms (expected improvement, information gain) that ranks what to try next. |
| Dueling bandit | A bandit that learns from pairwise comparisons (duels) instead of absolute rewards. |
| Strong regret | Per-round shortfall of the duel's best Copeland score versus the maximum Copeland score. |
| Copeland score | An arm's average pairwise win probability against the other arms. |
| Preferential Bayesian optimization | Optimization using only pairwise comparisons, no absolute evaluations. |
| Reward inversion | Inferring the latent reward from observed choices under a rationality model. |
| Bounded rationality | Humans choose with noise, bias, and computational limits. The choice rule is not exact maximization. |
| Misspecified rationality | The assumed choice model (temperature, bias form) is wrong, so inversion returns the wrong reward. |
| Voting rule | A function from ballot profiles to outcomes: plurality, Borda, Condorcet. |
| Condorcet winner | The candidate who beats every other candidate pairwise. Need not exist. |
| Condorcet cycle | A beats B, B beats C, C beats A by majority. No Condorcet winner. |
| Arrow's theorem | No ranked voting rule satisfies UD, Pareto, IIA, and non-dictatorship with 3+ candidates. |
| IIA (Arrow) | The A-vs-B social order depends only on the A-vs-B ballots. |
| Gibbard-Satterthwaite theorem | With 3+ outcomes, every non-dictatorial deterministic voting rule is manipulable. |
| Spoiler | A losing candidate whose entry flips the winner. A symptom of IIA violation. |
| Strategyproofness | No voter gains by misreporting their ballot. |
| Jury theorem | With independent voters each correct with p > 0.5, the majority correctness rises with n. |
| Subgroup fairness | Reporting preference-model accuracy per group, not just pooled. |
| Pareto frontier | The set of policies where no group can gain without another losing. |
| Pre-analysis plan | Hypotheses, metric, sample, success rule, failure rule, frozen before the run. |
| Data consent | Purpose, voluntariness, withdrawal, data-use limits, retention for preference-data collection. |
| Contribution statement | Baseline, intervention, evidence, scope: what is new relative to the strongest baseline. |
| Negative-result report | Hypothesis, numbers, diagnosis, what the report rules out. |
| Oral ladder | Eight follow-ups per claim: define, toy, derive, implement/complexity, compare, debug, critique, design. |

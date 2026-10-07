# Cheatsheet: cs329h

Formulas, definitions, and numbers for all ten units. Provenance: S01-S04 source-supported (title level), S05-S18 PLANNED / SOURCE ATTRIBUTION PENDING.

## Formulas

| Name | Formula |
| --- | --- |
| Bradley-Terry | P(i beats j) = 1 / (1 + exp(-(s_i - s_j))) |
| BT log-likelihood | sum log sigmoid(y * gap) over pairs |
| BT identifiability | scores identified up to +c, anchor (sum zero or first zero) |
| Softmax (Gumbel-max) | P(i) = exp(u_i) / sum exp(u_j) |
| IIA | P(A)/P(B) independent of C |
| KL-constrained optimum | pi*(y\|x) proportional to pi_ref(y\|x) exp(r(x,y)/beta) |
| DPO loss | -log sigmoid(beta * (log(pi(yw)/pi_ref(yw)) - log(pi(yl)/pi_ref(yl)))) |
| DPO implicit reward | beta * log(pi(y\|x)/pi_ref(y\|x)) |
| Value of information | E[max_a E[U\|answer]] - max_a E[U] |
| Bayes update (tea toy) | 0.8*0.5/(0.8*0.5+0.3*0.5) = 0.727 |
| Strong regret | C* - max(C_i, C_j), C = true Copeland scores |
| Jury theorem | sum_{k>n/2} C(n,k) p^k (1-p)^{n-k}, needs p > 0.5 and independence |
| CI (95%) | mean +/- 1.96 * SD / sqrt(n) |
| Borda (3 candidates) | 2/1/0 points per ballot |
| Pareto frontier (toy) | (sqrt(t), sqrt(1-t)), t in [0,1] |

## Key numbers

| Where | Number |
| --- | --- |
| U01 BT gap, 5-1 toy | 1.61 |
| U03 EM, Gaussian mixture | -775.9 to -738.2 |
| U06 VoI toy | 0.25 |
| U06 greedy vs eps-greedy regret | 29.6 vs 2.4 (T=300, seed 0) |
| U07 TS vs greedy vs eps-greedy | 4.0 vs 89.6 vs 5.2 (seed 23) |
| U08 inversion interval | [-0.94, 3.31], sign unsure |
| U09 cycle margins | +1/+1/+1 |
| U09 cycle frequency | 0.081 (11 voters, seed 0) |
| U09 spoiler | A 4-3-2, then B (A2 D2 B3 C2), D loses 2-7 to B |
| U09 jury | 0.6826 (n=5), 0.9791 (n=101) at p=0.6 |
| U09 subgroup | pooled 0.73, groups 0.87 / 0.59 |
| U09 welfare t=0.5 | (0.707, 0.707) |
| U10 capstone A | uniform 17.24 [16.56, 17.92], TS 3.11 [2.23, 3.99], leader 8.36 [6.86, 9.87] |
| U10 capstone B (HYPOTHETICAL) | funnel 6000/6000/5664/5664, drift 0.793 to 0.718, BLOCK |

## Definitions

- **Identifiability**: distinct parameters give distinct data distributions.
- **IIA (Arrow)**: the A-vs-B social order depends only on A-vs-B ballots.
- **Strategyproofness**: no voter gains by misreporting.
- **Strong regret**: per-round shortfall of the duel's best Copeland score vs the max.
- **Pre-analysis plan**: hypotheses, metric, sample, success rule, failure rule, frozen before the run.
- **Contribution**: baseline, intervention, evidence, scope.
- **Negative-result report**: hypothesis, numbers, diagnosis, what it rules out.
- **Data consent**: purpose, voluntariness, withdrawal, data-use limits, retention.
- **Oral ladder**: define, toy, derive, implement/complexity, compare, debug, critique, design.

## Axiom and escape maps

- **Arrow**: UD, Pareto, IIA, non-dictatorship. Escapes: 2 candidates, restricted domain (single-peaked), cardinal ballots, randomization.
- **Gibbard-Satterthwaite**: 3+ outcomes, deterministic, non-dictatorial rules are manipulable. Escape: random dictator (strategyproof in expectation).
- **Jury**: needs independence and p > 0.5. Correlated errors void it.
- **Interpersonal comparison**: needs a common scale. Rescaling one voter flips sums.

## Honest-number rules

1. Every computed number states its seed.
2. CIs are procedure properties, not bounds on the truth.
3. Hypothetical numbers stay labeled HYPOTHETICAL.
4. Negative results are reported with numbers, not buried.
5. The constant in BT scores is meaningless, anchor before quoting.

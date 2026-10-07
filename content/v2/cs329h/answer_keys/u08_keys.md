# U08 answer keys

## C01

- E1: Many goals map to the same choice, so the inverse maps one choice to many goals.
- E2: Both goals predict salad with probability 1. The likelihood ratio is 1, the posterior equals the prior.
- E3: Only salad remained. The choice reflects availability, not either goal.

## C02

- E1: G = 0.8 - 0.6t. S = 0.8 + 1.4t.
- E2: argmax G at t = 0 (0.8). argmax S at t = 1 (2.2, G = 0.2).
- E3: The 0.8 correlation was measured on random policies. Under optimization the policy leaves that region and the correlation reverses.

## C03

- E1: r1 = 0.9 + 0.3 = 1.2. r2 = 0.4 + 0.9 = 1.3.
- E2: The style gap 0.6 exceeds the content gap 0.5.
- E3: Correlated features make the split arbitrary. The least-squares attribution is then a modeling artifact.

## C04

- E1: Evaluate in random order, take the first option with utility above 0.8, stop at 10 evaluations.
- E2: 1 - 0.8^10 = 0.8926.
- E3: Unevaluated options look rejected to the maximizer model. Their inferred utility sinks though they were never seen.

## C05

- E1: P(pick A) = sigma(gap + b).
- E2: log(0.65/0.35) = 0.619.
- E3: Subtracting a wrong-signed b moves the estimate away from truth by 2|b|.

## C06

- E1: Truth: satisficing. Model: softmax maximization.
- E2: 0.449.
- E3: Without process data the likelihood cannot separate 'disliked' from 'unevaluated'.

## C07

- E1: L = sigma(s_A - s_B)^2 sigma(s_B - s_C) times the prior.
- E2: gap_AB = 0.802, gap_BC = 0.254.
- E3: 2-0 separates: the likelihood rises forever with the gap. No finite maximizer exists.

## C08

- E1: F(s,a,s') = gamma Phi(s') - Phi(s).
- E2: Q = [1.0, 0.9], Q' = [0.5, 0.40]. argmax both: go.
- E3: The shaping encodes task-specific potential. In a new MDP it rewards the wrong transitions.

## C09

- E1: Corrected = 0.8 - b_hat.
- E2: b_hat 0.3 -> 0.5 (error 0). b_hat 0.9 -> -0.1 (error 0.6).
- E3: One b_hat cannot match two true biases. Both annotators get a wrong correction.

## C10

- E1: Pooled: one gap fit to all 20 pairs. Per-user: one gap per 10 pairs.
- E2: +0.90 / -0.90.
- E3: Two pairs cannot pin a gap. The per-user MAP fits noise and loses on held-out.

## C11

- E1: p(gap | data) proportional to BT likelihood times N(0,1) priors, marginalized by grid.
- E2: [-0.94, 3.31].
- E3: The grid assumes BT. Under satisficing truth the interval is tight and wrong.

## C12

- E1: Model named, biases audited, pooling justified, uncertainty reported, counterfactuals flagged.
- E2: The honest sentence in c6.
- E3: 'Bias audited' with no experiment is a guess in a lab coat.

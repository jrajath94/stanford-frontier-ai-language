# U08 interview keys

## B1

Sufficient: the goal-to-choice map is many-to-one. Strong answer gives the salad example. Red flags: 'choices reveal goals'. Rubric: 2 points. Remediation: U08-C01.

## B2

Sufficient: measurable stand-in S for unobserved G, the gap is E[G | optimize S] vs max E[G]. Strong answer names Goodhart. Red flags: 'the surrogate is the goal'. Rubric: 2 points. Remediation: U08-C02.

## B3

Sufficient: F(s,a,s') = gamma Phi(s') - Phi(s). Strong answer states the preserved optimal policy. Red flags: any additive bonus. Rubric: 2 points. Remediation: U08-C08.

## B4

Sufficient: take the first option above an aspiration level within a search budget. Strong answer contrasts with maximizing. Red flags: 'lazy maximizing'. Rubric: 2 points. Remediation: U08-C04.

## B5

Sufficient: the bias is absorbed into the gap estimate as phantom preference. Strong answer computes 0.619. Red flags: 'more data fixes it'. Rubric: 2 points. Remediation: U08-C05.

## B6

Sufficient: |b - b_hat| < |b|. Strong answer shows the -0.1 case. Red flags: 'always debias'. Rubric: 2 points. Remediation: U08-C09.

## Ladder 1

- L1.1: Product over pairs of sigma(y*gap) times the prior.
- L1.2: MLE: gap -> +infinity (separation). MAP: finite (GAPAB).
- L1.3: sigma((s_a + c) - (s_b + c)) = sigma(s_a - s_b): the constant cancels.
- L1.4: Gradient ascent on the log posterior. O(pairs) per step.
- L1.5: MLE diverges on separation, MAP gives a finite point, the full posterior gives uncertainty.
- L1.6: Complete separation: the likelihood has no finite maximizer.
- L1.7: BT forces transitivity, cycles break the link.
- L1.8: A pair where the rewards predict opposite winners with high confidence.
- Red flags: treating the MAP gap as the truth without uncertainty
- Rubric: 2 points per rung for mechanism. Remediation: U08-C07/C11.

## Ladder 2

- L2.1: margin = gap + b.
- L2.2: Corrected = 0.8 - 0.9 = -0.1, error 0.6.
- L2.3: Correction helps iff the estimate is closer to b than 0 is.
- L2.4: Sweep b_hat, compute errors. O(grid).
- L2.5: Modeling subtracts an estimate, design removes the bias at the source.
- L2.6: The bias model is wrong (wrong sign or heterogeneous b).
- L2.7: One b_hat fits none of the annotators.
- L2.8: Randomize frames, test whether the bias term predicts held-out framed pairs.
- Red flags: debiasing without validating the bias model
- Rubric: 2 points per rung for mechanism. Remediation: U08-C05/C09.

## A1

Sufficient: the shaping telescopes in the Bellman equation, leaving the argmax unchanged, numerically Q = [1.0, 0.9], Q' = [0.5, 0.40], same argmax. Strong answer shows the telescoping step. Red flags: 'shaping never matters'. Rubric: 5 points. Remediation: U08-C08.

## A2

Sufficient: 1 - 0.8^10 = PFIND, mean chosen CHOSEN vs maximizer 0.99, the maximizer model reads unevaluated options as rejected. Strong answer explains the likelihood mismatch. Red flags: 'satisficing is just noisy maximizing'. Rubric: 5 points. Remediation: U08-C04/C06.

## D1

Sufficient: the choice model and the data, not the optimizer. Length bias is unmodeled, so the BT fit absorbs it as preference, MLE separation may also inflate gaps. Smallest principled fix: audit for length bias (randomize or model it) and add a prior before deploying. Strong answer names both issues. Red flags: blaming convergence. Rubric: 2 points diagnosis, 2 points fix. Remediation: U08-C05/C07.

## S1

Sufficient: replace softmax-maximization with the satisficing likelihood (evaluation order + threshold), infer threshold and utilities jointly. Strong answer notes the need for process data. Red flags: keeping the maximizer model. Rubric: 3 points. Remediation: U08-C04/C06.

## S2

Sufficient: pooled inversion averages opposite tastes to ~0, the single reward pleases nobody. Minimal change: per-group rewards or a mixture. Strong answer cites the +G1P/-G1P toy. Red flags: more pooled data. Rubric: 3 points. Remediation: U08-C10.

## R1

Sufficient objections: (1) length bias unmodeled. (2) no held-out test of the claim. (3) no debiasing or style control. Convincing: length-controlled pairs, bias audit, held-out prediction. Strong answer designs the distinguishing experiment. Red flags: accepting the inversion. Rubric: 2 points per objection, 2 points design. Remediation: U08-C03/C05/C09.

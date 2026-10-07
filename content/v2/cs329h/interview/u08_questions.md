# U08 interview bank: questions

## Breadth (6)

B1. Why does an observed choice underdetermine the latent goal?
B2. Define a surrogate objective and its gap.
B3. Write the potential-based shaping formula.
B4. Define satisficing in one sentence.
B5. How does an unmodeled bias corrupt the inverted reward?
B6. State the condition under which debiasing helps.

## Deep ladders (2 x 8)

### Ladder 1: BT inversion

L1.1 Define: write the inverse likelihood.
L1.2 Toy: A beats B twice. What does MLE say? What does MAP say?
L1.3 Derive: show the additive constant cancels (level non-identifiability).
L1.4 Implement and complexity: code the MAP fit, state its cost.
L1.5 Compare: MLE vs MAP vs full posterior.
L1.6 Debug: estimated gaps explode to +-20. Diagnose.
L1.7 Critique: the BT link is assumed. What if choices cycle?
L1.8 Design: design the distinguishing query between two candidate rewards.

### Ladder 2: bias and debiasing

L2.1 Define: write observed margin = true gap + bias.
L2.2 Toy: true gap 0.5, bias 0.3, b_hat 0.9. Compute the corrected gap and its error.
L2.3 Derive: the cure-beats-disease condition |b - b_hat| < |b|.
L2.4 Implement and complexity: code the correction sweep, state its cost.
L2.5 Compare: correct by modeling vs correct by design (randomization).
L2.6 Debug: debiased model worse on held-out. Diagnose.
L2.7 Critique: one b_hat for all annotators. What breaks?
L2.8 Design: design the experiment that validates the bias model.

## Analytical exercises (2)

A1. Show that potential-based shaping preserves the optimal policy: state the argument, then verify numerically on the 2-state toy (Q = [1.0, 0.9], Q' = [0.5, 0.40]).
A2. Compute the satisficing numbers: P(find above 0.8 in 10 draws) and mean chosen utility, and explain why the maximizer model misreads unevaluated options.

## Implementation / debug (1)

D1. The inverted reward below ranks verbose wrong answers first. The optimizer converged. Is the failure in the optimizer, the choice model, or the data? Justify in two sentences and give the smallest principled fix.

```python
gaps = fit_bt(pairs)          # MLE, no prior
reward = gaps                 # deploy
```

## Changed-constraint scenarios (2)

S1. The human is known to satisfice with threshold 0.7. How does the inversion change?
S2. The reward must serve two user groups with opposite tastes. What breaks in single-reward inversion?

## Research critique (1)

R1. A paper 'discovers' that users prefer longer answers, from an inverted reward on raw pairwise data. List three objections and design the convincing follow-up.

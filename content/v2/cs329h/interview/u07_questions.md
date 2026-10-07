# U07 interview bank: questions

## Breadth (6)

B1. State the Thompson sampling algorithm in two sentences.
B2. Write UCB, EI, and PI in symbols or words.
B3. What feedback does a dueling bandit observe?
B4. Define the Copeland winner.
B5. What is strong regret in a dueling bandit?
B6. Name two assumptions behind regret bounds.

## Deep ladders (2 x 8)

### Ladder 1: Thompson sampling

L1.1 Define: write the TS loop.
L1.2 Toy: Beta(8,3), Beta(4,4), Beta(2,6). Which arm does TS favor, and why?
L1.3 Derive: show probability matching: P(pull a) = P(a optimal | data).
L1.4 Implement and complexity: code Beta-Bernoulli TS, state its cost.
L1.5 Compare: TS vs epsilon-greedy vs UCB.
L1.6 Debug: TS never pulls arm 2 after round 50 though arm 2 is best. Diagnose.
L1.7 Critique: what breaks under non-stationary arms?
L1.8 Design: design the 20-seed comparison of TS vs UCB on the dueling toy.

### Ladder 2: dueling bandit inference

L2.1 Define: write P(i beats j) under BT.
L2.2 Toy: strengths [1.2, 0.8, 0.3, -0.5]. Compute P(arm1 beats arm4).
L2.3 Derive: derive the Copeland score from the win matrix.
L2.4 Implement and complexity: code Copeland, state its cost.
L2.5 Compare: Copeland vs Borda vs Condorcet winner.
L2.6 Debug: Copeland scores tie. What now?
L2.7 Critique: the BT link forces transitivity. What if the truth cycles?
L2.8 Design: design the test of BT fit on real duel data.

## Analytical exercises (2)

A1. Derive the EI closed form under a Gaussian posterior, then compute EI for the three toy candidates.
A2. Compute the Condorcet jury numbers for 5 voters at p = 0.6, and show the crowd accuracy increases with n.

## Implementation / debug (1)

D1. The preferential BO loop below stalls: best-so-far flat after iteration 5 though better points exist. The code is correct. Is the failure in the surrogate, the acquisition, or the feedback? Justify in two sentences and give the smallest principled fix.

```python
for t in range(20):
    fit_surrogate(duels)          # quadratic fit
    a, b = argmax_acquisition()   # EI duel pair
    w = observe_duel(a, b)
    duels.append((a, b, w))
```

## Changed-constraint scenarios (2)

S1. Duel feedback now allows ties ("no preference"). How do the BT model and Copeland change?
S2. The acquisition must run on a phone (tiny compute budget). Which method survives and what is lost?

## Research critique (1)

R1. A paper claims its dueling algorithm is 'optimal' because cumulative regret is lowest on one synthetic bandit with one seed. List three objections and design the evaluation that would convince you.

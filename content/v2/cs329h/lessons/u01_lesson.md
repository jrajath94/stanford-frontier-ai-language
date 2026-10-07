# U01: Preferences, choices, and generalization

Prerequisites: P06 (probability), P07 (estimation), P10 (ML foundations). Local remediation opens this lesson.

## Provenance

Concepts C01-C09 map to session S02, "Choice data, generalization, and Bradley-Terry" (28 Sep 2026): SOURCE-SUPPORTED at title level. Concepts C10-C12 are requested extensions: PLANNED / SOURCE ATTRIBUTION PENDING, taught as independent theory.

## Local remediation: the Bernoulli refresher

A comparison outcome is a Bernoulli trial. Write y = 1 when item a is chosen over item b, else y = 0. The PMF is P(y) = p^y (1-p)^(1-y). Expectation E[y] = p. Variance Var(y) = p(1-p). For n independent trials the count of wins is Binomial(n, p). The log-likelihood of p given outcomes y_1..y_n is l(p) = sum_i y_i log p + (1 - y_i) log(1 - p). Its maximizer is the sample mean. Every symbol in U01 builds on this.

## Russian-doll ladder for the major mechanism (Bradley-Terry)

- Shell 0: How do we turn pairwise win records into one score per item?
- Shell 1: Toy: A beat B five times, B beat A once. Two items, six numbers.
- Shell 2: Score s_i in utils, gap d = s_a - s_b, P(a beats b) = sigma(d).
- Shell 3: Rule: maximize the logistic log-likelihood over the scores.
- Shell 4: Derive the gradient. Implement gradient ascent in NumPy.
- Shell 5: Check: with no data the scores stay at the anchor. With infinite gap the probability saturates at 1.
- Shell 6: Change one factor: add label noise. Predict shrinkage, measure it (figure u01_f05).
- Shell 7: Counterexample: rock-paper-scissors cycles break the single-score model.
- Shell 8: Compare with Thurstone (Gaussian noise) under equal budgets.
- Shell 9: Extension: does a per-annotator noise rate improve held-out log-loss? Falsifiable: compare with a paired test.
- Shell 10: Production: miscalibrated scores misrank items in a recommender. Stakeholder decision is the ranking threshold.

## Not-yet-understood dependency list

1. Why the logistic shape and not another curve: answered in C05 via the Gumbel derivation in U02.
2. How to pick the score anchor: answered in C07.
3. What changes with more than two items per query: answered in U02.

---

### cs329h-U01-C01: preferences versus choices

**Contract 1. Source mapping, scope, objectives, dependencies.** Maps to S02. Objective: state the difference between a latent preference and an observed choice, and name one context factor that separates them. Depends on P06 only.

**Contract 2. Motivating question and tiny toy.** Question: if an annotator picks A over B, does that prove the annotator prefers A? Toy: the same annotator picks A in the morning and B in the evening. One person, two choices, one preference to infer.

**Contract 3. Plain-language mental model.** A preference is the answer to "what would you want with full information and no pressure." A choice is the answer to "what did you pick, here, now." The choice equals the preference plus context plus noise.

**Contract 4. Variables, units, shapes, assumptions.** Preference: latent ordering, no units observed. Choice y in {0,1}, dimensionless. Context c: a label like time of day or item order. Assumption: the preference is stable across the contexts we compare. The lesson tests this assumption in C11.

**Contract 5. Justified derivation or mechanism.** Write P(y=1 | preference, context) = sigma(s + b*c). The term b*c is the context effect. If b = 0, choices reveal preference directly. If b != 0, raw win counts confound the two. Figure u01_f01 fits s and b separately on a toy and recovers s = 0.80 with b = -0.66.

**Contract 6. Computed numerical example.** Toy data: 200 choices with A shown first, 200 with B shown first. Observed fraction choosing A: 0.69 versus 0.54. A naive read says preference changed. The fitted model says latent P(choose A) = 0.69 with context shift -0.66 utils. Same numbers as figure u01_f01.

**Contract 7. Algorithm and minimal implementation.** Grid search over (s, b) maximizing the logistic log-likelihood. Ten lines of NumPy. See the lab.

**Contract 8. Correctness checks and expected output.** Check 1: with b fixed at 0 the estimate of s equals log(p/(1-p)) of the pooled fraction. Check 2: shuffling the context labels drives the b estimate to 0. Expected output on the toy: s near 0.8, b near -0.66.

**Contract 9. Complexity, memory, statistical efficiency, stability, costs.** Grid search costs O(G_s * G_b * n). Memory O(n). Two parameters from 400 binary outcomes give standard errors near 0.1 utils. The sigmoid saturates, so extreme s values need many samples to pin down.

**Contract 10. Nearest alternatives and selection boundaries.** Alternative: a context-free Bradley-Terry model (b = 0). Choose the context model when the same pair is judged in multiple known contexts and the win rate moves with context. Choose the simpler model when each pair appears in one context only, because b is then unidentified.

**Contract 11. Failure case, broken assumption, counterexample.** Break the stability assumption: the annotator genuinely changes preference between morning and evening (a new movie released at noon). Then no context model recovers one stable s. The data needs a time-varying preference. Counterexample: with only one context observed, s and b collapse into one number and the split is arbitrary.

**Contract 12. Research reading and falsifiable extension.** Extension: measure order effects in a real annotation task by randomizing item order and testing b = 0 with a likelihood-ratio test. Falsifiable: if the test fails to reject b = 0 across three tasks, order effects are negligible there.

**Contract 13. Assessment.** Breadth: define preference and choice in one sentence each. Oral ladder: (1) define the split, (2) toy it with morning/evening, (3) derive why b confounds s, (4) implement the two-parameter fit, (5) compare with the b=0 model, (6) debug a fit where b explodes, (7) critique the stability assumption, (8) design the order-randomization experiment. Transfer: an LLM judge prefers the first answer shown. How do you estimate its true preference? Failure diagnosis: the b estimate is 5.0 with huge error bars. What went wrong? Counterfactual: what changes if contexts are unrecorded? Research: does b differ by annotator seniority?

**Contract 14. Lab and exercises.** Lab U01 task 1 fits s and b on synthetic data. Exercises: (E1) compute the pooled fraction from the toy. (E2) show b=0 reduces the model to plain Bradley-Terry. (E3) explain why one context cannot identify b. Keys in answer_keys/u01_keys.md.

**Contract 15. Visual units, provenance, accessibility, audit rows.** Figure visuals/u01_f01.png: lesson plate, source original toy, seed 0, alt text "Two bars of observed choice fractions by context, arrow labeled fit preference plus context, two bars of estimated preference and context effect." Audit: before state observed fractions, after state model parts, rule named, numbers computed. No unresolved visual conflict.

---

### cs329h-U01-C02: context

**Contract 1. Source mapping, scope, objectives, dependencies.** Maps to S02. Objective: list context factors that move choices without moving preferences, and show how to record them. Depends on C01.

**Contract 2. Motivating question and tiny toy.** Question: which details of the question change the answer? Toy: the same two summaries judged as "pick the better one" versus "pick the one with fewer errors" produce different winners.

**Contract 3. Plain-language mental model.** Context is everything around the comparison that is not the items: wording of the question, order on screen, time pressure, the annotator's recent history. Treat context as data, not as noise to ignore.

**Contract 4. Variables, units, shapes, assumptions.** Context vector c in R^k, each coordinate a recorded factor (binary or numeric). Assumption: the recorded factors capture the context shifts that matter. Unrecorded factors become noise.

**Contract 5. Justified derivation or mechanism.** Extend the score: effective gap = (s_a - s_b) + w . (c_a - c_b), where w weights context features. This is still logistic in an extended feature space, so all of C05-C06 applies unchanged.

**Contract 6. Computed numerical example.** Two summaries, gap s_a - s_b = 0.3. Wording factor adds w*c = 0.5 when the question mentions errors. P(A wins) moves from sigma(0.3) = 0.574 to sigma(0.8) = 0.690. One recorded factor flips a close call.

**Contract 7. Algorithm and minimal implementation.** Append context features to the design row for each comparison. Fit logistic regression. Same code as C06 with wider rows.

**Contract 8. Correctness checks and expected output.** Check: with all context features zero the fit matches the no-context fit. Expected: the wording coefficient near 0.5 on the toy.

**Contract 9. Complexity, memory, statistical efficiency, stability, costs.** Cost grows with k features: O(n*k) per gradient step. Each added feature needs roughly 10-20 informative comparisons to estimate without overfitting.

**Contract 10. Nearest alternatives and selection boundaries.** Alternative: randomize context away (shuffle order, fix wording) instead of modeling it. Model context when you cannot control it (logs from production). Randomize it away when you design the annotation task.

**Contract 11. Failure case, broken assumption, counterexample.** Unrecorded context that correlates with the items breaks the model: if harder prompts always show A first, the order coefficient absorbs item difficulty. Counterexample: a context factor with zero variation in the data gets an arbitrary coefficient.

**Contract 12. Research reading and falsifiable extension.** Extension: test whether time-of-day predicts choice residuals after fitting item scores. Falsifiable: a permutation test on the time coefficient.

**Contract 13. Assessment.** Breadth: name three context factors. Oral ladder from definition to the randomization experiment. Transfer: search results where position biases clicks. How do you correct it? Failure diagnosis: the context coefficient has the wrong sign. Name two causes. Counterfactual: what if context is recorded with error? Research: which contexts transfer across annotator pools?

**Contract 14. Lab and exercises.** Lab U01 task 2 adds a wording feature. Exercises: (E1) compute sigma(0.3) and sigma(0.8). (E2) write the extended design row. (E3) argue when to randomize instead of model. Keys in answer_keys/u01_keys.md.

**Contract 15. Visual units, provenance, accessibility, audit rows.** Covered by figure u01_f01 (context as the modeled factor). No new plate needed. Logged as shared.

---

### cs329h-U01-C03: pairwise data

**Contract 1. Source mapping, scope, objectives, dependencies.** Maps to S02. Objective: define the pairwise record, its schema, and its limits. Depends on P06.

**Contract 2. Motivating question and tiny toy.** Question: what is the smallest record that still teaches a model about preference? Toy: three records: (A,B,1), (B,C,1), (A,C,1).

**Contract 3. Plain-language mental model.** A pairwise record is one contest with one winner. Many contests rank the field the way many games rank teams.

**Contract 4. Variables, units, shapes, assumptions.** Record (a, b, y): a, b item ids in [m], y in {0,1} with y=1 meaning a won. Dataset D: n records. Assumption: records are independent given the scores (tested in C08).

**Contract 5. Justified derivation or mechanism.** The dataset is the sufficient input to the Bradley-Terry likelihood: only the win counts per ordered pair matter, not the record order. Proof sketch: the log-likelihood sums over records and regroups by pair.

**Contract 6. Computed numerical example.** Records: A beats B twice, B beats A once, B beats C three times. Win-count matrix: W[A,B]=2, W[B,A]=1, W[B,C]=3. The likelihood uses only these counts.

**Contract 7. Algorithm and minimal implementation.** Build the win-count matrix from a record list in one pass. Five lines of NumPy.

**Contract 8. Correctness checks and expected output.** Check: matrix is antisymmetric in counts (W[a,b] + W[b,a] = comparisons of that pair). Expected on the toy: the matrix above.

**Contract 9. Complexity, memory, statistical efficiency, stability, costs.** One pass, O(n) time, O(m^2) memory for the dense matrix. Use a sparse dict when m is large and pairs are few.

**Contract 10. Nearest alternatives and selection boundaries.** Alternative: listwise data (rank k items per query). Choose pairwise when annotators judge reliably only two at a time. Choose listwise when ranking k items costs little more than one pair and you need total orders.

**Contract 11. Failure case, broken assumption, counterexample.** Dependent records break independence: the same annotator judging the same pair ten times is not ten independent contests. Counterexample: duplicates from one annotator inflate confidence without adding information.

**Contract 12. Research reading and falsifiable extension.** Extension: test independence by comparing within-annotator and between-annotator agreement rates. Falsifiable: equal rates support independence.

**Contract 13. Assessment.** Breadth: write the schema. Oral ladder through the sufficiency argument. Transfer: chess tournament records. What breaks if rematches are common? Failure diagnosis: the win matrix has a zero row. What does fitting do? Counterfactual: what if y allows ties? Research: optimal pair sampling for a fixed budget (preview of U04).

**Contract 14. Lab and exercises.** Lab U01 task 3 builds the matrix. Exercises: (E1) build the toy matrix. (E2) prove sufficiency of pair counts. (E3) handle a tie code. Keys in answer_keys/u01_keys.md.

**Contract 15. Visual units, provenance, accessibility, audit rows.** Figure u01_f02 uses pair outcomes as its input. The record schema is a definition unit shown as a labeled triple in the lesson text. Logged: definition unit, no state change, no plate required.

---

### cs329h-U01-C04: ordinal/cardinal distinction

**Contract 1. Source mapping, scope, objectives, dependencies.** Maps to S02. Objective: distinguish order information from strength information and show what each identifies. Depends on C03.

**Contract 2. Motivating question and tiny toy.** Question: "A beats B" tells you the order. Does it tell you by how much? Toy: A beats B by a hair in task 1 and by a mile in task 2. Same order, different strength.

**Contract 3. Plain-language mental model.** Ordinal is the ranking. Cardinal is the ranking plus the gaps. Pairwise win records are ordinal. Scores with meaningful gaps are cardinal.

**Contract 4. Variables, units, shapes, assumptions.** Ordinal datum: a total order over items. Cardinal datum: a score vector s with gap units (utils). Assumption: the Bradley-Terry model upgrades ordinal data to cardinal scores by assuming the logistic link.

**Contract 5. Justified derivation or mechanism.** From ordinal data alone, any monotone transform of the scores preserves all orders. The logistic link picks one cardinal scale: gaps map to probabilities via sigma. Two pairs with the same order but gaps 0.8 and 3.0 give P = 0.69 and 0.95. Figure u01_f07 computes both.

**Contract 6. Computed numerical example.** Pair 1: gap 0.8, P(A beats B) = 0.690. Pair 2: gap 3.0, P = 0.953. Same order, different confidence. Same numbers as figure u01_f07.

**Contract 7. Algorithm and minimal implementation.** No new algorithm. This concept reframes the output of C05. Implementation: report gaps with confidence intervals, not just ranks.

**Contract 8. Correctness checks and expected output.** Check: permuting score labels preserves the predicted order but changes nothing else. Expected: rank order stable under monotone transforms of reported scores.

**Contract 9. Complexity, memory, statistical efficiency, stability, costs.** Cardinal claims cost more data: the standard error of a gap scales as 1/sqrt(n_pairs). Halving the error needs four times the comparisons.

**Contract 10. Nearest alternatives and selection boundaries.** Alternative: pure ranking methods (e.g., count wins, sort). Choose scores when downstream use needs confidence (thresholds, betting, active selection). Choose ranks when only the order matters and data is thin.

**Contract 11. Failure case, broken assumption, counterexample.** The logistic link is an assumption, not a fact. If the true noise is heavy-tailed, small gaps are overstated. Counterexample: with one comparison per pair, cardinal gaps are pure prior. The data only orders.

**Contract 12. Research reading and falsifiable extension.** Extension: elicit cardinal judgments directly (allocate 100 points between A and B) and test whether implied gaps match Bradley-Terry gaps. Falsifiable: correlation below 0.5 rejects the link on that task.

**Contract 13. Assessment.** Breadth: one sentence for ordinal, one for cardinal. Oral ladder through the monotone-transform argument. Transfer: movie ratings 1-5 stars. Ordinal or cardinal? Failure diagnosis: two items with identical win rates but the model reports different scores. Explain. Counterfactual: what if annotators report strengths directly? Research: when does cardinal elicitation beat pairwise?

**Contract 14. Lab and exercises.** Exercises: (E1) compute both probabilities. (E2) find a monotone transform that keeps order. (E3) state what one comparison per pair identifies. Keys in answer_keys/u01_keys.md.

**Contract 15. Visual units, provenance, accessibility, audit rows.** Figure visuals/u01_f07.png: lesson plate, source original toy, seed-free closed form, alt text "Two labeled pairs with the same order, arrow labeled sigma of gap, two probability bars 0.69 and 0.95." Audit: before state two gaps, after state two probabilities, rule named. No conflict.

---

### cs329h-U01-C05: Bradley-Terry

**Contract 1. Source mapping, scope, objectives, dependencies.** Maps to S02 (title level). Objective: state the model, derive its gradient, and fit it on a toy. Depends on C03, P07.

**Contract 2. Motivating question and tiny toy.** Question: how do many pairwise contests produce one score per item? Toy: three players, results: A beats B twice, B beats C three times, A beats C once.

**Contract 3. Plain-language mental model.** Each item has a hidden strength number. The chance A beats B depends only on the difference of strengths, pushed through an S-shaped curve. Big difference means near-certain win. Zero difference means a coin flip.

**Contract 4. Variables, units, shapes, assumptions.** Scores s in R^m, utils. P(a beats b) = sigma(s_a - s_b). Assumptions: (i) pairwise outcomes independent given scores. (ii) the logistic link. (iii) scores fixed during data collection.

**Contract 5. Justified derivation or mechanism.** The gradient of the log-likelihood with respect to s_a is sum over comparisons involving a of (y - p), where p is the current predicted probability. Reason: the derivative of log sigma(z) is 1 - sigma(z). So each comparison nudges the winner's score up by (1 - p) and the loser's down by p. Surprising wins move scores more than expected wins. This is the whole learning rule.

**Contract 6. Computed numerical example.** Toy scores start at 0. Records: (A,B,1) twice, (B,C,1) thrice, (A,C,1) once. After fitting with sum-zero anchor (2000 steps, lr 0.1, seed-free deterministic code): s = [2.72, 2.03, -4.76]. C never wins, so its score drops hard. Check: P(A beats B) = sigma(0.69) = 0.666, matching A's 2-0 record against B.

**Contract 7. Algorithm and minimal implementation.** Gradient ascent on the log-likelihood with the anchor sum(s) = 0 enforced after each step:

```python
import numpy as np

def fit_bradley_terry(pairs, m, steps=2000, lr=0.1):
    # pairs: list of (a, b, y). y = 1 if a won.
    s = np.zeros(m)
    for _ in range(steps):
        g = np.zeros(m)
        for a, b, y in pairs:
            p = 1.0 / (1.0 + np.exp(-(s[a] - s[b])))
            g[a] += y - p
            g[b] -= y - p
        s += lr * g
        s -= s.mean()
    return s
```

**Contract 8. Correctness checks and expected output.** Check 1: two items, A wins all n: the gap grows without bound. The code must cap or regularize (see C07/C08 in U03). Check 2: symmetric data (A beats B once, B beats A once) returns gap 0. Expected on the toy: scores sum to 0 and order A > B > C.

**Contract 9. Complexity, memory, statistical efficiency, stability, costs.** Each step costs O(n). Memory O(n + m). Logistic loss is convex in the scores, so gradient ascent converges to the global optimum. Overflow risk: exp of large gaps. Use the stable sigmoid or work in log space.

**Contract 10. Nearest alternatives and selection boundaries.** Alternative: Thurstone's model (Gaussian noise, probit link). The two agree closely on most data. Bradley-Terry has the simpler gradient and is the standard for preference data. Choose Thurstone when you need the Gaussian noise interpretation for a downstream model.

**Contract 11. Failure case, broken assumption, counterexample.** Cycles: A beats B, B beats C, C beats A, each strongly. One score per item cannot fit all three. The likelihood pushes scores together and the fit is poor. This is the intransitivity counterexample, expanded in U02-C12.

**Contract 12. Research reading and falsifiable extension.** Extension: fit a per-pair upset rate and test whether it is constant across pairs (a chi-square style check on residuals). Falsifiable: a pair with residual far outside the binomial range rejects the single-score model there.

**Contract 13. Assessment.** Breadth: write the model equation. Oral ladder: (1) define, (2) toy the three players, (3) derive the (y - p) update, (4) implement and state complexity, (5) compare with Thurstone, (6) debug the all-wins divergence, (7) critique the independence assumption, (8) design the residual experiment. Transfer: rank large language models from arena battles. What breaks? Failure diagnosis: scores drift by a constant each run. Name the cause (anchor). Counterfactual: what if ties are allowed? Research: does a per-item noise parameter improve held-out likelihood?

**Contract 14. Lab and exercises.** Lab U01 task 4 fits the three-player toy. Exercises: (E1) derive the (y - p) gradient. (E2) compute one gradient step by hand on (A,B,1) from zero. (E3) explain the all-wins divergence. Keys in answer_keys/u01_keys.md.

**Contract 15. Visual units, provenance, accessibility, audit rows.** Figure visuals/u01_f02.png: lesson plate, source original toy, closed form, alt text "Number line showing score gap 0.8, arrow labeled apply sigma of gap, logistic curve with marked point 0.69." Audit: before state gap, after state probability, rule named. No conflict.

---

### cs329h-U01-C06: logistic likelihood

**Contract 1. Source mapping, scope, objectives, dependencies.** Maps to S02. Objective: write the likelihood, plot it, and find its maximum on a toy. Depends on C05, P07.

**Contract 2. Motivating question and tiny toy.** Question: among all possible score gaps, which one makes the observed wins most probable? Toy: A beat B five times, B beat A once. One gap to estimate.

**Contract 3. Plain-language mental model.** The likelihood scores each candidate gap by how probable it makes the actual data. The log-likelihood turns the product into a sum. The best gap is the peak of that sum.

**Contract 4. Variables, units, shapes, assumptions.** Gap d in utils. Likelihood L(d) = sigma(d)^5 (1 - sigma(d))^1. Log-likelihood l(d) = 5 log sigma(d) + log(1 - sigma(d)). Assumptions: independent trials, correct link.

**Contract 5. Justified derivation or mechanism.** Differentiate: l'(d) = 5(1 - sigma(d)) - sigma(d). Set to zero: sigma(d) = 5/6, so d = log 5 = 1.609. The MLE gap is the log odds of the observed win rate. General rule: for one pair, the MLE gap equals log(wins_a / wins_b).

**Contract 6. Computed numerical example.** Grid search over d in [-3, 3] finds the minimum negative log-likelihood at d = 1.61 with value 2.703. Figure u01_f03 plots the curve and marks the minimum. Closed form agrees: log 5 = 1.609.

**Contract 7. Algorithm and minimal implementation.** One-dimensional grid search or Newton steps on l(d). The lab implements both and checks agreement.

**Contract 8. Correctness checks and expected output.** Check 1: grid optimum matches log(wins_a/wins_b) to grid resolution. Check 2: with 3-3 data the optimum is 0. Expected: d = 1.61, NLL = 2.703.

**Contract 9. Complexity, memory, statistical efficiency, stability, costs.** Grid search O(G) with G grid points. The log-likelihood is concave, so Newton converges in a handful of steps. Standard error of the gap estimate is about sqrt(1/wins_a + 1/wins_b) = 1.095 here. Wide, because n = 6 is small.

**Contract 10. Nearest alternatives and selection boundaries.** Alternative: Bayesian posterior over d (U03). Choose the MLE point when you need one number fast. Choose the posterior when the error bars matter for decisions.

**Contract 11. Failure case, broken assumption, counterexample.** Six wins and zero losses: the MLE gap is infinite. The likelihood keeps rising as d grows. Counterexample to "the MLE always exists": it fails on separable data. Fix: prior or regularization (U03-C08).

**Contract 12. Research reading and falsifiable extension.** Extension: compare the Wald interval from the curvature against a bootstrap interval on real annotation data. Falsifiable: systematic mismatch rejects the asymptotic approximation at that n.

**Contract 13. Assessment.** Breadth: write l(d) for the toy. Oral ladder through the log-odds derivation. Transfer: click-through rate estimation with 5 clicks in 6 shows. Give the MLE and its standard error. Failure diagnosis: Newton diverges. Name two causes (bad start, overflow). Counterfactual: what if trials are not independent? Research: when does the posterior beat the MLE for ranking?

**Contract 14. Lab and exercises.** Lab U01 task 5 reproduces figure u01_f03. Exercises: (E1) derive d = log 5. (E2) compute the standard error. (E3) show the 6-0 divergence. Keys in answer_keys/u01_keys.md.

**Contract 15. Visual units, provenance, accessibility, audit rows.** Figure visuals/u01_f03.png: lesson plate, source original toy, grid computation, alt text "Table of six comparison records, arrow labeled maximize log-likelihood, U-shaped negative log-likelihood curve with minimum marked at gap 1.61." Audit: before state records, after state curve, rule named. No conflict.

---

### cs329h-U01-C07: parameter identifiability

**Contract 1. Source mapping, scope, objectives, dependencies.** Maps to S02. Objective: prove the additive-constant non-identifiability and apply an anchor. Depends on C05.

**Contract 2. Motivating question and tiny toy.** Question: if two different score vectors predict exactly the same data, which one is right? Toy: s = [0.5, 0.0, -0.3] versus s = [1.5, 1.0, 0.7].

**Contract 3. Plain-language mental model.** Only gaps enter the model, so lifting every score by the same amount changes nothing observable. The data sees differences. The absolute level is invisible.

**Contract 4. Variables, units, shapes, assumptions.** Scores s in R^m. Shift invariance: P(a beats b | s) = P(a beats b | s + c*1) for any constant c. Assumption: the link depends only on gaps (true for Bradley-Terry and Thurstone).

**Contract 5. Justified derivation or mechanism.** sigma((s_a + c) - (s_b + c)) = sigma(s_a - s_b). The c cancels term by term. Hence the likelihood is flat along the all-ones direction, and the MLE is a line, not a point. Figure u01_f04 shows both score vectors giving identical probabilities 0.622, 0.690, 0.574.

**Contract 6. Computed numerical example.** s1 = [0.5, 0.0, -0.3], s2 = s1 + 1.0. Predicted P(A>B) = 0.622, P(A>C) = 0.690, P(B>C) = 0.574 under both. Equal to three decimals. Same numbers as figure u01_f04.

**Contract 7. Algorithm and minimal implementation.** Anchor after each gradient step: s -= s.mean() (sum-zero) or s[0] = 0 (pin first item). One line.

**Contract 8. Correctness checks and expected output.** Check: the anchored MLE is unique. Unanchored runs drift along the constant direction across seeds. Expected: identical probabilities, different raw scores.

**Contract 9. Complexity, memory, statistical efficiency, stability, costs.** Anchoring is O(m). Without it, the Hessian is singular and Newton steps fail. Gradient ascent still works but the solution wanders.

**Contract 10. Nearest alternatives and selection boundaries.** Alternative anchors: sum-zero (symmetric, good for reporting), pin-first (good when item 0 is a reference baseline), prior centered at zero (soft anchor, see U03-C08). Choose sum-zero for tables. Choose pin-first when comparing against a named baseline.

**Contract 11. Failure case, broken assumption, counterexample.** Partial identifiability: if items split into two groups with no cross-group comparisons, each group has its own free constant. Counterexample: A vs B data and C vs D data with no A/B vs C/D matches. The A-B gap to the C-D block is unidentified.

**Contract 12. Research reading and falsifiable extension.** Extension: test connectivity of the comparison graph before fitting. Report the number of connected components as the number of free constants. Falsifiable: a two-component graph must show a flat likelihood direction per component.

**Contract 13. Assessment.** Breadth: state the invariance in one line. Oral ladder through the cancellation proof. Transfer: word embeddings have a similar translation invariance. How is it handled? Failure diagnosis: Newton fails with singular matrix. Diagnose. Counterfactual: what if the link used absolute scores? Research: soft versus hard anchors under sparse data.

**Contract 14. Lab and exercises.** Exercises: (E1) prove the cancellation. (E2) construct the two-component counterexample. (E3) anchor both ways and compare. Keys in answer_keys/u01_keys.md.

**Contract 15. Visual units, provenance, accessibility, audit rows.** Figure visuals/u01_f04.png: lesson plate, source original toy, closed form, alt text "Bar chart of three probabilities for scores 0.5, 0.0, minus 0.3. Arrow labeled add 1.0 to all scores. Identical bar chart for scores 1.5, 1.0, 0.7." Audit: before state scores and probs, after state shifted scores and same probs, rule named. No conflict.

---

### cs329h-U01-C08: noise

**Contract 1. Source mapping, scope, objectives, dependencies.** Maps to S02. Objective: define label noise, derive its attenuation effect, and estimate it. Depends on C05-C06.

**Contract 2. Motivating question and tiny toy.** Question: what happens when the annotator clicks the wrong button sometimes? Toy: true gap 1.5, but each recorded label flips with probability q = 0.1.

**Contract 3. Plain-language mental model.** Label noise mixes the two outcomes toward 50/50. The model sees a weaker signal than the truth and estimates a smaller gap. Noise never creates preference. It only hides it.

**Contract 4. Variables, units, shapes, assumptions.** Flip probability q in [0, 0.5). Observed win rate p_obs = (1-q) p_true + q (1-p_true). Assumption: flips are independent of the items and of each other.

**Contract 5. Justified derivation or mechanism.** The observed log-odds shrink: logit(p_obs) = logit((1-2q) p_true + q). For small q this is approximately (1 - 2q) times the true gap. At q = 0.3 the factor is 0.4, so the estimated gap falls from 1.46 to 0.58 on the toy. Figure u01_f05 computes the full curve.

**Contract 6. Computed numerical example.** True gap 1.5, n = 2000, seed 0. Estimated gaps at q = 0.0, 0.1, 0.2, 0.3: 1.46, 1.11, 0.93, 0.58. Monotone shrinkage. Same numbers as figure u01_f05.

**Contract 7. Algorithm and minimal implementation.** Simulate flips with a Bernoulli(q) mask. Re-estimate the gap by log-odds. Ten lines of NumPy.

**Contract 8. Correctness checks and expected output.** Check 1: q = 0 recovers the clean estimate. Check 2: q = 0.5 gives gap 0 (pure noise). Expected: monotone decreasing estimates.

**Contract 9. Complexity, memory, statistical efficiency, stability, costs.** Simulation O(n) per q. Estimating q itself needs repeated items (same pair judged twice). Without repeats, q and the gap are confounded.

**Contract 10. Nearest alternatives and selection boundaries.** Alternative: model q as a parameter and fit it jointly (needs repeat judgments). Alternative: outlier-resistant losses that downweight surprising labels. Choose joint fitting when repeats exist. Choose outlier-resistant losses when they do not.

**Contract 11. Failure case, broken assumption, counterexample.** Adversarial noise breaks the model: if flips target close pairs, the attenuation is not uniform and the ranking itself can invert. Counterexample: q depending on the gap is not identifiable from win rates alone.

**Contract 12. Research reading and falsifiable extension.** Extension: plant gold pairs with known answers in the annotation queue and estimate q per annotator. Falsifiable: annotators with high q should show lower self-consistency on repeats.

**Contract 13. Assessment.** Breadth: write the p_obs formula. Oral ladder through the attenuation derivation. Transfer: noisy click logs in search. How do you debias? Failure diagnosis: the estimated gap is larger than the true gap. What assumption broke? Counterfactual: what if q > 0.5? Research: per-annotator noise models.

**Contract 14. Lab and exercises.** Lab U01 task 6 reproduces figure u01_f05. Exercises: (E1) derive p_obs. (E2) compute the q=0.1 estimate by hand from the formula. (E3) show the q/gap confound. Keys in answer_keys/u01_keys.md.

**Contract 15. Visual units, provenance, accessibility, audit rows.** Figure visuals/u01_f05.png: lesson plate, source original toy, seed 0, alt text "Two bars for true and estimated gap with clean labels. Arrow labeled flip labels with probability q. Decreasing curve of estimated gap versus q." Audit: before state clean estimate, after state noise curve, rule named. No conflict.

---

### cs329h-U01-C09: train/test generalization

**Contract 1. Source mapping, scope, objectives, dependencies.** Maps to S02. Objective: define generalization for choice models and measure it with held-out log-loss. Depends on C06, P10.

**Contract 2. Motivating question and tiny toy.** Question: the model fits the training comparisons well. Does it predict new comparisons? Toy: fit Bradley-Terry on 20 comparisons, test on 5000 fresh ones from the same true gap.

**Contract 3. Plain-language mental model.** Train loss measures fit to the past. Test loss measures skill on the future. With little data the two disagree. With much data they converge.

**Contract 4. Variables, units, shapes, assumptions.** Log-loss in nats per comparison. Train set n_train, test set n_test drawn from the same distribution (IID assumption). Assumption: the test pairs come from the same item distribution as training.

**Contract 5. Justified derivation or mechanism.** The MLE minimizes train loss by construction, so train loss is optimistic. Test loss is an unbiased estimate of the population risk. As n grows, both converge to the entropy of the true outcome distribution. Figure u01_f06 shows train loss below test loss at n = 20 and the curves meeting near n = 1000.

**Contract 6. Computed numerical example.** True gap 1.0, seed 0. Test log-loss at n = 20, 100, 1000: 0.627, 0.588, 0.582. Train log-loss at n = 20: 0.423, well below test. Same numbers as figure u01_f06.

**Contract 7. Algorithm and minimal implementation.** Split records by time or at random. Fit on train. Score log-loss on test. Fifteen lines reusing the C05 fitter.

**Contract 8. Correctness checks and expected output.** Check 1: train loss never exceeds test loss by more than noise on large n. Check 2: shuffling labels destroys test performance (loss near log 2 = 0.693). Expected: monotone test improvement with n.

**Contract 9. Complexity, memory, statistical efficiency, stability, costs.** Fitting cost O(n) per split. The test estimate has standard error about sigma/sqrt(n_test). Use n_test >= 1000 for stable comparisons.

**Contract 10. Nearest alternatives and selection boundaries.** Alternative: cross-validation when data is scarce. Alternative: information criteria (AIC) as an analytic proxy. Choose held-out test for final reporting. Choose CV for model selection on small data.

**Contract 11. Failure case, broken assumption, counterexample.** Distribution shift breaks the story: test pairs from new items or new annotators need not follow the training distribution. Counterexample: a model with great held-out loss on old annotators fails on a new annotator pool (preview of U02-C09 heterogeneity).

**Contract 12. Research reading and falsifiable extension.** Extension: measure generalization across annotator pools, not just across pairs. Falsifiable: if cross-pool loss exceeds within-pool loss by more than 0.05 nats, the pool shift matters.

**Contract 13. Assessment.** Breadth: define the two losses. Oral ladder through the optimism argument. Transfer: a recommender trained on US users launches in Japan. What do you measure first? Failure diagnosis: test loss rises with more data. Name two causes (shift, leakage). Counterfactual: what if test pairs reuse training items? Research: generalization bounds for pairwise models.

**Contract 14. Lab and exercises.** Lab U01 task 7 reproduces figure u01_f06. Exercises: (E1) explain the n=20 train/test gap. (E2) compute log 2 and interpret. (E3) design a leak-free split for time-ordered data. Keys in answer_keys/u01_keys.md.

**Contract 15. Visual units, provenance, accessibility, audit rows.** Figure visuals/u01_f06.png: lesson plate, source original toy, seed 0, alt text "Train log-loss curve falling with n on a log axis. Arrow labeled evaluate on held-out pairs. Test log-loss curve starting higher and converging." Audit: before state train curve, after state test curve, rule named. No conflict.

---

### cs329h-U01-C10: annotation protocol

**Contract 1. Source mapping, scope, objectives, dependencies.** PLANNED / SOURCE ATTRIBUTION PENDING. Requested extension. Objective: write a protocol that produces trustworthy pairwise data. Depends on C01-C09.

**Contract 2. Motivating question and tiny toy.** Question: what instructions and checks turn human clicks into data a model can trust? Toy: two instruction wordings for judging summaries. One produces 90% agreement on repeats, the other 65%.

**Contract 3. Plain-language mental model.** A protocol is a contract with the annotator: what to judge, how to judge it, and how you verify the work. Good protocols remove ambiguity before it becomes noise.

**Contract 4. Variables, units, shapes, assumptions.** Protocol elements: task definition, examples, gold questions with known answers, repeat rate, pay and timing, adjudication rule. Assumption: annotators follow written instructions when the instructions are concrete and checked.

**Contract 5. Justified derivation or mechanism.** Three mechanisms control quality. (i) Calibration examples anchor the scale: show five reference pairs with the intended verdicts. (ii) Gold questions estimate attention: intersperse pairs with obvious answers and track each annotator's gold accuracy. (iii) Repeats estimate noise: re-ask 10% of pairs and measure self-consistency, which bounds q from C08.

**Contract 6. Computed numerical example.** 1000 pairs, 10% repeats, 5% gold. Annotator A: gold accuracy 0.98, self-consistency 0.93. Annotator B: gold accuracy 0.80, self-consistency 0.70. Decision: keep A, retrain or drop B. Cost of quality control: 15% extra judgments.

**Contract 7. Algorithm and minimal implementation.** Pipeline: assign pairs, inject gold and repeats, compute per-annotator gold accuracy and consistency, filter below thresholds, refit. The lab implements the filter.

**Contract 8. Correctness checks and expected output.** Check 1: gold accuracy correlates with self-consistency across annotators. Check 2: dropping low-quality annotators improves held-out log-loss. Expected: a short report per annotator.

**Contract 9. Complexity, memory, statistical efficiency, stability, costs.** Overhead is linear in the gold and repeat rates. The binding cost is usually pay per judgment and reviewer time, not compute.

**Contract 10. Nearest alternatives and selection boundaries.** Alternative: expert-only annotation (high cost, low noise). Alternative: crowd with statistical filtering (low cost, needs the machinery above). Choose experts for safety-critical judgments. Choose filtered crowd for scale.

**Contract 11. Failure case, broken assumption, counterexample.** Gold questions that leak: annotators share the gold answers and game the metric. Counterexample: high gold accuracy with low self-consistency signals memorized golds, not careful work.

**Contract 12. Research reading and falsifiable extension.** Extension: test whether written rubrics beat example-only instructions on inter-annotator agreement. Falsifiable: a randomized trial with agreement as the outcome.

**Contract 13. Assessment.** Breadth: name the three quality mechanisms. Oral ladder from the toy to the trial design. Transfer: labeling medical images. What changes? Failure diagnosis: agreement is high but the model still fails. Name two causes (shared bias, wrong construct). Counterfactual: what if pay is per judgment with no quality check? Research: optimal gold rate under a fixed budget.

**Contract 14. Lab and exercises.** Lab U01 task 8 builds the annotator report. Exercises: (E1) compute the overhead rate. (E2) set a drop threshold and defend it. (E3) detect gold gaming. Keys in answer_keys/u01_keys.md.

**Contract 15. Visual units, provenance, accessibility, audit rows.** Definition and process unit. Shown as a numbered checklist in the lesson text. No state change, no plate required. Logged.

---

### cs329h-U01-C11: social meaning

**Contract 1. Source mapping, scope, objectives, dependencies.** PLANNED / SOURCE ATTRIBUTION PENDING. Requested extension. Objective: explain why a choice is not a preference and what that implies for modeling. Depends on C01-C02.

**Contract 2. Motivating question and tiny toy.** Question: an annotator picks the longer answer. Do they prefer longer answers? Toy: the same annotator picks the longer answer under time pressure but the shorter one when asked to judge carefully.

**Contract 3. Plain-language mental model.** Choices carry social meaning: politeness, effort, what the annotator thinks you want to hear. The recorded choice is a performance in a social situation, not a readout of an inner utility meter.

**Contract 4. Variables, units, shapes, assumptions.** Decompose the observed choice into preference, context, and social pressure. Assumption under test: the pressure term is zero. The lesson argues it rarely is.

**Contract 5. Justified derivation or mechanism.** Two mechanisms. (i) Demand effects: annotators infer the desired answer from the task framing and comply. (ii) Effort minimization: the easier option wins when attention is scarce. Both enter as context terms in the C02 model, but they are unrecorded, so they bias the preference estimate toward the compliant or lazy option.

**Contract 6. Computed numerical example.** True preference: shorter answers win 60% of careful judgments. Under time pressure, longer answers win 65% of judgments because they look more thorough. A model trained on the pressured data learns "longer is better," inverting the careful preference.

**Contract 7. Algorithm and minimal implementation.** No new estimator. Method: vary the social conditions (pressure, framing) across batches and test whether the fitted scores move. The lab runs the batch comparison.

**Contract 8. Correctness checks and expected output.** Check: scores fitted per batch differ by more than their standard errors. Expected: a report of score shift per condition.

**Contract 9. Complexity, memory, statistical efficiency, stability, costs.** The cost is experimental: running multiple conditions multiplies annotation spend. Analysis cost is the same O(n) fit per batch.

**Contract 10. Nearest alternatives and selection boundaries.** Alternative: treat the pressured judgments as the target (if the deployment is also pressured). The right target depends on the deployment context, not on abstract purity. Choose the condition that matches deployment.

**Contract 11. Failure case, broken assumption, counterexample.** The careful judgment may itself be artificial: nobody reads that carefully in real use. Counterexample: optimizing for careful-judgment preference can hurt the actual user experience.

**Contract 12. Research reading and falsifiable extension.** Extension: compare scores from lab-style careful annotation against scores from in-situ implicit feedback. Falsifiable: rank correlation below 0.7 means the two settings disagree materially.

**Contract 13. Assessment.** Breadth: define demand effects in one sentence. Oral ladder through the inversion example. Transfer: user satisfaction surveys. What social pressure shapes them? Failure diagnosis: two annotation vendors produce opposite rankings. How do you adjudicate? Counterfactual: what if annotators are told the true research goal? Research: measuring social pressure without changing it.

**Contract 14. Lab and exercises.** Exercises: (E1) construct the inversion numbers. (E2) design the batch comparison. (E3) argue which target matches a deployment. Keys in answer_keys/u01_keys.md.

**Contract 15. Visual units, provenance, accessibility, audit rows.** Interpretive unit. The inversion example is a worked table in the lesson text. No state change, no plate required. Logged.

---

### cs329h-U01-C12: fairness questions

**Contract 1. Source mapping, scope, objectives, dependencies.** PLANNED / SOURCE ATTRIBUTION PENDING. Requested extension. Objective: name the fairness questions a preference model must answer before deployment. Depends on C09-C11.

**Contract 2. Motivating question and tiny toy.** Question: the model is 87% accurate overall. Is it fair? Toy: two annotator groups. Accuracy 0.87 on group 1 and 0.59 on group 2. Same numbers as figure u01_f08.

**Contract 3. Plain-language mental model.** A pooled model serves the majority pattern. Fairness asks who the model serves, who it ignores, and who decided the target.

**Contract 4. Variables, units, shapes, assumptions.** Group label g, per-group accuracy, per-group calibration. Assumption under test: one score vector fits all groups. U02-C09 relaxes it.

**Contract 5. Justified derivation or mechanism.** Three questions with teeth. (i) Representation: whose judgments trained the model? (ii) Performance: does accuracy or calibration differ by group? (iii) Target: whose preference should the deployed system reflect? Figure u01_f08 shows (ii): pooled accuracy 0.73 hides 0.87 versus 0.59.

**Contract 6. Computed numerical example.** Same as figure u01_f08: overall 0.73, group 1 0.87, group 2 0.59, n = 600 per group, seed 0. The gap is 0.28, far above noise (standard error about 0.02).

**Contract 7. Algorithm and minimal implementation.** Split the test set by group, score per group, report the gap with a confidence interval. Ten lines of NumPy.

**Contract 8. Correctness checks and expected output.** Check: the pooled accuracy lies between the group accuracies. Check: the gap persists across random splits. Expected: a per-group report card.

**Contract 9. Complexity, memory, statistical efficiency, stability, costs.** O(n) scoring. The binding constraint is group labels, which may be unavailable or sensitive to collect.

**Contract 10. Nearest alternatives and selection boundaries.** Alternative: per-group models (better fit, fragmented). Alternative: one model with group-aware calibration (shared ranking, adjusted confidence). Choose per-group when groups genuinely disagree. Choose shared-plus-calibration when the ranking transfers but the noise differs.

**Contract 11. Failure case, broken assumption, counterexample.** Collecting group labels can itself harm: privacy risk and misuse. Counterexample: a fairness audit that publishes per-group error rates can leak group membership.

**Contract 12. Research reading and falsifiable extension.** Extension: test whether the group gap comes from noise (different q) or from genuine preference disagreement (different s). Falsifiable: fit per-group scores. If the score vectors agree up to noise, the gap is noise, not disagreement.

**Contract 13. Assessment.** Breadth: state the three questions. Oral ladder through the figure. Transfer: a hiring tool trained on past decisions. Which question bites first? Failure diagnosis: per-group models overfit small groups. What now? Counterfactual: what if group labels are banned? Research: fairness without demographics.

**Contract 14. Lab and exercises.** Lab U01 task 9 builds the per-group report. Exercises: (E1) compute the gap standard error. (E2) distinguish noise from disagreement. (E3) argue the privacy tradeoff. Keys in answer_keys/u01_keys.md.

**Contract 15. Visual units, provenance, accessibility, audit rows.** Figure visuals/u01_f08.png: lesson plate, source original toy, seed 0, alt text "One bar for pooled accuracy 0.73. Arrow labeled split by annotator group. Two bars 0.87 and 0.59." Audit: before state pooled number, after state group numbers, rule named. No conflict.

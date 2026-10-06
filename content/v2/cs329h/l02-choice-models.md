---
page_id: cs329h-l02
course_slug: cs329h
course_name: "CS329H: Machine Learning from Human Preferences"
course_order: 6
order: 2
nav: "L02 · Choice Models"
title: "Lecture 2: Bradley-Terry, Plackett-Luce, and the Preference Pair"
summary: "Random utility models, Gumbel noise and the logit, Bradley-Terry for pairs, Plackett-Luce for rankings, and the canonical preference-pair symbol."
date: "[uncertain] Spring 2026"
instructor: "Sanmi Koyejo"
offering: "Spring 2026"
duration: "1:19:20"
video_id: _bych9RfQvw
video_title: "Stanford CS329H Lecture 2: Choice Models"
video_caption: "Original lecture. Sanmi Koyejo covers choice models as the technical foundation of the human preference learning pipeline."
concepts: [bradley-terry, plackett-luce, random-utility, gumbel, logit, softmax, preference-pair, elo]
sources:
  - tag: video
    label: "Lecture 2 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=_bych9RfQvw
  - tag: notes
    label: "Official subtitle transcript (en-US)"
  - tag: notes
    label: "Course textbook, chapter 1.7 (Truong, Haupt, Koyejo, 2025)"
  - tag: paper
    label: "Bradley and Terry, Rank Analysis of Incomplete Block Designs (1952)"
---
## The problem: preferences jitter

Lecture 1 gave each item a fixed utility V_j. Ask an annotator to
compare the same two responses on Monday and on Friday. Sometimes
the answer flips. Monday: A beats B. Friday: B beats A. Nothing
about the responses changed. The person did.

A fixed utility predicts the same choice every time. Real data
shows reversals. Three readings of the noise exist, and the
textbook names all three. It can be population heterogeneity:
different decision-makers. It can be decision error: bounded
rationality, the person is noisy. It can be the designer's belief
about unknown preferences: the model is uncertain. The reading
decides what the data licenses. A model that cannot represent
jitter cannot fit, cannot report confidence, and cannot explain
disagreement. We need utility with noise built in.

## First attempt: fixed utilities, pick the max

Before the noise, see what breaks without it. Give each item a
fixed number. The choice rule: pick the item with the highest
number. Three items: V_A = 2.0, V_B = 1.0, V_C = 0.0.

The rule predicts: A always beats B. B always beats C. A always
beats C. Deterministic. Now watch it fail on three counts.

First, it cannot explain a reversal. If the annotator picks B over
A once in ten trials, the model says that trial was impossible.
Second, it reports no confidence. It treats "A barely beats B"
and "A crushes B" identically: both are just "A wins." Third, it
gives no likelihood. With no probabilities, there is nothing to
maximize when fitting. Lecture 4 would have no loss function.

The toy gap: V_A - V_B = 1.0 and V_B - V_C = 1.0. The model
cannot say the first gap is as strong as the second in any graded
way. It only says "wins." We need the gap to mean something
quantitative.

## The key question

What if the utility itself is random, so the same gap produces
the same win probability every time, and close contests come out
near 50/50?

## Random utility: add noise to the number

The **random utility model** writes:

H_j = V_j + noise_j

V_j is the mean utility of item j. noise_j is a random shock,
independent across items, drawn fresh on each choice. The choice
goes to the highest realized value H_j. Repeat the draw many
times. The win fractions settle into probabilities.

A concrete instance. Two chess players. Mean strengths V_A = 2.0,
V_B = 1.0. On any given day, each player's realized strength is
their mean plus a shock: form, luck, the opening. The stronger
player wins most days but not every day. The model now predicts a
number, not a verdict.

## Gumbel noise gives the softmax

The noise distribution decides the model. Choose independent
**Gumbel** noise and something remarkable happens. The choice
probabilities take a closed form, the **softmax**:

p(j chosen from set S) = e^{V_j} / sum_{k in S} e^{V_k}

The Gumbel distribution is the unique noise, up to regularity
conditions, that yields this form and the IIA property of Lecture
3. Other noises give other models: Gaussian noise gives the probit
(see the end of this chapter). Gumbel wins on tractability:
closed-form probabilities and easy gradients.

Watch the softmax on the toy. V = (2.0, 1.0, 0.0).

```ascii
e^2.0 = 7.39,  e^1.0 = 2.72,  e^0.0 = 1.00
total = 7.39 + 2.72 + 1.00 = 11.11

p(A) = 7.39 / 11.11 = 0.665
p(B) = 2.72 / 11.11 = 0.245
p(C) = 1.00 / 11.11 = 0.090
```

A wins two thirds of the time. B wins a quarter. C wins rarely.
The gaps now mean something quantitative: a one-unit gap is worth
about 0.42 of win probability at these values.

![Gumbel noise produces the softmax](assets/l02-gumbel.svg "Mean utilities plus independent Gumbel noise. The winner of each draw is chosen. Win fractions converge to the softmax. Source: original figure for Stanford Frontier AI.")

One property you will use constantly. Adding the same constant to
every V_j changes nothing. e^{V_j + c} = e^c e^{V_j}, and the e^c
cancels top and bottom. Only differences matter. This is the
identification problem of Lecture 3, previewed early because you
will trip on it otherwise.

> [!QA]
> Q: Why model utilities as random instead of just adding noise to responses?
> A: Both views give the same pairwise probabilities, so the choice is conceptual. The random utility view is the economics tradition: the evaluation itself fluctuates with mood, context, and unmeasured factors. It also unlocks the theory: with the right noise distribution, choice probabilities take a closed form, and that closed form is the softmax.
> Follow-up: What goes wrong with a purely deterministic model?
> A: It predicts the same choice every time. Real data shows reversals: the same annotator picks A over B on Monday and B over A on Friday. Without a noise model there is no likelihood, no fitting, and no way to say how confident the ranking is.

## Bradley-Terry: the two-item case

Take the softmax with two items. It reduces to a **sigmoid** of
the gap. This is the **Bradley-Terry model** (Bradley and Terry,
1952), the most used equation in this course.

p(j beats k) = sigma(V_j - V_k) = 1 / (1 + e^{-(V_j - V_k)})

The win probability depends only on the utility gap. Work the
numbers by hand.

```ascii
gap 0.0:  sigma(0) = 1 / (1 + 1)      = 0.500   (coin flip)
gap 1.0:  sigma(1) = 1 / (1 + 0.368)  = 0.731
gap 2.0:  sigma(2) = 1 / (1 + 0.135)  = 0.881
gap 3.0:  sigma(3) = 1 / (1 + 0.050)  = 0.953
```

Equal gap, equal probability, every time. Now the chess toy from
the textbook. V_A = 2.0, V_B = 1.0, V_C = 0.0.

p(A beats B) = sigma(1.0) = 0.731
p(A beats C) = sigma(2.0) = 0.881
p(B beats C) = sigma(1.0) = 0.731

The strongest beats the middle 73% of the time and the weakest
88%. Elo ratings are Bradley-Terry utilities on a scaled axis:
Elo's expected score is exactly this sigmoid.

![The sigmoid turns gaps into probabilities](assets/l02-sigmoid.svg "p(j > k) = sigma(V_j - V_k). Worked points: gap 1.0 gives 0.73, gap 2.0 gives 0.88. Source: original figure for Stanford Frontier AI.")

> [!QA]
> Q: Derive Bradley-Terry from the random utility model in one paragraph.
> A: Start with H_j = V_j + noise_j with i.i.d. Gumbel noise. Item j beats k when H_j > H_k. Condition on noise_j = t: then k loses when noise_k < V_j - V_k + t, which has Gumbel CDF probability. Integrate over t with the Gumbel density. The integral collapses to e^{V_j} / (e^{V_j} + e^{V_k}), which equals sigma(V_j - V_k). The full derivation is an exercise in the textbook.
> Follow-up: Why is this the same math as logistic regression?
> A: It is logistic regression. The log-loss on a pair (j, k) with label y is -[y log sigma(V_j - V_k) + (1-y) log(1 - sigma(V_j - V_k))]. Fitting Bradley-Terry by maximum likelihood is exactly logistic regression on pair differences. Lecture 4 builds on this.

## The preference pair: the atom of the course

One object powers everything downstream. Define it carefully now,
because every later lesson reuses it.

![The preference pair](assets/l02-pref-pair.svg "Prompt x, chosen response y_w, rejected response y_l. The update widens the reward gap. Defined in CS329H, reused by later courses. Source: original figure for Stanford Frontier AI.")

A **preference pair** is a triple (x, y_w, y_l). The prompt x. The
chosen response y_w, the winner. The rejected response y_l, the
loser. A human, or a model judge, said y_w beats y_l for this
prompt.

The update rule for every method in this course: widen the gap
between the winner and the loser. Reward models raise r(x, y_w)
and lower r(x, y_l). DPO raises the implicit reward gap directly.
PPO optimizes a policy against the learned gap. Same atom, three
algorithms. This symbol is first defined here in CS329H. Later
courses reuse it without redefinition.

## Plackett-Luce: rankings in stages

Pairs handle two items. Rankings need more. The **Plackett-Luce**
model builds a ranking in stages: pick the winner from the full
set, then the winner from the rest, and so on.

p(A > B > C) = [e^{V_A} / (e^{V_A} + e^{V_B} + e^{V_C})] x [e^{V_B} / (e^{V_B} + e^{V_C})]

Work it with V = (2, 1, 0), using the exponentials from above.

```ascii
stage 1, A wins the full set:  7.39 / 11.11 = 0.665
stage 2, B beats C of the rest: 2.72 / (2.72 + 1.00) = 2.72 / 3.72 = 0.731
full ranking A > B > C:  0.665 x 0.731 = 0.486
```

The most likely ranking puts A first, about 49% of the time. Each
stage is a softmax over the survivors. Binary Bradley-Terry is the
two-item special case. Accept-reject, item versus the outside
option, is the item-versus-one special case: sigma(V_j - V_0).
One model, four feedback types. The table at the end of the
chapter consolidates them.

![Plackett-Luce builds rankings in stages](assets/l02-pl.svg "Stage 1 picks the winner, stage 2 picks the winner of the rest. Worked: 0.665 x 0.731 = 0.486. Source: original figure for Stanford Frontier AI.")

> [!QA]
> Q: When would you use Plackett-Luce instead of Bradley-Terry?
> A: When annotators give full or partial rankings, not just pairs. A ranking of M items carries M-1 staged choices, more information per query than one pair. But rankings cost more annotator effort and the staged model assumes the same IIA structure at every stage. Most LLM pipelines stick to pairs: cheaper, faster, and BT is the special case anyway.
> Follow-up: Does the stage order matter?
> A: The model defines the ranking probability as the product over stages in rank order: first place, then second, and so on. Any total order has exactly one such factorization, so the probability is well defined. The stages are a modeling choice, not a claim about how humans actually rank.

## Where the noise model still fails

Three failures motivate the rest of the course. Each gets its own
lecture. Name them now so you can spot them.

**Context effects.** Preferences depend on what else is shown.
Adding a decoy changes choices. BT assumes the pair probability
never moves when the set changes. Lecture 3.

**Intransitivity.** A > B > C > A cycles exist in real data. No
utility vector explains a cycle, because numbers are transitive
and the data is not. Lectures 3 and 8.

**Heterogeneity.** Different annotators have different utilities.
BT learns a compromise that may match nobody. Lectures 3 and 8
through 10.

DPO inherits all three. It assumes the BT model is correctly
specified. When the assumption fails, DPO learns a compromise
policy. The textbook is explicit: check the assumption before
trusting the fit.

## Mapping back: what random utility buys

| Fixed-utility pain | Random-utility answer | How |
|---|---|---|
| Reversals are impossible | Reversals get a probability | sigma(1.0) = 0.731, so B beats A 27% of the time |
| No confidence | Gaps are graded | Gap 1.0 vs 2.0: 0.73 vs 0.88 |
| Nothing to fit | A likelihood exists | Log-loss on pairs; Lecture 4 maximizes it |

## The honest price: IIA

The softmax has a strong property hiding inside it. The ratio of
two choice probabilities ignores everything else in the set. Add
a third option and the j-versus-k odds do not move. This is the
**Independence of Irrelevant Alternatives**, IIA. The textbook
proves the equivalence: a random utility model satisfies IIA if
and only if the noise is i.i.d. Gumbel. Gumbel noise and IIA are
two names for the same assumption.

IIA is what makes learning tractable: M numbers determine every
choice probability. It is also the first thing reality breaks.
Clone an option and watch the odds move. That is Lecture 3, and
it starts with buses.

## Consolidation: one model, four feedback types

| Feedback type | Model name | Formula |
|---|---|---|
| Binary pair | Bradley-Terry | sigma(V_j - V_k) |
| Full ranking | Plackett-Luce | staged softmax |
| Accept / reject | Logistic regression | sigma(V_j - V_0) |
| Choice from set | Multinomial logit | softmax over S |

## Logit versus probit

Gumbel noise gives the logit. Gaussian noise gives the **probit**.
For binary comparisons the two are nearly identical after
rescaling: the logistic and normal CDFs have similar shapes. The
choice is empirically inconsequential for pairs.

The difference matters for multi-way choices. Gaussian noise
allows a covariance matrix over items, so similar items can have
correlated shocks. That correlation is exactly what the
red-bus/blue-bus problem needs, covered in Lecture 3. The price is
computation: probit choice probabilities have no closed form and
need numerical integration.

Rule of thumb: pairs, use logit. Many similar alternatives,
consider probit or nested logit.

> [!QA]
> Q: What is the Gumbel distribution doing in this model?
> A: It is the noise distribution that makes the math close. With i.i.d. Gumbel shocks, the probability that item j has the highest realized utility equals the softmax of the mean utilities. The textbook proves this by conditioning on one noise draw and integrating. No other common noise gives a closed form this clean.
> Follow-up: Is Gumbel a realistic model of human noise?
> A: It is a convenience, not a psychological fact. Gaussian noise, the probit model, is arguably as plausible and gives nearly identical binary predictions. Gumbel wins on tractability: closed-form probabilities, easy gradients, and the IIA structure that makes learning scale.

## Recap: the whole lesson on one screen

The story in eight steps. Each step answers the one before it.

1. **Preferences jitter.** Same annotator, Monday A beats B,
   Friday B beats A. Fixed utilities call the reversal
   impossible.
2. **Fixed max breaks three ways.** No reversals, no
   confidence, no likelihood to fit.
3. **Add noise to the utility.** H_j = V_j + noise_j. The
   choice goes to the highest realized value. Win fractions
   settle into probabilities.
4. **Gumbel gives the softmax.** p(j) = e^{V_j} / sum e^{V_k}.
   The toy: 0.665, 0.245, 0.090. Only differences matter.
5. **Two items give Bradley-Terry.** p(j beats k) =
   sigma(V_j - V_k). Gap 1.0 is 0.731. Gap 2.0 is 0.881.
   Chess toy: A beats B 73%, A beats C 88%.
6. **The preference pair is the atom.** (x, y_w, y_l).
   Every method widens the winner-loser gap. Defined here,
   reused everywhere.
7. **Rankings stage the softmax.** Plackett-Luce: 0.665 x
   0.731 = 0.486 for A > B > C. One model, four feedback
   types.
8. **The price is IIA.** The odds ignore the rest of the set.
   Tractable and strong. Clones break it. Lecture 3 shows how.

## Official sources and further reading

**Official:**
- Lecture 2 video (choice models): video id _bych9RfQvw. The
  lecture goal, stated at [00:08](ts:00:08): by the end you
  should have the technical tools to understand the human
  preference learning parts of modern pipelines.
- Course textbook, chapters 2.x: random utility, Gumbel, BT,
  Plackett-Luce, the worked chess example.

**Further reading:**
- Bradley and Terry (1952): the original paired-comparison
  paper.
- Train (2009), Discrete Choice Methods with Simulation: the
  reference for logit, probit, nested and mixed logit.

**Caveats from these sources.** The Gumbel-IIA equivalence holds
under regularity conditions the textbook states; the proof is
left as an exercise. The "unique noise" claim is up to those
conditions. The DPO-inherits-BT warning is the textbook's
explicit caution, not a proven failure rate.

## Connections to the other courses

- **CS329H L01:** fixed utilities and the Rasch model; pairs
  cancel the user. This lesson adds the noise.
- **CS329H L03:** IIA, identification, and where BT fails.
  Read next.
- **CS329H L04:** fitting BT by maximum likelihood, which is
  logistic regression on pair differences.
- **CS329H L07:** DPO as BT with the reward parameterized
  through the policy.
- **CS224N:** the same BT/DPO math from the language-modeling
  side.

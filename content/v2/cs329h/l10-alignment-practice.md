---
page_id: cs329h-l10
course_slug: cs329h
course_name: "CS329H: Machine Learning from Human Preferences"
course_order: 6
order: 10
nav: "L10 · Alignment in Practice"
title: "Lecture 10: Alignment in Practice, from Theory to Deployment"
summary: "The inversion problem, the four-stage fairness pipeline, value alignment, Polis and Community Notes, human-centered design, and running human experiments."
date: "[uncertain] Autumn 2024"
instructor: "Sanmi Koyejo"
offering: "[uncertain]"
duration: "[uncertain]"
video_id: VffFArrRSBE
video_title: "Stanford CS329H: Machine Learning from Human Preferences | Autumn 2024 | Human-centered Design"
video_caption: "Course lecture (Stanford Online, Autumn 2024). Human-centered design for preference systems. Verified live on YouTube."
concepts: [inversion-problem, fairness-pipeline, value-alignment, polis, community-notes, human-centered-design, paternalism, mutual-information]
sources:
  - tag: video
    label: "CS329H Autumn 2024: Human-centered Design (Stanford Online)"
    url: https://www.youtube.com/watch?v=VffFArrRSBE
  - tag: video
    label: "CS329H Autumn 2024: Voting, with Colin Megill (Polis) guest segment"
    url: https://www.youtube.com/watch?v=1QpNZXL35NM
  - tag: video
    label: "CS329H Autumn 2024: Ethics (Stanford Online, Daniel Webber guest)"
    url: https://www.youtube.com/watch?v=-kdR_7dCcyI
  - tag: video
    label: "CS329H guest lecture: Joseph Jay Williams [uncertain: title evidence from search index only]"
    url: https://www.youtube.com/watch?v=HFrCySzH9QI
  - tag: notes
    label: "Course textbook, chapters 4 and 5.6-5.11 (Truong, Haupt, Koyejo, 2025)"
    url: https://mlhp.stanford.edu/Machine-Learning-from-Human-Preferences.pdf
---
## The problem: the lab meets the world

Lectures 1 through 9 built the machinery: comparisons, choice
models, fitting, asking, acting, aggregating, incentives. This
chapter asks what the machinery is for and where it breaks
outside the lab. Whose preferences count? How does fairness fail?
How do deployed systems aggregate millions of voices? The
answers are less mathematical and more load-bearing than
everything before.

Carry one running example. A team builds an AI assistant for
peer review. It reads papers, suggests scores, drafts reviews.
Whose judgment does it encode? The senior faculty who labeled
the most data? The junior researchers who will live with its
decisions? The machinery from earlier lectures gives no answer.
It only executes the values smuggled into each stage.

## First attempt: invert behavior into preference

Every lesson so far observed behavior: clicks, choices, labels.
The goal was never behavior. The goal is preferences: what
people want, so the system can serve it. The gap between
observed behavior and true preference is the **inversion
problem**.

![The inversion problem](assets/l10-inversion.svg "Behavior is observed. Preferences are inferred. Errors, constraints, and strategy break the inversion. Source: original figure for Stanford Frontier AI.")

Inversion fails three ways. **Errors:** bounded rationality
means choices reflect mistakes, not wants. **Constraints:**
people choose from what is available, not what they want.
**Strategy:** annotators and users respond to the system
measuring them, Lecture 6's CIRL and Lecture 9's mechanisms.

The concrete failure: click data in recommenders. Clicks
reflect position bias, curiosity gaps, and outrage as much as
satisfaction. A model trained to maximize predicted clicks
learns clickbait, not quality. The inversion assumed clicks
reveal preference. They reveal engagement, which is a different
thing. Fixing it means modeling the bias in Lecture 4 or
changing the feedback in Lecture 5.

> [!QA]
> Q: What is the inversion problem?
> A: Observed choices do not equal underlying preferences. People err, face constrained options, and strategize against measurement. Inverting behavior to preferences requires assumptions about all three, and the assumptions are value choices. A system that optimizes learned "preferences" without auditing the inversion optimizes a distorted image of what people want.
> Follow-up: Give a concrete inversion failure.
> A: Click data in recommenders. Clicks reflect position bias, curiosity gaps, and outrage as much as satisfaction. A model trained to maximize predicted clicks learns clickbait, not quality. The inversion assumed clicks reveal preference. They reveal engagement, which is a different thing. Fixing it means modeling the bias or changing the feedback.

## The key question

If every technical choice smuggles in an answer to "whose
preferences matter," can we at least make the smuggling
explicit, stage by stage?

## The four-stage pipeline

The textbook's pipeline makes the values explicit. Trace the
peer-review assistant through four stages.

![The four-stage fairness pipeline](assets/l10-pipeline.svg "Elicitation, learning, aggregation, decision. Each stage embeds a value choice. Bias compounds across stages. Source: original figure for Stanford Frontier AI.")

1. **Elicitation.** Who gets asked? Querying the most
   "productive" reviewers, or the highest Fisher information
   from Lecture 5, oversamples senior faculty in mainstream
   areas. Efficiency has a demographic.
2. **Learning.** What counts as valid? Fitting Bradley-Terry
   assumes context-free preferences and discards the
   context-dependence of overburdened reviewers as noise. The
   model decides which variation is signal.
3. **Aggregation.** Whose preferences weigh more? Simple
   majority across annotators weights the overrepresented
   group equally per person, which is unequal per
   perspective. Lecture 8's impossibility lives here.
4. **Decision.** Defer or override? When the model is
   uncertain, does the system defer to a human or decide
   anyway? When is paternalism legitimate?

The compounding is the point. A 10% sampling bias at
elicitation can become a 40% performance gap after one
feedback loop: the model serves seniors well, seniors use it
more, their data dominates further. Each stage's small choice
multiplies. Interventions: stratified sampling at elicitation,
fairness-constrained learning, regular auditing at decision.

## Two fairnesses that cannot both hold

**Individual fairness** (Dwork et al., 2012): similar
individuals get similar outcomes. Formally, close features imply
close decisions. **Group fairness**: protected groups get equal
average outcomes. Formally, expected decisions match across
groups.

![Individual versus group fairness](assets/l10-fairness.svg "Similar treatment versus equal averages. In peer review they conflict: small subfields have few experts. Source: original figure for Stanford Frontier AI.")

The textbook proves by example they conflict. Peer review:
papers in small subfields have few available experts.
Individual fairness gives them less-expert reviewers: similar
papers get similar reviewers, and the similar reviewers are
scarce. Group fairness demands equally expert reviewers on
average across subfields. Both cannot hold. Satisfying group
fairness means giving small-subfield papers reviewers that
individual fairness would assign elsewhere.

The lesson generalizes: state which fairness you target, measure
it, and name what you sacrificed. "Fair" without a definition is
a slogan.

> [!QA]
> Q: Why cannot individual and group fairness both hold?
> A: They demand different things when feature distributions differ across groups. Individual fairness is local: treat similar cases similarly. Group fairness is aggregate: equalize group averages. When one group's cases systematically differ, small subfields with few experts, local similarity and aggregate equality pull in opposite directions. Dwork et al. proved the incompatibility formally. The peer review example shows it concretely.
> Follow-up: Which should a preference learning system target?
> A: It depends on the harm model. If the harm is disparate service quality across user groups, target group fairness on the outcome that matters: satisfaction, accuracy per group. If the harm is arbitrary treatment of individuals, target individual fairness with a defensible similarity metric. Most deployed systems need both partially: constrain one, optimize the other, and audit the tradeoff.

## Revealed versus informed preferences

Daniel Webber, post-doctoral researcher at Stanford University
(per the lecture video's description), gave the value alignment
guest lecture. His framing: alignment is about which preferences
count, not just how to learn them.

The key distinction is revealed versus informed preferences.
**Revealed preferences** are what behavior shows, discussed in
the Webber guest lecture (video id -kdR_7dCcyI).
**Informed preferences** are
what someone would want knowing the relevant facts, with time
to reflect. Alignment to raw revealed preferences bakes in
mistakes, ignorance, and manipulation. Alignment to informed
preferences requires extrapolation: from the cases you observed
to the cases the person never considered.

Webber also introduced the ESR statements, ethical and societal
reflection, required for the course's final projects: every
project must state its value assumptions explicitly. The habit
transfers. Any preference learning system should ship with its
inversion assumptions written down.

<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;max-width:100%;margin:16px 0;">
<iframe style="position:absolute;top:0;left:0;width:100%;height:100%;" src="https://www.youtube-nocookie.com/embed/-kdR_7dCcyI" title="CS329H Autumn 2024: Ethics, Daniel Webber guest lecture" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

> [!QA]
> Q: What is the difference between revealed and informed preferences?
> A: Revealed preferences are what choices show. Informed preferences are what someone would want with full information and reflection time. They differ whenever people err, lack facts, or face manipulation. Aligning to revealed preferences reproduces the errors. Aligning to informed preferences requires extrapolating beyond the data, which is itself a value-laden choice about how far to extrapolate and in which direction.
> Follow-up: Is not "informed preference" just the designer's preference in disguise?
> A: That is the central risk. Extrapolation needs a model of what the person would endorse, and the designer supplies it. The defense is transparency: state the extrapolation rule, make it contestable, and prefer the weakest extrapolation that resolves the clear errors. Webber's ESR requirement is the institutional version: write the assumption down so it can be argued with.

## Polis: bridging, not majority

Colin Megill, founder of Polis, gave the applied guest lecture.
Polis is an open-source platform for large-scale deliberation.
Its most famous descendant is Community Notes on X.

The problem with majority voting on notes: the largest
ideological group decides what is "helpful." The Polis answer is
**bridging**: find notes rated positively by people who disagree
with each other. Disagreement is the instrument, not the
obstacle.

The model is a factor model:

u = mu + alpha_j + beta_j + p^T q_j + noise

alpha is rater bias: some raters are generous. p^T q captures
ideological agreement between rater and note. beta_j is note
quality after ideology is removed. A note is selected when
beta_j clears a threshold: it must appeal across the divide,
not just within one camp.

Two details from the lecture. First, the goal is recovering
consensus signal from diversity of opinion. Second, the
factorization runs in continuous space, deliberately not
clustering raters into liberal/conservative buckets, to avoid
reifying the divisions it measures.

This is Lecture 1's factor model deployed at scale, and Lecture
8's social choice with a concrete rule: bridging beats majority
when the population is polarized.

### Subchapter: Community Notes, bridging at planetary scale

Polis is the prototype. Community Notes on X is the deployment:
millions of raters, millions of notes, one ranking question.
Which note gets shown under a post?

The mechanism is the same factorization, run at planetary
scale. A rating from rater i on note n decomposes as:

```ascii
r = mu + alpha_i + beta_n + p_i^T q_n + noise
```

mu is the global mean. alpha_i is the rater's baseline
generosity. beta_n is the note's bridging helpfulness. p_i and
q_n are latent ideology factors. The selection rule uses only
beta_n: a note is shown when its bridging helpfulness clears
the bar. In words: the note must be rated helpful by raters
from different sides of the latent disagreement axis, not just
by one faction. [uncertain: exact threshold and
regularization are implementation details not verified here.]

![Bridging selects notes both sides rate helpful](assets/plate-bridging.webp "Two rater clusters disagree on everything except one note. That note has high beta and gets shown. Shell 3. Source: original figure for bridging aggregation. Project: Stanford Frontier AI.")

Why this is the whole course in miniature. Lectures 1 to 3: a
choice model over ratings. Lecture 4: the fitted beta is a
latent reward. Lecture 5: which notes get rated is an
elicitation design. Lecture 8: aggregation without
majoritarianism, the bridge not the mob. Lecture 9: raters
strategize, and the factorization is the incentive-robust part,
because gaming your own alpha_i does not move the note's
beta_n.

The failure mode is structural. Bridging rewards
cross-faction agreement, so topics with no organized
disagreement generate little signal, and novel claims that one
side has not rated yet sit in limbo. Bridging is conservative
by construction: it surfaces what both sides already accept,
not what is true. That is the honest price of the mechanism.

## Human-centered design: the interface is the model

The human-centered design lecture, built on tutorials from the
HCI community, argues the elicitation interface is part of the
model. Understand users, prototype cheaply, test with humans,
iterate. For preference learning specifically: the question
format shapes the data. Pairwise versus ranked, forced choice
versus skip allowed, context shown versus hidden. Each choice
selects a different preference distribution.

![Human-centered design for preference systems](assets/l10-hcd.svg "Understand, prototype, iterate. The interface is part of the model. Source: original figure for Stanford Frontier AI.")

The connection to the pipeline: HCD operates at the elicitation
stage, where the fairness pipeline says values enter first.
Designing the question is designing the data. A "which is
better" button and a five-star widget do not measure the same
preference.

## Running experiments on humans

The final lecture covers running preference experiments with
humans: the applied setting where theory meets logistics. Its
distinctive content is Thompson sampling for experiments. When
comparing two treatments, which prompt elicits better
responses, allocate subjects adaptively. Early results suggest A
beats B 70/30. Thompson sampling assigns the next subject to A
with probability 0.7. Exploration and exploitation over human
subjects, minimizing regret while learning. Lecture 6's
algorithm, Lecture 10's stakes.

The ethics need stating. Adaptive allocation treats humans as
bandit arms. Subjects in the worse arm get worse treatment by
design. Human experiments need the statistical care of Lecture 6
plus the ethical care of Webber's lecture: consent, debriefing,
and a harm model for the exploration itself.

<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;max-width:100%;margin:16px 0;">
<iframe style="position:absolute;top:0;left:0;width:100%;height:100%;" src="https://www.youtube-nocookie.com/embed/HFrCySzH9QI" title="CS329H guest lecture: Joseph Jay Williams [uncertain: title from search-index evidence]" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## Closing topics: paternalism, privacy, peer prediction

**Paternalism.** When should the system override stated
preferences? The review-assistant example: help the harsh
reviewer be harsh, liberal assistance, or nudge toward
constructive feedback, illiberal assistance? Paternalism is the
decision stage of the pipeline. The textbook's stance: name the
override rule and its justification. Hidden paternalism is
manipulation.

**Privacy and personalization.** Personalization needs user
models. User models need data. The tension is fundamental: the
factor models of Lecture 1 identify individuals to serve them.
Aggregation and anonymization protect privacy at the cost of
personalization quality.

**Mutual information paradigm.** For peer prediction, eliciting
honest reports without ground truth, reward agents by the mutual
information between their report and others' reports.
Truth-telling maximizes shared information under the paradigm's
conditions. It is mechanism design from Lecture 9 for the
no-ground-truth case: pay for agreement that could only come
from shared signal.

### Subchapter: peer prediction, worked

No gold labels exist. Two labelers mark items positive or
negative. Pay each labeler by agreement with the other: a
bonus whenever their labels match. Does honesty pay? Work the
numbers. The true positive rate is 0.7. An honest labeler
reports the truth with 90% accuracy, errors independent.

If both report honestly, agreement needs both right or both
wrong: 0.9^2 + 0.1^2 = 0.82. Now the lazy strategy: always
report positive, no effort. The peer, still honest, reports
positive with probability 0.7 x 0.9 + 0.3 x 0.1 = 0.66. The
lazy labeler agrees 66% of the time. Honesty pays 0.82,
laziness pays 0.66. Output agreement beats the lazy
equilibrium.

But watch the collusion equilibrium. If both labelers go lazy
and always report positive, agreement is 1.0. Output agreement
pays perfect scores for coordinated laziness. This is where
the mutual information paradigm earns its name. Score by the
mutual information between the two reports instead of raw
agreement. Honest reports are correlated through the shared
truth: their joint distribution is (0.57, 0.09, 0.09, 0.25)
for (+,+) through (-,-), and the mutual information is 0.18
nats, computed in natural log. Constant reports carry zero
mutual information: a report that never varies shares nothing
with anything. The lazy-lazy equilibrium scores 0 under MI and
1.0 under raw agreement. MI kills exactly the equilibrium that
raw agreement rewards.

The precondition, stated honestly: the paradigm needs the
honest equilibrium to be the best one, which holds when
labelers' signals are genuinely correlated through the truth
and the MI estimator has enough data. With few shared items
the MI estimate is noise, and the mechanism misfires. Peer
prediction is Lecture 9's incentive compatibility for the
case where the designer has no ground truth to check against.

> [!QA]
> Q: Walk me through the Polis factor model on a toy rating matrix, by hand.
> A: Three raters, two notes. Rater 1 (left) rates note A 5 stars, note B 1 star. Rater 2 (right) rates note A 1 star, note B 5 stars. Rater 3 (left) rates note A 4 stars, note B 2 stars. Fit the model u = mu + alpha + beta + p^T q. mu, the global mean: (5+1+1+5+4+2)/6 = 3. Rater 1's alpha: his mean is 3, so alpha_1 = 0. Rater 2's mean is 3, alpha_2 = 0. Rater 3's mean is 3, alpha_3 = 0. Ideology: raters 1 and 3 lean left, rater 2 leans right. Note A's beta: after removing rater means and the ideology component, note A is loved by the left and hated by the right: high ideological loading, low beta. Note B: loved by the right, low beta too. Now add note C: all three raters give it 4 stars. Its beta is high, near 1 after centering, and its ideology loading is near zero. Note C is the bridge: selected, shown, trusted. Notes A and B are factional: suppressed. That is the whole mechanism in six numbers.
> Follow-up: What if rater 1 is just generous, rating everything 5?
> A: Then alpha_1 absorbs it: his mean is 5, alpha_1 = +2, and his ratings contribute no ideology or quality signal beyond the mean shift. The factorization separates generosity from belief. This is why the model beats raw averages: raw averages let the loudest and most generous raters set the outcome.

> [!QA]
> Q: Design a value audit for a peer-review assistant. It drafts reviews for conference submissions.
> A: Audit the four pipeline stages from the lesson. Elicitation: whose reviewing values are in the training pairs? If all pairs come from one subfield, the model encodes one subfield's taste. Sample pairs across subfields and seniority. Learning: check the fitted reward against held-out reviewer judgments, sliced by subfield, and publish the slices. Aggregation: the model aggregates its training population's preferences. If the population is 80% from two labs, say so in the model card. Decision: the deployment threshold, when a draft review ships without human edit, is the value choice. Set it where measured draft quality beats the human baseline on blind comparison, not where the demo looks good. Add the inversion check: run the pipeline backwards from model outputs to the implied reviewer values and ask whether any real reviewer would endorse them.
> Follow-up: What is the single most likely failure?
> A: Revealed-versus-informed drift. Reviewers in a hurry reveal a preference for short, confident drafts. Their informed preference is for careful, hedged reviews. The model trained on revealed pairs becomes a confident-sounding review generator. Catch it with a free-text "why" field on a sample of labels: if annotators say they prefer the draft because it was fast to read, the label is measuring the wrong thing.

> [!QA]
> Q: When does bridging fail? Give the failure taxonomy.
> A: Four failures. First, no disagreement, no signal: on topics where one side has not shown up, there is nothing to bridge, and notes sit unrated. Second, false balance: bridging treats both sides symmetrically by construction, so on settled questions it can elevate a "both sides" note over the correct one. Third, manufactured disagreement: coordinated raters can plant a fake faction, and the factorization will dutifully bridge to it. Fourth, speed: bridging needs ratings from both sides, which arrive slower than a majority vote, so breaking news outruns the mechanism. The common thread: bridging assumes the disagreement is real, symmetric, and already present. When the assumption fails, the mechanism fails with it.
> Follow-up: Which failure is hardest to fix?
> A: Manufactured disagreement. It attacks the mechanism's core assumption from the inside, and the factorization cannot distinguish a real faction from a coordinated one without an identity layer. The defense is the fraud model from Lecture 8's leaderboard answer: detect coordination, downweight it, and accept that perfect Sybil resistance is not available.

> [!QA]
> Q: Revealed or informed preferences: pick an extrapolation rule and defend it.
> A: Ship informed preferences with a stated, contestable rule: the preference the user would endorse on reflection with the relevant facts, extrapolated by the weakest model that fixes clear errors. Concretely: correct factual mistakes the user would retract, keep value judgments the user would defend, and flag every correction in the output log. Defend it: revealed preferences bake in manipulation, ignorance, and haste, and a system that optimizes those is a system that optimizes against the user. The cost is paternalism, which is why the rule must be stated and contestable: the user can see each extrapolation and veto it. An unstated rule is paternalism without appeal. A stated rule is a contract.
> Follow-up: Where does the rule break?
> A: Where the user's values are themselves the problem: informed preferences assume a reflective self worth extrapolating toward. For users whose informed preferences endorse harm to others, the extrapolation needs a boundary, and that boundary is a policy decision, not a preference inference. The pipeline can infer what the user would want. It cannot decide what the user should be allowed to want. That line is drawn by the deployment's values, stated openly.

## Mapping back: the course in one pipeline

| Stage | Machinery | Value choice smuggled in |
|---|---|---|
| Elicitation | Lectures 1, 5, 9. HCD | Who gets asked. what the interface measures |
| Learning | Lectures 2, 3, 4 | Which variation counts as signal. which model structure |
| Aggregation | Lecture 8. Polis bridging | Whose preferences weigh more. which axiom is relaxed |
| Decision | Lectures 6, 10 | Defer or override. the paternalism rule |

## The honest price: values all the way down

There is no value-free preference learning. The inversion
assumes behavior reveals preference. The model assumes IIA and
homogeneity. The aggregation relaxes some axiom. The decision
rule overrides or defers. Each step is defensible. None is
neutral. The honest practice is Webber's: write the assumptions
down, make them contestable, and prefer the weakest
extrapolation that resolves the clear errors. The machinery of
Lectures 1 through 9 is powerful exactly because it is
explicit. This chapter asks that the values be equally
explicit.

## Recap: the whole lesson on one screen

The story in eight steps. Each step answers the one before it.

1. **The lab meets the world.** A peer-review assistant.
   Whose judgment does it encode? The machinery gives no
   answer.
2. **Inversion: behavior is not preference.** Errors,
   constraints, strategy. Clicks reveal engagement, not
   satisfaction. Clickbait is the invoice.
3. **Four stages, four value choices.** Elicitation,
   learning, aggregation, decision. 10% sampling skew becomes
   a 40% gap. Bias compounds.
4. **Two fairnesses conflict.** Individual: similar cases,
   similar outcomes. Group: equal averages. Small subfields
   prove both cannot hold.
5. **Revealed versus informed.** Behavior shows revealed
   preferences. Informed preferences need extrapolation.
   State the rule. Make it contestable.
6. **Bridge, do not majoritize.** Polis: u = mu + alpha +
   beta + p^T q + noise. Select on beta after factoring out
   ideology. Disagreement is the instrument.
7. **The interface is the model.** Question format selects
   the preference distribution. Design elicitation with the
   users.
8. **Values all the way down.** No neutral pipeline exists.
   Write the assumptions down. Weakest extrapolation that
   fixes the clear errors.

## Go deeper

<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;max-width:100%;margin:16px 0;">
<iframe style="position:absolute;top:0;left:0;width:100%;height:100%;" src="https://www.youtube-nocookie.com/embed/VffFArrRSBE" title="Stanford CS329H Autumn 2024: Human-centered Design" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;max-width:100%;margin:16px 0;">
<iframe style="position:absolute;top:0;left:0;width:100%;height:100%;" src="https://www.youtube-nocookie.com/embed/HFrCySzH9QI" title="Stanford CS329H guest lecture: Joseph Jay Williams" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
- Human-centered Design lecture: https://www.youtube.com/watch?v=VffFArrRSBE
- Guest lecture, Joseph Jay Williams: https://www.youtube.com/watch?v=HFrCySzH9QI
- Guest lecture, Daniel Webber on value alignment (embedded above): https://www.youtube.com/watch?v=-kdR_7dCcyI
- Voting lecture, Colin Megill (Polis) guest segment: https://www.youtube.com/watch?v=1QpNZXL35NM
- Course textbook (Truong, Haupt, Koyejo): https://mlhp.stanford.edu/Machine-Learning-from-Human-Preferences.pdf
- Dwork et al. (2012): individual fairness.
- Polis, the open-source deliberation platform: https://pol.is

## Official sources and further reading

**Official:**
- Guest lecture: Daniel Webber on value alignment (video id
  -kdR_7dCcyI): revealed versus informed preferences, ESR
  statements.
- CS329H Autumn 2024: Voting (video id 1QpNZXL35NM), Colin
  Megill (Polis) guest segment: bridging, continuous
  factorization.
- Lecture on human-centered design (video id VffFArrRSBE).
- Guest lecture: Joseph Jay Williams (video id HFrCySzH9QI)
  [uncertain: title from search-index evidence only].
- Course textbook, chapters 4/5.6-5.11: the inversion problem, the
  four-stage pipeline, fairness, paternalism, privacy, the
  mutual information paradigm. [link](https://mlhp.stanford.edu/Machine-Learning-from-Human-Preferences.pdf)

**Further reading:**
- Dwork et al. (2012): individual fairness.
- Megill / Polis: [the open-source platform](https://pol.is).

**Caveats from these sources.** The 10%-to-40% compounding
figure is the textbook's illustration of feedback loops, not a
measured rate. The Polis factor model is stated as in the
lecture. estimation details are the platform's. The HCD
lecture is built on HCI community tutorials, per the lecturer.
Timestamps link to the guest lecture videos named above.

## Connections to the other courses

- **CS329H L01:** the factor model Polis deploys at scale.
- **CS329H L04:** systematic bias as the inversion problem in
  miniature.
- **CS329H L06:** Thompson sampling for human experiments.
  CIRL's strategic human.
- **CS329H L08:** aggregation impossibility. bridging as a
  concrete escape.
- **CS329H L09:** peer prediction as mechanism design without
  ground truth.
- **CS336:** the deployed systems these choices govern.

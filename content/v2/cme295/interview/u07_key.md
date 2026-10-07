# U07 interview key

Minimum sufficient explanation, strong answer, common red flags,
scoring rubric, remediation per question. Keep it closed-book.

## Breadth

- B1. Strong: recall = H/R (fraction of relevant docs found),
  precision = H/K (fraction of retrieved docs relevant). Red
  flag: swapping the two. Rubric: both formulas (2).
  Remediation: C01.
- B2. Strong: BM25 (exact terms) plus dense (meaning),
  normalize to [0,1] first or the larger scale eats alpha. Red
  flag: "average the raw scores". Rubric: what fuses (1), why
  normalize (1). Remediation: C03.
- B3. Strong: joint (query, doc) scoring, accurate but slow,
  precision@5 from the top-N candidates. Red flag: "it
  improves recall". Rubric: mechanism (1), what it cannot do
  (1). Remediation: C04.
- B4. Strong: Thought, Action, Observation, in that order,
  repeating until the answer. Red flag: "Action before
  Thought". Rubric: three + order (2). Remediation: C08.
- B5. Strong: trace (steps, short), session summary (medium),
  profile (durable). Red flag: "the context window".
  Rubric: three + lifetimes (2). Remediation: C09.
- B6. Strong: step cap, no-repeat rule, answer trigger. Red
  flag: "the model decides". Rubric: three (2).
  Remediation: C10.

## Deep ladders

- L1. Strong path: BM25 from term statistics, dense from
  cosine, fused [0.65, 0.60] ranks doc 1 first, normalization
  keeps alpha meaningful, hybrid with the rare-term check
  (exact term must surface), RRF uses ranks not scores,
  BM25-always-wins means unnormalized scales, "always helps"
  fails on single-signal corpora, experiment = sweep alpha,
  plot recall. Rubric: 2 per rung, 10 total.
- L2. Strong path: the three roles defined, a 2-tool trace
  written, triggers let the runtime inject observations,
  react_step with the no-action check (Thought without Action
  twice = stuck), plan-then-act is brittle to surprises, 6
  Thoughts 0 Actions means the trigger format is broken or
  the model avoids acting, thoughts can be decoration,
  experiment = strip thoughts, measure success. Rubric: 2
  per rung, 10 total.

## Analytical exercises

- E1. Strong: A: recall 0.80, precision 0.267. B: recall
  0.90, precision 0.150. Both give 5 x 0.80 = 4 relevant
  docs in the top 5 (precision@5 is the binding constraint,
  not recall@k). Ship A: same final quality at half the
  retrieval cost. Red flag: "B, higher recall". Rubric:
  four numbers (2), end-to-end (1), ship decision (1).
  Remediation: C01, C04.
- E2. Strong: budget 4400. 2000 + 1800 = 3800, 600 left:
  2 full docs plus 600 tokens of the third, the fourth is
  dropped. Compressed: [1200, 1080, 900, 720] = 3900, all
  4 fit. Red flag: forgetting the reserve. Rubric: first
  pack (2), compressed (2). Remediation: C05.

## Implementation/debug task

- D1. Strong: checks in order: (1) stopping config (is the
  repeat guard on?), (2) the trace for the repeated action
  and its observation (is search returning anything new?),
  (3) the tool schema (does search promise what it
  delivers?). Culprit: no repeat guard plus a search tool
  returning the same results, the agent rephrases forever.
  Fix: enable the no-repeat rule and add result
  dedup/diversity. Config: max_steps 10, no_repeat true,
  answer_trigger set. Red flag: "raise the temperature".
  Rubric: ordered checks (2), culprit (1), fix (1), config
  (1). Remediation: C08, C10.

## Changed-constraint scenarios

- S1. Strong: build hybrid with a high BM25 weight. Exact
  terms dominate this corpus, dense alone is the wrong tool
  and tuning it fights its nature. Rejected: dense-only
  (buries SKUs forever). First metric: recall@k on a
  part-number query set. Rubric: choice (1), rejected mode
  (2), metric (1). Remediation: C03.
- S2. Strong: (1) batch approvals: the agent prepares N
  refunds, one human click approves the batch, (2)
  pre-approval rules: auto-approve under a value threshold
  with full logging, human reviews the log daily. The gate
  stays, the per-ticket wait dies. Red flag: "make refunds
  green-tier". Rubric: two mechanisms (2 each).
  Remediation: C11.

## Research-critique question

- R1. Strong: strongest claim: more candidates raise the
  chance the answer is present, so quality rises with k.
  Counter: attention dilution and lost-in-the-middle mean
  extra docs bury the good ones, quality peaks then falls.
  Experiment: fix the window, vary k in {5, 10, 20, 40, 80}
  on a labeled set, plot answer accuracy. Falsification: if
  accuracy is monotone in k, the claim holds. A peak refutes
  it. Red flag: "recall is all that matters". Rubric: claim
  (1), counter (2), design (2). Remediation: C01, C05.

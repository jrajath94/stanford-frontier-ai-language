# U03 interview key

Minimum sufficient explanation, strong answer, common red flags,
scoring rubric, remediation per question. Keep it closed-book.

## Breadth

- B1. Strong: T -> 0 gives one-hot on argmax, T -> infinity gives
  uniform. Red flag: "T = 0 is the same as T small". Rubric: both
  limits (2). Remediation: C04.
- B2. Strong: top-p keeps the smallest set with mass >= p,
  top-k keeps exactly k regardless of mass. Red flag: "they are the
  same". Rubric: both rules (2). Remediation: C05.
- B3. Strong: router outputs gate probabilities over experts, the
  aux loss prevents collapse to one expert. Red flag: "the router
  picks the best expert, no loss needed". Rubric: output (1),
  prevention (1). Remediation: C02.
- B4. Strong: memory (T^2 scores, T cache), compute (T^2 d),
  position codes (unseen angles). Red flag: naming one. Rubric:
  three (2). Remediation: C03.
- B5. Strong: it walks into a high-probability cycle, sampling or
  a repetition penalty breaks it. Red flag: "higher temperature
  always fixes it". Rubric: mechanism (1), fix (1). Remediation:
  C06.
- B6. Strong: task behavior from prompt examples, no weight
  changes. Red flag: "ICL fine-tunes the model". Rubric: definition
  (1), no-update (1). Remediation: C08.

## Deep ladders

- L1. Strong path: formula, [0.979, 0.018, 0.002, 0.000] and
  [0.579, 0.213, 0.129, 0.078], limits from exp(z/T) behavior,
  order = temperature, then top-p, then sample (shape then cut
  then draw), high-T gibberish = tail junk, fix = truncation,
  "creative" critiqued via coherence loss, experiment = sweep T on
  factual vs story sets. Rubric: 2 per rung, 10 total.
- L2. Strong path: gates and top-2, E4/E3 weights [0.681, 0.319],
  collapse = rich-get-richer, aux penalizes f_i * P_i, moe
  implements routing + aux, active check = k/E per token, dense
  comparison on capacity vs compute, 90% expert = aux too small,
  "specialize" needs the loss, experiment = aux on/off, usage
  entropy. Rubric: 2 per rung, 10 total.

## Analytical exercises

- E1. Strong: p = [0.867, 0.117, 0.016], top-p = 0.75 keeps 1 ->
  [1, 0, 0], entropy 0.441 -> 0 nats. Truncation removed all
  choice. Red flag: forgetting renormalization. Rubric: distribution
  (2), truncation (1), entropies (2). Remediation: C04, C05.
- E2. Strong: per expert 2*4096*14336 = 117.4M, active = 2 experts
  = 234.9M, dense same-total = 939.5M active, fraction 25%.
  Red flag: "MoE has fewer parameters". Rubric: per-expert (2),
  active vs dense (2), fraction (1). Remediation: C02.

## Implementation/debug

- D1. Strong order: (1) render the tokenized prompt, check role
  markers are exact, (2) compare with the training template,
  (3) check the system role existed in training data. Culprit:
  template mismatch (markers differ from training). Fix: align the
  template, re-evaluate. Assert: tokenize(build_prompt(s, u))
  contains the exact marker ids in order. Red flag: "prompt harder"
  as the fix. Rubric: checks (3), culprit (2), assert (2).
  Remediation: C07, U04 C06.

## Changed-constraint scenarios

- S1. Strong: tool use wins (exact, 1 call), CoT fails on
  6-digit carries within 20 tokens and compounds errors, direct
  answer fails outright. Failure modes: tool = unavailable/latency,
  CoT = wrong carries, direct = low accuracy. Red flag: "CoT is
  enough". Rubric: choice + why (2), each failure mode (3).
- S2. Strong: guards = lower T (reshapes, keeps tail) then top-p
  (cuts tail). Order: temperature first, truncation second, sample
  last. Each change: T lowers entropy, top-p deletes the tail and
  renormalizes. Red flag: truncation before temperature (order
  still works but reasoning shows misunderstanding). Rubric: two
  guards (2), order (1), effects (2).

## Research-critique

- R1. Strong: claim = steps are the reasoning trace, counterexample
  = right answers with wrong steps, plausible steps to wrong
  answers, experiment = perturb an intermediate step, test whether
  the answer changes (causal test), falsification = answer
  invariant to step perturbations (steps are decoration). Red flag:
  "steps prove reasoning". Rubric: steelman (2), counterexample
  (2), experiment + falsification (3). Remediation: C10 item 11.

# U06 interview key

Minimum sufficient explanation, strong answer, common red flags,
scoring rubric, remediation per question. Keep it closed-book.

## Breadth

- B1. Strong: RL with verifiable rewards, the reward is computed
  by a program (exact match, unit tests), no human or learned
  model in the loop. Red flag: "a reward model trained on
  verifiable data". Rubric: expansion (1), verifiable meaning
  (1). Remediation: C01, C02.
- B2. Strong: A_i = (r_i - mu)/sigma over the prompt group, it
  replaces the learned critic/value baseline. Red flag:
  "it replaces the reward". Rubric: formula (1), replaced
  part (1). Remediation: C03.
- B3. Strong: 1 - (1 - p)^k, assumes near-independent attempts.
  Red flag: "it assumes the verifier is perfect". Rubric: law
  (1), assumption (1). Remediation: C07.
- B4. Strong: the policy maximizes the proxy reward by
  exploiting verifier gaps instead of solving the task. Red
  flag: "the model is misaligned". Rubric: proxy vs goal (2).
  Remediation: C06.
- B5. Strong: division standardizes the step size across
  prompts of different difficulty, raw gaps would make easy
  prompts dominate. Red flag: "it makes the math unbiased".
  Rubric: scale role (2). Remediation: C04.
- B6. Strong: paraphrase (surface brittleness), domain shift
  (skill transfer), verifier shift (referee fitting). Red flag:
  "train/val/test split". Rubric: three + what each isolates
  (2). Remediation: C12.

## Deep ladders

- L1. Strong path: group defined per prompt, A = [1,1,-1,-1],
  the critic gave the baseline and the group mean replaces it
  while sigma standardizes scale, group_adv with eps and the
  all-tied guard, PPO comparison on memory vs assumptions, tied
  groups mean no contrast (check sampler temperature and task
  difficulty), "no baseline" is false (the group mean is the
  baseline), experiment = sweep G, measure reward per step.
  Rubric: 2 per rung, 10 total.
- L2. Strong path: pass@k defined, best-of-k uses the verifier
  to select, pass@8 = 1 - 0.7^8 = 0.942, derivation from the
  complement of all-fail, best_of_k with the curve-tracking
  check, longer chains deepen one attempt while k widens
  coverage, far-below-curve means correlated failures or a
  weak selector, "always help" ignores diminishing returns and
  latency, experiment = vary verifier precision, measure
  realized gain. Rubric: 2 per rung, 10 total.

## Analytical exercises

- E1. Strong: mu = 0.75, sigma = sqrt(0.1875) = 0.968,
  A = [1.29 x3, -0.77 x5]. Single-winner group: mu = 0.125,
  sigma = 0.331, A_winner = 2.65. The first group teaches
  less per winner because three winners split the positive
  mass, contrast is diluted. Red flag: "more winners means
  more signal". Rubric: first group (2), second (2), why (1).
  Remediation: C04.
- E2. Strong: T = 60: k = 6, p = 0.632, score = 0.998. T = 90:
  k = 3, p = 0.777, score = 0.989. T = 60 wins: the extra test
  samples beat the extra skill. With k capped at 4, extra test
  budget is wasted, so the optimum moves to T = 80 (k = 4
  exactly, score 0.995 vs 0.982 at T = 60). Red flag: ignoring
  the cap. Rubric: both scores (2), cap reasoning (2).
  Remediation: C10.

## Implementation/debug task

- D1. Strong: checks in order: (1) verifier-vs-strict gap on
  fresh samples, (2) FP rate of the verifier on adversarial
  wrong answers, (3) n-gram scan of high-reward low-grade
  chains for a repeated exploit pattern. Culprit: reward
  hacking through a verifier keyway. Fix: harden the verifier
  (tighten the check the exploit abuses), roll back to the
  pre-hack checkpoint, resume. Logging: every 50 steps, log
  verifier pass, strict grade on a fixed 200-item probe set,
  and the gap. Red flag: "tune the KL". Rubric: ordered
  checks (2), culprit (1), fix (1), logging (1).
  Remediation: C06, C09.

## Changed-constraint scenarios

- S1. Strong: delay is the honest answer if the deadline
  allows, otherwise train with the weak verifier but cap
  optimization (early stop on the strict probe, small KL).
  A learned reward model inherits the same 18% leak plus its
  own biases and costs labels. Rejected: full training (farms
  the leak), pure delay without a probe (no information).
  First diagnostic: audit_verifier on adversarial wrongs to
  confirm the 18% and find the pattern. Rubric: choice (1),
  rejected failure modes (2), diagnostic (1). Remediation:
  C06, C09.
- S2. Strong: k = 2 with a perfect verifier roughly doubles
  effective coverage, so spend most of the budget on training
  to raise p, keep a small test-time reserve. At 70%
  verifier precision the selector misfires often, so shift
  budget back to train-time compute and consider k = 1 with a longer
  single chain. Red flag: "k = 2 always doubles accuracy".
  Rubric: allocation (2), precision effect (2). Remediation:
  C07, C10.

## Research-critique question

- R1. Strong: strongest claim: outcome supervision on
  checkable tasks teaches transferable procedures because
  the only way to pass many verifiers is to reason. Counter:
  the policy can memorize verifier quirks (format, parser
  gaps) that do not transfer. Experiment: train on verifier
  A, evaluate on a held-out verifier B built independently
  for the same skill, plus a paraphrase suite. Falsification:
  if B-score does not rise while A-score climbs, the gains
  are referee fitting, not reasoning. Red flag: "held-out
  prompts are enough". Rubric: claim (1), counter (2),
  design (2). Remediation: C09, C12.

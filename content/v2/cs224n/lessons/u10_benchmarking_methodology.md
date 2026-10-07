# U10 , Benchmarking and research methodology

## Local remediation

Bridges: `../shared/prerequisites/p07_estimation.md` (P07, intervals
and sample size), `../shared/prerequisites/p10_ml_foundations.md`
(P10, train/test discipline), `../shared/prerequisites/p22_experiments.md`
(P22, controls and falsification). This unit is about knowing what
a number means: a score without an interval is a rumor, a benchmark
without a contamination check is a hope.

R1. Wilson interval: 78/100 scores 0.780 with 95 percent interval
[0.689, 0.850]. Same score on 1000 items: [0.753, 0.805]. The
score is fixed, the claim shrinks with n.
R2. Contamination arithmetic: 200 items, 40 leaked at 0.95, 160
clean at 0.70. Reported 0.75, clean-only 0.70. The leak is worth
0.05 of pure inflation.
R3. Sample size: margin 0.02 at 95 percent needs n = 2401 for
p = 0.5. If your benchmark has 200 items, your margin is about
0.07, say so.

---

### C01: MMLU/HELM families

Leaf id `cs224n-U10-C01`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to the
   benchmarking session of the Winter 2026 schedule (S14 per
   `course_map.md`, readings MMLU, HELM, AlpacaEval per the
   schedule). Scope: what the big benchmark families measure.
   Objectives: describe the families, state what each aggregates,
   name the aggregation trap. Depends on P10.

2. **Motivating question and toy.** Question: model A scores 0.78,
   model B scores 0.76, did A win? Toy: the toy Wilson interval
   for 78/100 is [0.689, 0.850]. B's 0.76 sits inside A's
   interval. No winner declared.

3. **Mental model.** A benchmark is an exam with a fixed
   syllabus. MMLU-style exams test many subjects with
   multiple-choice questions. HELM-style harnesses test many
   scenarios with many metrics. The family name tells you the
   syllabus, not the truth.

4. **Objects, symbols, units, shapes, assumptions.** A score in
   [0, 1], an item count n, an aggregation (mean over tasks).
   Assumption: the items represent the deployment (the big
   assumption, C02 tests it).

5. **Derivation / mechanism.** The mechanism is the mean: task
   scores averaged, sometimes weighted. The trap: a mean hides
   the slices (C08). Two models with the same mean can differ
   by 0.26 on a slice, the toy shows exactly that.

6. **Computed example.** From `compute_u10.py`: 78/100 gives
   [0.689, 0.850], 780/1000 gives [0.753, 0.805]. Ten times the
   items, same point, honest claim.

7. **Algorithm and reference implementation.** `report(k, n)`:
   point estimate plus Wilson interval, never the point alone.
   ~8 lines. Test: the two toy intervals.

8. **Correctness checks and expected output.** The interval must
   contain the point and shrink with n. Check: at n = 2401 the
   half-width is about 0.02 (R3).

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** Scoring is cheap, the items are the cost. The
   practical cost is the items needed to make the interval
   useful.

10. **Nearest alternatives and selection boundaries.** Alternative:
    a bespoke eval for your task. Choose public benchmarks for
    comparability. Choose bespoke when the public syllabus
    misses your deployment.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "higher mean means better model." Counterexample:
    A wins the mean, B wins every slice that matters to you
    (C08). Read the slices before the mean.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: correlate benchmark rank with deployment-task
    rank on 12 models. Predict: positive but imperfect.
    Falsifier: near zero (then the benchmark is the wrong
    exam, build your own).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u10_answers.md` A1 (breadth), L1
    (ladder: describe the families, compute the interval,
    derive why n matters, diagnose the 0.78 vs 0.76 "win",
    design the rank-correlation test).

14. **Lab/exercises with answers separated.** E1: implement
    `report`, match both intervals. E2: find n for margin
    0.02, match 2401. Keys in `keys/u10_answers.md`.

15. **Visual units, provenance, accessibility, audit row.**
    `visuals/u10_fig01.png`: the two intervals, Shell 3,
    source original toy. Audit row in `visual_audit.md`.

---

### C02: task coverage

Leaf id `cs224n-U10-C02`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S14.
   Scope: whether the benchmark covers your task. Objectives:
   define coverage, build the coverage matrix, state the gap
   rule. Depends on C01.

2. **Motivating question and toy.** Question: the benchmark has
   50 tasks, yours is not among them, what does 0.78 mean for
   you? Toy: a 3-by-3 coverage matrix (tasks vs capabilities):
   your task needs capability X, no benchmark task tests X.
   The score says nothing about X.

3. **Mental model.** The benchmark is a map. Coverage asks
   whether your territory is on it. A high score on the map is
   not a high score on your ground.

4. **Objects, symbols, units, shapes, assumptions.** A task
   list, a capability list, a binary coverage matrix.
   Assumption: capabilities transfer (the optimistic one).

5. **Derivation / mechanism.** No new math. The mechanism is the
   matrix: rows are benchmark tasks, columns are capabilities
   your deployment needs. An empty column is an untested
   capability. Fill it or admit it.

6. **Computed example.** Toy (hand, labeled as such): 50 tasks,
   6 capabilities, column "tool use under permission errors"
   empty. The benchmark's 0.78 covers 5 of 6. Report 5/6
   covered, not 0.78.

7. **Algorithm and reference implementation.** `coverage(
   tasks, caps)`: build the matrix, list empty columns. ~8
   lines. Test: the toy finds the empty column.

8. **Correctness checks and expected output.** Every deployment
   capability appears as a column (completeness check). Check:
   no column is empty in the final report.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** The matrix is cheap. The cost is building the
   missing evals for empty columns.

10. **Nearest alternatives and selection boundaries.** Alternative:
    assume transfer from nearby tasks. Choose the matrix when
    the deployment is high-stakes. Choose assumption for
    exploration.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "nearby tasks predict mine." Counterexample:
    the model aces summarization benchmarks and fails your
    meeting-notes task (different length, different noise).
    Nearby is not same.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: predict deployment score from benchmark score
    across 10 deployments. Predict: weak correlation where
    coverage is thin. Falsifier: strong everywhere (then
    coverage analysis was too pessimistic, check).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u10_answers.md` A2 (breadth: define
    the coverage matrix and the gap rule).

14. **Lab/exercises with answers separated.** E3: build the
    matrix for the toy, find the empty column. E4: write the
    one-line coverage statement for a report. Keys in
    `keys/u10_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a matrix (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C03: judge calibration

Leaf id `cs224n-U10-C03`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S14
   (AlpacaEval-style judging per the schedule readings). Scope:
   using a model as an evaluator. Objectives: define agreement
   and calibration, compute kappa and Brier, state the bias
   risk. Depends on C01, P07.

2. **Motivating question and toy.** Question: the judge prefers
   A over B, do you believe it? Toy: two judges agree on 78 of
   100 items, kappa 0.551 (chance-corrected). The judge's
   confidence bins miscalibrate with Brier 0.17.

3. **Mental model.** A judge is a measuring instrument. Calibrate
   it like one: agreement with humans (kappa), confidence
   against reality (Brier, reliability). An uncalibrated judge
   is an opinion with a GPU.

4. **Objects, symbols, units, shapes, assumptions.** Observed
   agreement p_o, chance agreement p_e, kappa = (p_o - p_e) /
   (1 - p_e). Forecast f, outcome y, Brier = mean((f - y)^2).
   Assumption: the human labels are the reference (C09
   questions that).

5. **Derivation / mechanism.** Kappa removes the agreement that
   chance would give: with yes-rates 0.60 and 0.55, chance
   alone agrees 0.51 of the time, so 0.78 observed is kappa
   0.551. Brier is mean squared error of probabilities.

6. **Computed example.** From `compute_u10.py`: kappa 0.551,
   Brier 0.17. The reliability toy: stated 0.8, observed 0.67
   in one bin. Overconfident.

7. **Algorithm and reference implementation.** `judge_stats(
   a, b, f, y)`: kappa from the agreement table, Brier from
   forecasts. ~10 lines. Test: 0.551 and 0.17 on the toys.

8. **Correctness checks and expected output.** Kappa is 1 on
   perfect agreement, near 0 at chance. Check: Brier is 0 on
   perfect forecasts.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** The stats are cheap. The cost is the human labels
   for the calibration set.

10. **Nearest alternatives and selection boundaries.** Alternative:
    human eval for everything. Choose the judge for scale,
    with calibration. Choose humans for the final gate.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the judge is neutral." Counterexample: the
    judge prefers longer answers and the model's own style
    (C09). Calibrate on style-balanced pairs.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: blind the judge to answer length vs not, measure
    the win-rate shift. Predict: a shift toward concise.
    Falsifier: no shift (then length was not the bias, test
    style next).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u10_answers.md` A3 (breadth), L3
    (ladder: define kappa, compute 0.551, derive the chance
    correction, diagnose the length bias, design the blinding
    test).

14. **Lab/exercises with answers separated.** E5: implement
    `judge_stats`, match both numbers. E6: build the
    reliability table for the toy. Keys in `keys/u10_answers.md`.

15. **Visual units, provenance, accessibility, audit row.**
    `visuals/u10_fig03.png`: reliability bins, Shell 3, source
    original toy. Audit row in `visual_audit.md`.

---

### C04: contamination

Leaf id `cs224n-U10-C04`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S14.
   Scope: test items leaking into training. Objectives: define
   contamination, compute the inflation, state the quarantine
   rule. Depends on C01, U06 C08.

2. **Motivating question and toy.** Question: the model saw the
   test during training, what is the score worth? Toy: 40 of
   200 items leaked. Leaked solved at 0.95, clean at 0.70.
   Reported 0.75, honest 0.70.

3. **Mental model.** Contamination is an open-book exam graded
   as closed-book. The model is not smarter, the test is
   compromised. The inflation equals the leak share times the
   memorization gap.

4. **Objects, symbols, units, shapes, assumptions.** Leak share
   L, memorized score s_m, clean score s_c. Reported = L s_m +
   (1 - L) s_c. Assumption: leaked items are solved by memory
   (check with the canary test).

5. **Derivation / mechanism.** The formula is a mixture. With
   L = 0.20, s_m = 0.95, s_c = 0.70: 0.20 times 0.95 + 0.80
   times 0.70 = 0.75. The inflation is L times (s_m - s_c) =
   0.05. Small leaks, measurable lies.

6. **Computed example.** From `compute_u10.py`: reported 0.75,
   clean-only 0.70. Quarantine the 40, rescore, report 0.70
   with the quarantine noted.

7. **Algorithm and reference implementation.** `decontaminate(
   items, train)`: n-gram overlap probe, flag items above the
   threshold, rescore on the clean set. ~12 lines. Test: the
   toy flags the 40.

8. **Correctness checks and expected output.** The flagged set
   size matches the known leak on the toy. Check: the clean
   score is below the reported score.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** The probe matches strings over the train set,
   the expensive part. The cost of skipping it: publishing
   0.75.

10. **Nearest alternatives and selection boundaries.** Alternative:
    fresh held-out sets per evaluation. Choose quarantine when
    the train set is fixed. Choose fresh sets for high-stakes
    claims.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "overlap means memorization." Counterexample:
    common phrases overlap without the answer leaking. The
    probe needs the answer-bearing spans, not any n-gram.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: canary strings in training, measure extraction
    rate vs reported score. Predict: extraction tracks the
    inflation. Falsifier: no relation (then the probe is
    wrong, fix the spans).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u10_answers.md` A4 (breadth), L4
    (ladder: define contamination, compute the 0.05, derive
    the mixture, diagnose the n-gram false positive, design
    the canary test).

14. **Lab/exercises with answers separated.** E7: implement
    `decontaminate` on the toy, match 0.70. E8: show the
    common-phrase false positive. Keys in `keys/u10_answers.md`.

15. **Visual units, provenance, accessibility, audit row.**
    `visuals/u10_fig02.png`: reported vs clean bars, Shell 7,
    source original toy. Audit row in `visual_audit.md`.

---

### C05: uncertainty

Leaf id `cs224n-U10-C05`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S14.
   Scope: reporting what a score does not pin down. Objectives:
   compute intervals, state the n rule, refuse the bare number.
   Depends on C01, P07.

2. **Motivating question and toy.** Question: how many items make
   a claim? Toy: margin 0.02 at 95 percent needs n = 2401. Your
   200-item benchmark has margin about 0.07. Say that.

3. **Mental model.** Every score is a sample. The interval is the
   honest claim, the point is the center. Small n, wide claim.

4. **Objects, symbols, units, shapes, assumptions.** Wilson
   interval, standard error sqrt(p(1-p)/n), margin z times SE.
   Assumption: items are independent draws (often violated,
   say so).

5. **Derivation / mechanism.** SE at p = 0.5, n = 200: sqrt(0.25
   / 200) = 0.035, margin 1.96 times that = 0.069. The n for
   margin m: (z / m)^2 times 0.25 = 2401 at m = 0.02.

6. **Computed example.** From `compute_u10.py`: n = 2401. The
   toy intervals [0.689, 0.850] vs [0.753, 0.805] show the
   shrink.

7. **Algorithm and reference implementation.** `margin(n, p)`:
   Wilson half-width. ~6 lines. Test: 0.069 at n = 200,
   0.02 at n = 2401.

8. **Correctness checks and expected output.** The margin falls
   as 1/sqrt(n). Check: doubling n cuts the margin by about
   0.707.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** The math is free. The cost is the 2401
   good items.

10. **Nearest alternatives and selection boundaries.** Alternative:
    bootstrap intervals. Choose Wilson for proportions.
    Choose bootstrap for exotic metrics.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "items are independent." Counterexample: 200
    items from 10 templates, the effective n is near 10. The
    interval lies. Report the template count too.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: bootstrap vs Wilson on a templated benchmark.
    Predict: bootstrap wider. Falsifier: same (then the
    templates are diverse, check).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u10_answers.md` A5 (breadth: state
    the n rule and the template caveat).

14. **Lab/exercises with answers separated.** E9: implement
    `margin`, match 0.069 and 0.02. E10: show the templated
    effective-n trap. Keys in `keys/u10_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is an interval (carried in text, figure is C01's). Logged
    as an honest exception in `visual_audit.md`.

---

### C06: scoring formats

Leaf id `cs224n-U10-C06`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S14.
   Scope: how answers are scored (exact match, multiple choice,
   judge). Objectives: define the formats, state what each
   rewards, choose per task. Depends on C01, C03.

2. **Motivating question and toy.** Question: the model answers
   correctly but not exactly, what does exact match say? Toy:
   "Paris" vs "paris": exact match 0, normalized match 1. The
   format is a judgment, not a fact.

3. **Mental model.** The scorer is part of the benchmark. Exact
   match rewards formatting obedience. Multiple choice rewards
   the right letter. Judge scoring rewards whatever the judge
   likes (C03's biases).

4. **Objects, symbols, units, shapes, assumptions.** A
   normalization (lowercase, strip), a choice extractor, a
   judge prompt. Assumption: the format matches the task's
   real tolerance.

5. **Derivation / mechanism.** No new math. The mechanism is the
   pipeline: generate, normalize, compare. Each stage can flip
   the score. Report the normalization with the number.

6. **Computed example.** Toy (hand, labeled as such): 100
   answers, 12 differ only by case. Exact: 0.78, normalized:
   0.90. The 0.12 gap is format, not knowledge.

7. **Algorithm and reference implementation.** `score(ans,
   gold, mode)`: exact, normalized, choice-letter. ~10 lines.
   Test: the toy 0.78 vs 0.90.

8. **Correctness checks and expected output.** Normalized score
   is at least the exact score. Check: the choice extractor
   handles "B" and "B." alike.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** String work, cheap. The cost is disputes about the
   normalization.

10. **Nearest alternatives and selection boundaries.** Alternative:
    always use a judge. Choose exact/normalized for factual
    tasks. Choose the judge for open-ended tasks, calibrated.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "exact match is objective." Counterexample: the
    0.12 case gap. Objectivity needs the normalization
    stated.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: exact vs normalized vs judge on 100 items,
    correlate with human grades. Predict: normalized tracks
    humans best on factual tasks. Falsifier: judge wins (then
    the answers need semantic grading, use it).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u10_answers.md` A6 (breadth: name
    the three formats and what each rewards).

14. **Lab/exercises with answers separated.** E11: implement
    `score`, match the toy. E12: add a normalization and show
    the 0.12 move. Keys in `keys/u10_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a pipeline (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C07: baseline replication

Leaf id `cs224n-U10-C07`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S14.
   Scope: reproducing the baseline before claiming a win.
   Objectives: define the replication bar, run the check, state
   the tolerance. Depends on C01, P22.

2. **Motivating question and toy.** Question: your method beats
   the baseline by 0.03, did you run the baseline right? Toy:
   the paper claims 0.78, your rerun gives 0.74 with interval
   [0.65, 0.82]. The claim sits inside your interval: no
   evidence of a difference.

3. **Mental model.** The baseline is the control group. A win
   against a broken baseline is a win against nothing. Replicate
   first, then compare, with intervals on both sides.

4. **Objects, symbols, units, shapes, assumptions.** The
   reported number, your rerun, both intervals. Assumption: the
   baseline code and data are available (often half-true).

5. **Derivation / mechanism.** No new math. The mechanism is the
   comparison: if the reported number falls outside your
   rerun's interval, something differs (version, data, bug).
   Debug before you innovate.

6. **Computed example.** Toy (hand, labeled as such): reported
   0.78, rerun 0.74, interval [0.65, 0.82]. 0.78 is inside.
   Verdict: replicated within noise. Now the 0.03 win needs
   its own interval.

7. **Algorithm and reference implementation.** `replicate(
   reported, k, n)`: Wilson interval around your rerun, test
   membership. ~6 lines. Test: the toy verdict.

8. **Correctness checks and expected output.** The verdict is
   one of: replicated, differs, or underpowered (interval too
   wide to tell). Check: underpowered triggers more items,
   not a claim.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** One extra eval run. The cost of skipping it:
   months chasing a phantom win.

10. **Nearest alternatives and selection boundaries.** Alternative:
    trust the reported number. Choose replication for any
    claim you will defend. Choose trust for exploration only.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the baseline in the paper is the baseline in
    the code." Counterexample: the paper used a better prompt
    than the released code. Replicate the artifact, not the
    text.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: replicate 5 reported baselines, count how many
    fall inside your intervals. Predict: fewer than 5.
    Falsifier: all 5 (then the field is healthier than feared,
    good).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u10_answers.md` A7 (breadth: state
    the replication bar and the three verdicts).

14. **Lab/exercises with answers separated.** E13: implement
    `replicate`, match the toy verdict. E14: widen the
    interval until the verdict is underpowered. Keys in
    `keys/u10_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a verdict (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C08: slices

Leaf id `cs224n-U10-C08`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S14.
   Scope: scores broken down by subgroup. Objectives: define a
   slice, compute deltas, state the mean-trap. Depends on C01.

2. **Motivating question and toy.** Question: the mean is 0.78,
   is the model ready? Toy: slice A scores 0.52 (delta -0.26),
   slice B 0.86 (delta +0.08). The mean hid a failing slice.

3. **Mental model.** The mean is a summary, the slices are the
   story. A model can be excellent on average and unusable for
   your users. Cut by what matters: language, topic, difficulty.

4. **Objects, symbols, units, shapes, assumptions.** Slice
   scores, deltas from the mean, slice sizes. Assumption: the
   slice labels are correct (label noise ruins slices).

5. **Derivation / mechanism.** Delta = slice score minus
   overall. The mechanism is disaggregation: the same items,
   grouped. Small slices need their own intervals (C05's n
   rule applies per slice).

6. **Computed example.** From `compute_u10.py`: overall 0.78,
   slice A 0.52 (delta -0.26), slice B 0.86 (delta +0.08). The
   -0.26 is the headline, not the 0.78.

7. **Algorithm and reference implementation.** `slice_report(
   items)`: group, score, delta, interval per slice. ~10 lines.
   Test: the toy deltas.

8. **Correctness checks and expected output.** Deltas sum to
   zero when weighted by slice size. Check: no slice is
   reported without its n.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** Grouping is cheap. The cost is slice-labeled data.

10. **Nearest alternatives and selection boundaries.** Alternative:
    report the mean only. Choose slices for deployment
    decisions. Choose the mean for quick comparisons.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the mean represents the slices." Counterexample:
    the toy: -0.26 on slice A. The mean is a weighted lie when
    the weights are not your users.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: track the worst slice vs the mean across 12
    model versions. Predict: the mean rises faster than the
    worst slice. Falsifier: they move together (then the
    training is balanced, good).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u10_answers.md` A8 (breadth), L8
    (ladder: define a slice, compute the deltas, derive the
    weighted-zero check, diagnose the hidden -0.26, design the
    worst-slice tracking).

14. **Lab/exercises with answers separated.** E15: implement
    `slice_report`, match the toy. E16: drop slice labels and
    show the mean hiding the failure. Keys in `keys/u10_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a disaggregation (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C09: bias

Leaf id `cs224n-U10-C09`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S14.
   Scope: systematic skew in benchmarks and judges. Objectives:
   name the bias classes, design the control, state the
   correction limit. Depends on C03.

2. **Motivating question and toy.** Question: the judge prefers
   model A, is A better or longer? Toy: blinding the judge to
   length moves the win rate from 0.62 to 0.53. The 0.09 was
   length bias.

3. **Mental model.** Every instrument has a tilt. Benchmarks
   tilt toward their items, judges tilt toward style. Name the
   tilt, measure it, control it. Unmeasured tilt is a
   confounder.

4. **Objects, symbols, units, shapes, assumptions.** Bias
   classes: length, position, style, self-preference. A
   control: the blinded variant. Assumption: the control
   removes only the bias (the hard part).

5. **Derivation / mechanism.** The mechanism is comparison: the
   win rate with and without the suspected bias factor. The
   difference is the bias estimate. No fancy math, just the
   control.

6. **Computed example.** Toy (hand, labeled as such): unblinded
   win rate 0.62, blinded 0.53, bias 0.09. The correction:
   report the blinded number.

7. **Algorithm and reference implementation.** `bias_check(
   pairs, factor)`: score with and without the factor, report
   the delta. ~8 lines. Test: the toy 0.09.

8. **Correctness checks and expected output.** The blinded
   variant must be identical except for the factor (ceteris
   paribus check). Check: the delta's sign is stable.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** Two eval runs. The cost is the blinding design.

10. **Nearest alternatives and selection boundaries.** Alternative:
    ignore bias, report raw win rates. Choose controls for
    published claims. Choose raw for internal iteration.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "blinding fixed it." Counterexample: the judge
    prefers its own model's phrasing even blinded (style
    survives blinding). Controls are iterative.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: test 4 bias factors on 3 judges, rank them.
    Predict: length and self-preference top the list.
    Falsifier: position wins (then the judge is lazy, check
    the prompt order).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u10_answers.md` A9 (breadth: name
    the four bias classes and the control).

14. **Lab/exercises with answers separated.** E17: implement
    `bias_check`, match 0.09. E18: blind position instead and
    compare. Keys in `keys/u10_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a control (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C10: acceptance criteria

Leaf id `cs224n-U10-C10`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S14.
   Scope: deciding in advance what "good enough" means.
   Objectives: write a criterion, separate it from the metric,
   state the pre-registration rule. Depends on C01, C05, C08.

2. **Motivating question and toy.** Question: the model scores
   0.78, do you ship? Toy: the criterion was "worst slice above
   0.70 with n = 500 per slice". Slice A is 0.52. No ship,
   regardless of the mean.

3. **Mental model.** The criterion is the contract, the metric
   is the measurement. Write the contract before you measure,
   or you will negotiate with yourself after.

4. **Objects, symbols, units, shapes, assumptions.** A
   threshold, a slice set, a sample size, a decision rule.
   Assumption: the criterion reflects the deployment need
   (the product question, not the ML question).

5. **Derivation / mechanism.** No new math. The mechanism is
   pre-registration: the decision rule is fixed before the
   numbers arrive. This blocks HARKing (hypothesizing after
   the results are known).

6. **Computed example.** Toy (hand, labeled as such): criterion
   "mean above 0.75 and no slice below 0.70". Result: mean
   0.78, slice A 0.52. Verdict: fail. The mean passed, the
   contract did not.

7. **Algorithm and reference implementation.** `decide(
   report, criterion)`: check each clause, return pass/fail
   with the binding clause. ~8 lines. Test: the toy fails on
   slice A.

8. **Correctness checks and expected output.** The decision is
   reproducible from the report and the criterion (no judgment
   calls at decision time). Check: the binding clause is
   named.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** Writing the criterion is the cost. The cost of
   skipping it: shipping on a 0.78 that hides a 0.52.

10. **Nearest alternatives and selection boundaries.** Alternative:
    decide after seeing the numbers. Choose pre-registration
    for ship decisions. Choose exploration for research.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the criterion was right." Counterexample: the
    criterion demanded 0.95 on a task humans do at 0.80. The
    criterion needs its own review.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: pre-registered vs post-hoc decisions on 10
    ship calls, track regret. Predict: pre-registered has less
    regret. Falsifier: tie (then the criteria were vacuous,
    harden them).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u10_answers.md` A10 (breadth:
    state the pre-registration rule and the contract/metric
    split).

14. **Lab/exercises with answers separated.** E19: implement
    `decide`, match the toy verdict. E20: write a criterion
    for a support-bot deployment. Keys in `keys/u10_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a contract (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C11: project choice

Leaf id `cs224n-U10-C11`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S14
   and the project sessions. Scope: picking a project worth
   doing. Objectives: score a proposal, state the scope rule,
   kill a bad project early. Depends on P22, C10.

2. **Motivating question and toy.** Question: three project
   ideas, four weeks, which one? Toy: the scorecard (feasible,
   data, metric, compute, novelty): idea A scores 4/5, idea B
   2/5. Pick A, kill B on paper.

3. **Mental model.** A project is a bet. The scorecard prices
   the bet: can you do it, can you measure it, does it matter.
   Kill early, kill on paper, kill without shame.

4. **Objects, symbols, units, shapes, assumptions.** Five
   binary criteria, a total, a kill threshold. Assumption: the
   scores are honest (the failure mode is self-deception).

5. **Derivation / mechanism.** No new math. The mechanism is the
   checklist: feasibility in the time box, data available,
   metric defined, compute fits, some novelty. Below 3/5:
   rescope or drop.

6. **Computed example.** From `compute_u15.py` (shared toy):
   3/5. The missing two (metric defined, novelty) are exactly
   the rescope targets.

7. **Algorithm and reference implementation.** `scorecard(
   idea)`: five checks, total, verdict. ~8 lines. Test: the
   toy 3/5.

8. **Correctness checks and expected output.** Every "1" names
   the evidence (which data, which metric). Check: no idea
   scores 5/5 without a written metric.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** An hour of honesty. The cost of skipping it: four
   weeks on idea B.

10. **Nearest alternatives and selection boundaries.** Alternative:
    follow curiosity, no scorecard. Choose the scorecard for
    time-boxed work. Choose curiosity for open-ended research.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the scores are honest." Counterexample: "data
    available" checked for a dataset that needs a license you
    do not have. Evidence, not vibes.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: track scorecard vs outcome on 20 past projects.
    Predict: 4-5/5 predicts completion. Falsifier: no relation
    (then the criteria are wrong, revise).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u10_answers.md` A11 (breadth: name
    the five criteria and the kill rule).

14. **Lab/exercises with answers separated.** E21: score two
    ideas, kill one. E22: rescope the 3/5 idea to 4/5. Keys in
    `keys/u10_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a checklist (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C12: contribution accounting

Leaf id `cs224n-U10-C12`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S14.
   Scope: honest credit for what the project actually did.
   Objectives: separate the contribution from the scaffolding,
   write the accounting, state the novelty bar. Depends on
   P22, C07.

2. **Motivating question and toy.** Question: the project works,
   what did you contribute? Toy: the accounting lists: baseline
   (existing), data (existing), the one new thing (the
   rerank rule), the measurement (new). One new thing, named.

3. **Mental model.** The report has two columns: borrowed and
   built. Borrowed is fine, it must be labeled. Built is the
   contribution. A report with an empty built column is a
   tutorial, not a project.

4. **Objects, symbols, units, shapes, assumptions.** The
   borrowed list, the built list, the novelty claim.
   Assumption: the borrowed parts are cited (integrity).

5. **Derivation / mechanism.** No new math. The mechanism is the
   ledger: every component gets a source or an author. The
   novelty bar: one falsifiable claim that was not known
   before.

6. **Computed example.** Toy (hand, labeled as such): borrowed:
   transformer, BPE, Adam. Built: the chunk-size sweep and the
   split-rate rule. The rule is the contribution.

7. **Algorithm and reference implementation.** `account(
   components)`: label each borrowed/built, check the built
   list is non-empty. ~6 lines. Test: the toy passes.

8. **Correctness checks and expected output.** Every borrowed
   item has a citation. Check: the built list has at least one
   falsifiable claim.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** An honest hour. The cost of skipping it: a report
   that claims the transformer.

10. **Nearest alternatives and selection boundaries.** Alternative:
    claim everything. Choose the ledger for anything public.
    The alternative is misconduct.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "a negative result is not a contribution."
    Counterexample: the rerank sweep that found nothing is a
    contribution if the test was fair (negative results are
    data, C07's lesson).

12. **Research reading and falsifiable extension.** Falsifiable
    extension: none needed, this is the close. The extension
    is the habit: keep the ledger from day one.

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u10_answers.md` A12 (breadth:
    state the two columns and the novelty bar), L12 (ladder:
    define the ledger, build the toy, derive the bar,
    diagnose the empty-built report, design the negative-
    result writeup).

14. **Lab/exercises with answers separated.** E23: write the
    accounting for the U09 lab. E24: label one borrowed item
    you almost claimed. Keys in `keys/u10_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a ledger (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

## Unit visual map

| Figure | Claim | Shell | Source |
|--------|-------|-------|--------|
| `visuals/u10_fig01.png` | 0.78 on 100 items is [0.689, 0.850] | 3 | original toy |
| `visuals/u10_fig02.png` | 40 leaked items inflate 0.70 to 0.75 | 7 | original toy |
| `visuals/u10_fig03.png` | judge stated 0.8, observed 0.67 | 3 | original toy |

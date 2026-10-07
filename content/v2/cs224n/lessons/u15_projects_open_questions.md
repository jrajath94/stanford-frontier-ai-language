# U15 , Project tutorials and open questions

## Local remediation

Bridges: `../shared/prerequisites/p02_python.md` (P02, the
working language), `../shared/prerequisites/p12_pytorch.md` (P12,
tensors and stability), `../shared/prerequisites/p22_experiments.md`
(P22, falsifiable projects), `../shared/prerequisites/p24_production.md`
(P24, stakeholder foundations). This unit turns the course into a
project: the tutorials (C01-C03), the default project recipe
(C04-C05), the project machinery (C06-C10), and the open horizon
(C11-C12).

R1. Minimal GPT-2: V = 1000, d = 8, L = 4. Total 44,928 params.
The recipe scales: the default project is this, bigger.
R2. Seed variance: 5 seeds, mean 0.8045, std 0.0203. One seed
is an anecdote, five are evidence.
R3. Pipeline: proposal, milestone, poster, report. Four gates
from SRC-04. Each gate needs a falsifiable claim, not a demo.

---

### C01: Python review

Leaf id `cs224n-U15-C01`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to the
   Python review session of the Winter 2026 schedule (S03 per
   `course_map.md`). Scope: the Python you need for the
   course. Objectives: write clean numeric code, state the
   vectorization rule, avoid the loop trap. Depends on P02.

2. **Motivating question and toy.** Question: the loop over
   1M tokens takes minutes, the vectorized version takes a
   second, why? Toy: numpy dot on 1M floats: 0.002 s. Python
   loop: 0.4 s. The loop is the trap.

3. **Mental model.** Python is the conductor, numpy is the
   orchestra. The conductor waves (calls), the orchestra
   plays (C loops). Conduct, do not play.

4. **Objects, symbols, units, shapes, assumptions.** Arrays,
   shapes, dtypes. Assumption: the data fits in memory (or
   batch it).

5. **Derivation / mechanism.** The mechanism is where the loop
   runs: C for numpy, Python for the for-loop. Same
   arithmetic, 200x the speed.

6. **Computed example.** Toy (hand, labeled as such): the
   0.002 s vs 0.4 s above. Measure on your machine, the
   ratio holds.

7. **Algorithm and reference implementation.** `fast_dot(a,
   b)`: numpy dot, shape check. ~4 lines. Test: matches the
   loop's answer.

8. **Correctness checks and expected output.** The answers
   match to 1e-10. Check: shapes agree before the call.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** Same FLOPs, different constant. The cost of the
   loop: your afternoon.

10. **Nearest alternatives and selection boundaries.** Alternative:
    PyTorch (C02). Choose numpy for CPU labs. Choose torch
    for GPUs and autograd.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "vectorized is always faster." Counterexample:
    tiny arrays: the call overhead dominates. Vectorize the
    big loops.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: none, this is craft. The extension is the
    habit: profile before optimizing.

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u15_answers.md` A1 (breadth), L1
    (ladder: state the rule, time the toy, derive the loop
    location, diagnose the tiny-array case, design the
    profile habit).

14. **Lab/exercises with answers separated.** E1: time both
    versions. E2: find the crossover size. Keys in
    `keys/u15_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a timing (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C02: PyTorch tutorial

Leaf id `cs224n-U15-C02`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to the
   PyTorch tutorial session (S06 per `course_map.md`). Scope:
   tensors, autograd, and stability. Objectives: build the
   graph, read a gradient, keep it stable. Depends on P12,
   U03.

2. **Motivating question and toy.** Question: who computes the
   gradient? Toy: x = tensor(2.0, requires_grad=True), y =
   x**2, y.backward(): x.grad is 4.0. Autograd did the chain
   rule.

3. **Mental model.** PyTorch records what you did (the graph)
   and replays it backwards (the gradient). You write the
   forward, it writes the backward.

4. **Objects, symbols, units, shapes, assumptions.** Tensors,
   the graph, .grad. Assumption: the ops are differentiable
   (or the graph stops).

5. **Derivation / mechanism.** The mechanism is U03's chain
   rule, automated: each op knows its local derivative, the
   engine multiplies them.

6. **Computed example.** Toy (hand, labeled as such): x = 2,
   y = x^2 = 4, dy/dx = 2x = 4. The .grad reads 4.0.

7. **Algorithm and reference implementation.** `grad_check(
   f, x)`: autograd vs finite difference. ~8 lines. Test:
   match to 1e-5.

8. **Correctness checks and expected output.** Finite
   differences agree (U03 C10's check). Check: no NaN in the
   grad.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** Backward costs about 2x forward. Stability: U03
   C03's softmax shift, C08's clipping.

10. **Nearest alternatives and selection boundaries.** Alternative:
    numpy with hand gradients. Choose torch for models.
    Choose numpy for the course labs (this box has no torch).

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the graph is free." Counterexample: keeping
    the graph for 1000 steps eats memory. Detach or lose
    the graph.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: none, this is craft. The extension is grad
    checks on every new op.

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u15_answers.md` A2 (breadth: state
    the graph rule and the check).

14. **Lab/exercises with answers separated.** E3: the x^2 toy.
    E4: grad-check a softmax. Keys in `keys/u15_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a graph (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C03: Hugging Face tutorial

Leaf id `cs224n-U15-C03`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to the
   Hugging Face tutorial session (S13 per `course_map.md`).
   Scope: the standard model-and-data plumbing. Objectives:
   load a tokenizer and model, run a batch, name the
   versioning rule. Depends on C02, U12.

2. **Motivating question and toy.** Question: you need a
   pretrained model in 5 lines, how? Toy: the tokenizer +
   model pair, one forward pass, logits out. The plumbing is
   the product.

3. **Mental model.** The hub is a library, the API is the
   librarian. You ask for a model by name and version, you
   get weights and config. Pin the version or the library
   moves under you.

4. **Objects, symbols, units, shapes, assumptions.** Model
   names, revisions, tokenizers, batches. Assumption: the
   name resolves to what you think (pin the revision).

5. **Derivation / mechanism.** No new math. The mechanism is
   the pipeline: tokenize, batch, forward, decode. Each
   stage has a contract (shapes, padding).

6. **Computed example.** Toy (hand, labeled as such): batch of
   4, length 7, vocab 50: logits (4, 7, 50). The shapes are
   the contract (U03 C11's lesson).

7. **Algorithm and reference implementation.** `encode_batch(
   texts, tok)`: pad, mask, tensor. ~8 lines. Test: the
   shapes.

8. **Correctness checks and expected output.** The attention
   mask zeros the padding. Check: decode(encode(s)) is s.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** The hub download is the cost. The risk: an
   unpinned revision changes results.

10. **Nearest alternatives and selection boundaries.** Alternative:
    raw PyTorch, own data loading. Choose the hub for
    standard models. Choose raw for custom architectures.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the name is the model." Counterexample: the
    revision moved, your numbers changed. Pin everything.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: none, this is craft. The extension is the
    version log.

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u15_answers.md` A3 (breadth: state
    the pipeline and the pin rule).

14. **Lab/exercises with answers separated.** E5: write the
    5-line load. E6: show the mask on padding. Keys in
    `keys/u15_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a pipeline (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C04: minimalist GPT-2 objective

Leaf id `cs224n-U15-C04`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to the
   final project description (SRC-04: default project implements
   parts of GPT-2). Scope: the smallest honest GPT-2. Objectives:
   count the params, write the loss, train the toy. Depends on
   U04, U05, R1.

2. **Motivating question and toy.** Question: what is the least
   you can build and call it GPT-2? Toy: V = 1000, d = 8,
   L = 4: 44,928 params. Next-token cross-entropy. The
   recipe, miniature.

3. **Mental model.** GPT-2 is embeddings plus transformer
   blocks plus a head, trained to predict the next token.
   The miniature has every part, just small. Scale is a
   dial, not a different machine.

4. **Objects, symbols, units, shapes, assumptions.** V, d, L,
   the param count, the loss. Assumption: the parts compose
   the same at scale (the scaling bet, U06).

5. **Derivation / mechanism.** Params = embeddings (Vd + nd)
   + L times (attention + MLP). The loss is U04's next-token
   cross-entropy. The mechanism is the stack.

6. **Computed example.** From `compute_u15.py`: 44,928 params.
   The toy loss falls 1.164 to 1.043 on random data (the
   probe learns the mean, honestly labeled).

7. **Algorithm and reference implementation.** `tiny_gpt2(
   V, d, L)`: the parts list. ~10 lines. Test: the param
   count.

8. **Correctness checks and expected output.** The count
   matches the formula. Check: the loss falls on the toy.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** U06's scaling: the miniature teaches the
   plumbing, not the capabilities.

10. **Nearest alternatives and selection boundaries.** Alternative:
    the full default project (parts of real GPT-2). Choose
    the miniature for understanding. Choose the default for
    the grade.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the miniature predicts the full model."
    Counterexample: capabilities emerge with scale (U06).
    The miniature teaches mechanics, not emergence.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: scale d in {8, 32, 128}, measure the loss
    curve shape. Predict: same shape, lower floor.
    Falsifier: different shape (then scale changes the
    dynamics, interesting).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u15_answers.md` A4 (breadth), L4
    (ladder: define the parts, count 44,928, derive the
    formula, diagnose the emergence gap, design the scale
    sweep).

14. **Lab/exercises with answers separated.** E7: implement
    the param counter, match 44,928. E8: train the probe,
    match the loss fall. Keys in `keys/u15_answers.md`.

15. **Visual units, provenance, accessibility, audit row.**
    `visuals/u15_fig01.png`: the param plate, Shell 2,
    source original toy. Audit row in `visual_audit.md`.

---

### C05: downstream tasks

Leaf id `cs224n-U15-C05`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to
   SRC-04 (default project: 3 downstream tasks). Scope: using
   the miniature on tasks. Objectives: define the transfer,
   report with uncertainty, state the fine-tune rule. Depends
   on C04, U06 C09.

2. **Motivating question and toy.** Question: the model is
   trained, now what? Toy: three tasks with toy scores:
   paraphrase 0.81 +/- 0.03, sentiment 0.93 +/- 0.01, QA 0.68
   +/- 0.05. Labeled toy, the template is the lesson.

3. **Mental model.** Pretraining is school, downstream tasks
   are the jobs. The template: attach a head, fine-tune a
   little, report with intervals (U10's discipline).

4. **Objects, symbols, units, shapes, assumptions.** Task
   scores with standard errors. Assumption: the scores are
   labeled toy (they are, say so).

5. **Derivation / mechanism.** The mechanism is U06 C09's
   transfer: the pretrained stack plus a task head. The
   reporting is U10's: intervals, slices.

6. **Computed example.** From `compute_u15.py` (labeled toy):
   the three scores above. QA's 0.05 SE is the widest: the
   task needs more items.

7. **Algorithm and reference implementation.** `downstream(
   model, task)`: head, fine-tune, report. ~8 lines. Test:
   the template runs.

8. **Correctness checks and expected output.** Every score has
   its SE. Check: the fine-tune uses a dev split.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** Fine-tuning is cheap. The cost is the eval items
   for honest SEs.

10. **Nearest alternatives and selection boundaries.** Alternative:
    prompt the base model (U08). Choose fine-tuning for
    reliability. Choose prompting for speed.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the toy scores are real." Counterexample:
    they are template numbers. Never cite them as findings.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: run the template for real on the miniature.
    Predict: the pattern holds (sentiment > paraphrase >
    QA). Falsifier: any order (then the miniature differs,
    investigate).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u15_answers.md` A5 (breadth: state
    the template and the toy label).

14. **Lab/exercises with answers separated.** E9: run the
    template. E10: widen QA's items, show the SE shrink.
    Keys in `keys/u15_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a template (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C06: proposal/milestone/report

Leaf id `cs224n-U15-C06`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to
   SRC-04 (proposal, milestone, poster, report). Scope: the
   four gates. Objectives: write each gate's one-pager, state
   the falsifiable rule, pass the gates. Depends on P22, U10
   C10-C12.

2. **Motivating question and toy.** Question: the project is 4
   weeks, how do you not drift? Toy: the proposal names the
   claim ("k=5 beats k=1 on our QA"), the milestone shows the
   first numbers, the poster tells the story, the report
   proves it. Each gate has a falsifiable claim.

3. **Mental model.** The gates are the project's spine. The
   proposal promises, the milestone shows progress, the
   poster communicates, the report archives. A gate without
   a claim is a status meeting.

4. **Objects, symbols, units, shapes, assumptions.** Four
   documents, each with a claim. Assumption: the claim
   survives contact with data (or it changes, honestly).

5. **Derivation / mechanism.** No new math. The mechanism is
   the gate: each document answers "what did you claim, what
   did you find, what changed". The report is U10 C12's
   accounting plus U13 C12's impact.

6. **Computed example.** Toy (hand, labeled as such): the
   proposal claim above, milestone numbers (k=5: 0.667 vs
   k=1: 0.000), the report's verdict. The numbers from U09's
   toys, reused honestly.

7. **Algorithm and reference implementation.** `gate_doc(
   kind, claim, evidence)`: the template. ~6 lines. Test: all
   four kinds render.

8. **Correctness checks and expected output.** Every gate names
   its claim and its evidence. Check: the report's claims
   match the proposal's (or the changes are logged).

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** Writing time. The cost of skipping a gate: drift.

10. **Nearest alternatives and selection boundaries.** Alternative:
    one final report. Choose the gates for a team or a grade.
    Choose the single report for solo exploration.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the proposal survives." Counterexample: the
    data kills the claim at the milestone. Change the claim,
    log the change, continue.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: none, this is operations. The extension is
    doing it.

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u15_answers.md` A6 (breadth), L6
    (ladder: define the gates, write the toy claim, derive
    the gate logic, diagnose the killed claim, design the
    change log).

14. **Lab/exercises with answers separated.** E11: write the
    proposal one-pager. E12: write the milestone with the
    U09 numbers. Keys in `keys/u15_answers.md`.

15. **Visual units, provenance, accessibility, audit row.**
    `visuals/u15_fig03.png`: the four gates, Shell 10,
    source SRC-04. Audit row in `visual_audit.md`.

---

### C07: project scope

Leaf id `cs224n-U15-C07`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to
   SRC-04. Scope: sizing the project. Objectives: score the
   idea, cut it to fit, state the kill rule. Depends on U10
   C11.

2. **Motivating question and toy.** Question: the idea needs 8
   weeks, you have 4, what goes? Toy: the scorecard 3/5
   (missing: metric defined, novelty). Define the metric:
   4/5. Cut the novelty: ship the measurement.

3. **Mental model.** U10 C11's scorecard, applied: scope is the
   art of the shippable. A project that does one thing well
   beats a project that does three things badly.

4. **Objects, symbols, units, shapes, assumptions.** The five
   criteria, the cut list. Assumption: the cut preserves the
   claim (or the claim changes).

5. **Derivation / mechanism.** The mechanism is subtraction:
   list everything, cut until it fits the time box, keep the
   falsifiable core.

6. **Computed example.** From `compute_u15.py`: 3/5. The two
   missing are the rescope targets: define the metric
   (+1), cut novelty (ship at 4/5).

7. **Algorithm and reference implementation.** `rescope(idea,
   weeks)`: score, cut, rescore. ~8 lines. Test: the toy
   3/5 to 4/5.

8. **Correctness checks and expected output.** The rescoped
   idea fits the weeks. Check: the claim is still falsifiable.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** An honest hour. The cost of skipping it: the 8-
   week idea in 4 weeks.

10. **Nearest alternatives and selection boundaries.** Alternative:
    do it all, sleep less. Choose scoping always. The
    alternative is the crunch.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the cut kept the claim." Counterexample: the
    cut removed the control group. Cut scope, never the
    control.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: none, this is craft. The extension is the
    habit.

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u15_answers.md` A7 (breadth: state
    the cut rule).

14. **Lab/exercises with answers separated.** E13: score and
    cut the toy idea. E14: find the control in the cut list.
    Keys in `keys/u15_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a cut (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C08: original implementation

Leaf id `cs224n-U15-C08`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to
   SRC-04 (default project: implement parts of GPT-2). Scope:
   writing your own code. Objectives: implement from the
   spec, test each part, state the originality rule. Depends
   on C04, the course integrity policy (SRC-06).

2. **Motivating question and toy.** Question: the project says
   "implement", what does that allow? Toy: you write the
   attention from U05's spec, test the shapes, cite the
   spec. Original code, cited ideas.

3. **Mental model.** Implementation is translation: spec to
   code. The spec is cited, the code is yours. Copying the
   translation is not translation.

4. **Objects, symbols, units, shapes, assumptions.** The spec,
   your code, the tests, the citations. Assumption: the spec
   is complete enough to implement from.

5. **Derivation / mechanism.** The mechanism is the test: shape
   checks (U03 C11), grad checks (U03 C10), the toy numbers.
   Tests are the proof of understanding.

6. **Computed example.** Toy (hand, labeled as such): your
   attention matches the spec's shapes (4, 7, 50) and the
   grad check passes. The code is yours, the idea is cited.

7. **Algorithm and reference implementation.** The rule, not
   code: spec in, tests green, citations attached. Test: the
   toy passes.

8. **Correctness checks and expected output.** All tests green.
   Check: no copied code without attribution.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** Your time. The cost of copying: the policy
   violation (SRC-06).

10. **Nearest alternatives and selection boundaries.** Alternative:
    use the library. Choose implementation for the parts you
    must understand. Choose the library for the plumbing.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "clean code means my code." Counterexample:
    renamed variables on copied code. The policy cares about
    provenance, not cosmetics.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: none, this is integrity. The extension is the
    habit.

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u15_answers.md` A8 (breadth: state
    the originality rule).

14. **Lab/exercises with answers separated.** E15: implement one
    part from spec, test it. E16: cite the spec. Keys in
    `keys/u15_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a rule (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C09: reproducibility

Leaf id `cs224n-U15-C09`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to
   SRC-04. Scope: making the project rerunnable. Objectives:
   seed everything, log everything, state the tolerance.
   Depends on R2, P22.

2. **Motivating question and toy.** Question: your partner
   reruns your code, gets a different number, what broke? Toy:
   5 seeds give mean 0.8045, std 0.0203. The "different
   number" is inside the noise. Report the spread.

3. **Mental model.** Reproducibility is the receipt: seeds,
   versions, data, commands. A result without a receipt is a
   story.

4. **Objects, symbols, units, shapes, assumptions.** Seeds, the
   spread, the receipt. Assumption: the code is deterministic
   given the seed (true on CPU, mostly true on GPU).

5. **Derivation / mechanism.** The mechanism is the seed
   sweep: run k seeds, report mean and std. The tolerance is
   the std: differences inside it are noise.

6. **Computed example.** From `compute_u15.py`: mean 0.8045,
   std 0.0203. A rerun at 0.79 is consistent, at 0.70 is
   not.

7. **Algorithm and reference implementation.** `seed_sweep(
   run, seeds)`: the stats. ~6 lines. Test: the toy numbers.

8. **Correctness checks and expected output.** The seeds are
   fixed and logged. Check: the receipt reruns.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** k runs. The cost of one run: an anecdote.

10. **Nearest alternatives and selection boundaries.** Alternative:
    one seed, one number. Choose the sweep for any claim.
    Choose one seed for debugging.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "same seed, same number." Counterexample: GPU
    nondeterminism, library versions. Log those too.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: none, this is craft. The extension is the
    receipt habit.

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u15_answers.md` A9 (breadth), L9
    (ladder: define the receipt, compute the spread, derive
    the tolerance, diagnose the GPU case, design the
    receipt).

14. **Lab/exercises with answers separated.** E17: implement
    `seed_sweep`, match the toy. E18: write the receipt for
    the U09 lab. Keys in `keys/u15_answers.md`.

15. **Visual units, provenance, accessibility, audit row.**
    `visuals/u15_fig02.png`: the seed bars, Shell 5, source
    original toy. Audit row in `visual_audit.md`.

---

### C10: limitations

Leaf id `cs224n-U15-C10`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to
   SRC-04 (the report). Scope: the honest limits section.
   Objectives: list the limits, rank them, state what each
   would take to fix. Depends on U10 C12, U12 C12.

2. **Motivating question and toy.** Question: the project
   worked, what did it not do? Toy: the list: toy data (not
   real), 5 seeds (not 50), one language (not twelve), no
   human eval. Each with its fix cost.

3. **Mental model.** Limitations are the report's immune
   system: they stop the reader from overclaiming your
   work. A missing limitations section is a claim without a
   leash.

4. **Objects, symbols, units, shapes, assumptions.** The limit
   list with fix costs. Assumption: the list is complete
   (audit it).

5. **Derivation / mechanism.** No new math. The mechanism is
   the ranking: the limit that most threatens the claim goes
   first.

6. **Computed example.** Toy (hand, labeled as such): the four
   limits above, ranked: toy data first (it threatens
   everything).

7. **Algorithm and reference implementation.** `limitations(
   project)`: the ranked list. ~6 lines. Test: the toy list.

8. **Correctness checks and expected output.** Every claim in
   the report maps to a limit or a defense. Check: the top
   limit is named in the abstract.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** An honest hour. The cost of skipping it: the
   reviewer writes it for you, unkindly.

10. **Nearest alternatives and selection boundaries.** Alternative:
    bury the limits. Choose honesty always. The alternative
    is the rebuttal.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the list is complete." Counterexample: the
    limit you forgot. Have someone else read it.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: none, this is the close of the machinery.
    The extension is the habit.

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u15_answers.md` A10 (breadth:
    state the ranking rule).

14. **Lab/exercises with answers separated.** E19: write the
    limits for the U11 lab. E20: rank them. Keys in
    `keys/u15_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a list (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C11: open questions 2026

Leaf id `cs224n-U15-C11`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Scope: the
   horizon as of the October 2026 baseline. Objectives: name
   the open questions, mark each as open (not answered), state
   what would close it. Depends on the whole course.

2. **Motivating question and toy.** Question: what is still
   unknown? Toy: the list: the ICL mechanism (U08), faithful
   reasoning (U11), the scaling limit (U06), multilingual
   parity (U12), mechanistic understanding (U13). Each open,
   each with a closing experiment.

3. **Mental model.** Open questions are the field's todo list.
   This course ends where research begins. The list is dated:
   October 2026. It will age.

4. **Objects, symbols, units, shapes, assumptions.** Questions
   with closing criteria. Assumption: the list is partial
   (it is).

5. **Derivation / mechanism.** No new math. The mechanism is
   the criterion: each question names the experiment that
   would answer it.

6. **Computed example.** Toy (hand, labeled as such): "does
   ICL implement gradient descent?" Closing: the
   construction or the refutation, with the experiment.

7. **Algorithm and reference implementation.** `open_list()`: the
   dated list. ~6 lines. Test: 5 questions, each with a
   criterion.

8. **Correctness checks and expected output.** Every question
   is marked open. Check: the date is on the list.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** Reading time. The cost of skipping it: thinking
   the field is done.

10. **Nearest alternatives and selection boundaries.** Alternative:
    present answers. Choose open questions for honesty. The
    alternative is the textbook that aged badly.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the list is the field." Counterexample: the
    question nobody asked yet. Lists are partial.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: the questions are the extensions. Pick one.

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u15_answers.md` A11 (breadth:
    name 3 open questions and their closing criteria), L11
    (ladder: name the questions, mark them open, derive the
    criteria, diagnose the answered-in-disguise, design the
    experiment for one).

14. **Lab/exercises with answers separated.** E21: write the
    dated list. E22: add one question of your own with its
    criterion. Keys in `keys/u15_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a list (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C12: oral defense

Leaf id `cs224n-U15-C12`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Scope: the
   course's close. Objectives: defend the project orally,
   handle the ladders, state the defense rubric. Depends on
   the whole course.

2. **Motivating question and toy.** Question: can you defend
   your project out loud? Toy: the rubric: claim stated (1),
   evidence shown (1), limit named (1), transfer answered
   (1). Four points, no slides needed.

3. **Mental model.** The defense is the course in miniature:
   say what you did, show the numbers, name the limits,
   answer the changed scenario. If you can defend it, you
   know it.

4. **Objects, symbols, units, shapes, assumptions.** The
   rubric, the ladders (interview banks). Assumption: the
   defender did the work (the integrity rule).

5. **Derivation / mechanism.** No new math. The mechanism is
   the oral ladder: define, compute, derive, diagnose,
   design. The interview banks are the practice ground.

6. **Computed example.** Toy (hand, labeled as such): the
   defense of the U09 lab: claim (recall at k), evidence
   (0.000/0.333/0.667/1.000), limit (toy corpus), transfer
   (hourly updates). Four points.

7. **Algorithm and reference implementation.** `defend(
   project)`: the four-point check. ~6 lines. Test: the toy
   scores 4.

8. **Correctness checks and expected output.** Every point has
   evidence. Check: the transfer answer is not "I do not
   know" (it can be "here is the experiment").

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** An hour of talking. The cost of skipping it: the
   gap between writing and knowing.

10. **Nearest alternatives and selection boundaries.** Alternative:
    written exam. Choose oral for depth. Choose written for
    scale.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "fluent means correct." Counterexample: the
    fluent rationalization (U11 C12). The rubric scores
    evidence, not fluency.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: none, this is the close. The extension is the
    defense itself.

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u15_answers.md` A12 (breadth:
    state the rubric), L12 (ladder: state the claim, show
    the evidence, name the limit, answer the transfer,
    score the defense).

14. **Lab/exercises with answers separated.** E23: defend the
    U09 lab aloud. E24: defend the U11 lab aloud. Keys in
    `keys/u15_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a rubric (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

## Unit visual map

| Figure | Claim | Shell | Source |
|--------|-------|-------|--------|
| `visuals/u15_fig01.png` | tiny GPT-2: 44,928 params | 2 | original toy |
| `visuals/u15_fig02.png` | 5 seeds: mean 0.8045, std 0.0203 | 5 | original toy |
| `visuals/u15_fig03.png` | 4 project gates (SRC-04) | 10 | SRC-04 |

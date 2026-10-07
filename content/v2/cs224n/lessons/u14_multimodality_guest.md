# U14 , Multimodality and guest research

## Local remediation

Bridges: `../shared/prerequisites/p11_neural_nets.md` (P11, the
blocks being fused), `../shared/prerequisites/p14_transformer.md`
(P14, the shared stack), `../shared/prerequisites/p22_experiments.md`
(P22, reading claims honestly). This unit adds a second sense:
images next to text. The last three concepts are about honesty
itself: how to handle guest research you have not inspected.

R1. Fusion arithmetic: d = 1024, L = 24. Late fusion adds 5.03e7
params (two projections per layer). Early fusion adds 0: the
shared stack does the work.
R2. Routing: 8 experts, loads [3,2,3,2,3,2,2,3], aux loss 1.04.
Balanced routing keeps every expert useful.
R3. Token budget: 256 image tokens + 512 text tokens, image share
0.333. Images eat the window first.

---

### C01: image/text representation

Leaf id `cs224n-U14-C01`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to the
   multimodality guest session of the Winter 2026 schedule (S20
   per `course_map.md`, guest Luke Zettlemoyer). Scope: how images
   become tokens. Objectives: describe the patch encoding, state
   the token budget, name the resolution trade. Depends on P14.

2. **Motivating question and toy.** Question: the model reads
   text as tokens, what is an image to it? Toy: a 224x224 image
   cut into 16x16 patches: 196 patch tokens, each a linear
   projection of pixels. The image becomes a sentence of 196
   "words".

3. **Mental model.** The vision encoder is a translator: pixels
   in, tokens out. After translation the transformer cannot
   tell which tokens came from eyes and which from text. That
   blindness is the point.

4. **Objects, symbols, units, shapes, assumptions.** Patches
   (P x P), patch tokens (N = HW/P^2, d), a projection matrix.
   Assumption: the grid preserves what matters (fine detail
   dies at 16x16).

5. **Derivation / mechanism.** The mechanism is the patchify:
   reshape the image into N patches, flatten, project to d.
   Position embeddings mark the grid. After that it is tokens
   all the way down.

6. **Computed example.** Toy (hand, labeled as such): 224/16 =
   14 per side, 196 patches. At d = 1024, the image is 196
   tokens before a word is read.

7. **Algorithm and reference implementation.** `patchify(img,
   P)`: reshape, flatten, project. ~8 lines. Test: 196 tokens
   on the toy.

8. **Correctness checks and expected output.** The patch count
   is HW/P^2 exactly. Check: the projection output has dim d.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** 196 tokens per image at n^2 attention cost (U05
   C10). The practical cost: images eat the window (R3).

10. **Nearest alternatives and selection boundaries.** Alternative:
    a separate vision tower with late fusion (C02). Choose
    patch tokens for a unified stack. Choose the tower when
    the vision task needs its own capacity.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the grid preserves detail." Counterexample:
    small text in the image blurs past reading at 16x16.
    Resolution is a choice, not a given.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: patch 16 vs 8 on a reading-heavy task.
    Predict: 8 wins. Falsifier: tie (then the task does not
    need the detail).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u14_answers.md` A1 (breadth), L1
    (ladder: define patchify, compute 196, derive the token
    cost, diagnose the blurred text, design the patch test).

14. **Lab/exercises with answers separated.** E1: implement
    `patchify`, match 196. E2: halve P, show the 4x tokens.
    Keys in `keys/u14_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a reshape (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C02: early/late fusion

Leaf id `cs224n-U14-C02`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S20.
   Scope: where the modalities meet. Objectives: define both
   fusions, compute the param gap, choose per budget. Depends
   on C01, R1.

2. **Motivating question and toy.** Question: do image and text
   mix at the input or at the output? Toy: early fusion: 0
   extra params. Late fusion: 5.03e7 extra (d = 1024, L = 24).
   The meeting point has a price.

3. **Mental model.** Early fusion is one room: both modalities
   enter the same stack as tokens. Late fusion is two rooms
   with a door: separate towers, joined by projections at the
   end.

4. **Objects, symbols, units, shapes, assumptions.** The
   projection matrices (d x d each), the layer count.
   Assumption: the shared stack can learn both (capacity
   question).

5. **Derivation / mechanism.** Late fusion adds 2 projections
   per layer: 2 d^2 L params = 5.03e7. Early fusion adds the
   patch projection only. The mechanism is the wiring
   diagram.

6. **Computed example.** From `compute_u14.py`: 5.03e7 vs 0.
   On a 7B model that is 0.7 percent: small but not free.

7. **Algorithm and reference implementation.** `fuse(early)`: the
   wiring choice. ~6 lines. Test: the param counts.

8. **Correctness checks and expected output.** The counts match
   the formula. Check: early fusion's token stream mixes both
   modalities.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** Late fusion costs params, early fusion costs
   context (image tokens in the shared window).

10. **Nearest alternatives and selection boundaries.** Alternative:
    frozen towers with adapters. Choose early for tight
    integration. Choose late when the towers are pretrained
    and frozen.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "early fusion is always cheaper." Counterexample:
    the image tokens bloat every layer's attention to n^2.
    Params are not the only bill.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: early vs late at equal total params, measure
    the task gap. Predict: early wins on tight tasks.
    Falsifier: tie (then the wiring matters less than the
    data).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u14_answers.md` A2 (breadth), L2
    (ladder: define both, compute 5.03e7, derive the wiring,
    diagnose the attention bill, design the equal-param
    test).

14. **Lab/exercises with answers separated.** E3: compute both
    param counts. E4: price the attention bill for 196 image
    tokens. Keys in `keys/u14_answers.md`.

15. **Visual units, provenance, accessibility, audit row.**
    `visuals/u14_fig01.png`: the wiring and the param gap,
    Shell 8, source original toy. Audit row in
    `visual_audit.md`.

---

### C03: mixed-modal tokens

Leaf id `cs224n-U14-C03`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S20.
   Scope: one token stream for both senses. Objectives: build
   the mixed stream, compute the budget share, state the
   ordering rule. Depends on C01, C02.

2. **Motivating question and toy.** Question: the input is an
   image plus a question, what does the model see first? Toy:
   256 image tokens then 512 text tokens, image share 0.333.
   Order matters: the question can attend to the image.

3. **Mental model.** The stream is a mixed tape: image tokens,
   then text, or interleaved. The transformer reads left to
   right (causal) or all at once (bidirectional): the order
   decides what can see what.

4. **Objects, symbols, units, shapes, assumptions.** The token
   budget, the image share, the order. Assumption: the
   position embeddings distinguish the modalities (or the
   model confuses them).

5. **Derivation / mechanism.** Share = image tokens over total.
   The mechanism is concatenation with modality markers. In a
   causal model, put the image first so the text can attend
   to it.

6. **Computed example.** From `compute_u14.py`: 256 + 512 =
   768, image share 0.333. A third of the window is vision.

7. **Algorithm and reference implementation.** `mixed_stream(
   img_toks, txt_toks)`: concat with markers. ~5 lines.
   Test: the share is 0.333.

8. **Correctness checks and expected output.** The markers are
   present and unambiguous. Check: the total fits the window.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** The stream length sets the n^2 bill. The cost of
   images: fewer text tokens fit.

10. **Nearest alternatives and selection boundaries.** Alternative:
    cross-attention (text attends to image features, no image
    tokens). Choose the stream for simplicity. Choose cross-
    attention when the image is large and the text is long.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "order does not matter." Counterexample: text
    before image in a causal model: the question cannot see
    the image. Order is architecture.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: image-first vs text-first, measure the VQA
    gap. Predict: image-first wins. Falsifier: tie (then the
    model is bidirectional, check).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u14_answers.md` A3 (breadth: state
    the share and the ordering rule).

14. **Lab/exercises with answers separated.** E5: build the
    stream, match 0.333. E6: flip the order, show the causal
    break. Keys in `keys/u14_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a stream (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C04: joint generation

Leaf id `cs224n-U14-C04`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S20.
   Scope: generating in two modalities at once. Objectives:
   define the joint objective, price the toy, state the
   alignment problem. Depends on C01-C03.

2. **Motivating question and toy.** Question: the model writes a
   caption and draws the image, how? Toy: autoregressive text
   (128 tokens at cost 1) plus diffusion image (20 steps at
   cost 8): total 288. Two engines, one bill.

3. **Mental model.** Joint generation is two craftsmen sharing
   a shop: the text engine writes, the image engine draws,
   and they must agree on the subject. Agreement is the hard
   part.

4. **Objects, symbols, units, shapes, assumptions.** Token
   costs, step costs, the total. Assumption: the costs are
   comparable units (the toy's simplification).

5. **Derivation / mechanism.** Total = text tokens times 1 +
   diffusion steps times 8. The mechanism is the schedule:
   generate text, condition the image on it (or jointly).

6. **Computed example.** From `compute_u14.py`: 128 + 160 =
   288. The image costs more than the text.

7. **Algorithm and reference implementation.** `joint_cost(
   toks, steps)`: the sum. ~4 lines. Test: 288 on the toy.

8. **Correctness checks and expected output.** The cost splits
   by modality (auditable). Check: zero steps gives text-
   only cost.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** Diffusion steps dominate. The practical cost: the
   image engine's latency.

10. **Nearest alternatives and selection boundaries.** Alternative:
    text only, image retrieved (U09 C04's machinery). Choose
    generation when the image must be novel. Choose retrieval
    when it exists already.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the modalities agree." Counterexample: the
    caption says "red car", the image shows blue. Joint
    training, not just joint billing, fixes agreement.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: joint vs pipeline (text then image), measure
    agreement. Predict: joint wins. Falsifier: tie (then the
    conditioning is enough).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u14_answers.md` A4 (breadth), L4
    (ladder: define the joint, price 288, derive the split,
    diagnose the red/blue case, design the agreement test).

14. **Lab/exercises with answers separated.** E7: implement
    `joint_cost`, match 288. E8: price text-only vs joint.
    Keys in `keys/u14_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a bill (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C05: diffusion/autoregressive mix

Leaf id `cs224n-U14-C05`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S20.
   Scope: the two generation paradigms side by side. Objectives:
   contrast the paradigms, state when each wins, name the
   interface. Depends on C04, U04.

2. **Motivating question and toy.** Question: text is
   autoregressive, images are diffusion, why? Toy: text has a
   natural order (left to right), images do not. The paradigm
   follows the data's structure.

3. **Mental model.** Autoregression tells a story: one token
   at a time, each conditioned on the past. Diffusion is
   sculpting: start from noise, refine everywhere at once.
   Stories need order, pictures need global coherence.

4. **Objects, symbols, units, shapes, assumptions.** Next-token
   likelihood (AR), denoising steps (diffusion). Assumption:
   the modality's structure matches the paradigm.

5. **Derivation / mechanism.** AR factorizes the joint as a
   product of conditionals. Diffusion learns to reverse a
   noise process. The interface: AR text conditions the
   diffusion (the prompt guides the sculpting).

6. **Computed example.** Toy (hand, labeled as such): AR text
   at 128 tokens, diffusion at 20 steps. The text is the
   plan, the image is the execution.

7. **Algorithm and reference implementation.** `generate_mix(
   prompt)`: AR text, then diffusion conditioned on it. ~8
   lines. Test: both outputs produced.

8. **Correctness checks and expected output.** The image
   respects the text prompt (the agreement check). Check:
   the text is fluent alone.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** AR is sequential, diffusion is parallel per step
   but needs many steps. The bill: C04's 288.

10. **Nearest alternatives and selection boundaries.** Alternative:
    AR for everything (image as tokens). Choose diffusion
    for image quality. Choose AR tokens for a unified
    model.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the paradigms compose cleanly." Counterexample:
    the diffusion ignores the AR text (weak conditioning).
    Strengthen the conditioning, measure agreement.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: strong vs weak conditioning, measure the
    agreement rate. Predict: strong wins. Falsifier: tie
    (then the text was not the bottleneck).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u14_answers.md` A5 (breadth: name
    the two paradigms and the interface).

14. **Lab/exercises with answers separated.** E9: sketch the
    mix pipeline. E10: list the failure if conditioning is
    weak. Keys in `keys/u14_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a contrast (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C06: sparse modality experts

Leaf id `cs224n-U14-C06`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S20.
   Scope: experts per modality. Objectives: define the
   routing, compute the aux loss, state the collapse rule.
   Depends on C02, R2.

2. **Motivating question and toy.** Question: one stack for both
   senses, or specialists? Toy: 8 experts (4 image, 4 text),
   10 tokens routed as [3,2,3,2,3,2,2,3], aux loss 1.04.
   Balanced, no collapse.

3. **Mental model.** Experts are specialists, the router is the
   receptionist. Image tokens go to image experts, text to
   text experts. If the receptionist naps, one expert does
   everything: collapse.

4. **Objects, symbols, units, shapes, assumptions.** Expert
   loads, the aux loss (mean squared normalized load).
   Assumption: the router learns the split (it needs the aux
   loss to do so).

5. **Derivation / mechanism.** Aux = mean((load / mean)^2).
   Perfect balance gives 1.0. The mechanism: add the aux loss
   to train time, the router spreads the load.

6. **Computed example.** From `compute_u14.py`: aux 1.04 on
   the toy. Near 1.0: healthy.

7. **Algorithm and reference implementation.** `route(tokens,
   experts)`: top-1 per token, aux loss. ~8 lines. Test: the
   toy aux.

8. **Correctness checks and expected output.** Loads sum to the
   token count. Check: aux near 1.0.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** Sparse means cheap per token (one expert), dear
   in total params (all experts stored).

10. **Nearest alternatives and selection boundaries.** Alternative:
    dense shared stack. Choose experts for capacity without
    per-token cost. Choose dense for simplicity.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the router balances." Counterexample: aux
    weight 0, one expert takes all. The aux loss is
    load-bearing.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: aux weight in {0, 0.01, 0.1}, measure the max
    expert share. Predict: 0 collapses. Falsifier: no
    collapse (then the data routes itself, check).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u14_answers.md` A6 (breadth), L6
    (ladder: define routing, compute 1.04, derive the aux,
    diagnose the collapse, design the weight sweep).

14. **Lab/exercises with answers separated.** E11: implement
    `route`, match the aux. E12: set aux weight 0, show the
    collapse. Keys in `keys/u14_answers.md`.

15. **Visual units, provenance, accessibility, audit row.**
    `visuals/u14_fig02.png`: the expert loads, Shell 3,
    source original toy. Audit row in `visual_audit.md`.

---

### C07: multimodal retrieval

Leaf id `cs224n-U14-C07`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S20.
   Scope: finding images with text and vice versa. Objectives:
   define the joint space, compute the fused score, state the
   weight rule. Depends on U09 C04, C01.

2. **Motivating question and toy.** Question: the query is text,
   the answer is an image, how do they meet? Toy: text score
   0.72, image score 0.41, fused at w = 0.5: 0.565. The joint
   space is the meeting room.

3. **Mental model.** U09 C07's fusion, across senses: embed
   text and images into one space, score by similarity. The
   weight w dials which sense leads.

4. **Objects, symbols, units, shares, assumptions.** Text
   score, image score, fusion weight w. Assumption: the joint
   space aligns the senses (trained, not given).

5. **Derivation / mechanism.** Fused = w times image + (1 - w)
   times text. The mechanism is the convex mix, same as U09
   C07.

6. **Computed example.** From `compute_u14.py`: w = 0.0 gives
   0.720, w = 0.5 gives 0.565, w = 1.0 gives 0.410. Tune w on
   a dev set.

7. **Algorithm and reference implementation.** `mm_retrieve(
   query, index, w)`: score both, fuse, rank. ~8 lines.
   Test: the three toy scores.

8. **Correctness checks and expected output.** w = 0
   reproduces text-only ranking. Check: the fused pool is the
   union.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** Two indexes, one fusion. The cost is the joint
   embedding training.

10. **Nearest alternatives and selection boundaries.** Alternative:
    text-only retrieval with captions. Choose joint when
    images lack captions. Choose captions when they exist
    (cheaper).

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the space aligns." Counterexample: the text
    query matches the wrong image's caption-like region.
    Alignment needs the training, check it.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: sweep w, measure recall. Predict: interior
    optimum. Falsifier: w = 0 wins (then the images add
    nothing, check the index).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u14_answers.md` A7 (breadth: write
    the fusion and the weight rule).

14. **Lab/exercises with answers separated.** E13: implement
    `mm_retrieve`, match the scores. E14: sweep w. Keys in
    `keys/u14_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a fusion (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C08: instruction tuning

Leaf id `cs224n-U14-C08`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S20.
   Scope: teaching the multimodal model to follow instructions.
   Objectives: define the format, state the data need, name
   the eval. Depends on U07 C01, C03.

2. **Motivating question and toy.** Question: the model sees
   images and text, how does it learn to answer questions?
   Toy: instruction triples (image, question, answer), SFT
   loss on the answer tokens. Same recipe as U07, new
   modality.

3. **Mental model.** Instruction tuning is the same school,
   new students: the format (image + instruction + response)
   teaches the behavior. The vision encoder learns what the
   questions need.

4. **Objects, symbols, units, shapes, assumptions.** The triple
   format, the SFT loss. Assumption: the triples cover the
   behaviors (the data question again).

5. **Derivation / mechanism.** The mechanism is U07 C01's: next-
   token loss on the response, image tokens in context. The
   gradients reach the vision encoder through the stack.

6. **Computed example.** Toy (hand, labeled as such): 10k
   triples, loss falls, VQA accuracy rises 0.12. The format
   teaches the task.

7. **Algorithm and reference implementation.** `mm_sft(
   triples)`: format, loss on answers. ~8 lines. Test: the
   loss falls on the toy.

8. **Correctness checks and expected output.** The loss masks
   the instruction tokens (answers only). Check: image tokens
   get gradients.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** SFT is cheap next to pretrain. The cost is the
   triples: human-written multimodal data.

10. **Nearest alternatives and selection boundaries.** Alternative:
    zero-shot from the pretrained model. Choose tuning when
    the behavior must be reliable. Choose zero-shot for
    exploration.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "tuning teaches vision." Counterexample: the
    model learns to answer from the text alone, ignoring the
    image (the language prior). Test with image-ablated
    inputs.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: ablate the image at test, measure the drop.
    Predict: a drop (it uses vision). Falsifier: no drop
    (then it is a text model with pictures).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u14_answers.md` A8 (breadth: state
    the format and the prior trap).

14. **Lab/exercises with answers separated.** E15: format the
    triples. E16: ablate the image, show the drop (or not).
    Keys in `keys/u14_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a format (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C09: reward evaluation

Leaf id `cs224n-U14-C09`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Maps to S20.
   Scope: judging multimodal outputs. Objectives: define the
   judge's job, state the agreement problem, name the
   calibration need. Depends on U10 C03, C08.

2. **Motivating question and toy.** Question: the model
   describes an image, who grades it? Toy: a judge model
   scores 100 descriptions, agrees with humans at kappa
   0.55 (U10's number, reused honestly). The judge is an
   instrument, calibrate it.

3. **Mental model.** U10 C03's judge, new modality: the judge
   must see the image too. A blind judge grades text, not
   the pair.

4. **Objects, symbols, units, shapes, assumptions.** Judge
   scores, human scores, kappa. Assumption: the humans are
   the reference.

5. **Derivation / mechanism.** The mechanism is U10 C03's:
   kappa for agreement, Brier for calibration. The new part:
   the judge needs the image input.

6. **Computed example.** Toy (hand, labeled as such): kappa
   0.55 with image, 0.31 without. The blind judge is worse.

7. **Algorithm and reference implementation.** `mm_judge(
   desc, img)`: score with the image. ~6 lines. Test: the
   toy kappas.

8. **Correctness checks and expected output.** The judge sees
   the same image as the model. Check: kappa reported with
   n.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** The judge costs a multimodal forward pass per
   grade. The cost of skipping calibration: U10's lesson.

10. **Nearest alternatives and selection boundaries.** Alternative:
    human grading. Choose the judge for scale, humans for the
    gate.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the judge sees what the model saw."
    Counterexample: different resolutions, different crops.
    Match the inputs.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: judge with vs without image, measure kappa.
    Predict: with wins. Falsifier: tie (then the text
    carries the grade, check the task).

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u14_answers.md` A9 (breadth: state
    the judge's input requirement).

14. **Lab/exercises with answers separated.** E17: score the
    toy with and without image. E18: compute both kappas.
    Keys in `keys/u14_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a judge (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C10: guest Tinker/LoRA artifacts

Leaf id `cs224n-U14-C10`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** The
   inventory names "guest Tinker/LoRA artifacts" as a requested
   leaf. No artifact was inspected in this build: no slides,
   no video, no paper. Scope: title-only handling. Objectives:
   state exactly what is known (the title), mark everything
   else unknown, write the card. Depends on C11, P22.

2. **Motivating question and toy.** Question: the leaf names a
   guest artifact, what do you teach? Toy: the card reads
   "title: guest Tinker/LoRA artifacts (inventory label),
   content: not inspected, claims: none". The card is the
   whole lesson.

3. **Mental model.** A title is a pointer to work you have not
   seen. Teaching the pointer honestly means saying what it
   points at (unknown) and how you would verify it (the
   artifact list). This is U11 C03's reading card, stricter.

4. **Objects, symbols, units, shapes, assumptions.** The card:
   label, source (the inventory), inspection status (none),
   claims (none). Assumption: none about the content.

5. **Derivation / mechanism.** No new math. The mechanism is
   the boundary: the leaf stays open until an artifact is
   inspected. C12 tracks it.

6. **Computed example.** Toy (hand, labeled as such): the card
   above. Nothing is computed because nothing is known.

7. **Algorithm and reference implementation.** `guest_card(
   label)`: the four fields, status "not inspected". ~5
   lines. Test: the toy card.

8. **Correctness checks and expected output.** No content claim
   appears under this leaf. Check: the card is referenced
   wherever the label appears.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** Five minutes. The cost of skipping it: inventing
   a guest talk.

10. **Nearest alternatives and selection boundaries.** Alternative:
    drop the leaf. Choose the card: the leaf stays visible
    and honest. Choose dropping only if the inventory
    retires it.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "Tinker means the product." Counterexample:
    the label could be a codename, a demo, or a typo. Do not
    resolve ambiguity by guessing.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: none until inspection. The extension is the
    inspection itself.

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u14_answers.md` A10 (breadth:
    state the card and the no-claim rule), L10 (ladder:
    define the card, write the toy, derive the boundary,
    diagnose the codename guess, design the inspection).

14. **Lab/exercises with answers separated.** E19: write the
    card. E20: find one place the label tempts a claim, mark
    it. Keys in `keys/u14_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a card (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C11: distinguish actual talk content

Leaf id `cs224n-U14-C11`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Scope: the
   method for handling guest sessions honestly. Objectives:
   define the three content classes, classify a claim, state
   the citation rule. Depends on P22.

2. **Motivating question and toy.** Question: someone says "the
   guest showed X", how do you check? Toy: three classes:
   (a) inspected artifact, (b) schedule title, (c) hearsay.
   "The guest showed X" needs (a). The schedule gives (b).
   Everything else is (c).

3. **Mental model.** Claims have passports. Class (a) travels
   anywhere, class (b) travels with its title stamp, class
   (c) does not travel. Check the passport before repeating
   the claim.

4. **Objects, symbols, units, shapes, assumptions.** The three
   classes, the citation each requires. Assumption: the
   classifier is strict (when in doubt, demote).

5. **Derivation / mechanism.** No new math. The mechanism is
   the routing: each claim about a talk gets a class, each
   class has a citation format. Demote on doubt.

6. **Computed example.** Toy (hand, labeled as such): "S20 was
   a guest session on multimodality": class (b), cite the
   schedule. "The guest showed early fusion": class (c),
   do not repeat.

7. **Algorithm and reference implementation.** `classify_claim(
   claim, evidence)`: the three classes. ~6 lines. Test: the
   toy routing.

8. **Correctness checks and expected output.** No class-(c)
   claim appears in a lesson as fact. Check: every guest
   mention carries its class.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** A minute per claim. The cost of skipping it: the
   telephone game.

10. **Nearest alternatives and selection boundaries.** Alternative:
    trust the retelling. Choose the classes for anything
    cited. The alternative is how myths start.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the schedule title is the talk." Counterexample:
    tentative schedules change (the course says so). The
    title is a plan, not a transcript.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: none, this is method. The extension is the
    habit.

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u14_answers.md` A11 (breadth:
    name the three classes), L11 (ladder: define the
    classes, classify the toy, derive the demote rule,
    diagnose the title-as-talk, design the citation
    check).

14. **Lab/exercises with answers separated.** E21: classify 6
    toy claims. E22: demote one class-(c) claim. Keys in
    `keys/u14_answers.md`.

15. **Visual units, provenance, accessibility, audit row.** Claim
    is a method (carried in text). Logged as an honest
    exception in `visual_audit.md`.

---

### C12: source gaps

Leaf id `cs224n-U14-C12`. Claim class REQUESTED-BRANCH.
Status PLANNED / SOURCE ATTRIBUTION PENDING.

1. **Source mapping, scope, objectives, dependencies.** Scope: the
   unit's close and the course's gap ledger. Objectives: list
   the U14 gaps, state the closure rule, keep the ledger.
   Depends on C10, C11.

2. **Motivating question and toy.** Question: what is still
   unknown after this unit? Toy: the ledger: S20 talk content
   (uninspected), Tinker/LoRA artifacts (uninspected), all
   C01-C09 mechanisms (requested-branch, toy-taught). Each
   with its closure rule.

3. **Mental model.** The gap ledger is the course's honesty
   file. Every open item names what would close it. A course
   without a ledger is a course that forgot what it does not
   know.

4. **Objects, symbols, units, shapes, assumptions.** Gap rows:
   item, status, closure rule. Assumption: the ledger is
   complete (audit it).

5. **Derivation / mechanism.** No new math. The mechanism is
   the row: status open until the named artifact is
   inspected. C10's card feeds this ledger.

6. **Computed example.** Toy (hand, labeled as such): 3 open
   rows for U14. The count is the deliverable.

7. **Algorithm and reference implementation.** `gap_ledger(
   unit)`: rows with closure rules. ~6 lines. Test: the 3
   toy rows.

8. **Correctness checks and expected output.** Every PLANNED
   leaf appears in the ledger. Check: no row closes without
   the artifact.

9. **Complexity, memory, statistical efficiency, stability, practical
   costs.** Maintenance time. The cost of skipping it: the
   audit fails.

10. **Nearest alternatives and selection boundaries.** Alternative:
    close gaps by assumption. Choose the ledger always. The
    alternative is the audit finding.

11. **Failure case, broken assumption, counterexample.** Broken
    assumption: "the ledger is complete." Counterexample: the
    gap you forgot. The audit (different agent) checks.

12. **Research reading and falsifiable extension.** Falsifiable
    extension: none, this is the close. The extension is the
    audit.

13. **Breadth recall, deep oral ladder, unfamiliar transfer.**
    Assessment block: `keys/u14_answers.md` A12 (breadth:
    state the ledger rule), L12 (ladder: define the ledger,
    list the toy rows, derive the closure rule, diagnose the
    assumed-closure, design the audit check).

14. **Lab/exercises with answers separated.** E23: write the
    U14 ledger. E24: close one row honestly (or mark why
    not). Keys in `keys/u14_answers.md`.

15. **Visual units, provenance, accessibility, audit row.**
    `visuals/u14_fig03.png`: the claim-class plate, Shell 10,
    source this build's lessons. Audit row in
    `visual_audit.md`.

---

## Unit visual map

| Figure | Claim | Shell | Source |
|--------|-------|-------|--------|
| `visuals/u14_fig01.png` | early 0 vs late 5.03e7 params | 8 | original toy |
| `visuals/u14_fig02.png` | routing aux loss 1.04 | 3 | original toy |
| `visuals/u14_fig03.png` | U14: 12 requested-branch leaves | 10 | this build |

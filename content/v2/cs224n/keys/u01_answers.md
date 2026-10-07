# Answer key , U01 NLP history, tasks, and data

Attempt the exercises before reading. Ladders are oral: answer aloud,
then check.

## Remediation

R1. c(cat) = 5, c(cat, sat) = 2. MLE: 2/5 = 0.4. Add-1: (2 + 1) /
(5 + 14) = 3/19 = 0.1579. Matches `compute_u01.py`.

## Breadth

A1. Three levels: word ("bank": river edge vs money store), phrase
("visiting relatives": noun phrase vs verb + object), discourse or
intent ("great, another meeting": sincere vs sarcastic). Ambiguity
breaks rule pipelines because one input maps to several structures and
a rule that commits early cannot recover.

A2. Symbolic: rules written by hand, updated by editing. Statistical:
counts from a corpus, updated by recounting. Neural: vectors and
differentiable maps, updated by gradient steps. Word2vec (2013) sits
at the statistical-to-neural boundary: count-based objective,
vector output.

A3. A corpus is a fixed document collection used as evidence. Three
design decisions: source selection (which genres), cleaning level
(what gets deleted), split plan (how documents divide into
train/validation/test). Two biases: genre over-representation, time
drift.

A4. Naive Bayes: P(c | d) proportional to P(c) x product of P(w | c),
scored in log space with add-1 smoothing. Two failures: the
independence assumption ignores word order, tiny data lets smoothing
dominate evidence (the toy predicts "finance" for "cat bank").

A5. Dependency parse: directed arcs head -> dependent, one head per
word, no cycles. Constituency: nested phrases. Ambiguity makes parsing
a search problem: exponentially many trees, so the parser must score,
not enumerate.

A6. Three subproblems: sense (which meaning), lexical choice (which
target words), reordering (which order). BLEU: geometric mean of
clipped n-gram precisions (n = 1..4) times a brevity penalty
exp(1 - ref_len/cand_len) when the candidate is short.

A7. A language model is a distribution over sequences, factored by the
chain rule into next-word predictions. Perplexity = exp(mean negative
log-likelihood): the effective number of equally likely choices per
word. Toy: 5.137, about a 5-sided die.

A8. "You shall know a word by the company it keeps." Build
cooccurrence vectors in a fixed window, cosine measures neighbor
overlap. Window 2 around the first "cat": {the:1, sat:1, on:1}, slid
one right: {the:2, cat:1, on:1}.

A9. Train fits parameters, validation picks settings, test reports the
claimed number. Two leaks: vocabulary or counts built on test data,
near-duplicate documents across splits. Fix: dedup first, split by
document, build everything from train only.

A10. Five slots: input, output, data (with splits), metric, decision
(the action the output drives and the cost of being wrong). A task
with a blank slot is a wish, not a task.

A11. Accuracy: fraction right. Precision: TP/(TP+FP). Recall:
TP/(TP+FN). F1: harmonic mean 2PR/(P+R), which is 0 when either is 0.
On the 99/1 toy with a constant classifier: accuracy 0.99, precision
0.0, recall 0.0, F1 0.0.

A12. Honesty note, in your own words: the Winter 2026 schedule,
assignments, tutorials, and project descriptions anchor scope, 2024
public videos need separate edition tags, I claim no viewing of
restricted 2026 videos. A leaf is PLANNED / SOURCE ATTRIBUTION PENDING
until an inspected artifact verifies it.

## Oral ladders

L1 (ambiguity). Define: one input, several valid interpretations.
Toy: "visiting relatives can be dull", two parses. Rule: -ing forms
attach as noun or verb. New sentence: "flying planes can be
dangerous" , same branch. Experiment: 200 -ing sentences, parser
errors vs human slowdown, predict correlation.

L2 (eras). Define the three eras by stored knowledge and update rule.
Toy: the "not X" rule cascade. Derive: 196 bigram cells, 22 nonzero,
sparsity 0.888. Zip-code task: regex wins, exact and auditable.
Critique: "neural always wins" fails where constraints are hard and
data is zero.

L3 (corpora). Define corpus as sample, not language. Profile the toy:
6 docs, 35 tokens, 14 types, ratio 0.40. Derive ratio meaning: lexical
variety. Predict: SEO spam triples tokens, drops quality. Experiment:
vary dedup threshold, fixed model, measure accuracy.

L4 (classification). Define scoring + argmax. Hand-score "cat bank":
pets -5.416, finance -5.011, predict finance. Derive log form from
Bayes rule. Diagnose: 100 duplicate "the cat sat" docs shift the
"the" likelihood and break finance docs. Experiment: accuracy vs data
size for NB vs logistic regression, five seeds.

L5 (parsing). Define arc set and tree constraints. Wire "the cat
sat": sat -> cat (nsubj), cat -> the (det). Search blowup: Catalan
growth. Diagnose garden-path: greedy cannot backtrack. Experiment:
greedy vs beam-4 on garden-path vs normal.

L6 (translation). Define the three subproblems. Hand-compute BLEU-2:
precisions 1.0, 1.0, BP exp(1 - 6/3) = 0.368, score 0.368. Explain BP:
punishes short perfect guesses. Diagnose: scrambled candidate keeps
unigram precision. Experiment: BLEU vs human rank correlation by
genre.

L7 (language modelling). Define distribution + chain rule. Hand-score
"the cat sat": 0.24 x 0.1579, H = 1.636 nats, ppl 5.137. Derive the
"effective choices" reading from the uniform case. Diagnose:
memorization gives great perplexity on leaked test. Experiment:
shuffled vs intact training, grammaticality separation.

L8 (distributional hypothesis). State the hypothesis. Hand-build the
"cat" vector. Derive cosine. Antonym failure: "hot"/"cold" share
contexts, cosine calls them similar. Experiment: cosine ranking vs
human judgments, antonyms as predicted outliers.

L9 (splits). Assign the three jobs. Compute OOV: 2/6 = 0.333 clean,
0.0 leaky. Two leak mechanisms: test-built vocab, cross-split
duplicates. Diagnose: 95% test, 70% production from duplicate crawl.
Experiment: random vs time split predicting a future slice.

L10 (formulation). Name the five slots. Fill for the toy NB task.
Derive loss vs metric: loss is differentiable, metric is
task-meaningful. Diagnose the 99% accuracy trap: constant classifier,
minority recall 0. Experiment: audit three sentiment datasets for
slot mismatches.

L11 (metrics). Define TP/FP/FN/TN and the four formulas. Compute the
toy: (0.0, 0.0, 0.0) vs accuracy 0.99. Derive the harmonic choice:
arithmetic mean hides R = 0. Diagnose BLEU Goodhart. Experiment:
F1-tuned vs accuracy-tuned thresholds under a 10x minority cost.

## Exercises

E1. Accept any three sentences with two stated readings each, e.g.
word: "I saw the bank", phrase: "old men and women", intent: "nice
job on the report" (sincere/sarcastic). One mark per valid pair.

E2. `first_diff` returns 0 on the toy pair: same tokens, structure
differs from word 0. A duplicate-structure third reading must keep the
count at 2.

E3. 6/6: decision tree over counts -> statistical, transformer ->
neural, regex cascade -> symbolic, Naive Bayes over embeddings ->
statistical (decision rule decides), bigram LM -> statistical,
fine-tuned BERT classifier -> neural.

E4. 22 nonzero cells of 196. Sparsity = 1 - 22/196 = 0.888.

E5. Profiler output: docs 6, tokens 35, types 14, ratio 0.40, top
"the": 11, duplicates 0. Matches `compute_u01.py`.

E6. Genre A (pets) neighbors of "bank": none present , genre B
(finance) neighbors: {stock, rate, loan}. Disjoint by construction,
the point is that meaning follows the corpus.

E7. Scores match (-5.416, -5.011) to 3 decimals, prediction
"finance". With pets prior 0.9 the log-prior shift flips it to pets.

E8. Flip at p = 0.60: at 0.55 scores are -5.321 vs -5.117 (finance),
at 0.60 both -5.234 (tie), at 0.65 pets wins -5.154 vs -5.368.

E9. "the cat sat": arcs listed in L5. "a dog barked": barked -> dog
(nsubj), dog -> a (det). Labels may vary if justified.

E10. Action replay yields exactly the gold arc set, no cycles, one
head per word.

E11. `bleu2` returns 0.368 on the toy.

E12. Example: candidate "mat the on sat cat the" vs reference "the
cat sat on the mat": unigram precision 6/6 = 1.0, bigram matches 0
("mat the", "the on", ... none in reference) so BLEU-2 = 0.0 < 0.5.

E13. `bigram_ppl` returns 5.137 on ("the cat sat", toy corpus).

E14. MLE: doubling all counts leaves every ratio c(u,v)/c(u)
unchanged, so perplexity is unchanged. Smoothed: (2c+1)/(2c(u)+V)
moves toward the MLE ratio as counts grow, so perplexity falls toward
the MLE value. Smoothing matters less with more data.

E15. `cooccur` reproduces {the:1, sat:1, on:1} and {the:2, cat:1,
on:1}. Cosine self-check 1.0, disjoint 0.0.

E16. "rug"/"sat": cosine 0.983, yet a rug is not an act of sitting.
Corpus cause: both sit in the fixed frame "sat on the __", the toy
corpus has only two such frames, so the vectors coincide.

E17. Clean vocab (train on sentences 1-4, test on sentence 5): OOV 2/6 = 0.333. Leaky vocab: 0.0. The
assertion on vocab-builder input must fail if a test doc is passed.

E18. With the duplicate planted, test accuracy (or test perplexity)
improves while a fresh paraphrase set does not move: the signature of
memorization, not generalization.

E19. Spam spec: input email text, output {spam, ham}, data with dated
splits, metric F1 on spam, decision quarantine. QA spec: input
(question, passage), output answer span, data SQuAD-style splits,
metric exact match / F1, decision show to user. Both need all five
slots.

E20. Three specs: the C04 toy passes, label-mismatch spec rejected,
no-decision spec rejected.

E21. `prf(0, 0, 1)` = (0.0, 0.0, 0.0) by the zero-division convention.

E22. Pair 1 BLEU-2 0.368, pair 2 BLEU-2 1.0. Mean sentence: 0.684.
Corpus: clipped counts 9/9 unigram, 7/7 bigram, BP exp(1 - 12/9) =
0.7165, corpus BLEU-2 = 0.7165. Corpus BLEU micro-averages, so the
long perfect sentence dominates, mean sentence BLEU weights sentences
equally.

E23. Five eras in order, each naming the bottleneck it removed:
rules -> statistics (authoring cost), statistics -> neural
(sparsity), neural -> pretraining (labeled data), pretraining ->
scale/alignment (objective mismatch with human judgment).

E24. "A2 covers dependency parsing": OFFICIAL-SOURCE (assignment title
on the inspected site). "Lecture 3 teaches arc-standard parsing":
REQUESTED-BRANCH (no lecture artifact inspected). "The 2026 videos
show X": RESTRICTED-UNVIEWED. "The 2024 videos show Y":
EDITION-2024, PLANNED until viewed.

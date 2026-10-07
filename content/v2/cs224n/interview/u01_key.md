# Interview key , U01

Each answer gives the minimum sufficient explanation, a strong answer,
common red flags, a scoring rubric, and remediation.

## B1

Minimum: word ("bank"), phrase ("visiting relatives"), discourse or
intent (sarcasm). Strong: adds why each level breaks a different
pipeline stage. Red flag: only word-level examples. Rubric: 2 marks
for three levels with examples, 1 for one level. Remediation: C01.

## B2

Minimum: rules edited by hand, counts recounted on new data, vectors
moved by gradient steps. Strong: names the generalization mechanism
of each (intent, smoothing, geometry). Red flag: "neural is just
better". Rubric: 2 for store+update of all three. Remediation: C02.

## B3

Minimum: a fixed document collection used as evidence. "Bigger is
better" is false because size without curation adds spam, duplicates,
and bias, the model learns the sample, not the language. Strong:
cites the SEO-spam counterexample and dedup. Red flag: "more data
always helps". Rubric: 2 for definition plus a concrete failure.
Remediation: C03.

## B4

Minimum: train fits parameters, validation picks settings, test
reports the claimed number. Strong: adds the leak mechanisms (test
vocabulary, cross-split duplicates) and the document-level split rule.
Red flag: "validation and test are the same". Rubric: 2 for three
jobs plus one leak. Remediation: C09.

## B5

Minimum: precision TP/(TP+FP), recall TP/(TP+FN), F1 harmonic mean.
Accuracy lies under class imbalance: 99% accuracy with 0% minority
recall. Strong: derives why the harmonic mean refuses partial credit.
Red flag: "F1 is the average of precision and recall". Rubric: 2 for
formulas plus the 99/1 numbers. Remediation: C11.

## B6

Minimum: perplexity = exp(mean negative log-likelihood), the effective
number of equally likely choices per word. 5.137 means the model is as
confused as a 5-sided die on the test sentence. Strong: derives it
from the uniform case and states the same-tokenizer comparison rule.
Red flag: "perplexity is accuracy". Rubric: 2 for definition plus the
die reading. Remediation: C07.

## D1 ladder

D1.1 Reading 1: "visiting relatives" as one noun phrase (relatives who
visit). Reading 2: "visiting" as verb, "relatives" as object.
D1.2 Arcs: chased -> cat (nsubj), chased -> mouse (obj), the -> cat,
the -> mouse (det). One head per word, no cycles.
D1.3 Binary trees over n words follow Catalan growth: the count is
exponential, so enumeration is impossible and the parser must score.
D1.4 "raced" attaches as main verb early, the true main verb is
"fell". Greedy decoding cannot backtrack, so the error is structural,
not lexical. Beam or global inference recovers.
D1.5 Joint parsing degrades less: legal contracts shift both syntax
and vocabulary, and a syntax-first pipeline compounds its errors,
while a joint model can lean on the lexical signal. Caveat: with zero
legal training data both degrade, the claim is relative, not absolute.

## D2 ladder

D2.1 Words in similar contexts have similar meanings (operationally:
similar cooccurrence distributions).
D2.2 {the:1, sat:1, on:1} at width 2 around the first "cat".
D2.3 cos = u.v/(||u|| ||v||). Raw count vectors live in R^V with V
huge and mostly zeros: O(V^2) memory for all pairs, sparse and
expensive.
D2.4 The error: reading substitutability as synonymy. Both sit in
the frame "sat on the __", the toy corpus has two such frames, so
their vectors coincide. Cosine measures shared company, not shared
meaning.
D2.5 Assumption needed: context distributions determine meaning.
Antonym counterexample: "hot"/"cold" share contexts and get high
cosine. The honest claim is "captures substitutability", not
"captures meaning".

## Q1

Sparsity = 1 - 22/196 = 0.888 (88.8% empty). At 50,000 types the table
has 2.5 x 10^9 cells. The pain: the table grows quadratically while
real text covers a vanishing fraction, so most probability mass sits
on unseen events and smoothing dominates. Strong answer names the
quadratic growth and the smoothing consequence.

## Q2

Accuracy 0.99, precision 0.0 (0/0 by convention), recall 0.0, F1 0.0.
Rebuttal: "The 99% comes from never predicting finance, finance recall
is 0, so the system fails every finance document it was built for."

## I1

Two bugs. Bug 1: `match / sum(cn.values())` divides by zero when the
candidate is shorter than n (empty Counter), giving ZeroDivisionError
or NaN downstream. Bug 2: `np.log(p)` with p = 0 gives -inf with a
warning, and the brevity penalty `np.exp(1 - len(ref)/len(cand))`
divides by zero when the candidate is empty. Fix: guard empty
candidates (return 0.0), use max(1, sum(cn.values())), and map p = 0
to -inf explicitly with np.errstate. Red flag: fixing only one bug.
Rubric: 2 for both bugs plus guards. Remediation: C06 lab.

## S1

Changes: add aggressive dedup and quality filtering before counting,
re-weight or downsample the SEO slice, rebuild the vocabulary from
the clean slice only. Watch: validation perplexity or F1 on a
human-curated slice, plus the type-token ratio and duplicate rate from
the profiler. If the curated metric drops while the full-corpus
metric holds, the SEO slice poisons the model.

## S2

What breaks: the output slot (binary vs 5 labels), the metric
(accuracy on 5 classes hides ordinal errors: predicting 1 star for a
5-star review is worse than predicting 4), and the decision threshold.
Smallest fix: change the output layer to 5 classes with cross-entropy
loss and report mean absolute error alongside accuracy, since stars
are ordered.

## R1

Gap 1: single test set, no uncertainty , rerun with 5 seeds and report
mean and standard deviation. Gap 2: weak baseline , add logistic
regression and a fixed train-size learning curve. Gap 3: no error
analysis , slice errors by class and length, and test significance
(paired bootstrap) before claiming the 2% is real.

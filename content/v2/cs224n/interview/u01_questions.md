# Interview bank , U01 NLP history, tasks, and data

Test mode: answer closed-book before opening `interview/u01_key.md`.
Quotas per the course prompt: 6 breadth, 2 deep ladders of 5
follow-ups, 2 analytical/quantitative, 1 implementation/debug, 2
changed-constraint, 1 research critique.

## Breadth (B1-B6)

B1. Name three levels of language ambiguity and give one example each.
B2. Contrast symbolic, statistical, and neural NLP by what each stores
and how each updates.
B3. What is a corpus, and why is "bigger is better" false?
B4. State the three data splits and the one job each owns.
B5. Define precision, recall, and F1. When does accuracy lie?
B6. Define perplexity and explain what "5.137" means for a model.

## Deep ladder 1 , from ambiguity to parses (D1.1-D1.5)

D1.1 Define ambiguity and give the two readings of "visiting relatives
can be dull".
D1.2 Toy: draw the dependency arcs for "the cat chased the mouse".
D1.3 Derive why the number of parses grows faster than linear in
sentence length.
D1.4 Debug: a greedy transition parser fails on "the horse raced past
the barn fell" but scores well on news text. Explain.
D1.5 Compare syntax-first and joint parsing under a changed domain
(news to legal contracts). Which degrades less, and why?

## Deep ladder 2 , from counts to vectors (D2.1-D2.5)

D2.1 Define the distributional hypothesis in one sentence.
D2.2 Toy: build the width-2 cooccurrence vector for "cat" in the toy
corpus.
D2.3 Derive cosine similarity and explain why raw counts make it
expensive.
D2.4 Debug: "rug" and "sat" get cosine 0.983 in the toy corpus. A
learner concludes they are synonyms. Explain the error.
D2.5 Critique: a paper claims distributional vectors "capture meaning".
State the assumption this needs and the antonym counterexample.

## Analytical / quantitative (Q1-Q2)

Q1. The toy corpus has 35 tokens and 14 types. A bigram table has 196
cells, 22 nonzero. Compute the sparsity. If the vocabulary grows to
50,000 types, how many cells does the table have, and why is this the
statistical era's core pain?
Q2. A classifier on 100 documents (99 pets, 1 finance) always predicts
pets. Compute accuracy, precision, recall, F1 for the finance class.
A stakeholder says "99% accuracy, ship it." Give the one-sentence
rebuttal with the numbers.

## Implementation / debug (I1)

I1. This BLEU-2 implementation returns NaN on some inputs. Find the
bug and fix it.

```python
def bleu2(cand, ref):
    import numpy as np
    from collections import Counter
    prec = []
    for n in (1, 2):
        cn = Counter(tuple(cand[i:i+n]) for i in range(len(cand)-n+1))
        rn = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
        match = sum(min(cn[g], rn[g]) for g in cn)
        prec.append(match / sum(cn.values()))
    bp = np.exp(1 - len(ref)/len(cand))
    return bp * np.exp(np.mean([np.log(p) for p in prec]))
```

## Changed-constraint scenarios (S1-S2)

S1. Your corpus doubles in size but is now 80% machine-generated SEO
text. You may not collect new data. What changes in the pipeline, and
what metric do you watch to detect damage?
S2. The task changes from binary sentiment to 5-star rating, but the
model and loss stay the same. What breaks in the formulation, and what
is the smallest fix?

## Research critique (R1)

R1. A preprint reports a new classifier beating Naive Bayes by 2% on
one 200-document test set, with no seed variation and no baseline
beyond NB. List three methodological gaps and the experiment that
would close each.

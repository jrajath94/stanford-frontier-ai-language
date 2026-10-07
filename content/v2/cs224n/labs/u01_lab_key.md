# Lab key , U01

Execution-verified outputs from `labs/u01_lab_run.py`, run 2026-10-06
(CPython, numpy 1.26.4, CPU). Re-run the script to confirm, any drift
fails the key.

## T1 , corpus profile

docs 6, tokens 35, types 14, type-token ratio 0.40, top type "the" with
11, duplicate documents 0.

## T2 , bigram perplexity

Unsmoothed (MLE): 2.3452. Smoothed (add-1): 5.1370. Smoothing raises
perplexity here because it steals probability mass from the seen
bigrams and spreads it over all 14 types, the MLE is overconfident on
a tiny corpus and would assign probability 0 to any unseen bigram
(perplexity infinite on novel text).

## T3 , naive bayes

Prior 0.5: pets -5.416, finance -5.011, prediction finance.
Prior 0.9 (pets): pets -4.828, finance -6.620, prediction pets.
The prior moves both scores by its log, the argmax flips near
prior 0.60.

## T4 , BLEU-2

Unigram precision 1.0, bigram precision 1.0, brevity penalty 0.3679,
BLEU-2 0.3679. The penalty dominates: a short perfect candidate still
scores low.

## T5 , OOV leak demo

Clean vocabulary (sentences 1-4): OOV rate 0.3333 on sentence 5
("ran", "from" unknown). Leaky vocabulary (all six sentences): 0.0.
The leaky number is the lie the split discipline prevents.

## T6 , window slide

Before (window on first "cat", width 2): {the:1, sat:1, on:1}.
After (slid one right, window on "sat"): {the:2, cat:1, on:1}.
"the" rose from 1 to 2 because the window now covers two "the"
tokens, "sat" left the context and "cat" entered it.

# Lab U01 , counts, splits, and scores

Six tasks on the toy corpus. Run `python3 labs/u01_lab_run.py` from the
`cs224n/` directory. The script uses numpy only (no torch on this box)
and prints every number the tasks ask for. Record the numbers, then
read `labs/u01_lab_key.md` to verify.

T1. Profile the toy corpus: document, token, type counts, type-token
ratio, top type and its count, exact-duplicate document count.
T2. Build the add-1 smoothed bigram model and report the perplexity of
"the cat sat". Also report the unsmoothed (MLE) perplexity of the same
sentence and explain the difference.
T3. Train Naive Bayes on the four toy documents (C04) and report both
class scores and the prediction for "cat bank". Then set the pets prior
to 0.9 and report the new prediction.
T4. Compute BLEU-2 for candidate "the cat sat" against reference "the
cat sat on the mat". Show the two precisions and the brevity penalty.
T5. Build the vocabulary from sentences 1-4 only, then from all six.
Report the OOV rate of sentence 5 ("the mouse ran from the cat")
under each vocabulary.
T6. Compute the width-2 cooccurrence counts for the first "cat", then
slide the window one position right and report the new counts. Name the
word whose count changed and say why.

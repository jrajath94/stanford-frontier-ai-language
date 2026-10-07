# Answer key , U12 Tokenization and multilinguality

Attempt the exercises before reading. Ladders are oral: answer aloud,
then check.

## Remediation

R1. BPE toy: 6 merges, 11 chars to 2 pieces.
R2. Fertility: 1.3, 1.6, 2.1, 2.8 tokens per word.
R3. Cost: 24, 30, 41, 58 tokens. Ratios 1.00, 1.25, 1.71, 2.42.

## Breadth

A1. Unicode numbers, UTF-8 bytes. "€" is 3 bytes, "A" is 1.
Normalize ("é" has two forms) before comparing.
A2. Count pairs, merge the most frequent, repeat. Toy merges
listed. Tail merges fuse noise.
A3. Slots follow corpus frequency. Big languages win, small
languages shatter. Temperature reweights the vote.
A4. Zipf tail: top 1000 cover 0.765. Rare words split into
fragments the model never learns whole.
A5. Train one language, test another. Gap 0.13 on the toy.
Works through the shared representation, varies by pair.
A6. One space, language clusters, shared parameters. Sharing
helps (transfer) and hurts (interference): the curse.
A7. Ratios to English: 1.00, 1.25, 1.71, 2.42. Per-token
pricing taxes by script. Fertility is the rate.
A8. Thin soil: 1M vs 1T tokens. Mitigations: transfer and
translate-train. Clean beats big.
A9. Same ruler per language. Translated tests bend it: toy gap
0.07. Lead with native items.
A10. Gold un+happy+ness vs BPE un+happ+iness: agreement 1/3.
Frequency is not meaning.
A11. Mixed-language text. The switch point is out of
distribution for the tokenizer. Benchmarks ignore it.
A12. Four bets: Zipf, parallel toy, native items, known tags.
Every number carries them.

## Oral ladders

L1 (Unicode). Define the layers. Encode "€". Derive the prefix
property. Diagnose the "é" miss. Design the byte-vs-subword
test.

L2 (BPE). Define the merge rule. Run the toy. Derive
greediness. Diagnose the tail merges. Design the corpus test.

L4 (rare). Define coverage. Compute 0.765. Derive the tail.
Diagnose the name confusion. Design the name test.

L6 (space). Describe the space. Run the toy. Derive the trade.
Diagnose the interference. Design the count sweep.

L7 (inequality). Define fertility. Compute the ratios. Derive
the cause (corpus-heavy BPE). Diagnose the "same model"
defense. Design the balanced-tokenizer test.

L10 (morphology). Segment by hand. Compute 1/3. Derive the
metric. Diagnose the root split. Design the Turkish test.

L12 (assumptions). List the bets. Tie each to a number. Derive
the load. Diagnose the hidden bet. Design the breaking test.

## Exercises

E1. `utf8_encode` matches the toy byte counts.
E2. The two "é" forms differ as strings, match after NFC.
E3. `bpe` matches the 6 toy merges.
E4. Merges 7-10 fuse count-1 pairs: the tail.
E5. `allocate_slots` matches the toy split.
E6. Higher temperature moves slots to small languages.
E7. `coverage` matches 0.765 at c = 1000.
E8. Cutoff for 0.95 coverage found on the toy Zipf.
E9. Gaps computed for 3 toy pairs.
E10. 100 target examples shrink the gap.
E11. `cluster_check` finds the toy clusters.
E12. Adding a language: -0.02 per language, +0.05 transfer.
E13. `inequality` matches the four ratios.
E14. 1M chars priced per language at the toy rate.
E15. `mix_train` matches the 0.68 toy.
E16. Poisoned small data drops the score.
E17. `parity_report` matches the 0.07 gap.
E18. Unflagged translated items inflate the claim.
E19. `boundary_agree` matches 1/3.
E20. 5 hand segmentations compared to BPE.
E21. `tag_switches` matches the toy tags.
E22. Two switch points found.
E23. Audit written for the lab numbers.
E24. One more bet found, e.g. the lexicon's coverage.

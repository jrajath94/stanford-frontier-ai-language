# Interview key , U12

## Breadth

A1. Byte-pair encoding: merge frequent pairs. Rare words shatter
into pieces.
A2. Tokens per word. Differs because BPE merges followed the
training corpus, which was English-heavy.
A3. Train one language, test another through the shared
representation. Limited by alignment and language distance.
A4. Same meaning costs up to 2.42x the tokens across scripts,
so per-token pricing taxes by script.
A5. The same ruler per language. Translationese: translated
tests are easier than the language (toy gap 0.07).
A6. More languages share fixed capacity: per-language quality
falls past some count.

## Deep ladders

L1. (1) a+a, aa+a, aaa+b, then three count-1 merges. (2) Merge
the most frequent adjacent pair. (3) Each step is locally
optimal, no lookahead. (4) The tail fuses noise: stop merges
where counts die. (5) Balanced vs English-heavy corpus,
measure fertility spread per language.

L2. (1) 1.00, 1.25, 1.71, 2.42. (2) Tokens per word. (3) Merges
follow corpus frequency. (4) Same model, different bill: the
price is per token, not per meaning. (5) Price 1M chars both
ways per language, compare the bills.

## Analytical

A7. The 0.765 is irrelevant: your task lives in the 0.235 tail.
The names shatter (C04). Change: byte-level handling for names
or a name-aware tokenizer, and eval on names specifically.
A8. (i) Language distance: Y is farther from the source. (ii)
Script/tokenizer: Y's script shatters (fertility). Cheapest
tests: the fertility ratio per language, and the transfer gap
vs a distance measure.

## Implementation/debugging

A9. (1) Normalize to NFC before comparing (cheapest). (2) Audit
where the strings come from (mixed forms). (3) Normalize at
ingest, not at match time.

## Changed-constraint

A10. Small languages get few slots and shatter. Mitigations:
tempered corpus sampling for the tokenizer, byte fallback,
and per-language fertility monitoring.
A11. Build a code-switched eval: native mixed transcripts with
human grades. Report monolingual and switched scores
separately, lead with the switched.

## Research critique

A12. Steelman: one model, one API, all languages work to some
degree. Counterexample: 2.42x the token bill, shattered rare
words, translated-test inflation. Equal API, unequal service.

# Interview bank , U12 Tokenization and multilinguality

## Breadth (6)

Q1. What is BPE, and what does it do to rare words?
Q2. What is fertility, and why does it differ by language?
Q3. What is cross-lingual transfer, and what limits it?
Q4. What is the token cost inequality?
Q5. What is evaluation parity, and what is translationese?
Q6. What is the curse of multilinguality?

## Deep ladders (2 x 5)

L1 (BPE). (1) Run 6 merges on the toy corpus. (2) State the
merge rule. (3) Derive why the rule is greedy. (4) The tail
merges fuse count-1 pairs: diagnose. (5) Design the balanced-
corpus tokenizer test.

L2 (inequality). (1) Compute the four ratios. (2) Define
fertility. (3) Derive the cause: corpus-heavy BPE. (4) "Same
model, same price": rebut. (5) Design the per-token vs per-
character pricing comparison.

## Analytical (2)

Q7. Top 1000 words cover 0.765 of tokens. Your NER task is all
rare names. What does the coverage number tell you, and what
do you change?
Q8. Transfer gap is 0.13 on language X but 0.31 on language Y.
Both had zero target training data. Name two mechanisms and
the cheapest test for each.

## Implementation/debugging (1)

Q9. Your "é" string matches fail intermittently. List the fixes
in order, cheapest first.

## Changed-constraint (2)

Q10. You must serve 100 languages on one model with a fixed
vocabulary. What breaks, and what is the mitigation?
Q11. The deployment is code-switched speech transcripts. Your
benchmark is monolingual. What is the honest eval plan?

## Research critique (1)

Q12. "Our multilingual model serves all languages equally."
Steelman, then give the strongest counterexample from this unit.

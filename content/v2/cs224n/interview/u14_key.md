# Interview key , U14

## Breadth

A1. Patchify: 196 tokens for 224x224 at 16x16. Cost: the tokens
eat the window at n^2 attention.
A2. Early: one shared stack, 0 extra params. Late: separate
towers joined by projections, 5.03e7 extra at d = 1024, L = 24.
A3. Image share 0.333 on the toy. Put the image first so the
text can attend to it.
A4. A specialist FFN per modality. Collapse: the router sends
everything to one expert. The aux loss prevents it.
A5. Title known, content not inspected, claims none. The card
is the whole lesson until inspection.
A6. Inspected artifact, schedule title, hearsay. Demote on
doubt.

## Deep ladders

L1. (1) 2 x 1024^2 x 24 = 5.03e7. (2) Early: concat at input.
Late: towers then project. (3) Two d x d projections per
layer. (4) 196 image tokens in every layer's attention: the
n^2 bill. (5) Equal total params, measure the task gap.

L2. (1) Label, source, status, claims. (2) No content claim
without inspection. (3) The leaf's only honest content is its
boundary. (4) Codename guessing: the label could be anything.
(5) Inspect the artifact, then re-teach the leaf.

## Analytical

A7. Collapse: aux 5.28 vs healthy 1.04, one expert takes 8 of
10 tokens. Fix: turn on the aux loss (weight > 0) and retrain
the router.
A8. (i) Shrink the image: fewer patches, lose detail. (ii)
Cross-attention instead of image tokens: no image tokens in
the window, added machinery. Price detail vs complexity.

## Implementation/debugging

A9. (1) The language prior: test with image ablation (cheapest,
already the symptom). (2) The triples never needed the image:
audit the data. (3) The vision encoder is frozen or broken:
check gradients. (4) Retrain with image-necessary triples.

## Changed-constraint

A10. The video becomes an inspectable artifact. Watch it, take
notes with anchors, re-teach C01-C09 from it, and update the
claim classes from requested-branch to official where the
video supports them. C10's card gets its content.
A11. Cut the image tokens first (fewer patches or a smaller
image share): latency is n^2 in stream length. What breaks:
fine detail (C01) and the text budget.

## Research critique

A12. Steelman: one model reads both, joint training, cross-
modal transfer. Counterexample: the image-ablation test, no
accuracy drop means it is a text model with pictures. Input
format is not multimodality.

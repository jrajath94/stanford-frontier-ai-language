# Answer key , U14 Multimodality and guest research

Attempt the exercises before reading. Ladders are oral: answer aloud,
then check.

## Remediation

R1. Late fusion: 5.03e7 extra params. Early fusion: 0.
R2. Routing aux loss 1.04: balanced.
R3. Image share: 256/768 = 0.333.

## Breadth

A1. Patches projected to tokens: 196 for 224x224 at 16x16. The
transformer cannot tell eyes from text after translation.
A2. Early: one stack, 0 extra params. Late: two towers joined,
5.03e7 extra. Early costs context, late costs params.
A3. Image share 0.333 on the toy. In a causal model put the
image first.
A4. Joint bill 288: 128 text + 20 diffusion steps at 8 each.
Agreement is the hard part.
A5. AR tells a story, diffusion sculpts. The interface:
AR text conditions the diffusion.
A6. Route tokens to modality experts, aux loss keeps balance:
1.04 on the toy. Aux weight 0 collapses.
A7. Fused = w image + (1-w) text: 0.720, 0.565, 0.410. Tune w
on dev.
A8. Format (image, question, answer), SFT on answers. Trap:
the language prior answers without the image.
A9. The judge must see the image. Blind judge kappa 0.31 vs
0.55 with image.
A10. The card: title known, content not inspected, claims none.
No claim outruns the card.
A11. Three classes: inspected artifact, schedule title,
hearsay. Demote on doubt.
A12. The ledger: every open item names its closure rule. No
row closes without the artifact.

## Oral ladders

L1 (patches). Define patchify. Compute 196. Derive the token
cost. Diagnose the blurred text. Design the patch test.

L2 (fusion). Define both. Compute 5.03e7. Derive the wiring.
Diagnose the attention bill. Design the equal-param test.

L4 (joint). Define the joint. Price 288. Derive the split.
Diagnose the red/blue case. Design the agreement test.

L6 (experts). Define routing. Compute 1.04. Derive the aux.
Diagnose the collapse. Design the weight sweep.

L10 (guest card). Define the card. Write the toy. Derive the
boundary. Diagnose the codename guess. Design the inspection.

L11 (claims). Define the classes. Classify the toy. Derive the
demote rule. Diagnose the title-as-talk. Design the citation
check.

L12 (gaps). Define the ledger. List the toy rows. Derive the
closure rule. Diagnose the assumed closure. Design the audit
check.

## Exercises

E1. `patchify` matches 196 tokens.
E2. P = 8 gives 784 tokens (4x).
E3. Param counts match 0 and 5.03e7.
E4. Attention bill priced for 196 image tokens.
E5. Stream built, share 0.333.
E6. Text-first breaks the causal read.
E7. `joint_cost` matches 288.
E8. Text-only 128 vs joint 288.
E9. The mix pipeline sketched.
E10. Weak conditioning: the image ignores the text.
E11. `route` matches aux 1.04.
E12. Aux weight 0: one expert takes all.
E13. `mm_retrieve` matches the three scores.
E14. w sweep finds the interior optimum.
E15. Triples formatted correctly.
E16. Image ablation shows the drop (or its absence).
E17. Toy scored with and without image.
E18. Kappas 0.55 vs 0.31 computed.
E19. The guest card written.
E20. One tempting claim marked as class (c).
E21. 6 toy claims classified.
E22. One class-(c) claim demoted.
E23. The U14 ledger written: 3 open rows.
E24. One row closed honestly, or marked why not.

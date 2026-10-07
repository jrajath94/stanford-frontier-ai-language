# Answer key , U15 Project tutorials and open questions

Attempt the exercises before reading. Ladders are oral: answer aloud,
then check.

## Remediation

R1. Tiny GPT-2: 44,928 params (V=1000, d=8, L=4).
R2. Seeds: mean 0.8045, std 0.0203. Report the spread.
R3. Four gates: proposal, milestone, poster, report (SRC-04).

## Breadth

A1. Vectorize the big loops: 0.002 s vs 0.4 s on the toy.
Python conducts, numpy plays.
A2. The graph records, backward replays. x=2, x^2: grad 4.0.
Grad-check every new op.
A3. Tokenize, batch, forward, decode. Pin the revision.
A4. Embeddings + blocks + head, next-token loss. 44,928 params
in the miniature. Scale is a dial.
A5. Template: head, fine-tune, report with SE. Toy scores
labeled toy, never cited as findings.
A6. Four gates, each with a falsifiable claim. The report is
accounting plus impact.
A7. Score, cut, rescore. Cut scope, never the control. Toy:
3/5 to 4/5.
A8. Spec in, tests green, citations attached. Copying with
renamed variables is still copying.
A9. Seed sweep: mean 0.8045, std 0.0203. Differences inside
the std are noise. Keep the receipt.
A10. Rank the limits by threat to the claim. The top limit
goes in the abstract.
A11. ICL mechanism, faithful reasoning, scaling limit,
multilingual parity, mechanistic understanding. Dated
October 2026, each with a closing criterion.
A12. Rubric: claim, evidence, limit, transfer. Four points.

## Oral ladders

L1 (Python). State the rule. Time the toy. Derive the loop
location. Diagnose the tiny-array case. Design the profile
habit.

L4 (GPT-2). Define the parts. Count 44,928. Derive the
formula. Diagnose the emergence gap. Design the scale sweep.

L6 (gates). Define the gates. Write the toy claim. Derive the
gate logic. Diagnose the killed claim. Design the change log.

L9 (reproducibility). Define the receipt. Compute the spread.
Derive the tolerance. Diagnose the GPU case. Design the
receipt.

L11 (open). Name the questions. Mark them open. Derive the
criteria. Diagnose the answered-in-disguise. Design the
experiment for one.

L12 (defense). State the claim. Show the evidence. Name the
limit. Answer the transfer. Score the defense.

## Exercises

E1. Both versions timed, ratio about 200x.
E2. Crossover size found on your machine.
E3. x^2 toy: grad 4.0.
E4. Softmax grad-check passes to 1e-5.
E5. The 5-line load written.
E6. The mask zeros the padding.
E7. Param counter matches 44,928.
E8. Probe loss falls 1.164 to 1.043.
E9. The template runs.
E10. More QA items shrink the SE.
E11. Proposal one-pager written.
E12. Milestone with the U09 numbers written.
E13. Toy idea scored and cut to 4/5.
E14. The control identified in the cut list.
E15. One part implemented from spec, tests green.
E16. The spec cited.
E17. `seed_sweep` matches mean 0.8045, std 0.0203.
E18. Receipt written for the U09 lab.
E19. Limits written for the U11 lab.
E20. Limits ranked by threat.
E21. Dated list of 5 open questions written.
E22. One own question with its criterion added.
E23. U09 lab defended aloud, 4 points.
E24. U11 lab defended aloud, 4 points.

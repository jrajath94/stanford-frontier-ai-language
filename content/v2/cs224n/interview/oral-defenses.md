# Oral defenses , 10 deep ladders x 8 follow-ups

Answer aloud, closed-book. Each ladder goes 8 deep: define, toy,
derive, implement, complexity, compare, debug, critique, design.
Keys in `interview/keys-oral.md`. Test mode: do not read the keys
first.

## D1 , attention is not explanation (U05, U13)

1. Define attention weights. 2. Compute one row on a toy. 3.
Derive what the weights can and cannot show. 4. Implement the
row. 5. State the cost of reading them. 6. Compare attention
maps vs ablations as evidence. 7. Debug: the map highlights the
wrong token but the answer is right. 8. Design the experiment
that tests whether the map caused the answer.

## D2 , the ReAct loop (U09)

1. Define the three steps. 2. Run the 7-step toy. 3. Derive why
observations are the only new information. 4. Implement the step
cap. 5. State the per-step cost growth. 6. Compare interleave vs
plan-then-execute. 7. Debug: 50 steps of "no results". 8. Design
the equal-budget comparison.

## D3 , benchmark honesty (U10)

1. Define the Wilson interval. 2. Compute it for 78/100. 3.
Derive the 1/sqrt(n) shrink. 4. Implement the report function.
5. State the n for margin 0.02. 6. Compare Wilson vs bootstrap.
7. Debug: 200 items from 10 templates. 8. Design the
contamination canary.

## D4 , test-time scaling (U11)

1. Define the majority vote. 2. Compute the curve at p = 0.6. 3.
Derive the binomial tail. 4. Implement the vote. 5. State the
cost per sample. 6. Compare sampling vs training a bigger
model. 7. Debug: the vote never changes the answer. 8. Design
the correlation test.

## D5 , tokenization inequality (U12)

1. Define fertility. 2. Compute the four ratios. 3. Derive the
corpus cause. 4. Implement the BPE toy. 5. State the price per
language. 6. Compare subword vs byte models. 7. Debug: the "é"
match fails. 8. Design the balanced-corpus test.

## D6 , causal interpretability (U13)

1. Define ablation. 2. Compute the 1.5 effect. 3. Derive the
counterfactual logic. 4. Implement the projection. 5. State the
cost per intervention. 6. Compare ablation vs patching. 7.
Debug: no effect, but the behavior needs the component. 8.
Design the backup-circuit test.

## D7 , multimodal fusion (U14)

1. Define early vs late fusion. 2. Compute the 5.03e7 gap. 3.
Derive the param formula. 4. Implement the wiring choice. 5.
State the attention bill for 196 tokens. 6. Compare stream vs
cross-attention. 7. Debug: the model ignores the image. 8.
Design the image-ablation test.

## D8 , RAG end to end (U09)

1. Define the pipeline. 2. Compute recall at k. 3. Derive the
top-k factorization. 4. Implement the answer function. 5. State
the index freshness cost. 6. Compare RAG vs long-context
stuffing. 7. Debug: rank-1 chunk is a keyword trap. 8. Design
the k sweep.

## D9 , reward design (U07, U11)

1. Define verifiable vs learned reward. 2. Run the toy split.
3. Derive why the verifier resists hacking. 4. Implement the RL
step. 5. State the sparsity cost. 6. Compare RLHF vs DPO vs
verifiable RL. 7. Debug: the model exploits a weak test. 8.
Design the verifier-strength test.

## D10 , the project defense (U15)

1. State your claim. 2. Show the numbers with intervals. 3.
Derive the tolerance from the seed sweep. 4. Implement the
receipt. 5. State the cost of the claim. 6. Compare your method
to the baseline at equal budget. 7. Debug: the rerun differs.
8. Design the limitation list, ranked.

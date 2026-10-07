# Oral defense keys

## D1

1. Weights = softmax(QK^T/sqrt(d_k)) rows, one distribution per
query. 2. Toy row computed in U05. 3. They show where the model
looked, not why: one factor in a deep stack (U05 T3, U13 C02).
4. Row implementation matches the lesson. 5. Reading is free,
misreading is dear. 6. Ablations are causal, maps are
correlational. 7. The map is not the mechanism: run the
intervention. 8. Corrupt the attended token, measure the answer
shift (U11 C12's test).

## D2

1. Thought, Action, Observation. 2. 7 steps, 92 tokens. 3. The
model authors everything except Observations. 4. Cap B, stop
reason logged. 5. Context grows per step, attention is n^2. 6.
Interleave wins when observations change the plan. 7. No
progress signal: add the cap and the no-new-info detector. 8.
40 two-hop tasks, equal tool budget.

## D3

1. Wilson interval for a proportion. 2. [0.689, 0.850]. 3. SE =
sqrt(p(1-p)/n). 4. report() matches both toy intervals. 5. n =
2401. 6. Wilson for proportions, bootstrap for exotic metrics.
7. Effective n near 10: report the template count. 8. Plant
canary strings, measure extraction vs inflation.

## D4

1. Majority over n samples. 2. 0.6000, 0.6826, 0.7535, 0.8256.
3. Binomial tail sum. 4. Vote implementation matches. 5. n times
one generation. 6. Sampling: fixed model, per-query cost.
Training: one cost, every query cheap. 7. Check temperature
(theater), diversity, p. 8. Pairwise wrong-answer agreement vs
independence.

## D5

1. Tokens per word. 2. 1.00, 1.25, 1.71, 2.42. 3. BPE merges
follow the training corpus. 4. Toy merges match. 5. Per-token
pricing taxes by script. 6. Subwords: efficient, unequal. Bytes:
equal, dearer. 7. Normalize to NFC first. 8. Balanced vs
English-heavy corpus, measure the fertility spread.

## D6

1. Project out the direction, remeasure. 2. 2.1 - 0.6 = 1.5. 3.
Same input minus the component: the difference is its
contribution. 4. Ablation matches the toy. 5. One forward pass
per intervention. 6. Patching (clean activations) is cleaner
than zeroing. 7. Backup circuits: the test is one-sided. 8.
Ablate pairs and triples, watch for the drop.

## D7

1. Early: one stack. Late: towers plus projections. 2. 5.03e7
at d = 1024, L = 24. 3. Two d x d projections per layer. 4.
Wiring choice implemented. 5. 196 tokens in every layer's n^2.
6. Stream: simple, eats the window. Cross-attention: complex,
saves the window. 7. The language prior: ablate the image. 8.
Image on vs off, measure the drop.

## D8

1. Retrieve top-k, prepend, generate with citations. 2. 0.000,
0.333, 0.667, 1.000. 3. Sum over chunks approximated by top-k.
4. rag_answer matches. 5. Incremental index plus quarantine.
6. RAG: fresh, cited, retrieval can fail. Stuffing: simple,
window-bound. 7. Retrieve more, rerank, require citations. 8. k
in {1, 5, 20}, predict rise then fall.

## D9

1. Verifiable: a program checks. Learned: a model scores. 2.
63/37 toy split. 3. The reward is the task: hacking needs the
right answer. 4. rl_step matches. 5. Binary rewards are sparse.
6. RLHF: general, hackable. DPO: simple, offline. Verifiable:
honest, narrow. 7. The verifier is the bug: verify the
verifier. 8. Weak vs strong verifier, measure true
correctness.

## D10

1. One falsifiable sentence. 2. Numbers with Wilson intervals
and slices. 3. Tolerance = seed std (0.0203 on the toy). 4.
Receipt: seeds, versions, data, commands. 5. The bill: time,
compute, eval items. 6. Equal data, equal tuning, vary one
thing. 7. Inside the std: noise. Outside: a bug. 8. Ranked by
threat to the claim, top limit in the abstract.

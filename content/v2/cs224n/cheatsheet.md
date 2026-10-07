# Cheatsheet , cs224n U01-U15

One page per area. Numbers are toy-scale from the unit scripts.

## Core formulas

- Perplexity: exp(mean negative log-likelihood), per tokenizer.
- Attention: softmax(QK^T / sqrt(d_k)) V, mask before softmax.
- LoRA: dW = BA, rank <= r, params 2dr per matrix.
- Majority vote: sum C(n,k) p^k (1-p)^(n-k), k > n/2.
- Wilson: point +/- z sqrt(p(1-p)/n + z^2/4n^2), over 1 + z^2/n.
- Kappa: (p_o - p_e) / (1 - p_e).
- Brier: mean((f - y)^2). ECE: mean|conf - correct|.
- Contamination: reported = L s_m + (1 - L) s_c.
- Split rate: max(0, s - o) / (c - o).
- Chunks: 1 + ceil((doc - c) / (c - o)).
- Fusion: w dense + (1 - w) sparse.
- Disparity: min rate / max rate.
- Spec decode E: (1 - a^(g+1)) / (1 - a).
- Aux loss: mean((load / mean)^2), 1.0 is balanced.

## Key numbers

- LoRA d=4096 r=8: 65536/matrix, 0.060% of 7B.
- Vote p=0.6: 0.6000, 0.6826, 0.7535, 0.8256 (n=1,5,11,21).
- Vote p=0.4: 0.3174 (n=5), 0.2465 (n=11): do not vote.
- 78/100: [0.689, 0.850]. 780/1000: [0.753, 0.805].
- n = 2401 for margin 0.02 at 95%.
- Contamination: 0.75 reported, 0.70 clean.
- Kappa 0.551, Brier 0.17 on the toys.
- RAG recall@k: 0.000, 0.333, 0.667, 1.000.
- Chunk 200/50: 7 chunks, split 0.00. No overlap: 5, 0.15.
- Fertility: 1.3, 1.6, 2.1, 2.8. Cost ratios: 1.00-2.42x.
- BPE toy: 6 merges, 11 chars to 2 pieces.
- Disparity 0.667. Intervention effect 1.5. ECE 0.425.
- Late fusion 5.03e7 params. Routing aux 1.04. Image share 0.333.
- Tiny GPT-2: 44,928 params. Seeds: 0.8045 +/- 0.0203.
- ReAct toy: 7 steps, 92 tokens. Spec decode: 2.94/pass.

## Decision rules

- Conditioning is not learning: prompts steer, they do not train.
- Report the interval, never the bare number.
- Attribute before fixing: R/T/P, biggest class first.
- Spread concave gains: 10x10 beats 1x100, except thresholds.
- Check p before voting: below 0.5 voting hurts.
- Cap the agent loop: valid answer or step cap, both exits.
- Enforce permissions in code, not in the prompt.
- Calibrate the judge: kappa and Brier before trust.
- Quarantine leaks: rescore on the clean set.
- Read the slice, not the mean.
- Pre-register the acceptance criterion.
- Pin the model revision. Keep the receipt (seeds, versions).
- Cut scope, never the control group.
- Guest content: title only until inspected. Demote on doubt.
- Traces are evidence, not proof: run the corruption test.

## Claim classes

- OFFICIAL-SOURCE: inspected artifact (schedule S01-S20, SRC-01-06).
- REQUESTED-BRANCH: prompt/inventory request, toy-taught, PLANNED /
  SOURCE ATTRIBUTION PENDING.
- RESTRICTED-UNVIEWED: exists behind login, not viewed.
- EDITION-2024: 2024 public videos, tagged.

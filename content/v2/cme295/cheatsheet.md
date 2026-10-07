# CME295 cheatsheet

One page per unit. Computed toys, not measurements. Baseline
2026-10-06.

## Formulas

| What | Formula |
|------|---------|
| Attention | softmax(QK^T / sqrt(d)) V |
| BT loss | -log sigmoid(r_w - r_l) |
| PPO objective | E[min(rho A, clip(rho, 1-eps, 1+eps) A)] |
| DPO loss | -log sigmoid(beta x margin) |
| GRPO advantage | A_i = (r_i - mu) / sigma |
| pass@k | 1 - (1 - p)^k |
| Recall / precision | H/R , H/K |
| RRF | sum 1/(k + rank_i) |
| Debiased win rate | (w1 + 1 - w2)/2 |
| ECE | sum_b (n_b/n) x \|acc_b - conf_b\| |
| Wilson | center (p + z^2/2n)/(1 + z^2/n), half-width z sqrt(p(1-p)/n + z^2/4n^2)/(1 + z^2/n), z = 1.96 |
| LoRA params | r(d + k) vs dk |

## Key numbers (toys)

- BT: r_w=1.2, r_l=0.4 -> P=0.69, loss=0.371
- PPO: rho=2.0, A=1.0, eps=0.2 -> obj=1.2
- DPO: margin 0.14 -> P=0.535, loss=0.626
- GRPO: [1,1,0,0] -> A=[1,1,-1,-1]
- pass@16 at p=0.2 -> 0.972
- Retrieval: R=8,K=20,H=6 -> recall 0.75, precision 0.30
- Bias: orders 0.70/0.45 -> debiased 0.625
- ECE toy -> 0.074
- n=100,w=62 -> [0.522, 0.709] (Wilson)

## Failure -> fix

| Failure | Fix |
|---------|-----|
| Quadratic attention | GQA, KV cache, FlashAttention |
| Reward hacking | harden verifier, audit gap |
| KL drift | KL anchor, early stop |
| Position bias | judge both orders |
| Verbosity bias | length caps in rubric |
| Contamination | perplexity gap, quarantine |
| Runaway agent | step cap, no-repeat guard |
| Split-table retrieval | overlapping chunks |
| Tied GRPO groups | raise temperature |
| Saturated benchmark | retire it, build harder items |

## Decision rules

- DPO vs RLHF: verifier exists -> RLVR/DPO, subjective -> RLHF
- Train vs test-time: rare hard queries + verifier -> test-time
- Hybrid vs dense: part numbers + prose -> hybrid
- Pointwise vs pairwise: ranking matters -> pairwise
- Full vs PEFT: new domain or big shift -> full, else LoRA
- RAG vs long context: corpus >> window -> RAG
- Ship or not: lower Wilson band clears 0.5 after debiasing

## Oral ladder rungs

Define, toy, derive, implement/complexity, compare, debug,
critique assumptions, design experiment/transfer.

## Never forget

No naked scores. Dated claims only. Orphans are decisions.
The final tests the graph, not the nodes.

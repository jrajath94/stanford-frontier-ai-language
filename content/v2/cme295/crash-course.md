# CME295 crash course , all nine units

Baseline 2026-10-06. Synthesized from the nine unit lessons. Every
leaf stays PLANNED / SOURCE ATTRIBUTION PENDING until artifact
extraction verifies it. Numbers below are the course's computed
toys, not measurements.

## U01 , NLP to transformer foundations

Language becomes numbers through tokenization (BPE merges frequent
pairs), embeddings (vectors per token), and training objectives
(next-token prediction). RNNs read serially and forget, LSTMs gate
the forgetting, attention lets every position read every other
position directly. The transformer block is attention plus a
feedforward network, with residual links and normalization. Causal
masks enforce left-to-right order. Shapes rule everything: know
(batch, seq, dim) at every step.

## U02 , attention variants and positions

Multi-head attention runs h parallel attentions. MQA shares one KV
head across query heads (small cache, weaker), GQA groups them (the
middle path). Positions enter through RoPE: rotate q/k pairs by
position-dependent angles so relative distance survives the dot
product. KV caching makes decoding O(1) per step after a quadratic
prefill. Attention is O(n^2 d): the tax on long context.

## U03 , LLM architecture and generation

An LLM is a decoder-only transformer trained to predict the next
token, scaled until in-context learning appears. MoE routes each
token to a few experts, buying capacity at fixed compute. Sampling:
temperature reshapes sharpness (T->0 is argmax, T->infinity is
uniform), top-k/top-p cut the tail. Prompting, ICL, and
self-consistency (vote over samples) steer behavior without
training. Evaluation limits: benchmarks measure proxies, not skill.

## U04 , training and efficient adaptation

Pretraining minimizes next-token loss over trillions of tokens.
SFT installs the behavior format on curated pairs. LoRA freezes W
and trains BA (rank r, r(d+k) params vs dk). Quantization (INT8)
and bf16 cut memory, the optimizer states dominate the budget.
Full fine-tune is expressive and forgetful, PEFT is cheap and
stable. Regression tests guard what training might break.

## U05 , preference tuning and policy optimization

Human comparisons become a scalar reward via the Bradley-Terry
model: loss = -log sigmoid(r_w - r_l). RLHF: SFT, reward model,
PPO. PPO clips the policy ratio (eps 0.2) for a pessimistic
trust-region step. Advantages (return minus baseline) cut
variance. DPO skips RL: the policy itself is the reward, loss =
-log sigmoid(beta x margin). Pathologies: reward hacking, KL
drift, likelihood displacement. Evaluate with win rates and
human spot checks.

## U06 , reasoning, RL, and scaling

RLVR trains on verifiable rewards (exact match, unit tests), no
human labels. GRPO samples G chains per prompt and normalizes
advantages within the group: A = (r - mu)/sigma, no critic. The
verifier is the single point of failure: hack it and the policy
farms the hack. Test-time scaling: pass@k = 1 - (1-p)^k, a
verifier turns samples into answers. Split fixed budgets between
training (skill) and test-time (chances) at the interior optimum.
Ablate one factor at a time, probe generalization on paraphrase,
domain, and verifier shifts.

## U07 , RAG and agentic systems

RAG: chunk, embed, index, retrieve top-k, rerank, pack into the
window. Hybrid search fuses BM25 (exact terms) with dense
(meaning), normalize before fusing. Rerankers trade latency for
precision@5. Agents loop Thought -> Action -> Observation (ReAct),
with tools behind JSON schemas. State lives in the trace, session
summaries, and a profile. Guards: step caps, no-repeat rules,
green/yellow/red permission tiers, human approval for the red.
Debug upstream first: retrieval, then tools, then the model.

## U08 , LLM evaluation and judging

Write anchored rubrics or the judge invents its own law. Pairwise
judging is O(n^2) but stable, pointwise is O(n) but needs
calibration. Correct position bias by judging both orders:
(w1 + 1 - w2)/2. Control verbosity with length caps. Blind the
judge to model identity. Report ECE for calibration honesty,
Wilson intervals for uncertainty, contamination status for every
benchmark. No naked scores: every number carries its band.

## U09 , trends, exams, synthesis

Masked diffusion LMs unmask by confidence in parallel steps,
trading serial coherence for parallel latency. Map every exam
objective to a lesson home. Orphans are study priorities. The
final tests the graph: attention is the keystone (15 downstream),
then policy gradients and retrieval. Production needs five gates:
data match, latency, cost, shadow, rollback. Defend any concept
on the 8-rung ladder: define, toy, derive, implement, compare,
debug, critique, design.

## The one-page spine

Tokens -> attention -> sampling -> training -> preferences ->
reasoning -> retrieval -> agents -> judging. Each stage has one
equation, one failure, and one fix. The equations: softmax(QK^T /
sqrt(d))V, -log sigmoid(r_w - r_l), A = (r - mu)/sigma,
1 - (1-p)^k, (w1 + 1 - w2)/2. The failures: quadratic cost,
reward hacking, verifier gaps, position bias, contamination.
The fixes: GQA and KV cache, KL anchors and audits, swap
correction, quarantine, and dated honesty about what is known.

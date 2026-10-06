---
page_id: cme295-cheatsheet
course_slug: cme295
course_name: "CME295: Transformers and Large Language Models"
course_order: 5
order: 900
nav: "CME295 · Cheatsheet"
title: "CME295 Cheatsheet"
summary: "Every key fact from CME295 on one dense page: the toys, the formulas, and the numbers that carry each idea."
instructor: "Afshine Amidi, Shervine Amidi"
offering: "Autumn 2025"
---

<div class="cheat-cols" markdown="1">

<div class="cheat-block" markdown="1">

### Attention, built from zero

**The toy.** Three tokens: counselor [1,0], helped [0,1], frame
[1,1]. Query for "frame" = [1,1]. Scores: 1, 1, 2. Softmax:
[0.21, 0.21, 0.58]. New "frame" = 0.79, 0.79.

**Q, K, V.** Query asks what the token seeks. Key advertises what
it offers. Value is the payload. Three learned projections per
layer. Roles change per layer.

**The formula.** Attention = softmax(QK^T / sqrt(d_k)) V. QK^T
scores every pair. Sqrt(d_k) keeps the softmax responsive (without
it, dot products spread with dimension, the softmax saturates, and
gradients die). Softmax makes weights. V mixes.

**Causal mask.** Decoder sets future scores to -inf before softmax.
Training parallelizes. Inference does not.

**Multi-head.** H heads, each d_model/H. Same input, several views
at once: syntax, coreference, position.

**Interview line.** "Attention is content-based weighted lookup:
similarity of query to keys gives weights. Values get summed."

</div>

<div class="cheat-block" markdown="1">

### The transformer block

**Block.** x + Sublayer(LayerNorm(x)) per sublayer (post-norm,
2017). Modern LLMs normalize first (pre-norm) so the residual
highway stays clean for 96-layer stacks. RMSNorm drops the mean:
x / rms(x).

**Positions.** Sinusoids: PE(p).PE(q) depends only on p - q.
Relatives: T5 adds a learned bias per distance bucket. ALiBi
subtracts m * distance (extrapolates, no parameters). RoPE rotates
Q and K by position so the dot product keeps only distance. RoPE
is the default.

**KV cache.** Store past keys and values. Append one row per step.
GQA shares 32 query heads over 8 KV heads: 4x smaller. MQA shares
one.

**BERT.** Encoder-only. WordPiece ~30k, [CLS]/[SEP], segment
encodings. MLM: mask 15% of tokens, 80/10/10 split (of 512 tokens:
77 masked, ~61 become [MASK], ~8 random, ~8 unchanged). NSP: 50/50
next-sentence. Bidirectional. Cannot generate.

**T5.** Span corruption: mask spans, insert <X>/<Y> sentinels,
decode each sentinel plus its missing text. Everything is
text-to-text.

</div>

<div class="cheat-block" markdown="1">

### LLMs, decoding, inference

**The LLM.** Next-token probabilities at scale: hundreds of
billions of parameters, up to tens of trillions of tokens, >90%
decoder-only.

**Decoding on one distribution** (lit 0.50, read 0.30, slept 0.12,
ate 0.08). Greedy: argmax, "lit". Beam: B hypotheses, summed
log-probs with length norm. Sampling: draw from it. Top-K (K=2):
renormalize to 0.625/0.375. Top-P (P=0.9): smallest set reaching
0.9 = {lit, read, slept}. Temperature: p_i ~ exp(z_i/T). On logits
[3,2,1], T=0.5 gives [0.87, 0.12, 0.02], T=2 gives [0.51, 0.31,
0.19]. Sampling is the only randomness in the machine.

**Guided decoding.** Model valid outputs as a grammar. Mask invalid
tokens to zero each step. JSON always parses.

**MoE.** y-hat = sum g_i * E_i(x). Sparse top-1/2 routing per token
per layer. Routing collapse: the loop converges to one expert.
fix with auxiliary load-balance loss plus noisy gating. Switch:
1.6T params at top-1.

**Inference is memory-bound.** 70B at 2 bytes = 140 GB moved per
token. KV cache avoids O(t) recompute. PagedAttention kills
fragmentation (fixed blocks). MLA compresses KVs to latents.
Speculative decoding: draft k, verify in one pass, keep the
accepted prefix. Multi-token prediction learns it in.

**Prompting.** Context, instructions, inputs, constraints.
Zero-shot, few-shot, chain-of-thought (tokens as compute),
self-consistency (sample 5, majority wins). Beware context rot.
Cache repeated prefixes (~1/10 price).

</div>

<div class="cheat-block" markdown="1">

### Pre-training and scaling

**FLOPs vs FLOPS.** FLOPs = work. FLOPS = rate. GPT-3: 6 * 175B *
300B = 3.15e23 FLOPs. At 34 TFLOPS: 294 GPU-years, or 10.7 days on
10,000 GPUs at fantasy efficiency.

**Chinchilla.** 20 tokens per parameter. GPT-3 needed 3.5T tokens,
got 300B: 11.7x undertrained. 100B params want 2T tokens.

**Parallelism.** Data parallel: each GPU holds the 2.1 TB replica
(Adam states alone are 1.4 TB for 175B). ZeRO-1/2/3 shards
optimizer, gradients, params. Tensor: split layers. Pipeline: split
stages, pay bubbles.

**FlashAttention.** Tile QKV to SRAM, running softmax, recompute
instead of re-reading HBM. Exact, not approximate. ~10x fewer HBM
accesses.

**Quantization.** 70B FP16 = 140 GB. INT8 = 70 GB. Mixed precision:
FP32 masters, FP16 compute.

**SFT.** Loss on output tokens only. Mixture teaches format. Facts
come from pre-training. 13K examples then, ~10M now.

**LoRA.** W = W0 + BA, rank 4: 32,768 params vs 16.7M on a 4096
block (512x fewer). ~10x learning rate. FFN blocks help most.
QLoRA: NF4 base, BF16 adapters, double quantization: ~16x less
VRAM, 65B-class fine-tuning on one GPU.

</div>

<div class="cheat-block" markdown="1">

### Preference tuning (RLHF)

**Why.** SFT imitates good examples. It never says "not this".
Pairs add the negative signal: raise the chosen, lower the
rejected. Shapes tone, not facts.

**Bradley-Terry.** P(i beats j) = sigma(r_i - r_j). Toy: r = 2 vs 1
gives P = 0.731, loss 0.313. Reversed gives 1.313. Train pairwise,
score pointwise. Loss: -E[log sigma(r_w - r_l)].

**RL framing.** Agent = LLM. State = tokens so far. Action = next
token. Reward = one sparse number at the end. ~100k+ rollouts.

**Reward hacking.** Reward climbs while human judgment falls (the
clapping toy). Fix: KL to the SFT model, small steps, monitor
average reward.

**PPO-clip.** L = min(rA, clip(r, 1-eps, 1+eps)A), maximized. Toy:
eps = 0.2, A = 2, r = 1.5 -> min(3.00, 2.40) = 2.40. r is a ratio
(pi_theta/pi_old), not a reward. Beta * KL(pi || pi_ref) keeps the
leash. Four models in memory.

**Advantage.** A = reward - baseline. Reward 5 with baseline 3 is
+2. With baseline 7 is -2. GAE blends horizons. Value head predicts
per token.

**Best-of-N.** Sample 4, keep the max reward. No training. N times
the inference cost. Beat this first.

**DPO.** Solve RLHF for the optimal policy, plug into
Bradley-Terry, train supervised on pairs. Two models, beta ~ 0.1.
Cheaper than PPO. Watch distribution shift.

**Interview line.** "Bradley-Terry turns human pairwise labels into
a pointwise reward. PPO optimizes it with a KL leash. DPO solves
the same objective in closed form."

</div>

<div class="cheat-block" markdown="1">

### Reasoning (GRPO)

**Think-then-answer.** Hidden chain first, answer second. Billed as
output tokens. o1 Sep 2024 to R1 Jan 2025.

**pass@k.** 1 - C(n-c,k)/C(n,k). Toy: n=10, c=3, k=2 ->
1 - 21/45 = 0.533, up from pass@1 = 0.3. T=0 flat, 0.2-0.8 sweet,
1.2 wild. Report temperature or do not compare.

**Verifiable rewards.** Code runs tests. Math parses answers. The
checker is free, so RL runs without humans.

**GRPO.** Sample g completions. A_i = (r_i - mean)/std. Toy:
rewards [0,0,1,0] -> +1.73 for the winner, -0.58 for each loser.
No value function. The group is the baseline. KL to the reference
stays explicit.

**Length bias.** The 1/|o_i| term downweights a 50-token failure 10x
harder than a 500-token one: the model learns to ramble. DAPO
equalizes token weights. Dr. GRPO drops the term. Clip-higher:
asymmetric epsilons.

**R1.** R1-Zero: RL only on the base, accuracy + format rewards,
messy chains. R1: cold-start SFT -> RL (+language reward) -> big
SFT (rejection sampling, 3:1, ~200k) -> final RL (harmlessness on
think tokens). Distill: SFT the teacher's tokens, beats small RL.

**Budget.** Dynamic budgets, "wait" forcing, continuous thoughts.
Shortest chain that still solves the problem.

</div>

<div class="cheat-block" markdown="1">

### RAG, tools, agents

**RAG.** Retrieve, augment, generate. The cutoff forces it.
retraining risks regressions, dumping hits limits. Retrieval is the
whole game.

**Chunking.** ~500 tokens, low-hundreds overlap, ~1500-dim
embeddings. 1M tokens -> 2,000 chunks, 12 MB.

**Retrieval funnel.** Stage 1: bi-encoder, cosine, ANN, recall to
~100. Stage 2: cross-encoder, query + chunk together, precision to
top k.

**Semantic vs BM25.** Embeddings match meaning ("Cuddly" returns
"Huggy", cosine 0.95). BM25 guarantees keywords. Hybrid: 0.5/0.5
scores B 0.80 over A 0.475. HyDE: embed a fake answer doc. Prompt
caching: ~1/10 price on repeated prefixes.

**Metrics.** NDCG on the toy: 4.0/4.262 = 0.939. Reciprocal rank,
precision@k, recall@k. MTEB is the benchmark.

**Tool calling.** API + docs in. Arguments out. Execute. Respond.
The model sees the interface, never the implementation. Two SFT
pair types, or a tuned explanation. Categories: informational,
computation, actions.

**Selection and MCP.** Router picks 2 of 50 tools (RAG over tools).
MCP: servers, tools, prompts, resources. Write once, use
everywhere.

**ReAct.** Observe, plan, act, repeat. Exit on the goal. The
thermostat: get_temp() = 65F, set_temp(+5), done.

**Seven failure modes.** Punt, hallucinated tool, wrong tool,
wrong args, bad output, no output (empty JSON beats silence), bad
synthesis. Fix in groups, not one-offs.

**Safety.** Training-time harmlessness plus inference-time
classifiers. Exfiltration is the attack to name.

</div>

<div class="cheat-block" markdown="1">

### Evaluation

**Scope.** Output quality. Free-form resists metrics.

**Humans.** Ideal, slow, expensive, subjective. Agreement rate lies:
chance = P_A P_B + (1-P_A)(1-P_B). Observed 0.85 is kappa 0.70 at
balanced rates, 0.17 at 90/10. Cohen/Fleiss/Krippendorff correct.
Align raters when kappa drops.

**Rules.** BLEU: precision + brevity penalty (the toy: 0.39 despite
perfect unigram precision). ROUGE: recall-flavored. METEOR:
F-score x ordering penalty. Punish paraphrase. Weak human
correlation. Need references.

**LLM-as-a-judge.** Prompt + response + criteria -> rationale THEN
score. Binary scale. Structured output parses. Single or pairwise
(synthetic preference labels).

**Biases.** Position: ask both orders, vote. Verbosity: guidelines,
counter-examples, length penalty. Self-enhancement: different,
bigger judge.

**Practice.** Crisp guidelines. Temperature 0.1-0.2. Calibrate vs
humans. Do not overoptimize the proxy.

**Factuality.** Extract facts -> check each binary via RAG/search ->
weighted aggregate. Two of four facts wrong, caught individually.

**Agents.** Punt, hallucinated tool, wrong tool/args, bad/no output
(empty JSON beats none), bad synthesis. Categorize in groups, fix
in groups.

</div>

<div class="cheat-block" markdown="1">

### Benchmarks and frontiers

**Families.** MMLU: ~60 tasks, 4-choice, knowledge. AIME: integer
math, reasoning. PIQA: 20k 2-choice physical common sense.
SWE-bench: real GitHub issues + PR tests, coding. HarmBench:
standard/copyright/contextual/multimodal harm, classifier judge.
tau-bench: airline/retail agents. Pass-hat@k = all k succeed.

**Reading.** Pareto: best per dollar (10x cheaper at 95% quality
sits on the frontier). Contamination: hashes, blocklists, fresh
tests. Goodhart: optimized measures stop measuring. Try the models
yourself.

**ViT (2020).** 224x224 image, 16x16 patches: 196 tokens + [CLS].
Weak inductive bias + big data beats CNNs.

**MDM/DLLM.** Mask = noise. Forward masks, reverse unmasks in N
fixed steps: 1,000 tokens in 32 passes (~10x end-to-end).
Fill-in-the-middle. LLaDA works the math.

**Data.** ~80% of search results LLM-generated [uncertain]. Model
collapse: each generation trains on a narrower shadow. Curation and
mid-training answer.

**Open knobs.** Muon optimizer, pre/RMS norm, attention variants,
activations, MoE vs dense.

**Open problems.** Continuous learning, hallucination by design,
personalization, safety.

</div>

<div class="cheat-block" markdown="1">

### Memory aids: never-confuse pairs

| Pair | Distinction |
|---|---|
| FLOPs / FLOPS | Work vs rate. Capital S is speed. |
| r (PPO) / reward | r is the ratio pi_new/pi_old. Never a reward. |
| Top-K / top-P | K fixes count. P fixes probability mass. P adapts. |
| pass@k / pass-hat@k | Any-of-k succeeds vs all-of-k succeed. Capability vs reliability. |
| SFT / preference tuning | Imitate good vs also punish bad. Format vs taste. |
| DPO / PPO | Closed-form on pairs vs online RL. Two models vs four. |
| GRPO / PPO | Group z-score vs value function. Coarse vs fine credit. |
| RAG / long context | Retrieve-then-read vs dump-everything. Cost vs simplicity. |
| Agreement rate / kappa | Raw percent vs chance-corrected. 0.85 can be 0.70 or 0.17. |
| BLEU / ROUGE | Precision-flavored vs recall-flavored. Translation vs summary. |
| Pre-norm / post-norm | Norm before sublayer (deep stacks) vs after (2017 original). |
| MQA / GQA / MHA | 1 KV head / 8 KV heads / 32 KV heads. Cache 32x / 4x / 1x smaller. |
| Mask (BERT) / mask (diffusion) | Corrupt 15% to learn vs corrupt all to generate. |

**Mnemonics.** "Retrieve, augment, generate" = RAG's three verbs in
order. "Observe, plan, act" = ReAct's loop. "Extract, check,
aggregate" = factuality's three steps. "Cold-start, RL, big SFT,
final RL" = R1's four stages. "Punt, hallucinate, wrong tool, wrong
args, bad output, no output, bad synthesis" = the seven failures,
in pipeline order.

**If this, then that.** If the distribution is sharp, top-P shrinks
the set. If flat, it grows: use top-P with a top-K cap. If reward
climbs but humans disagree, it is hacking: tighten beta, add
punishing pairs. If all GRPO rewards tie, the group teaches
nothing: filter it. If kappa drops below 0.6, fix the rubric, not
the raters. If the corpus updates hourly, use HNSW for the hot
shard. If the answer is verifiable, use RLVR. If not, preference
tuning. If the tool returns silence, the agent will hallucinate:
always return something. If the benchmark is older than the
cutoff, assume contamination.

</div>

</div>

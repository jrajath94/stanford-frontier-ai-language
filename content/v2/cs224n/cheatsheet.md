---
page_id: cs224n-cheatsheet
course_slug: cs224n
course_name: "CS224N: NLP with Deep Learning"
course_order: 4
order: 900
nav: "CS224N · Cheatsheet"
title: "CS224N Cheatsheet"
summary: "Every key fact from CS224N on one dense page: definitions, formulas, numbers, decisions, mistakes, interview lines."
---

<div class="cheat-cols" markdown="1">

<div class="cheat-block" markdown="1">

### Core formulas

**Skip-gram:** P(w(t+j) | w(t)) for j in window, j != 0.

**Objective:** J(theta) = -(1/T) sum over t, sum over j of log P(w(t+j) | w(t)).

**Softmax:** P(o|c) = exp(u_o . V_c) / sum over w of exp(u_w . V_c).

**RNN:** h(t) = tanh(W_h h(t-1) + W_e x(t) + b). H(0) = zeros.

**LSTM:** cell = old x forget + candidate x input. Hidden = output x tanh(cell).

**Attention:** scores = QK^T / sqrt(d_k). Weights = softmax(scores). Output = weights x V.

**Bradley-Terry:** P(y1 > y2) = sigma(r1 - r2).

**LoRA:** W + (alpha/r) x B x A. B: d x r. A: r x k.

</div>

<div class="cheat-block" markdown="1">

### Numbers to memorize

- Learning rates: 1e-3, 1e-4, 1e-5.
- SGD mini-batch: 16 or 32.
- Naive softmax: 400K vocab x 100/300-dim dots per prediction.
- N-gram demo: company/bank 0.153, price 0.077 (Reuters trigram).
- Gradient through 30 steps: 0.9^30 = 0.042 (vanishes), 1.1^30 = 17.4 (explodes).
- Clip norm: 5, 10, 20.
- UAS: Chen and Manning 92, Parsey McParseface 94.6.
- BERT: 110M base, 340M large, 15% masking (80/10/10).
- GPT: 117M -> 1.5B -> 175B. Pretraining data: 1.4T (2022) -> ~15T tokens (2024).
- CoT zero-shot: 17.7 -> 78.7.
- FLAN: 3M+ examples (+6.1 to +26.6). LIMA: 1,000.
- DPO: 9 of 10 HF leaderboard models (2024).
- Memory: 16 bytes per param per GPU (2+2+4+4+4).
- BCI: 60-70 wpm vs 150 natural. WER 25% -> near zero (UC Davis).
- AlpacaFarm agreement: 67% (50% random). AlpacaEval: 98% rank correlation.
- Chatbot Arena: 200k votes. Meta bought 1.5M comparisons.
- LLaMA 65B MMLU by harness: 63.7, 63.6, 48.8.

</div>

<div class="cheat-block" markdown="1">

### Toy numbers from the deep dives

- Subsampling: "the" kept 1.4% of the time. "zebra" kept 100%.
- Perplexity toy: P = 0.50, 0.25 -> perplexity 2.83 (about 3 live choices).
- Temperature toy (0.66/0.24/0.10): T=0.5 -> 0.80/0.16/0.04. T=2 -> 0.42/0.32/0.26.
- Top-p=0.9 on (0.66/0.24/0.10): keeps cat + dog, cuts zebra.
- GRU toy: [1.0, 0.5] -> [0.92, 0.77] with z = [0.9, 0.1].
- DPO toy: margin 1.38, loss 0.40, beta = 0.5.
- Elo toy: 1500 vs 1600, upset win -> 1520.5.
- BERTScore toy: feline/rested vs cat/sat -> 0.80. BLEU on the same pair: 0.

</div>

<div class="cheat-block" markdown="1">

### What is used where (Oct 2026)

- Attention: GQA everywhere open (Llama 4, Mistral, Qwen). MHA survives in small models. Closed labs [unknown].
- Position: RoPE is the open standard. Sinusoidal and learned are history.
- Inference: FlashAttention family via vLLM and TensorRT-LLM. The KV cache is the cost center.
- Objectives: CLM for all frontier LMs. MLM retired with encoders. RTD and span corruption are history.
- Fine-tuning: LoRA/QLoRA for open adaptation. Full fine-tune only with huge data. Prefix and adapters faded.
- Decoding: temperature + top-p for chat. Beam for translation (GNMT legacy). Greedy for eval.
- Alignment: DPO family dominates open work (Tulu 3). Closed labs run online RL (DeepSeek-R1). Recipes [unknown].
- Eval: LMArena Elo for vibes. AlpacaEval for dev loops. HELM for breadth. SWE-bench for code agents.

</div>

<div class="cheat-block" markdown="1">

### Decisions

**Word2vec or GloVe?** Prediction versus counting. Similar geometry. Rarely decides results.

**Greedy or beam parsing?** Greedy is linear and fast. Beam recovers accuracy. Google added beam to reach 94.6.

**RNN or transformer?** RNN for streaming short sequences. Transformer for everything at scale: constant interaction distance, full parallelism.

**Sinusoidal or learned positions?** Sinusoidal extrapolates in theory, not in practice. Learned works within length n, crashes beyond.

**Full fine-tune or LoRA?** LoRA when batch-1 barely fits or data is small (65,536 vs 16.7M at r=8). Full when the task needs large global changes.

**ReLU or GELU?** ReLU for simple nets: cheap, sparse, exact. GELU for transformers: smooth everywhere, slightly better gradients.

**Temperature or top-p?** Temperature reshapes the whole distribution. Top-p cuts the unreliable tail. Standard chat recipe: temperature 0.7, top-p 0.9.

**KTO or DPO?** DPO for real pairs (chosen vs rejected on the same prompt). KTO for one-sided thumbs.

**GQA or MHA?** GQA when the KV cache threatens memory (long context, high batch). MHA when quality is all that matters and the cache fits.

**Human or LLM judges?** LLM for iteration (100x faster/cheaper). Humans for final claims. Detailed rubrics either way.

**Online or offline alignment?** Offline is cheap and stable. Online is better: fresh data from the policy, refreshed labels.

</div>

<div class="cheat-block" markdown="1">

### Common mistakes

- Initializing with zeros. False symmetries: nothing learns.
- Oversized learning rate. Overshoot and diverge.
- Reading analogies as proof. Cherry-picked demos, not production.
- Forgetting the causal mask. The model reads the answer. Training is "too easy."
- Normalizing LayerNorm across the batch. It is per word.
- Trusting BLEU on open-ended tasks. Overlap is not meaning.
- Comparing human evals across papers. Different rubrics, different numbers.
- Citing a metric without its harness. LLaMA 65B MMLU: 63.7, 63.6, or 48.8.
- Optimizing a learned reward without a KL leash. Reward hacking breeds gibberish.
- Assuming one vector = one sense. Modern vectors are superpositions.

</div>

<div class="cheat-block" markdown="1">

### Mnemonics

- **Q-S-S-W-C:** attention steps. Query, Score, Softmax, Weighted-average,
  Concatenate.
- **F-I-O:** LSTM gates. Forget (remember), Input (writes), Output
  (reveals).
- **T-M-O:** RLHF stages. Tune (SFT), Model (reward), Optimize (RL).
- **2-2-4-4-4:** bytes per parameter. Params, grads, master, momentum,
  variance. Total 16.
- **A-L-H:** eval ladder. Automatic (speed), LLM judge (dev loops), Humans
  (final claims).

</div>

<div class="cheat-block" markdown="1">

### Never-confuse pairs

- **Perplexity vs entropy:** perplexity = e^entropy per word. Read as live
  choices.
- **Temperature vs top-p:** temperature reshapes, top-p cuts the tail.
- **BLEU vs ROUGE:** precision (was it right?) vs recall (did it cover?).
- **Forward vs reverse mode:** one sweep per input vs one per output.
- **Vanishing vs forgetting:** backward gradient dies vs forward memory
  fades.
- **DPO vs PPO:** offline classification vs online RL on fresh generations.
- **MLM vs CLM:** both directions (reads) vs past only (writes).
- **Greedy vs beam:** one path commits vs k paths compare, k times cost.
- **MHA vs GQA vs MQA:** H KV heads vs G groups vs 1 shared.
- **Fine-tune vs continued pretraining:** labeled task data vs unlabeled
  domain text.

</div>

<div class="cheat-block" markdown="1">

### If this, then that

- KV cache eats GPUs -> GQA (MLA if training from scratch).
- Batch-1 does not fit -> LoRA first, ZeRO-3 second.
- Annotators disagree on scores -> collect pairwise preferences.
- Benchmark saturates -> the ruler is dead, move on.
- Serving long context -> RoPE, and test extrapolation first.
- Tail is crazy -> top-p, not just lower temperature.
- Task is local (autocomplete) -> smoothed n-gram, not transformer.
- Gradient explodes -> clip the norm. Vanishes silently -> change the
  architecture.
- Reward climbs, human preference stalls -> you are hacking the proxy.
  Stop.
- Harness differs -> never compare the numbers.

</div>

<div class="cheat-block" markdown="1">

### Interview one-liners

- "Word meaning is a vector: know a word by the company it keeps."
- "Negative sampling replaces the 400K-word softmax with k+1 logistic regressions."
- "Backprop is the chain rule applied efficiently: store, never recompute."
- "Vanishing is worse than exploding: it is silent."
- "The LSTM adds instead of multiplying: cell = old x forget + candidate x input."
- "Attention is more human-like: the decoder looks back at the source."
- "Self-attention is a set operation: position must be injected."
- "Pretraining reconstructs the input: mask, predict, repeat over trillions of words."
- "RLHF: tune, model preferences with Bradley-Terry, optimize with a KL leash."
- "DPO skips the reward model: Z(x) cancels, binary classification remains."
- "Subsampling throws away 'the' before training: frequent words teach little."
- "Temperature reshapes. Top-p cuts the tail."
- "GRU blends with one gate. The LSTM guards a cell."
- "RoPE rotates queries and keys: the dot product sees only relative distance."
- "Perplexity is the number of live choices the model hesitates over."
- "Never just believe numbers: the harness is half the score."

</div>

<div class="cheat-block" markdown="1">

### Word Vectors (L01-L02)

One-hot: orthogonal, no similarity. Distributional hypothesis (Firth 1957).
Skip-gram: P(context | center). Softmax over vocab. Two vectors per word (v, u).
Average at end. SGD: sample windows, noise helps. Negative sampling: 1 real + k
negatives, sigmoid. GloVe: ratios of co-occurrence. Dot ~ log prob + biases.
Intrinsic: analogies (cherry-picked), similarity ratings. Extrinsic: NER.
Senses: one vector = superposition. NER: 5x100=500 input, 8x500 matrix, cross-entropy.

</div>

<div class="cheat-block" markdown="1">

### Neural Nets and Parsing (L03-L04)

Layer: z = Wx + b, h = sigma(z). Chain rule: ds/dz = ds/dh x dh/dz.
Jacobian: matrix of partials. Forward: apply and store. Backward: reuse.
Ambiguity: PP attachment ("space whales"). Dependency: head-dependent arcs.
Transitions: SHIFT, LEFT-ARC, RIGHT-ARC. Stack + buffer. Greedy. Linear. Oracle.
Eval: unlabeled arcs, labeled arcs+labels. Old: millions of sparse indicators.
New: dense word/POS/label embeddings, concat, ReLU, softmax. Graph-based: n^2 scores,
MST, 50x slower. Stanza (2017) in production.

</div>

<div class="cheat-block" markdown="1">

### RNNs, LSTMs, Attention (L05-L07)

LM: P(next word | history). N-gram: count, sparsity (0/1), storage.
RNN: one weight set, h(t) = tanh(...), h(0) = zeros. Loss: avg NLL per position.
Problems: sequential for-loop (unparallelizable), forgetting ((0.5)^6 < 2%).
Gradients: eigenvalues <1 vanish, >1 explode. Clip norm at 5/10/20. LSTM 1997:
3 gates (forget = remember). Additive cell: gradients flow. BiLSTM: concat both
directions. Seq2seq: encoder -> final state -> decoder + start token.
Bottleneck = one fixed vector. MT: rules -> phrase-based -> 2014 NMT.
Attention (Bahdanau 2015): decoder queries encoder, scores, softmax, weighted
average, concat, predict. Shorter gradient paths. BLEU: n-gram precision + brevity
penalty.

</div>

<div class="cheat-block" markdown="1">

### Transformers (L08)

Fixes: linear interaction distance, O(n) sequential. Fuzzy key-value lookup.
Separate Q/K = low-rank bilinear. Self-attention: set op. Needs position.
Sinusoidal vs learned d-by-n. Mask: -inf future, softmax 0. Decoder masks.
Scale: /sqrt(d_k). Heads: 8, d/H each, 64+ dims. Residual: gradient 1.
LayerNorm: per word. Mu, sigma, gamma, beta. Block: attn -> add&norm -> FFN ->
add&norm. Cross-attention: Q decoder, K/V encoder. Cost: quadratic. N=30 fine,
n=50000 not.

</div>

<div class="cheat-block" markdown="1">

### Pretraining (L09)

Objective: reconstruct masked input. Data: ~5T words vs ~1M labeled.
Teaches: trivia, syntax, coreference, semantics, sentiment, world models.
BERT: 15% mask, 80/10/10, segments, CLS, 110M/340M. NSP unnecessary.
GLUE sea change. PEFT: prefix, prompt, LoRA. GPT: 117M -> 1.5B -> 175B.
In-context: examples, no updates. CoT: scratch pad. Chinchilla: smaller + more
data. Cost ~ params x tokens. Recipe: pretrain -> continue -> fine-tune.
Warning: fluent but frequently wrong.

</div>

<div class="cheat-block" markdown="1">

### Post-Training (L10)

Few-shot: examples in, no gradient updates. CoT: 17.7 -> 78.7 zero-shot.
Instruct: FLAN 3M+ (+6.1 -> +26.6), LIMA 1000. RLHF: SFT -> reward model -> RL.
Bradley-Terry: P = sigma(r1-r2). Pairwise because humans are noisy.
Hacking: gibberish, authoritative > truthful, longer wins. KL penalty.
DPO: closed form, Z cancels, binary classification. Matches RLHF. 9/10 HF models.
ChatGPT = dialogue InstructGPT.

</div>

<div class="cheat-block" markdown="1">

### Evaluation (L11)

Purposes: train, dev, deploy, publish. Closed <10 answers. Open-ended.
MMLU 25% -> 90% in ~4y. BLEU: precision + brevity. ROUGE: recall.
"heck yes": yes = 67%, yep = 0 (false negative), heck no ~7x (false positive).
Human: gold but noisy. 67% agreement. Never cross-compare.
Arena: 200k votes, Elo. LLM judge: 100x faster/cheaper. GPT-4 > human-human.
98% rank correlation. Length bias ~70%. Monoculture risk. HELM: look at everything.
Harness moves numbers: 63.7 / 63.6 / 48.8. Rule: never just believe numbers.

</div>

<div class="cheat-block" markdown="1">

### Efficient Training (L12)

fp32 4B. Fp16 2B + scalers. Bf16 2B, 8 exp bits, no scalers, Ampere+.
Memory: 2+2+4+4+4 = 16B/param/GPU. Optimizer states are 12 of 16.
DDP: full copies, all-reduce 2B/param. ZeRO: 1 optimizer, 2 +grads, 3 +params.
Reduce-scatter. FSDP. Checkpointing: trade compute for activation memory.
LoRA: W + (a/r)BA. Train A/B. Attention matrices. R slider. Merge at inference.
PEFT: fit batch-1. Small-data generalization. Sustainability: demand > capacity.

</div>

<div class="cheat-block" markdown="1">

### Speech BCI (L13)

Locked-in: brain works, body does not. Letter board: minutes/sentence.
Eye-tracking: tiring. 2017: imagined movement, 40/20 chars/min.
T12: 4 arrays (2 motor cortex, 2 Broca's). Motor cortex carries signal.
WER: 25% (25/100 wrong). UC Davis near zero with continuous training.
Daily use with family. Speed: 60-70 wpm vs 150 natural. Handwriting 13-14.
Eye-track ~5. Frontier: inner speech (no ground truth). UCSF avatar
(phonemes + articulation -> 3D avatar).

</div>

<div class="cheat-block" markdown="1">

### Reasoning and Agents (L14)

Deductive: certain. Inductive: probable. Abductive: best explanation.
CoT: steps first. Self-consistency: sample, majority vote.
Distill rationales. Iterate on own (beats human ones).
Counterfactual: base-9 vs base-10 separates memory from reasoning.
Agent: network + environment + observation + action + goal G.
Pre-LM: semantic parsers, plan inference, RL. 2024: trajectory modeling.
CoT in a loop. Train: synthetic demos, hindsight relabeling.
Benchmarks: MiniWoB (<3 actions, far from perfect), WebArena (sandbox web),
WebLinx (real web, human action). Failures: email in password field.
Repeated searches. No recovery. Long horizons: 0.95^20 = 0.36.

</div>

<div class="cheat-block" markdown="1">

### Life After DPO (L15)

Lambert: Berkeley PhD, HF, AI2. RL background. Moment: post-training is the action.
Data: Arena 800k. Meta 1.5M. Labs more. Online vs offline: fresh data + fresh
labels vs static. UltraFeedback distills many models. Self-rewarding: judge own
data, iterate DPO. Beyond pairwise: KTO one-sided. Starling k-wise. SteerLM
fine-grained. Open Qs: distribution matching? search + synthetic to exceed humans?
reward models as moat?

</div>

</div>

---
page_id: cme295-l03
course_slug: cme295
course_name: "CME295: Transformers and Large Language Models"
course_order: 5
order: 3
nav: "L03 · LLMs and decoding"
title: "Lecture 3: Large Language Models, Decoding, and Inference"
summary: "From probabilities to tokens and from tokens to affordable serving: the LLM definition, mixture of experts with a worked routing toy, decoding strategies demonstrated on one distribution, temperature arithmetic, prompting as programming, and the inference toolbox (KV cache, paging, MLA, speculative decoding) grounded in the memory-bandwidth bottleneck."
date: "2025-10-10"
instructor: "Afshine Amidi, Shervine Amidi"
offering: "Autumn 2025"
duration: "1:48:36"
video_id: Q5baLehv5So
video_title: "CME295 Lecture 3, Autumn 2025"
video_caption: "Original lecture. LLM definition, MoE, decoding, temperature, prompting, and inference efficiency."
sources:
  - tag: video
    label: "Lecture 3 recording (YouTube)"
    url: https://www.youtube.com/watch?v=Q5baLehv5So
  - tag: slides
    label: "Lecture 3 slides (PDF), CME295 Autumn 2025"
  - tag: paper
    label: "Fedus et al., Switch Transformers: Scaling to Trillion Parameter Models (2021)"
    url: https://arxiv.org/abs/2101.03961
  - tag: paper
    label: "Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention (2023)"
    url: https://arxiv.org/abs/2309.06180
concepts: [llm-definition, decoder-only, next-token-prediction, mixture-of-experts, sparse-routing, routing-collapse, switch-transformer, greedy-decoding, beam-search, sampling, top-k, top-p, temperature, guided-decoding, prompting, zero-shot, few-shot, chain-of-thought, self-consistency, context-rot, prompt-caching, kv-cache, paged-attention, mla, speculative-decoding, multi-token-prediction, memory-bound]
---

## The problem: what does "large" mean

A **language model** is a next-token probability machine. At every
position it computes P(x_t | x_1 ... x_{t-1}): the probability of
each vocabulary word given the words so far. **Large** means three
magnitudes at once: hundreds of billions of parameters, hundreds of
billions to tens of trillions of training tokens, and the
engineering to serve both.

Architecturally, the field converged: over 90% of large models are
**decoder-only transformers**. One stack, one objective (predict the
next token), no encoder-decoder plumbing. The decoder-only pattern
won because it is simple and it scales.

![LLM definition](assets/l03-llm-def.svg "Next-token probabilities at scale: parameters, tokens, decoder-only. Stanford Frontier AI.")

## The problem: every token runs the full model

In a dense transformer, every token passes through every
feed-forward network in every layer. The FFN holds roughly two
thirds of the parameters. Serving millions of tokens through all of
that is expensive. The lecture asks: do you need all parameters for
every token?

## The key question

Training buys you a probability distribution over next tokens. How do
you turn that distribution into answers, and how do you serve those
answers fast enough and cheap enough to use?

## Mixture of experts: route tokens to specialists

The metaphor is a room of experts: route each token to the
specialists it needs, and let the rest sleep. Formally, the output
is a weighted sum over experts:

**y-hat = sum of g_i * E_i(x)**

A **gating network** produces weights g_i per token. Each **expert**
E_i is a feed-forward network. In a dense MoE all experts run and
the weights blend them. In a **sparse MoE** only the top-K (K = 1 or
2) experts run per token. Sparse is the point: capacity without
proportional compute.

Watch routing on a toy. Four experts, one token, gating scores:

```ascii
token "bear", gating network outputs:
  expert 1: 0.60    expert 2: 0.25    expert 3: 0.10    expert 4: 0.05

top-2 routing: run expert 1 and expert 2 only.
output = 0.60 * E1(x) + 0.25 * E2(x)   (renormalized over the chosen two)
experts 3 and 4 sleep: zero compute, zero memory traffic.
```

Three details matter:

- **Routing is per token, per layer.** Each token picks its experts
  fresh at every layer. Tokens also scatter across GPUs, which
  parallelizes the expert computation.
- **Routing collapse**
  ([21:16](https://www.youtube.com/watch?v=Q5baLehv5So&t=1276s)).
  The router discovers that one expert is good enough and sends
  everything there. The other experts starve. Demonstrate it: start
  the toy with gating [0.40, 0.30, 0.20, 0.10]. Expert 1 gets 40% of
  the training signal, so it improves fastest. Next round its
  gating rises to 0.55, it gets more signal, it improves further.
  The loop converges to [1.00, 0, 0, 0]: you paid for four experts
  and use one. Fixes: an auxiliary load-balancing loss plus noisy
  gating, so the router keeps exploring.
- **Switch Transformer**
  ([28:34](https://www.youtube.com/watch?v=Q5baLehv5So&t=1714s)).
  The recommended read: top-1 routing pushed to about 1.6 trillion
  parameters. It proved sparse models scale.

![MoE](assets/l03-moe.svg "Token-level routing to top-2 experts per layer. Experts are FFNs. Stanford Frontier AI.")

> [!QA]
> Q: Why are experts feed-forward networks and not attention layers?
> A: The FFN holds most of a transformer's parameters (roughly two
> thirds), so sparsifying it saves the most compute. Attention is
> comparatively cheap and benefits from seeing the full context, so
> it stays dense.
> Follow-up: What breaks if routing collapses?
> A: You paid for N experts and use one. Capacity is wasted, and the
> single active expert becomes a bottleneck that every token queues
> through. The auxiliary loss exists to keep the router honest.

## The problem: the model outputs probabilities, not tokens

The model's output at each step is a distribution over the whole
vocabulary: 50,000 numbers summing to 1. **Decoding** turns that
distribution into a token. Five strategies cover the space. Watch
all five on one toy distribution:

```ascii
vocabulary:  "lit" 0.50,  "read" 0.30,  "slept" 0.12,  "ate" 0.08
```

- **Greedy.** Take the argmax: "lit". Fast, deterministic, and prone
  to dull loops: the most likely token at every step is rarely the
  best sequence.
- **Beam search**
  ([41:36](https://www.youtube.com/watch?v=Q5baLehv5So&t=2496s)). Keep
  B hypotheses alive. With B = 2, extend "lit" and "read", keep the
  two best by summed log-probability, extend again. Length
  normalization divides by a length factor, because log-probabilities
  are negative: without it, a 3-token sequence summing to -2.1 always
  loses to a 2-token sequence summing to -1.8, regardless of quality.
  Better quality, B times the cost.
- **Sampling.** Draw from the distribution. On the toy, "lit" 50% of
  the time, "read" 30%, "slept" 12%, "ate" 8%. Diverse, occasionally
  surprising. The only randomness in the whole transformer lives
  here.
- **Top-K.** Sample from the K most likely tokens only. With K = 2,
  renormalize over "lit" (0.50) and "read" (0.30): 0.625 and 0.375.
  "slept" and "ate" get zero. Cuts the weird tail.
- **Top-P (nucleus).** Sample from the smallest set whose cumulative
  probability reaches P. With P = 0.9: "lit" + "read" = 0.80, not
  enough. Add "slept" = 0.92, enough. The set is {lit, read, slept}.
  It adapts automatically: sharp distributions give small sets, flat
  distributions give large ones.

![Decoding](assets/l03-decoding.svg "Greedy, beam search, sampling, top-K, top-P. Stanford Frontier AI.")

## Temperature reshapes the distribution

**Temperature** T reshapes the softmax: p_i is proportional to
exp(z_i / T)
([52:42](https://www.youtube.com/watch?v=Q5baLehv5So&t=3162s)).
Watch it on toy logits z = [3, 2, 1]:

```ascii
T = 0.5:  exp([6, 4, 2]) = [403, 55, 7]    -> [0.87, 0.12, 0.02]  (spiky)
T = 1.0:  exp([3, 2, 1]) = [20.1, 7.4, 2.7] -> [0.67, 0.24, 0.09] (trained)
T = 2.0:  exp([1.5, 1, 0.5]) = [4.5, 2.7, 1.6] -> [0.51, 0.31, 0.19] (flatter)
T -> inf: -> [0.33, 0.33, 0.33]  (uniform noise)
```

As T approaches 0, the argmax takes everything and outputs turn
deterministic. At T = 1 you get the trained distribution. As T grows
large, the distribution flattens toward uniform.

Two subtleties the lecture stresses. First, nothing in the
transformer is probabilistic except the sampling step. The model is a
deterministic function of its input. Randomness enters only when you
sample. Second, T = 0 is deterministic in theory but not always in
practice: GPU floating-point nondeterminism can still change outputs
between runs.

![Temperature](assets/l03-temperature.svg "T to 0: spiky. T = 1: trained. T to infinity: uniform. Stanford Frontier AI.")

> [!QA]
> Q: When would you use greedy decoding over sampling?
> A: When there is one right answer: math, code, factual lookup.
> Sampling adds variance you do not want. Use sampling (with
> temperature) when you want variety: brainstorming, dialogue,
> creative writing.
> Follow-up: Why does beam search need length normalization?
> A: Log-probabilities are negative, so longer sequences sum to more
> negative totals. Without normalization the search prefers short
> sequences regardless of quality. Dividing by a length penalty
> levels the field.

## The problem: sometimes the output must parse

A JSON API call, a form, a code skeleton: free text is not
acceptable. **Guided decoding**
([65:12](https://www.youtube.com/watch?v=Q5baLehv5So&t=3912s))
constrains sampling to tokens the grammar allows. Model the valid
outputs as a finite state machine or grammar
([66:48](https://www.youtube.com/watch?v=Q5baLehv5So&t=4008s)). At
each step, mask every token that would break the grammar and
renormalize over the rest.

The toy: the schema expects an opening brace. The model proposes
"lit" 0.50, "read" 0.30, "{" 0.12, "ate" 0.08. Mask the invalid
three, renormalize: "{" gets 1.0. The output always parses because
invalid tokens had zero probability.

![Guided decoding](assets/l03-guided-decoding.svg "Mask invalid tokens at each step. The output always parses. Stanford Frontier AI.")

## The problem: steer the model without retraining

Retraining is expensive. **Prompting** steers behavior with text
alone. A prompt has an anatomy: **context** (background facts),
**instructions** (what to do), **inputs** (the data), **constraints**
(format, length, tone). Getting each part right is most of applied
LLM work.

The technique ladder, on one running task ("classify this review"):

- **Zero-shot**
  ([74:59](https://www.youtube.com/watch?v=Q5baLehv5So&t=4499s)). No
  examples. "Classify: This teddy bear is SO CUTE!" Works when the
  instruction alone suffices.
- **Few-shot.** A few input-output examples in context. Shows the
  pattern to copy: format, style, edge cases.
- **Chain-of-thought**
  ([78:50](https://www.youtube.com/watch?v=Q5baLehv5So&t=4730s)). Ask
  the model to reason step by step before answering. More tokens
  means more compute spent on the problem, and intermediate steps
  catch errors early.
- **Self-consistency**
  ([82:16](https://www.youtube.com/watch?v=Q5baLehv5So&t=4936s)).
  Sample N reasoning paths, take the majority answer. The toy: N =
  5 paths give answers [A, B, A, A, C]. Majority: A, with 3 of 5
  votes. Independent noise cancels. The modal answer is usually the
  careful one.

Two warnings come with scale. Context windows reach hundreds of
thousands of tokens, but stuffing them hurts: irrelevant context
degrades performance, a phenomenon the lecture calls **context rot**
([69:09](https://www.youtube.com/watch?v=Q5baLehv5So&t=4149s)), citing
a needle-in-a-haystack paper from summer 2025 [uncertain: paper not
named in the transcript]. And repeated prompt prefixes can be cached:
**prompt caching** computes the shared prefix once and reuses it,
cutting cost and latency for repeated scaffolding.

![Prompting](assets/l03-prompting.svg "Anatomy plus the four techniques. CoT spends tokens as compute. Stanford Frontier AI.")

## The problem: serving is where the bills land

Each forward pass streams the full weight matrices from HBM memory to
the chip to produce one token. The arithmetic per byte moved is tiny.
So the bottleneck is **memory bandwidth, not FLOPS**: decoding is
**memory-bound**. Every efficiency trick here either moves less
memory or amortizes one memory move over more tokens. Watch the
arithmetic:

```ascii
70B parameters at 2 bytes each = 140 GB moved per token, per request.
Generate 100 tokens: 100 memory moves of 140 GB = 14 TB of traffic.
```

**KV cache**
([89:04](https://www.youtube.com/watch?v=Q5baLehv5So&t=5344s)).
Without caching, step t recomputes keys and values for all t-1
previous tokens: O(t) work per step, O(n^2) total. The cache stores
them and appends one row per step. This is why GQA and MQA from
Lecture 2 matter: fewer KV heads means a smaller cache. The lecture
notes the stored quantity is non-negligible, roughly eight KV head
copies per block [uncertain: exact referent of "eight copies" at
[96:51](https://www.youtube.com/watch?v=Q5baLehv5So&t=5811s)].

**Paged attention.** The KV cache fragments like OS memory: internal
fragmentation (reserved but unused space inside blocks) and external
fragmentation (scattered free space,
[94:34](https://www.youtube.com/watch?v=Q5baLehv5So&t=5674s)). The
toy: a request reserves 256 slots but uses 137. 119 slots sit idle
inside the reservation. Across thousands of requests the waste adds
up. Paging the cache into fixed blocks (block size around 16 in the
paper's setting [uncertain: speaker hedged]) packs memory tightly
and raises throughput.

**MLA** ([97:21](https://www.youtube.com/watch?v=Q5baLehv5So&t=5841s),
multi-head latent attention, DeepSeek V2). Compress keys and values
into small latent vectors instead of storing full heads. Less memory
per token, same information.

![KV cache and paging](assets/l03-kv-paged.svg "Cache the KVs. Page them. Or compress them with MLA. Stanford Frontier AI.")

**Speculative decoding**
([101:37](https://www.youtube.com/watch?v=Q5baLehv5So&t=6097s)). A
small draft model proposes k tokens fast. The large target model
verifies all k in one forward pass, accepting the good prefix and
resampling at the first rejection. The toy: k = 3 drafts, the target
accepts 2, rejects the third, and resamples it. Two memory moves
produced 3 tokens instead of 3 moves for 3 tokens. Verification moves
the weights once for k tokens, so accepted drafts are nearly free
speed.

**Multi-token prediction**
([106:43](https://www.youtube.com/watch?v=Q5baLehv5So&t=6403s)).
Train the model to emit several tokens per forward pass. Same idea
as speculation, but learned into the model instead of bolted on.

![Speculative decoding](assets/l03-speculative.svg "Draft fast, verify in one pass, keep the accepted prefix. Stanford Frontier AI.")

> [!QA]
> Q: Why is decoding memory-bound rather than compute-bound?
> A: Each forward pass streams the full weight matrices from HBM to
> the chip to produce one token. The arithmetic per byte moved is
> tiny. So the bottleneck is memory bandwidth, not FLOPS. Every
> efficiency trick here either moves less (KV cache, MLA, GQA) or
> amortizes the move over more tokens (speculative decoding,
> multi-token prediction).
> Follow-up: When does speculative decoding fail to help?
> A: When the draft model is wrong too often. Rejected drafts waste
> the verification pass, and resampling adds latency. The draft must
> be close enough to the target that acceptance rates stay high.

## Mapping back: one bottleneck rules serving

| Problem | Trick | What it saves |
|---|---|---|
| Every token runs every FFN | Sparse MoE | Compute: only top-1/2 experts run per token |
| Dense FFN per token per layer | Switch Transformer | Scale: 1.6T parameters at top-1 cost |
| Probabilities, not tokens | Decoding strategies | The right bias per task: greedy for facts, sampling for variety |
| Flat or spiky outputs | Temperature | Reshape exp(z_i / T): 0.5 spiky, 1 trained, 2 flat |
| Free text where JSON is needed | Guided decoding | Validity by construction: invalid tokens get 0 |
| Retraining to steer | Prompting | Zero/few-shot, CoT, self-consistency: no weights change |
| Recompute per step | KV cache | One row appended per step instead of O(t) recompute |
| Cache fragmentation | Paged attention | Fixed blocks: no idle slots inside reservations |
| Cache too big | MLA | Latent compression instead of full heads |
| One token per memory move | Speculative decoding, multi-token prediction | Amortize: k tokens per weight move |

## The honest price

Every trick trades something. Sparse MoE buys capacity and pays in
routing collapse risk and expert-parallel complexity. Beam search
buys quality and pays B times the compute. Sampling buys diversity
and pays in reproducibility. Guided decoding buys validity and pays
in expressiveness: the model cannot say what the grammar forbids.
Prompt caching buys speed and pays in prefix rigidity. Speculative
decoding buys speed and pays a draft model plus wasted verification
on rejections. The memory-bandwidth bottleneck is a fact of physics.
The tricks are ways to spend less of it per token.

## Recap: the whole lesson on one screen

The story in eight steps. Each step answers the one before it.

1. **Large means three magnitudes.** Next-token probabilities, with
   hundreds of billions of parameters and up to tens of trillions of
   tokens. Over 90% of LLMs are decoder-only.
2. **Route tokens to specialists.** y-hat = sum of g_i * E_i(x).
   Sparse top-1/2 routing per token per layer. Collapse is the
   failure mode: the router converges to one expert. Auxiliary loss
   plus noisy gating is the fix.
3. **Decoding turns probabilities into tokens.** Greedy (argmax),
   beam (B hypotheses with length norm), sampling (the only
   randomness), top-K (cut the tail), top-P (adapt the set to the
   distribution).
4. **Temperature reshapes.** exp(z_i / T). On logits [3,2,1]: T =
   0.5 gives [0.87, 0.12, 0.02]. T = 1 gives [0.67, 0.24, 0.09]. T =
   2 gives [0.51, 0.31, 0.19]. T = 0 is deterministic in theory, not
   always on GPUs.
5. **Constrain when output must parse.** Grammar plus mask:
   invalid tokens get zero probability. JSON comes out valid by
   construction.
6. **Prompting is programming in text.** Context, instructions,
   inputs, constraints. Zero-shot, few-shot, chain-of-thought,
   self-consistency (5 paths, majority wins). Beware context rot.
   Cache repeated prefixes.
7. **Decoding is memory-bound.** 70B at 2 bytes: 140 GB moved per
   token. KV cache appends one row per step. Paging kills
   fragmentation. MLA compresses.
8. **Amortize the memory move.** Speculative decoding: draft k,
   verify in one pass, keep the accepted prefix. Multi-token
   prediction learns it into the model.

## Official sources and further reading

**Official:**
- Lecture 3 recording (YouTube): timestamped above.
- Lecture 3 slides (PDF), CME295 Autumn 2025.
- Fedus et al., "Switch Transformers" (2021):
  https://arxiv.org/abs/2101.03961 — top-1 MoE at 1.6T parameters.
- Kwon et al., "PagedAttention" (2023):
  https://arxiv.org/abs/2309.06180 — paged KV cache management.

**Further reading:**
- Holtzman et al., "The Curious Case of Neural Text Degeneration"
  (2019): nucleus (top-P) sampling.
- DeepSeek-V2 paper (2024): multi-head latent attention.
- Leviathan et al., "Fast Inference from Transformers via Speculative
  Decoding" (2023).

**Caveats from these sources.** The "eight copies per block" figure is
the speaker's live estimate [uncertain]. The context-rot paper is
unnamed in the transcript [uncertain]. The PagedAttention block size
(~16) is hedged by the speaker [uncertain]. MoE load-balancing
recipes vary by lab. The lecture gives the standard one.

## Connections to the other courses

- **CS336 L01:** the token chip and tokenization. Every decoding
  decision here operates on its IDs.
- **CS336 L04:** MoE in depth, plus linear attention as an
  alternative efficiency story.
- **CS336 L10/L18:** inference systems: KV cache management, batching,
  and serving at scale.
- **CS336 L02:** resource accounting: why memory bandwidth dominates
  decoding cost.
- **CME295 L02:** GQA/MQA, the architectural side of the KV cache
  story.

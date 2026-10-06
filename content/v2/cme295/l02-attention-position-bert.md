---
page_id: cme295-l02
course_slug: cme295
course_name: "CME295: Transformers and Large Language Models"
course_order: 5
order: 2
nav: "L02 · Attention and BERT"
title: "Lecture 2: Attention Maps, Positions, and the BERT Family"
summary: "The transformer after 2017, rebuilt from its problems: reading attention maps on an anaphora toy, why positions went relative (worked sinusoid, ALiBi, RoPE toys), why norms moved before sublayers, how attention got cheaper with numbers, and BERT trained end to end with the 80/10/10 and 50/50 arithmetic."
date: "2025-10-03"
instructor: "Afshine Amidi, Shervine Amidi"
offering: "Autumn 2025"
duration: "1:47:11"
video_id: yT84Y5zCnaA
video_title: "CME295 Lecture 2, Autumn 2025"
video_caption: "Original lecture. Attention maps, positional encodings, normalization, efficient attention, and encoder-decoder families."
sources:
  - tag: video
    label: "Lecture 2 recording (YouTube)"
    url: https://www.youtube.com/watch?v=yT84Y5zCnaA
  - tag: slides
    label: "Lecture 2 slides (PDF), CME295 Autumn 2025"
  - tag: paper
    label: "Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers (2018)"
    url: https://arxiv.org/abs/1810.04805
  - tag: paper
    label: "Su et al., RoFormer: Rotary Position Embedding (2021)"
    url: https://arxiv.org/abs/2104.09864
concepts: [attention-map, anaphora, positional-encoding, sinusoidal, relative-position, alibi, rope, layer-norm, pre-norm, rmsnorm, longformer, sliding-window, mqa, gqa, kv-cache, t5, span-corruption, bert, mlm, nsp, distilbert, roberta]
---

## The problem: we cannot see what attention reads

Lecture 1 built the attention arrow: query against keys, softmax,
mix the values. A model with 12 layers and 12 heads computes 144
attention maps per sentence. Each map is an N x N matrix: rows are
queries, columns are keys, brightness is weight. Can we read one and
check whether the model reads like we do?

The lecture's test case is **anaphora**, the linguistic term for a
pronoun pointing back to its noun. Take the sentence "The animal did
not cross the street because it was too tired." The pronoun "it"
refers to "animal". Now look at one attention head's weights for the
query token "it", over the eight tokens:

```ascii
query "it" attends to:
  The      0.03
  animal   0.72
  did      0.02
  not      0.01
  cross    0.03
  the      0.02
  street   0.06
  because  0.04
  too      0.03
  tired    0.04
```

The weight on "animal" is 0.72. The map resolves the pronoun the way
a reader does. Two lessons follow. First, attention weights are
inspectable: you can check whether the model's reading matches yours.
Second, no single head tells the whole story. Different heads track
different relations, which is why the architecture uses many of them.

![Attention map](assets/l02-attention-map.svg "The query 'it' attends most strongly to 'animal'. Attention maps make coreference visible. Stanford Frontier AI.")

> [!QA]
> Q: Do attention maps explain the model's decision?
> A: They show information flow, not reasoning. A bright cell says
> token A read token B's representation. It does not say why that
> helped, or that the model "understood" coreference. Treat maps as a
> debugging lens, not a proof of understanding.
> Follow-up: Why look at maps at all, then?
> A: Because they catch failures cheaply. If "it" attended to
> "street" instead of "animal", you would suspect the head before
> blaming the data. Interpretability starts with checking the
> obvious.

## The problem: positions encode addresses, attention needs distances

The 2017 transformer adds one position vector per index to each
embedding. Two flavors: learned (a trainable vector per position, the
BERT style) and sinusoidal (fixed sin/cos waves, the original paper).
The sinusoidal version has a mathematical property the lecture
highlights. Write the position embedding at position p as pairs of
sin and cos at several frequencies. The dot product of the embeddings
at positions p and q works out to a sum of cos terms of (p - q).
Only the difference survives:

```ascii
PE(p) . PE(q) = sum over frequencies of cos((p - q) * omega)
```

The model can read relative distance straight out of the dot
product. Whether "bear" sits at position 5 or 500, "teddy" is one
step before it, and the waves say so. High-frequency dimensions
change fast and track local order. Low-frequency dimensions change
slowly and carry long-range information.

That property is a hint of what the field wanted: attention cares
about distance, not address. Absolute encodings answer "where am I".
Attention needs "how far apart are we". Three successors encode
relative position directly, each a small worked idea.

**T5 relative bias.** Add a learned scalar to the attention score,
chosen by distance bucket. The toy: score("bear", "teddy") gets +0.8
because distance 1 has bias 0.8. score("bear", "street") gets +0.1
because distance 5 has bias 0.1. Close tokens get a boost from the
bucket, far tokens do not.

**ALiBi** (Attention with Linear Biases,
[30:22](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=1822s)). Subtract
a fixed penalty proportional to distance. The toy with penalty slope
m = 0.5:

```ascii
distance 1: score - 0.5
distance 2: score - 1.0
distance 4: score - 2.0
```

No learned parameters. Far tokens are pushed down by arithmetic, not
by learned weights. Because the penalty is a formula, it keeps working
past the longest training sequence: a model trained on 2,048 tokens
still penalizes distance 3,000 correctly. It extrapolates.

**RoPE** (rotary position embeddings,
[31:37](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=1897s)). Rotate
the query and key vectors by an angle proportional to their
positions. The toy in two dimensions: q = [1, 0] at position 3,
k = [1, 0] at position 5, rotation angle = position * theta. Rotate q
by 3*theta, k by 5*theta. Their dot product is cos((5-3)*theta):
only the distance 2 remains. The rotation happens inside the QK dot
product, where it belongs. RoPE is the default choice in current
models: no extra parameters, relative by construction, clean long
context extension.

![Position embeddings](assets/l02-pos-embeddings.svg "Learned vs sinusoidal. Dot products encode relative distance. Stanford Frontier AI.")
![RoPE](assets/l02-rope.svg "Rotate Q and K by position. Their dot product keeps only relative distance. Stanford Frontier AI.")

> [!QA]
> Q: Why did the field move from absolute to relative positions?
> A: Absolute encodings answer "where am I", but attention needs "how
> far apart are we". Relative schemes bake the distance into the
> score computation itself, and they generalize better: a model
> trained on length-2048 sequences can still reason about distance
> 3000, because distance 3000 is just another bucket, angle, or
> penalty, not an unseen vector.
> Follow-up: Why is RoPE the default today?
> A: It needs no extra parameters, it lives inside the QK dot product
> rather than as an add-on, and it extends cleanly to long contexts.
> Simplicity plus generality wins.

## The problem: deep stacks choke on post-norm

The 2017 paper normalizes *after* each sublayer: sublayer, then layer
norm, then the residual add. This is **post-norm**. Picture the
residual stream, the highway that carries the input of each layer
straight through to the next. With post-norm, the gradient flowing
back down the highway passes through a layer norm at every layer.
Each normalization rescales and recenters, and the product of many
such rescalings destabilizes deep stacks.

Modern stacks flip it ([46:36](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=2796s)):
normalize *before* the sublayer, with the residual stream flowing
untouched. This is **pre-norm**. The highway stays clean, the
gradient travels down it without passing through 48 normalizations,
and 96-layer stacks train stably.

The norm itself also changed
([47:04](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=2824s)). **Layer
norm** centers by the mean and scales by the variance: for a vector
x, subtract the mean, divide by the standard deviation. **RMSNorm**
skips the centering: scale by the root mean square only.

```ascii
layer norm:  x -> (x - mean(x)) / std(x)
RMSNorm:     x ->  x / rms(x)
```

Fewer parameters, same stability in practice, cheaper to compute.
Pre-norm plus RMSNorm is the current stack.

![Pre-norm and RMSNorm](assets/l02-norm.svg "Post-norm (2017) normalizes after the sublayer. Pre-norm normalizes before it. RMSNorm drops the mean. Stanford Frontier AI.")

## The problem: full attention costs O(n^2)

Every token attends to every token, so the score matrix is n x n.
Watch the numbers:

```ascii
n = 4,096:  full attention = 4,096^2 = 16,777,216 scores
sliding window w = 512: n * w = 4,096 * 512 = 2,097,152 scores
ratio: 8x fewer
```

For long documents the quadratic price hurts. Several patterns attend
to less. **Longformer** (2020,
[51:30](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=3090s)): each
token attends to w neighbors, plus a few global tokens that attend
everywhere. Cost drops to O(n*w). **Mistral 7B**
([54:33](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=3273s)) stacks
sliding-window layers: each layer sees w tokens, but the receptive
field grows with depth, like dilated convolutions. A token at layer 3
indirectly reaches 3*w tokens back. The pattern is general: pay full
attention where it matters, approximate elsewhere.

![Attention approximations](assets/l02-attention-approx.svg "Full, sliding window, and stacked windows with growing receptive field. Stanford Frontier AI.")

## The problem: the KV cache eats inference memory

**Multi-head attention** gives every head its own key and value
projections. At inference, autoregressive decoding generates one
token at a time. Without a cache, step t would recompute keys and
values for all t-1 previous tokens. The **KV cache**
([57:59](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=3479s)) stores
them once and appends one row per step. The cache grows with heads,
layers, and sequence length, and it dominates inference memory.

The fix is sharing. Count the KV heads stored per layer:

```ascii
MHA (multi-head attention):  32 query heads -> 32 KV heads. Cache = 32 units.
GQA (grouped-query attention,
[59:24](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=3564s)):
                             32 query heads -> 8 KV heads.  Cache = 8 units (4x smaller).
MQA (multi-query attention): 32 query heads -> 1 KV head.   Cache = 1 unit (32x smaller).
```

Fewer KV heads means a smaller cache and faster decoding, at a small
quality cost. Most current models ship GQA as the compromise.

![MQA GQA MHA](assets/l02-mqa-gqa.svg "MHA: h KV heads. GQA: groups share. MQA: one shared head. The cache shrinks left to right. Stanford Frontier AI.")

> [!QA]
> Q: Why does the KV cache exist at all?
> A: Autoregressive decoding generates one token at a time. Without a
> cache, step t would recompute keys and values for all t-1 previous
> tokens. The cache stores them once and appends one row per step.
> Memory trades against quadratic recompute, and memory wins.
> Follow-up: When does GQA hurt?
> A: When heads genuinely need different keys and values. Sharing
> forces heads to read the same projections. Empirically the quality
> drop is small, which is why GQA is the default, but MQA's single
> head is a coarser cut used mainly for maximum throughput.

## The key question

The 2017 machine works on translation. What breaks when you aim it at
512-token documents and scale it to billions of parameters, and what is
the smallest fix for each break?

## The three families, then BERT end to end

Keeping the encoder, the decoder, or both gives three model families:

- **Encoder-only** (BERT). Reads the whole sequence bidirectionally.
  Produces embeddings for downstream tasks. Cannot generate text.
- **Decoder-only** (GPT). Reads left to right, predicts the next
  token. Generates text. This family becomes the LLM of Lecture 3.
- **Encoder-decoder** (T5). Reads the input bidirectionally,
  generates the output autoregressively. Text in, text out, for every
  task.

**T5** frames every task as text-to-text. Its pre-training objective
is **span corruption**. Random spans of the input are masked out and
replaced by sentinel tokens (<X>, <Y>, and so on
[66:14](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=3974s)):

```ascii
input:  "A cute teddy bear is reading <X> the park <Y> today"
target: "<X> in <Y> and"
```

The target is the sequence of sentinel tokens followed by their
missing spans. The model learns to fill blanks of arbitrary length,
which exercises the encoder (understand the corrupted input) and the
decoder (generate the fills) at once.

![T5 span corruption](assets/l02-t5.svg "Mask spans, insert sentinels, predict the fills. Stanford Frontier AI.")

**BERT** (2018) is the landmark encoder-only model. The input
pipeline: WordPiece tokenizer with about 30k tokens
([83:11](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=4991s));
special tokens [CLS] opening the sequence and [SEP] separating and
closing segments. **Segment encodings** (sentence A gets one learned
vector, sentence B another). Position encodings are added per position.
(The lecture notes uncertainty about whether BERT's position
encodings were learned or hard-coded. Implementations commonly use
learned absolute encodings [uncertain].)

Two pre-training objectives, trained jointly:

- **MLM (masked language modeling).** Mask 15% of tokens. Of those,
  80% become [MASK], 10% become a random token, 10% stay unchanged.
  Work the numbers on a 512-token sequence: 0.15 * 512 = 77 tokens
  masked. Of the 77, about 61 become [MASK], about 8 become random
  tokens, about 8 stay as-is. The 80/10/10 split keeps the model
  honest: it cannot just learn "predict something whenever you see
  [MASK]", because [MASK] never appears at fine-tuning time.
- **NSP (next sentence prediction,
  [78:25](https://www.youtube.com/watch?v=yT84Y5zCnaA&t=4705s)).** 50%
  of the time sentence B truly follows sentence A. 50% of the time it is random.
  The [CLS] embedding predicts which. This teaches sentence-level
  relationships.

Notation in BERT papers: L layers, H hidden size, A attention heads.
BERT-base is L=12, H=768, A=12. Fine-tuning adds a small head: the
[CLS] embedding feeds a classifier for sentence tasks, or per-token
heads for tagging tasks. The whole stack trains end to end on the
labeled data.

![BERT](assets/l02-bert.svg "WordPiece, [CLS]/[SEP], segment encodings, MLM 80/10/10, NSP 50/50. Stanford Frontier AI.")

Two descendants close the lecture:

- **DistilBERT.** Knowledge distillation: a small student trains to
  match a large BERT teacher's output distribution (KL divergence),
  not just hard labels. The teacher's soft probabilities carry extra
  signal about class similarities. Result: most of the accuracy at a
  fraction of the size and speed.
- **RoBERTa.** A replication study that found BERT was undertrained.
  It drops NSP, masks dynamically (a fresh mask pattern each epoch
  instead of one fixed mask), trains longer on much more data, with
  bigger batches. Same architecture, better training, better results.

> [!QA]
> Q: Why mask only 15% of tokens, and why the 80/10/10 split?
> A: 15% keeps most of the context intact so prediction stays
> feasible. The 80/10/10 split fixes a train-test mismatch: [MASK]
> appears in pre-training but never in fine-tuning. By replacing 10%
> with random tokens and leaving 10% unchanged, the model must produce
> good representations for every token, not just react to the [MASK]
> symbol.
> Follow-up: Why did RoBERTa drop NSP?
> A: Follow-up studies found NSP too easy and not helpful: the model
> could solve it from topic overlap alone, and removing it while
> training on longer contiguous sequences worked as well or better.
> An objective that does not teach anything is dead weight.

## Mapping back: each refinement answers a 2017 limitation

| 2017 limitation | Refinement | How |
|---|---|---|
| Attention is opaque | Attention maps | Read the N x N weights: "it" puts 0.72 on "animal" |
| Absolute positions answer the wrong question | Relative schemes | T5 bias per distance bucket, ALiBi penalty m * distance, RoPE rotation |
| Post-norm destabilizes deep stacks | Pre-norm + RMSNorm | Clean residual highway. RMSNorm drops the mean |
| Full attention is O(n^2) | Sliding windows | n*w scores: 8x fewer at n = 4,096, w = 512 |
| KV cache dominates inference memory | GQA | 32 KV heads to 8: 4x smaller cache |
| Translation-only, hard to inspect | BERT and T5 | Bidirectional understanding (MLM 80/10/10, NSP 50/50) and text-to-text span corruption |

## The honest price

Every refinement trades something. ALiBi's penalty is fixed, so the
model cannot learn when far tokens matter. Sliding windows lose
direct long-range links and recover them only through depth. GQA and
MQA force heads to share keys and values, which costs a little
quality. Pre-norm trains stably but can underperform post-norm on
shallower stacks. MLM trains a model that never generates text. The
2017 design was coherent. The refinements are compromises with the
real world, and each one names what it gave up.

## Recap: the whole lesson on one screen

The story in eight steps. Each step answers the one before it.

1. **Read the attention.** An N x N map per head: rows are queries,
   columns are keys. "it" attends 0.72 to "animal": anaphora made
   visible. A debugging lens, not a proof of reasoning.
2. **Positions should encode distance.** Sinusoidal dot products
   depend only on p - q. Absolute answers "where am I". Attention
   needs "how far apart are we".
3. **Three relative schemes.** T5 adds a learned bias per distance
   bucket. ALiBi subtracts m * distance, no parameters,
   extrapolates. RoPE rotates Q and K. The dot product keeps only
   relative distance. RoPE is the default.
4. **Normalize before the sublayer.** Post-norm chokes the residual
   highway in deep stacks. Pre-norm leaves it clean. RMSNorm drops
   the mean-centering: fewer parameters, same stability.
5. **Attention gets cheaper.** O(n^2) hurts long documents. Sliding
   windows: n*w, 8x fewer scores at n = 4,096, w = 512. Mistral 7B
   stacks windows so the receptive field grows with depth.
6. **Share keys and values.** The KV cache dominates inference
   memory. GQA shares 32 query heads over 8 KV heads: 4x smaller.
   MQA goes to one. GQA is the standard compromise.
7. **T5 corrupts spans.** Mask spans, insert <X>/<Y> sentinels, train
   the decoder to output each sentinel plus its missing text. Every
   task becomes text-to-text.
8. **BERT masks and predicts.** WordPiece ~30k, [CLS]/[SEP], segment
   encodings. MLM: 15% masked with 80/10/10. NSP: 50/50. Fine-tune
   with a small head. DistilBERT distills. RoBERTa trains harder and
   drops NSP.

## Official sources and further reading

**Official:**
- Lecture 2 recording (YouTube): timestamped above.
- Lecture 2 slides (PDF), CME295 Autumn 2025.
- Devlin et al., "BERT" (2018):
  https://arxiv.org/abs/1810.04805 — the encoder-only landmark.
- Su et al., "RoFormer: Rotary Position Embedding" (2021):
  https://arxiv.org/abs/2104.09864 — RoPE.

**Further reading:**
- Raffel et al., "Exploring the Limits of Transfer Learning with T5"
  (2019): span corruption and the text-to-text frame.
- Press et al., "Train Short, Test Long: ALiBi" (2021): length
  extrapolation.
- Beltagy et al., "Longformer" (2020): sliding windows plus global
  tokens.
- Sanh et al., "DistilBERT" (2019).
- Liu et al., "RoBERTa" (2019).

**Caveats from these sources.** The lecture flags uncertainty about
whether BERT's position encodings were learned or hard-coded. Treat
that detail as [uncertain]. Attention maps show weight, not causation.
ALiBi and RoPE length extrapolation is empirical, not guaranteed.

## Connections to the other courses

- **CS336 L04:** linear attention, more efficient attention variants,
  and the MoE continuation of the sharing story.
- **CS336 L03:** the transformer architecture derivation this lecture
  modifies.
- **CS224N L08-L09:** transformers and pre-training from the NLP
  sequence-modeling perspective.
- **CME295 L01:** the attention arrow this lecture reads and refines.

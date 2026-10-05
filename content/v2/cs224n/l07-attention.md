---
page_id: cs224n-l07
course_slug: cs224n
course_name: "CS224N: NLP with Deep Learning"
course_order: 4
order: 7
nav: "L07 · Attention"
title: "Lecture 7: Attention"
summary: "MT evaluation with BLEU, the seq2seq bottleneck, and Bahdanau attention: the decoder queries the encoder states, scores, weights, and averages."
instructor: "Christopher Manning"
offering: "Spring 2024"
duration: "1:17:30"
video_id: J7ruSOIzhrE
video_title: "Lecture 7: Attention and Final Projects; Practical Tips"
video_caption: "Original lecture. Christopher Manning introduces BLEU evaluation and Bahdanau attention for neural machine translation."
concepts: [bleu, evaluation, attention, bahdanau, seq2seq, query, attention-scores, context-vector]
sources:
  - tag: video
    label: "Lecture 7 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=J7ruSOIzhrE
  - tag: notes
    label: "Official subtitle transcript"
  - tag: paper
    label: "Bahdanau, Cho, Bengio, Neural Machine Translation by Jointly Learning to Align and Translate (2015)"
    url: https://arxiv.org/abs/1409.0473
  - tag: paper
    label: "Papineni et al., BLEU: a Method for Automatic Evaluation of Machine Translation (2002)"
    url: https://aclanthology.org/P02-1040/
---

## How to read this lesson

This lesson has two levels. **Level 1 (Core)** covers BLEU and the attention
mechanism. **Level 2 (Deep)** generalizes attention and explains why it helps
gradients.

## Level 1: BLEU evaluates translations

Before attention, the lecture covers **BLEU** ([00:26](ts:00:26)): the
standard automatic metric for machine translation. BLEU compares n-gram
overlap between the model's output and reference translations.

![BLEU](assets/l07-bleu.svg "BLEU counts matching n-grams against references and penalizes brevity.")

Automatic metrics are cheap and reproducible. They are also crude: they count

Automatic metrics are cheap and reproducible. They are also crude: they count
word overlap, not meaning.

## Level 1: The bottleneck

Seq2seq stuffs the whole source sentence into **one fixed vector**. "It
still seems a very questionable thing to do," says the lecture
([14:09](ts:14:09)). "It is certainly not like what a human being does."

![Bottleneck](assets/l07-bottleneck.svg "Many source words compress into one fixed vector; detail is lost.")

A human translator reads the sentence, holds its meaning, then **looks back**
at earlier parts while translating ([14:14](ts:14:14)). The network should do
the same: attend to different source words as needed.

## Level 1: Attention, step by step

On each decoder step, insert direct connections to the encoder
([14:51](ts:14:51)):

![Attention steps](assets/l07-attention-steps.svg "Query, compare, softmax scores, weighted average, concatenate and predict; Bahdanau et al., 2015.")

1. Use the decoder's hidden state as a **query** to look back
([15:20](ts:15:20)).
2. Compare it with every encoder hidden state. Compute **attention scores**:
where to look ([15:50](ts:15:50)).
3. Softmax the scores into an **attention distribution**.
4. Take the **weighted average** of the encoder states ([16:25](ts:16:25)).
5. Concatenate with the decoder state and predict.

Each step gets its own context vector. The model reads what it needs, when
it needs it. "More human-like."

![Attention scores](assets/l07-scores.svg "Attention weights over 'he hit me with a pie': the context vector is the weighted average of encoder states.")

## Level 2: The general form

Attention generalizes beyond translation. The general form ([25:20](ts:25:20)):

- A set of **values** H(1)..H(n).
- A **query** vector.
- Scores between query and values, softmaxed into weights.
- Output = weighted average of values.

Any problem with "look at the relevant parts" fits this shape.

```ascii
values:   H_1   H_2   H_3  ...  H_n
query:    q
scores:   s_i = score(q, H_i)
weights:  a = softmax(s)
output:   sum over i of a_i * H_i
```

## Level 2: Why attention helps gradients

![Shorter paths](assets/l07-paths.svg "Direct decoder-to-encoder connections shorten the paths gradients travel.")

Attention solves the bottleneck: "you now no longer" compress everything
into one vector ([23:06](ts:23:06)). It also helps the **vanishing gradient**
problem ([23:22](ts:23:22)): direct connections from decoder to each encoder
state shorten the paths gradients travel. Shorter paths, healthier gradients.

> [!QA]
> Q: Why is attention "more human-like"?
> A: Humans translating look back at the source sentence as they write. Seq2seq forces the model to memorize the sentence in one vector and never look again. Attention restores the looking-back behavior.
> Follow-up: What did attention cost?
> A: Computation per step. Each decoder step now compares against every encoder state: quadratic in sequence length for self-attention. The transformer (L08) pays this cost everywhere and wins anyway.

> [!QA]
> Q: Is attention the same as alignment?
> A: Related, not identical. Classical alignment maps source words to target words as a hard preprocessing step. Attention learns soft weights jointly with translation. The weights often look like alignments, but they are learned for the task, not imposed.
> Follow-up: Can attention weights be read as explanations?
> A: Cautiously. High weight means the model used that state, not that the word caused the output. The weights are one factor in a large computation. Treat them as hints, not proofs.

## Recap: the whole lesson on one screen

Eight ideas carry this lecture. Read each card. Say the core sentence out
loud. If you can, you own the lesson.

<div class="recap-grid">
<div class="recap-card">
<img src="assets/l07-bottleneck.svg" alt="Bottleneck">
<div class="rc-body">
<strong>1. One fixed vector cannot hold a sentence</strong>
<p>Seq2seq compresses everything into one vector. Detail is lost. Humans
look back. Models should too.</p>
<p class="rc-num">Key: the bottleneck is implausible</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l07-attention-steps.svg" alt="Attention steps">
<div class="rc-body">
<strong>2. Query, score, weight, average</strong>
<p>Decoder state as query. Compare with encoder states. Softmax scores.
Weighted average. Concatenate and predict.</p>
<p class="rc-num">Key: Bahdanau et al., 2015</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l07-scores.svg" alt="Attention scores">
<div class="rc-body">
<strong>3. Scores become a distribution</strong>
<p>Softmax turns raw scores into weights. The context vector is the weighted
average of encoder states.</p>
<p class="rc-num">Key: one context per step</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l07-attention-steps.svg" alt="Human-like">
<div class="rc-body">
<strong>4. Attention is more human-like</strong>
<p>A human translator looks back at the source. Attention gives the network
the same ability: read what it needs, when it needs it.</p>
<p class="rc-num">Key: look back while translating</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l07-attention-steps.svg" alt="General form">
<div class="rc-body">
<strong>5. The general form: values plus query</strong>
<p>Values H(1)..H(n), a query, scores, softmax, weighted average. Any
"look at the relevant parts" problem fits.</p>
<p class="rc-num">Key: [25:20](ts:25:20)</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l07-bottleneck.svg" alt="Gradients">
<div class="rc-body">
<strong>6. Shorter paths help gradients</strong>
<p>Direct decoder-to-encoder connections shorten gradient paths. Healthier
gradients, less vanishing.</p>
<p class="rc-num">Key: [23:22](ts:23:22)</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l07-scores.svg" alt="BLEU">
<div class="rc-body">
<strong>7. BLEU measures n-gram overlap</strong>
<p>Automatic, cheap, reproducible. Counts word overlap against references,
not meaning. Standard for MT of this era.</p>
<p class="rc-num">Key: cheap and crude</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l07-attention-steps.svg" alt="What is next">
<div class="rc-body">
<strong>8. Attention is the bridge to transformers</strong>
<p>L08 removes recurrence entirely: attention everywhere, all at once.
This lecture's mechanism becomes the whole architecture.</p>
<p class="rc-num">Key: attention is all you need, next</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- Lecture 7 video and transcript.
- Bahdanau, Cho, Bengio (2015): attention for NMT.
- Papineni et al. (2002): BLEU.

**Further reading:**
- Luong, Pham, Manning (2015), "Effective Approaches to Attention-based Neural Machine Translation": the local/global attention variants.
- Vaswani et al. (2017): the transformer. Covered in L08.

**Caveats from these sources.** BLEU correlates imperfectly with human judgment. The lecture's evaluation lecture (L11) revisits this. Attention weights are hints, not explanations.

## Connections to the other courses

- **This course:** L06's bottleneck is this lecture's motivation. L08 turns attention into the transformer.
- **CS336:** L03 builds the transformer architecture. L12 evaluation: BLEU's limits generalize to all automatic metrics.
- **CS229S:** alignment algorithms are the classical predecessor of attention.

> [!CHEAT]
> **Attention cheatsheet.** BLEU: n-gram overlap vs references. Bottleneck: one fixed vector, implausible. Fix: decoder queries encoder states. Steps: query -> scores -> softmax -> weighted average -> concat -> predict. General: values + query -> weights -> output. Bahdanau 2015. Bonus: shorter gradient paths. Cost: per-step comparison.

> [!MEMORY]
> **Look back.** Humans look back at the source. Attention lets networks look back too. One mechanism, two wins: better context, healthier gradients.

---
page_id: cs224n-l05
course_slug: cs224n
course_name: "CS224N: NLP with Deep Learning"
course_order: 4
order: 5
nav: "L05 · Language Modeling and RNNs"
title: "Lecture 5: Language Modeling and RNNs"
summary: "Language models as the course's central concept, n-gram counting with its sparsity and storage problems, and recurrent neural networks: one weight set, hidden memory, and their two fatal problems."
instructor: "Christopher Manning"
offering: "Spring 2024"
duration: "1:19:00"
video_id: fyc0Jzr74y4
video_title: "Lecture 5: Language Modeling and RNNs"
video_caption: "Original lecture. Christopher Manning introduces language modeling, n-grams, and recurrent neural networks."
concepts: [language-modeling, n-gram, sparsity, rnn, hidden-state, sampling, perplexity]
sources:
  - tag: video
    label: "Lecture 5 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=fyc0Jzr74y4
  - tag: notes
    label: "Official subtitle transcript"
  - tag: supplement
    label: "Stanford CS224N course site"
    url: https://web.stanford.edu/class/cs224n/
---

## How to read this lesson

This lesson has two levels. **Level 1 (Core)** defines language modeling,
builds the n-gram model, and derives the RNN. **Level 2 (Deep)** trains the
RNN as a language model and explains its two fatal problems.

## Level 1: Language modeling is the central concept

A **language model** assigns a probability to every possible next word. The
lecture calls it "the most important concept in the class". It leads to BERT,
GPT-3, and ChatGPT.

![Language model](assets/l05-lm-def.svg "A language model assigns a probability to every possible next word.")

Everything downstream is language modeling: translation, question answering,

Everything downstream is language modeling: translation, question answering,
summarization. Learn this well.

## Level 1: N-gram language models

The classical approach: **count**. A trigram model over 1.7 million Reuters
words ([29:06](ts:29:06)) estimates P(word | previous two words) from
frequencies.

![N-gram probabilities](assets/l05-ngram.svg "Trigram probabilities for 'today the ____': company 0.153, bank 0.153, price 0.077, italian 0.039, emirate 0.039.")

Given "today the", the model outputs a distribution: company 0.153, bank
0.153, price 0.077, italian 0.039, emirate 0.039. Sample from it to generate
text.

Two problems kill n-grams.

![Sparsity](assets/l05-sparsity.svg "Unseen n-grams get probability 0; n-grams seen once get probability 1. Both are absurd.")

1. **Sparsity.** Unseen n-grams get probability 0. N-grams seen once get
probability 1. The distribution has no granularity.
2. **Storage.** Every n-gram needs a count. The table grows with the corpus.

> [!QA]
> Q: Why do n-grams fail on unseen word sequences?
> A: They only know what they counted. An unseen trigram gets probability zero, which kills any sentence containing it. Smoothing redistributes mass, but it is a patch on a fundamentally discrete representation.
> Follow-up: What replaced counting?
> A: Neural models. An RNN learns continuous representations, so unseen sequences still get sensible probabilities through similarity to seen ones. The representation generalizes where the table cannot.

## Level 1: Recurrent neural networks

A **recurrent neural network** applies **one set of weights** at every time
step ([51:43](ts:51:43)). The network keeps a **hidden state**: a vector that
carries memory forward.

![RNN cell](assets/l05-rnn-cell.svg "h_t = tanh(W_h h_{t-1} + W_e x_t + b): the same weights combine the old memory with the new word.")

h(t) = tanh(W(h) h(t-1) + W(e) x(t) + b)

- x(t): the current word's vector.
- h(t-1): the memory from the previous step.
- The same matrices W(h), W(e) apply at every step.

h(0) starts as zeros. Each step updates the memory. The hidden state is the
model's memory of everything it has read.

## Level 2: The RNN as a language model

Unroll the RNN across the sequence. At each position, the hidden state feeds
a **softmax over the vocabulary**: one probability distribution per step.
Sample a word from it to generate text.

![Unrolled RNN](assets/l05-rnn-lm.svg "Memory flows left to right through shared cells; each step outputs a vocabulary distribution.")

The **loss** is the average negative log likelihood per position: at each
step, penalize the model for the probability it assigned to the actual next
word.

## Level 2: Two problems killed RNNs

![RNN problems](assets/l05-problems.svg "Sequential: a for loop from 1 to T that cannot parallelize. Forgetting: distant context fades.")

1. **Sequential.** Step t needs step t-1. This is a for loop from t=1 to T.
It cannot be parallelized ([56:56](ts:56:56)). GPUs sit idle. This problem
"led them to fall out of favor."
2. **Forgetting.** "They tend to forget stuff from further back as well"
([58:08](ts:58:08)). Distant context fades too quickly. The memory is a
leaky summary.

> [!QA]
> Q: Why did sequentiality kill RNNs at scale?
> A: Training speed. A GPU computes thousands of operations in parallel, but the RNN forces a strict order: step 1000 waits for step 999. Long sequences train slowly no matter how much hardware you add. The transformer (L07-L08) fixed this by removing the order entirely.
> Follow-up: Is forgetting the same as the vanishing gradient?
> A: Related but not identical. Forgetting is about the forward pass: the hidden state is a fixed-size summary, so old information gets overwritten. Vanishing gradients (L06) are about the backward pass: gradients shrink through long chains. The LSTM fixes both with the same mechanism.

## Recap: the whole lesson on one screen

Eight ideas carry this lecture. Read each card. Say the core sentence out
loud. If you can, you own the lesson.

<div class="recap-grid">
<div class="recap-card">
<img src="assets/l05-ngram.svg" alt="N-gram probabilities">
<div class="rc-body">
<strong>1. Language modeling is the central concept</strong>
<p>Assign a probability to every next word. It leads to BERT, GPT-3,
ChatGPT. Everything downstream is language modeling.</p>
<p class="rc-num">Key: the most important concept in the class</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l05-ngram.svg" alt="N-gram">
<div class="rc-body">
<strong>2. N-grams count</strong>
<p>Trigram on 1.7M Reuters words: "today the" gives company 0.153, bank
0.153, price 0.077. Sample from the distribution to generate.</p>
<p class="rc-num">Key: count, normalize, sample</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l05-sparsity.svg" alt="Sparsity">
<div class="rc-body">
<strong>3. Sparsity and storage kill n-grams</strong>
<p>Unseen gets 0, seen-once gets 1. No granularity. The count table grows
with the corpus. Smoothing is a patch.</p>
<p class="rc-num">Key: 0 and 1 are both absurd</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l05-rnn-cell.svg" alt="RNN cell">
<div class="rc-body">
<strong>4. One weight set, reused every step</strong>
<p>h(t) = tanh(W(h) h(t-1) + W(e) x(t) + b). Same matrices at every
position. h(0) is zeros. The hidden state is memory.</p>
<p class="rc-num">Key: recurrence shares parameters</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l05-rnn-lm.svg" alt="Unrolled RNN">
<div class="rc-body">
<strong>5. Each step outputs a distribution</strong>
<p>Unroll across the sequence. Softmax per step. Sample to generate. Loss:
average negative log likelihood per position.</p>
<p class="rc-num">Key: one softmax per step</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l05-problems.svg" alt="RNN problems">
<div class="rc-body">
<strong>6. Sequential: the for loop kills scale</strong>
<p>Step t needs step t-1. No parallelization. GPUs idle. This problem ended
the RNN era at scale.</p>
<p class="rc-num">Key: t=1..T, strictly ordered</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l05-problems.svg" alt="Forgetting">
<div class="rc-body">
<strong>7. Forgetting: memory is leaky</strong>
<p>Distant context fades too quickly. The fixed-size hidden state
overwrites old information. LSTMs protect it.</p>
<p class="rc-num">Key: leaky summary</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l05-rnn-cell.svg" alt="Memory">
<div class="rc-body">
<strong>8. Architecture families so far</strong>
<p>Word2vec (simple encoder), feedforward nets (L03), RNNs (this lecture),
transformers (L08). Each family fixes the last one's flaw.</p>
<p class="rc-num">Key: [51:09](ts:51:09)</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- Lecture 5 video and transcript.

**Further reading:**
- Mikolov et al. (2010), "Recurrent neural network based language model": the RNNLM that scaled n-grams to continuous representations.
- Jozefowicz et al. (2016), "Exploring the Limits of Language Modeling": large-scale RNN LMs and their limits.

**Caveats from these sources.** The sparsity numbers (0, 1) describe raw unsmoothed counts. Real n-gram systems always smooth. RNNs remain useful for short sequences and streaming. The lecture's "fell out of favor" refers to large-scale training.

## Connections to the other courses

- **This course:** L06 fixes the two problems (LSTM for forgetting, and the gradient story). L07-L08 remove sequentiality with attention.
- **CS336:** L03 builds the transformer that replaced the RNN. L10 inference: the KV cache is the transformer's answer to the hidden state.
- **CS229:** maximum likelihood estimation is the same objective in both courses.

> [!CHEAT]
> **Language modeling cheatsheet.** LM: P(next word | history). N-gram: count trigrams, P = count/total. "today the": company/bank 0.153, price 0.077. Sparsity: unseen 0, seen-once 1. Storage: table grows with corpus. RNN: one weight set, h(t)=tanh(W(h)h(t-1)+W(e)x(t)+b), h(0)=zeros. Output: softmax per step. Loss: avg NLL. Problems: sequential for-loop (unparallelizable), forgetting distant context.

> [!MEMORY]
> **Count, then recur, then attend.** N-grams count. RNNs share weights across time. Both hit walls: sparsity for n-grams, order for RNNs. Attention removes the order.

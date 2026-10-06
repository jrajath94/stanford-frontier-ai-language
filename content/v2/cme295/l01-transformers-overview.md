---
page_id: cme295-l01
course_slug: cme295
course_name: "CME295: Transformers and Large Language Models"
course_order: 5
order: 1
nav: "L01 · Transformers overview"
title: "Lecture 1: From Words to Transformers"
summary: "The transformer rebuilt from zero: one-hot vectors, word2vec, RNNs, and self-attention with a hand-worked toy, then the full 2017 architecture walked end to end on one translation. Built from the Lecture 1 slides: no transcript exists."
date: "2025-09-26"
instructor: "Afshine Amidi, Shervine Amidi"
offering: "Autumn 2025"
sources:
  - tag: slides
    label: "Lecture 1 slides (PDF), CME295 Autumn 2025"
  - tag: site
    label: "CME295 course site: cme295.stanford.edu"
    url: https://cme295.stanford.edu
  - tag: supplement
    label: "Super Study Guide: Transformers and Large Language Models (Amidi)"
    url: https://superstudy.guide
  - tag: paper
    label: "Vaswani et al., Attention Is All You Need (2017)"
    url: https://arxiv.org/abs/1706.03762
concepts: [nlp-tasks, tokenization, one-hot, word2vec, cbow, skip-gram, rnn, vanishing-gradient, attention, self-attention, query-key-value, transformer, positional-encoding, multi-head-attention, label-smoothing, autoregressive]
---

> [!QA]
> Q: Why does this lesson carry a source warning?
> A: No transcript or recording reference survives for Lecture 1. Everything
> below is rebuilt from the Lecture 1 slide deck alone. Slide decks show
> conclusions, not the spoken reasoning. Treat explanations here as faithful
> to the slides, and treat anything beyond the slides as absent, not as
> implied. Later lectures in this course have full transcripts and are
> sourced to the spoken word.

## The problem: translate a sentence

Take the sentence "A cute teddy bear is reading." and turn it into
French: "Un ours en peluche mignon lit." A human does this easily. A
computer sees neither letters nor meaning. It sees whatever numbers we
give it. The entire history of this course is the search for the right
numbers.

Before any machinery, name the job. Language tasks come in three
families, and the family decides what the machine outputs:

- **Classification.** Text in, one label out. "This teddy bear is SO
  CUTE!" becomes *positive*. Intent detection, language detection, and
  topic modeling live here.
- **Multi-classification.** Text in, one label per token out.
  Part-of-speech tagging and named entity recognition live here: in "A
  cute teddy bear is reading", the model tags "teddy bear" as an entity.
- **Generation.** Text in, text out. Translation, question answering,
  and summarization live here. The output is open-ended, not a fixed
  label.

![NLP task families](assets/l01-nlp-tasks.svg "Classification maps text to a label. Multi-classification maps text to per-token tags. Generation maps text to text. Stanford Frontier AI.")

Machine translation is generation: at each step the model picks one
word from the vocabulary, so generation is repeated classification over
the vocabulary, one token at a time. Everything in this chapter is the
machinery that makes that picking work.

## First attempt: one-hot vectors

The computer needs numbers for words. The simplest scheme is the
**one-hot vector**: a vector with a 1 in the position of the word and
0 everywhere else. With a toy vocabulary of five words:

```ascii
"bear"   = [1, 0, 0, 0, 0]
"teddy"  = [0, 1, 0, 0, 0]
"quantum"= [0, 0, 0, 0, 1]
```

One-hot works as an encoding. Nothing is ambiguous. But it says
nothing about meaning. Compute the dot product, the simplest measure
of similarity: "bear" dot "teddy" = 0. "bear" dot "quantum" = 0.
"Bear" and "teddy" are as far apart as "bear" and "quantum". Every
word is equally distant from every other word. A model fed one-hot
vectors has no reason to think "teddy bear" behaves differently from
"quantum bear".

Worse, the vectors grow with the vocabulary. A 50,000-word vocabulary
gives 50,000-dimensional vectors, mostly zeros. Storing them is
wasteful. Computing with them is slow.

## The key question

What if a word's vector encoded the company it keeps? "Teddy" appears
near "bear", "soft", "cute". "Quantum" appears near "physics",
"particle". If two words share neighbors, their vectors should sit
close together. Then the model would *see* that "teddy" relates to
"bear" before it ever answers a question.

## Word2vec: learning vectors from a proxy task

**Word2vec** (Mikolov et al., 2013) turns that idea into training. The
trick is a **proxy task**: train a network to do something easy, then
steal its hidden layer. The proxy task is prediction of neighbors.

Two versions exist:

- **CBOW** (continuous bag of words): given the surrounding words,
  predict the center word. Given "a cute ___ bear is reading", predict
  "teddy".
- **Skip-gram**: given the center word, predict the surrounding
  words. Given "teddy", predict "cute", "bear".

Watch a CBOW training step on the toy sentence "the teddy bear is
cute", vocabulary of 5, embedding dimension d = 2:

```ascii
input:  context words "teddy" and "is"  (average their one-hots)
hidden: 2 numbers, e.g. [0.3, -0.5]     (this is the embedding!)
output: 5 scores, softmax -> probabilities over the vocabulary
target: "bear" gets 1.0
```

The network adjusts its weights so "bear" gets high probability when
surrounded by "teddy" and "is". After training on billions of words,
words with similar neighbors have been pushed through similar weight
updates, and their hidden-layer rows land near each other. "teddy"
dot "bear" is now large. "teddy" dot "quantum" is small. The vector
space encodes meaning as geometry.

The network itself is disposable. The prize is the hidden layer: each
row is a **word embedding**, a dense vector that captures
distributional meaning. The original setup: input size V (the
vocabulary), hidden size d (a few hundred), output size V.

![Word2vec](assets/l01-word2vec.svg "A shallow network with a proxy task. The hidden layer becomes the embedding. Stanford Frontier AI.")

## Where word2vec breaks

Two cracks, each easy to demonstrate.

**No context.** Every word gets exactly one vector. "Bear" in "a
teddy bear" and "bear" in "the bear market crashed" get the same
embedding, the average of both worlds. The representation cannot tell
which sentence it sits in.

**No order.** CBOW averages the context, so "dog bites man" and "man
bites dog" produce the same context vector. Word order carries the
meaning of the sentence and the embedding throws it away.

So the job is now sharper. Build a representation that knows the
surrounding words *and* their order. The representation of "bear" must
change with the sentence around it. Call this a **contextual
representation**. Everything from here serves it.

## Tokenization: cutting text into units

Before any vector exists, the text must be cut into units. That cut is
called **tokenization**, and the choice matters. Take "reading":

```ascii
word-level:      "reading"
subword-level:   "read" + "##ing"
character-level: "r" "e" "a" "d" "i" "n" "g"
```

Three levels, each with a price:

- **Word-level** is simple and readable, but any unseen word breaks it
  (out-of-vocabulary), and it cannot reuse knowledge of roots.
- **Subword-level** (WordPiece, BPE) reuses common prefixes and
  suffixes. New words decompose into known parts. Small vocabulary,
  small out-of-vocabulary risk, learned from the data.
- **Character-level** survives misspellings and casing, but a sentence
  becomes many times longer, so computation slows.

Modern models use subword tokenizers with vocabularies around 30,000
to 50,000 tokens.

![Tokenization levels](assets/l01-token-levels.svg "Word, subword, and character splits of one sentence. Stanford Frontier AI.")

## First attempt at context: the RNN

The **recurrent neural network** (RNN) reads a sentence one token at
a time and keeps notes in a **hidden state**, a vector updated at
every step. The update rule:

```ascii
h_t = tanh(W_h * h_{t-1}  +  W_x * x_t)
       ^^^^^^^^^^^^^^^^^^^    ^^^^^^^^^
       faded old notes        new token
```

W_h and W_x are learned matrices. tanh squashes the result into a calm
range. The formula says: fade the old notes, add the new token, squash.
The hidden state is the model's memory of everything read so far.

Watch it on a toy. Two-dimensional vectors. W_h halves the old notes.
W_x passes the token through.

```ascii
tokens:   x1 = "the" = [1, 0]      x2 = "cat" = [0, 1]
start:    h_0 = [0, 0]

step 1:   h_1 = tanh(0.5 * [0,0] + [1,0]) = tanh([1, 0]) = [0.76, 0]
step 2:   h_2 = tanh(0.5 * [0.76,0] + [0,1]) = tanh([0.38, 1]) = [0.36, 0.76]
```

Read h_2. The second number (0.76) is "cat", fresh and strong. The
first number (0.36) is "the", faded but present. Recent tokens are
louder than old ones. The same body serves all three task families
with different heads: a sentiment head for classification, per-token
tags for multi-classification, and step-by-step prediction for
translation. **LSTMs** (1997) keep the same idea with a more
structured hidden state, so memory survives longer.

![RNN unrolled](assets/l01-rnn.svg "Tokens enter left to right. The hidden state carries the past forward. Stanford Frontier AI.")

## Where the RNN breaks

Three cracks, each demonstrated with numbers.

**Long distances fade.** In "The counselor helped frame the
situation", the model needs "The" to represent "situation", six steps
later. Each update halves its trace. After six steps, "The" survives
at (0.5)^6 = 0.016 of its original strength. Under 2 percent remains.
The model is trying to remember the start of the sentence through six
rounds of dilution.

**Gradients break too.** Training nudges weights by the **gradient**,
the size of the correction each weight deserves. In an RNN the gradient
for an early token travels backward through every timestep, multiplying
by roughly the same number at each step:

```ascii
fading multiplier 0.9, 10 steps:  0.9^10  = 0.35  (vanishes)
growing multiplier 1.1, 10 steps: 1.1^10  = 2.59  (explodes)
```

Below 1, the learning signal decays to zero: early tokens stop
learning. Above 1, it grows exponentially and training destabilizes.
Long sequences are exactly the ones that break.

**The chain is serial.** Step t waits for step t-1. A 1,000-token
sequence means 1,000 sequential steps. A GPU with thousands of cores
gets one step at a time and watches the rest idle. Training on
billions of words becomes an exercise in patience. This bottleneck is
the crack that mattered most in practice.

## The key question

What if tokens could talk to each other *directly*, skipping the chain
entirely? What if "situation" could look straight back at "The"
without passing through six rounds of dilution? That question is the
transformer.

## Attention: a lookup, not a chain

The idea: every token puts two things on the table, a **key** (a
label saying what it contains) and a **value** (the content it
offers). Every token also forms a **query** (a description of what it
needs). To build the new representation of "frame", compare its query
against every key, and mix the values in proportion to the match.

Now watch it by hand. Three tokens, two-dimensional vectors. The
projections are identity here, so the vectors are their own queries,
keys, and values. (The real model learns the projections. The
mechanism is the same.)

```ascii
tokens:   counselor = [1, 0]    helped = [0, 1]    frame = [1, 1]
query for "frame": q = [1, 1]

step 1, scores (dot products of q with each key):
  q . k_counselor = [1,1] . [1,0] = 1
  q . k_helped    = [1,1] . [0,1] = 1
  q . k_frame     = [1,1] . [1,1] = 2

step 2, softmax (turn scores into weights that sum to 1):
  e^1 = 2.72,  e^1 = 2.72,  e^2 = 7.39,  total = 12.83
  weights = [0.21, 0.21, 0.58]

step 3, mix the values:
  new "frame" = 0.21 * [1,0] + 0.21 * [0,1] + 0.58 * [1,1]
              = [0.79, 0.79]
```

Read the result. "frame" pulled 21% from "counselor", 21% from
"helped", and 58% from itself. The query-key match decided the mix.
That is **attention**, and because it compares tokens inside one
sequence, this is **self-attention**.

In matrix form, with N tokens of dimension d stacked into X:

**Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V**

QK^T scores every query against every key. The division by sqrt(d_k)
keeps dot products in the range where softmax stays sensitive. As
dimensions grow, raw dot products spread wider, the softmax saturates
(one weight goes to 1, the rest to 0), and gradients die. One line of
numerical hygiene. Softmax turns scores into weights. Multiplying by V
mixes the values. Every token does this against every other token,
simultaneously.

![QKV attention](assets/l01-qkv-attention.svg "The attention arrow: softmax(QK^T / sqrt(d_k)) V. The canonical symbol for Stanford Frontier AI: query at left, keys with score bars, weights, then the weighted value mix.")

This figure defines the attention arrow for the whole learning system.
CME295 is its first home. Later courses reuse it instead of redrawing
it.

> [!QA]
> Q: Why does attention divide by sqrt(d_k)?
> A: Dot products grow with dimension: for random vectors of dimension
> d, the variance of the dot product is d, so scores spread wider as
> models get bigger. A wide-spread softmax saturates: one weight nears
> 1, the rest near 0, and the gradients through the tiny weights die.
> Dividing by sqrt(d_k) brings the variance back to 1 and keeps the
> softmax in its responsive range.
> Follow-up: What if you forget it?
> A: In small models, little. In large models (d in the thousands),
> attention collapses onto a single token per query early in training
> and never recovers. A classic "trains but learns nothing" bug.

## Attention lost the order. Put it back

Attention compares every token to every other token. Shuffle the input
and the scores shuffle identically. The operation sees a *set*, not a
sequence. Word order, the thing the RNN had for free, is gone.

The fix adds a **positional encoding** to each input embedding. Two
flavors: learned (one trainable vector per position) or hard-coded
fixed sin/cos waves. The waves have a property the slides highlight:
the dot product of two position embeddings depends only on the
relative distance between positions, not their absolute indices. High
frequency dimensions change fast and track local order. Low frequency
dimensions change slowly and carry long range information.

![Positional encoding](assets/l01-positional-encoding.svg "Token embedding plus position embedding. Learned or hard-coded. Stanford Frontier AI.")

So the input to the machine is: token embedding plus position
embedding, per position. Order restored.

## The transformer stacks it all

The 2017 paper "Attention Is All You Need" stacks self-attention into
an encoder-decoder machine for translation. Walk it on "A cute teddy
bear is reading."

**Input.** Tokenize and wrap: [BOS] A cute teddy bear is reading .
[EOS]. ([BOS] marks the start, [EOS] the end.) Embed each token, add
the position encoding.

**Encoder.** N stacked layers. Each layer: multi-head self-attention
over the input tokens, then a **feed-forward network** applied to each
position independently, each wrapped with add-and-norm (add the
sublayer's input back to its output, then normalize). Out come
context-aware encoded embeddings. Each token now knows the whole
sentence.

**Decoder input.** The French output, shifted right: training starts
the decoder with [BOS] so it learns to predict the first real token.
The target is the same sequence shifted one step left, ending with
[EOS].

**Decoder.** N stacked layers. Each layer has three sublayers: masked
multi-head self-attention (decoder tokens attend only to earlier
decoder tokens, never the future: scores for future positions are set
to negative infinity before the softmax, the **causal mask**), then
encoder-decoder attention (decoder queries attend to the encoder
outputs), then the feed-forward network. Each with add-and-norm.

**Output.** A linear projection plus softmax over the vocabulary: a
classification problem where each class is a word. A probability per
word, e.g. [0.001, 0.0003, ..., 0.4, ..., 0.002]. The top word is
"Un". Feed "Un" back in and repeat: "ours", "en", "peluche",
"mignon", "lit", then [EOS]. Final output: "Un ours en peluche mignon
lit."

That loop, predict one token, feed it back, repeat until [EOS], is
**autoregressive generation**. It returns in Lecture 3 as the defining
property of large language models.

![Transformer](assets/l01-transformer-arch.svg "Encoder and decoder stacks. Post-norm, as in the 2017 paper. Stanford Frontier AI.")
![End-to-end translation](assets/l01-end-to-end.svg "Tokenize, embed, encode, decode, softmax. One sentence through the whole machine. Stanford Frontier AI.")

## Two computational tricks

**Multi-head attention.** Run h self-attention operations in parallel,
each with its own QKV projections, then concatenate and project with a
final matrix Wo. Each head can track a different kind of relationship:
one tracks syntax, one tracks coreference, one tracks position. The
slides compare heads to the multiple filters of a convolutional layer
in vision. Same input, several views at once.

![Multi-head attention](assets/l01-multihead.svg "Four heads in parallel, concatenated, projected with Wo. Stanford Frontier AI.")

**Label smoothing.** Training against hard targets (the correct word
gets 1.0, everything else 0.0) makes the model overconfident and prone
to overfitting. The fix mixes noise into the targets: the correct word
gets 1 - eps, and the remaining eps is spread over all other words.
With eps = 0.1 and a 10,000-word vocabulary, the target becomes 0.9 for
the right word and 0.00001 for each other word. Small change, better
generalization: accuracy and BLEU both improve.

![Label smoothing](assets/l01-label-smoothing.svg "Hard targets breed overconfidence. Smoothed targets generalize. Stanford Frontier AI.")

## Mapping back: what each piece fixed

Each new idea answers one crack from the chain of first attempts, by
name:

| Crack | Answer | How |
|---|---|---|
| One-hot says nothing about meaning | Word2vec embeddings | Neighbors decide geometry: "teddy" dot "bear" grows, "teddy" dot "quantum" stays small |
| Static vectors ignore the sentence | Contextual representations (RNN, then attention) | Each token's vector is rebuilt from its surroundings every pass |
| RNN fades over distance: (0.5)^6 < 2% | Self-attention | Every pair of tokens meets in one step, no chain, no dilution |
| RNN gradients break: 0.9^10 = 0.35, 1.1^10 = 2.59 | Short paths | The learning signal crosses one attention layer, not N chained steps |
| RNN is serial: step t waits for t-1 | Parallel matrix ops | All N^2 pair scores compute at once |
| Attention sees a set, not a sequence | Positional encoding | Add position info to each embedding. Sinusoids encode relative distance |

## The honest price: quadratic

Deleting the chain has a price. The score matrix is N x N: every token
pairs with every token. Double the sequence length and the work
quadruples. For N = 4,096, that is 16.7 million scores per layer per
head, most of which must sit in memory. This single fact, attention is
quadratic in sequence length, drives Lecture 2's cheaper attention
patterns, Lecture 3's KV cache engineering, and much of the rest of
the course.

## Recap: the whole lesson on one screen

The story in eight steps. Each step answers the one before it.

1. **Translate a sentence.** Generation is repeated classification over
   the vocabulary, one token at a time. The machine needs numbers for
   words that carry meaning.
2. **One-hot says nothing.** Dot products are all 0 or 1. Every word
   is equally distant from every other.
3. **Word2vec learns geometry.** CBOW and skip-gram train on neighbor
   prediction. The hidden layer becomes the embedding. But one vector
   per word: no context, no order.
4. **Tokenization cuts the text.** Word, subword, character. Subword
   (WordPiece, BPE) is the modern compromise, ~30k-50k tokens.
5. **The RNN reads in order.** h_t = tanh(W_h h_{t-1} + W_x x_t).
   Hidden state carries the past. Three cracks: distance fades
   ((0.5)^6 < 2%), gradients break (0.9^10 = 0.35, 1.1^10 = 2.59), the
   chain is serial.
6. **Attention deletes the chain.** Queries meet keys, softmax makes
   weights, values mix. "frame" = 0.21 counselor + 0.21 helped + 0.58
   self. O(1) distance, fully parallel.
7. **Order is added back.** Positional encoding: learned or sin/cos.
   The transformer stacks it: encoder, masked decoder, encoder-decoder
   attention, softmax over words.
8. **The price is quadratic.** N x N scores, 16.7M at N = 4,096. The
   rest of the course is about paying that bill.

## Official sources and further reading

**Official:**
- Lecture 1 slides (PDF), CME295 Autumn 2025: the sole source for this
  lesson. No transcript exists.
- cme295.stanford.edu: syllabus, logistics, and posted recordings.
- Super Study Guide (superstudy.guide): the course textbook by Amidi.
- VIP cheatsheet (github.com/afshinea/stanford-cme-295-transformers-large-language-models):
  translated into 11 languages.

**Further reading:**
- Vaswani et al., "Attention Is All You Need" (2017):
  https://arxiv.org/abs/1706.03762 — the transformer paper. Read the
  architecture section against the slides.
- Bahdanau et al., "Neural Machine Translation by Jointly Learning to
  Align and Translate" (2014): the original attention paper.
- Mikolov et al., "Efficient Estimation of Word Representations in
  Vector Space" (2013): word2vec.

**Caveats from these sources.** This lesson is slide-bound: the spoken
explanations, demos, and board work from the live lecture are lost.
The slides adapt figures from the CS 230 cheatsheets and the 2017
paper. Check those for full detail. The timeline (1980s-2020s) is
pedagogical, not a complete history of NLP.

## Connections to the other courses

- **CS336 L01:** tokenization derived from first principles, including
  the token chip. CME295 uses the conclusions. CS336 owns the
  mechanics.
- **CS336 L03:** the transformer architecture with full derivations.
  The Amidi walkthrough here is the conceptual version. CS336 is the
  mathematical one.
- **CS224N L05-L08:** language modeling with RNNs, LSTMs, NMT,
  attention, and transformers from the linguistics side.
- **CS224N L01-L02:** word vectors before and after word2vec.
- **CS229S L02:** the same attention toy from the systems side, with
  the training instability worked out in numbers.

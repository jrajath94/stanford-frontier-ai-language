---
page_id: cs224n-l01
course_slug: cs224n
course_name: "CS224N: NLP with Deep Learning"
course_order: 4
order: 1
nav: "L01 · Intro and Word Vectors"
title: "Lecture 1: Introduction and Word Vectors"
summary: "Course scope, why word meaning as a symbol fails, one-hot vectors, the distributional hypothesis, word2vec skip-gram setup, softmax, and optimization basics."
instructor: "Christopher Manning"
offering: "Spring 2024"
concepts: [word-vectors, distributional-semantics, one-hot, wordnet, word2vec, skip-gram, softmax, gradient-descent, stochastic-gradient-descent]
sources:
  - tag: slides
    label: "CS224N Spring 2024 Lecture 1 slide deck, official (word vectors introduction)"
  - tag: paper
    label: "Mikolov et al., Efficient Estimation of Word Representations in Vector Space (2013)"
    url: https://arxiv.org/abs/1301.3781
  - tag: supplement
    label: "Firth, A synopsis of linguistic theory, 1930-1955 (distributional hypothesis)"
  - tag: supplement
    label: "Stanford CS224N course site"
    url: https://web.stanford.edu/class/cs224n/
---

> [!WARN]
> **This lesson was built from the slide deck only.** The lecture video was bot-blocked when the course was archived, so no transcript exists for Lecture 1. Everything below comes from the official slides. No timestamp links appear because no lecture video is available.

## How to read this lesson

This lesson has two levels. **Level 1 (Core)** contains what you need to
follow the rest of the course. **Level 2 (Deep)** derives the math and the
training mechanics.

## Level 1: The course arc

CS224N teaches deep learning for natural language processing. It starts with
the simplest idea and rebuilds the field from there.

![Course arc](assets/l01-course-map.svg "Roadmap: L01-L02 word vectors, L03-L04 neural nets and parsing, L05-L06 RNNs, L07 attention, L08 transformers, L09 pretraining, L10 post-training.")

The course builds in this order. Word vectors. Feed-forward networks.
Recurrent networks. Attention. Transformers. Pretraining. Post-training:
RLHF and SFT. Then evaluation, efficient training, and applications.

> [!QA]
> Q: What is the single key result of this lecture?
> A: Word meaning can be represented well by a high-dimensional vector of real numbers. That is the astounding result. Every neural NLP system since 2013 builds on it.
> Follow-up: Why is that result astounding?
> A: Before 2013, meaning lived in hand-built resources: synonym lists, thesaurus entries, taxonomies. Nobody proved that counting which words sit near each other would capture meaning. It worked anyway. The whole field pivoted to learning representations from raw text.

## Level 1: Meaning as a symbol is brittle

The standard linguistic view of meaning: a **signifier** (the word "tree")
denotes a **signified** (the idea of a tree). This is denotational semantics.

![Signifier and signified](assets/l01-signifier.svg "The word 'tree' denotes the idea of a tree; WordNet stores meaning as synonym lists and hypernyms, which fail on nuance and new senses.")

Computers need usable meaning. The old solution was WordNet: a thesaurus of
synonym sets and hypernym ("is a") relationships. It is useful. It has four
problems.

1. It misses nuance. "Proficient" is listed as a synonym for "good". That is
only true in some contexts.
2. It misses new meanings. Words like wicked, ninja, or badass gained senses
that no editor kept up with. Keeping the list current is impossible.
3. It is subjective. Human editors decide the senses.
4. It took human labor to build and to adapt.

Most important for this course: you cannot compute word similarity from lists.
Lists tell you words are synonyms or not. They give no graded notion of
"banking" being closer to "monetary" than to "river".

## Level 1: One-hot vectors are localist

Traditional NLP treated words as discrete symbols. A symbol can be encoded as
a **one-hot vector**: one 1, the rest 0s. Dimension equals vocabulary size,
often 500,000 or more.

![One-hot vectors](assets/l01-onehot.svg "motel and hotel as one-hot vectors: orthogonal, so a search for 'Seattle motel' misses 'Seattle hotel'.")

The problem: one-hot vectors are **orthogonal**. Every pair has dot product
zero. "Motel" and "hotel" are as different as "motel" and "catastrophe".

Consequence for search: a query for "Seattle motel" will not match a document
containing "Seattle hotel". There is no natural notion of similarity. The
encoding itself carries no meaning.

> [!QA]
> Q: Why not fix similarity with a synonym list?
> A: It fails in practice: incompleteness, stale entries, and no graded scores. The slides note this is well known to fail badly. The fix is to learn similarity into the vectors themselves.
> Follow-up: What does "learn similarity into the vectors" mean concretely?
> A: Assign each word a dense vector of real numbers. Train the vectors so that words appearing in similar contexts get high dot products. Similarity becomes geometry: a number you can compute, not a list you must maintain.

## Level 1: Distributional semantics

The distributional hypothesis: **a word's meaning is given by the words that
frequently appear close by**. "You shall know a word by the company it keeps"
(J. R. Firth, 1957). One of the most successful ideas in modern NLP.

![Distributional contexts](assets/l01-distributional.svg "Three contexts of 'banking' all concern money, not rivers; the contexts teach the meaning.")

The context of a word is the set of words appearing nearby, within a fixed
window. Collect many contexts of "banking". They concern debt, regulation,
systems. Not rivers. The contexts build the representation.

This is why word vectors are also called **embeddings** or **neural word
representations**. They are a **distributed representation**: meaning is spread
across all dimensions, not stored in one slot.

## Level 1: Word2vec skip-gram

Word2vec (Mikolov et al., 2013) is a framework for learning word vectors.
The idea:

1. Take a large corpus: a long list of words.
2. Represent every word in a fixed vocabulary by a vector.
3. Walk through each position t in the text. It has a **center word** c and
**context (outside) words** o.
4. Use the vector similarity of c and o to compute the probability of o
given c.
5. Adjust the vectors to maximize this probability.

The **skip-gram** model predicts context words from the center word.

![Skip-gram window](assets/l01-skipgram.svg "Center word 'banking' with window size 2 predicts the four context words: turning, into, crises, as.")

For a window of size m around position t, predict P(w(t+j) | w(t)) for each
offset j in the window.

## Level 2: The objective function

Data likelihood over all positions:

L(theta) = product over t=1..T, over j in window, of P(w(t+j) | w(t); theta)

The **objective** J(theta) is the average negative log likelihood:

J(theta) = -(1/T) * sum over t, sum over j in window, of log P(w(t+j) | w(t); theta)

Minimizing J(theta) is the same as maximizing predictive accuracy. It is
sometimes called the cost or loss function.

## Level 2: The softmax prediction function

How do we compute P(o | c)? Three steps.

![Softmax](assets/l01-softmax.svg "Softmax: dot product for similarity, exponentiate to make scores positive, normalize over the vocabulary to get probabilities.")

1. **Dot product** u(o) . v(c) compares similarity of o and c. Larger dot
product means larger probability.
2. **Exponentiation** makes every score positive.
3. **Normalization** over the entire vocabulary gives a probability
distribution.

P(o | c) = exp(u(o) . v(c)) / sum over all words w of exp(u(w) . v(c))

This is the **softmax** function. "Soft" because it still assigns some
probability to smaller scores, unlike a hard max. "Max" because it amplifies
the largest score. The slides note the name is a bit odd: the output is a
distribution, not a maximum.

> [!QA]
> Q: Why normalize over the whole vocabulary? That is expensive.
> A: You need a real probability distribution, so the scores must sum to one. The normalization is what makes training tractable in theory and expensive in practice. Lecture 2 introduces negative sampling to avoid the full sum.
> Follow-up: What does each part of softmax buy you?
> A: The dot product measures similarity. Exponentiation guarantees positivity. Normalization guarantees the probabilities sum to one. Remove any step and the output stops being a valid distribution.

## Level 2: Two vectors per word

The model keeps **two vectors per word**:

![Two vectors per word](assets/l01-two-vectors.svg "Each word w has a center vector v_w used as c and an outside vector u_w used as o; the final vector averages both.")

- v(w): used when w is the **center** word.
- u(w): used when w is the **context (outside)** word.

Both are subparts of the big parameter vector theta. Gradients flow into
both during training. The final word vector for w is the average of the two.

## Level 2: Optimization basics

To train, walk down the gradient of J(theta): compute the gradient, take a
small step in its negative direction, repeat. The step size alpha is the
**learning rate**.

![GD versus SGD](assets/l01-gd-sgd.svg "Full gradient descent uses the whole corpus per step and moves slowly; SGD samples windows, moves fast, and its noise helps.")

**Gradient descent** computes the gradient over the whole corpus. That is
extremely expensive: J(theta) is a function of all windows, potentially
billions of them. You would wait a very long time for one update.

**Stochastic gradient descent (SGD)** samples windows and updates after each
sample. Fast. Noisy. The noise is a feature, not a bug: it helps the optimizer
escape poor local minima. Mini-batch gradient descent sits between the two.

Note: the objective is not convex. Life turns out to be okay anyway.

> [!QA]
> Q: Why is plain gradient descent a bad idea for neural nets?
> A: Each step needs a full pass over the corpus before a single parameter update. With billions of windows, that is impossibly slow. SGD updates after every sample, so it makes progress thousands of times faster per pass.
> Follow-up: Does SGD find the same answer as gradient descent?
> A: Not exactly. SGD follows a noisy gradient, so it does not converge to the exact minimum. In practice the noise helps: it escapes sharp minima and often finds solutions that generalize better.

![Vector space](assets/l01-vectorspace.svg "Similar words sit near each other; dot products measure the similarity.")

## Level 2: Looking at word vectors

The slides close with a Jupyter notebook: inspect the trained vectors. Words
with similar contexts sit near each other. The vectors encode semantic
relationships as geometry. Lecture 2 verifies this with analogies and
evaluations.

## Recap: the whole lesson on one screen

Eight ideas carry this lecture. Read each card. Say the core sentence out
loud. If you can, you own the lesson.

<div class="recap-grid">
<div class="recap-card">
<img src="assets/l01-course-map.svg" alt="Course arc">
<div class="rc-body">
<strong>1. The course arc runs from words to agents</strong>
<p>Word vectors, neural nets, RNNs, attention, transformers, pretraining,
post-training, evaluation, training systems, speech, reasoning, alignment.</p>
<p class="rc-num">Key: L01-L02 vectors, L08 transformers, L09 pretraining, L10 RLHF</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l01-signifier.svg" alt="Signifier and signified">
<div class="rc-body">
<strong>2. Symbol lists cannot capture meaning</strong>
<p>WordNet misses nuance, misses new senses, is subjective, and needs human
labor. Lists give no graded similarity between words.</p>
<p class="rc-num">Key: proficient is not always good; ninja gained senses</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l01-onehot.svg" alt="One-hot vectors">
<div class="rc-body">
<strong>3. One-hot vectors are orthogonal</strong>
<p>Motel and hotel have dot product zero. Search for "Seattle motel" misses
"Seattle hotel". No similarity lives in the encoding.</p>
<p class="rc-num">Key: dimension = vocabulary size, 500,000+</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l01-distributional.svg" alt="Distributional semantics">
<div class="rc-body">
<strong>4. Know a word by the company it keeps</strong>
<p>Firth, 1957. A word's meaning comes from the words that frequently appear
nearby. Contexts of "banking" teach money, not rivers.</p>
<p class="rc-num">Key: distributional hypothesis</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l01-skipgram.svg" alt="Skip-gram window">
<div class="rc-body">
<strong>5. Skip-gram predicts context from center</strong>
<p>Slide a window of size m over the corpus. At each position, predict the
context words from the center word.</p>
<p class="rc-num">Key: P(w(t+j) | w(t)), window size m</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l01-softmax.svg" alt="Softmax">
<div class="rc-body">
<strong>6. Softmax turns scores into probabilities</strong>
<p>Dot product for similarity. Exponentiate for positivity. Normalize over
the vocabulary so probabilities sum to one.</p>
<p class="rc-num">Key: P(o|c) = exp(u_o.v_c) / sum exp(u_w.v_c)</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l01-two-vectors.svg" alt="Two vectors per word">
<div class="rc-body">
<strong>7. Two vectors per word, one objective</strong>
<p>v(w) for center use, u(w) for context use. Objective: average negative log
likelihood over all windows. Final vector: the average of both.</p>
<p class="rc-num">Key: J(theta) = average NLL</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l01-gd-sgd.svg" alt="GD versus SGD">
<div class="rc-body">
<strong>8. SGD updates on samples, not the corpus</strong>
<p>Full gradient descent needs a whole corpus pass per step. SGD samples
windows and updates immediately. The noise helps optimization.</p>
<p class="rc-num">Key: noise is a feature, not a bug</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- CS224N Spring 2024 Lecture 1 slide deck: the sole source for this lesson.
- Stanford CS224N course site: syllabus, assignments, logistics.

**Further reading:**
- Mikolov et al. (2013), "Efficient Estimation of Word Representations in Vector Space": the original word2vec paper. The slides follow its skip-gram formulation.
- Firth (1957), "A synopsis of linguistic theory, 1930-1955": the distributional hypothesis source. Dense linguistics. Read the quoted sentence and the surrounding paragraph only.

**Caveats from these sources.** The slides are the only record of what was taught. Without the video, spoken explanations and demos are lost. Word2vec's original paper trains on the full softmax. Practical training uses negative sampling (Lecture 2).

## Connections to the other courses

- **CS336:** L01 of CS336 assumes the embedding interface that word2vec created. Tokenization replaced word tokens with subwords. The dense-vector idea survived unchanged.
- **CS229:** gradient descent, stochastic gradient descent, and cross-entropy appear here in their NLP setting. The optimization theory is shared.
- **This course:** L02 evaluates the vectors (analogies, similarity, NER) and fixes the expensive softmax.

> [!CHEAT]
> **Word vectors cheatsheet.** One-hot: one 1, rest 0s, dimension = vocab, orthogonal, no similarity. Distributional hypothesis: meaning from nearby words (Firth 1957). Skip-gram: P(context | center) over window m. Objective: average NLL J(theta). Softmax: exp(u.v)/sum exp. Two vectors per word: v center, u outside. Average at end. GD: whole corpus per step, too slow. SGD: sample windows, noise helps. Not convex. Life is okay.

> [!MEMORY]
> **The ladder.** Lists fail (WordNet). Symbols fail (one-hot). Counts work (contexts). Prediction works (skip-gram). Sampling works (SGD).

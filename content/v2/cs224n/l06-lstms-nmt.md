---
page_id: cs224n-l06
course_slug: cs224n
course_name: "CS224N: NLP with Deep Learning"
course_order: 4
order: 6
nav: "L06 · LSTMs and NMT"
title: "Lecture 6: LSTMs and Neural Machine Translation"
summary: "Vanishing and exploding gradients, gradient clipping, the LSTM's three gates, bidirectional and stacked RNNs, and seq2seq for machine translation."
instructor: "Christopher Manning"
offering: "Spring 2024"
duration: "1:17:00"
video_id: Ba6Fn1-Jsfw
video_title: "Lecture 6: LSTM RNNs and Neural Machine Translation"
video_caption: "Original lecture. Christopher Manning covers vanishing gradients, LSTMs, and neural machine translation."
concepts: [vanishing-gradient, exploding-gradient, gradient-clipping, lstm, gates, bidirectional-rnn, stacked-rnn, seq2seq, nmt, machine-translation]
sources:
  - tag: video
    label: "Lecture 6 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=Ba6Fn1-Jsfw
  - tag: notes
    label: "Official subtitle transcript"
  - tag: paper
    label: "Hochreiter and Schmidhuber, Long Short-Term Memory (1997)"
    url: https://www.bioinf.jku.at/publications/older/2604.pdf
  - tag: paper
    label: "Sutskever, Vinyals, Le, Sequence to Sequence Learning with Neural Networks (2014)"
    url: https://arxiv.org/abs/1409.3215
---

## How to read this lesson

This lesson has two levels. **Level 1 (Core)** diagnoses the gradient
disease and prescribes the LSTM. **Level 2 (Deep)** applies sequence models
to machine translation.

## Level 1: Vanishing and exploding gradients

Backprop through 30 time steps multiplies 30 copies of the local Jacobian
([09:25](ts:09:25)). With a linear activation, that is the 30th power of the
weight matrix. The eigenvalues decide everything:

![Vanishing versus exploding](assets/l06-vanish-explode.svg "Eigenvalues below 1 shrink the gradient to zero across 30 steps; above 1 it blows up.")

- Eigenvalues below 1: the gradient **vanishes**. Thirty multiplications
shrink the signal to nothing. Distant words cannot train.
- Eigenvalues above 1: the gradient **explodes**. Thirty multiplications blow
the update up. Training diverges.

**Vanishing is worse.** An exploded gradient is visible: the loss jumps. A
vanished gradient is silent: nothing learns.

> [!QA]
> Q: Why do eigenvalues control the gradient?
> A: Repeated multiplication by a matrix scales each eigendirection by its eigenvalue, raised to the number of steps. Below 1 shrinks to zero, above 1 blows up. Thirty steps make even small deviations extreme.
> Follow-up: Why is vanishing worse than exploding?
> A: Exploding is loud: NaN losses, obvious divergence, and clipping fixes it. Vanishing is quiet: the model trains, the loss falls a little, but long-range dependencies never get learned. You cannot fix what you cannot see.

## Level 1: Gradient clipping

The exploding fix is **gradient clipping** ([19:15](ts:19:15)). Before the
update, cap the gradient norm at 5, 10, or 20 ([19:37](ts:19:37)):

![Gradient clipping](assets/l06-clipping.svg "Cap the gradient norm at 5, 10, or 20; keep the direction, limit the size.")

g = g x limit / ||g||, when ||g|| exceeds the limit.

Keep the direction. Cap the size. The lecture calls it "a crude hack that
works really well." It fixes exploding gradients. It does nothing for
vanishing.

## Level 1: The LSTM

The **LSTM** (Hochreiter and Schmidhuber, 1997, [24:55](ts:24:55)) fixes
vanishing with a **cell state**: a protected memory highway. Three gates, each
outputting 0-to-1 probabilities, guard it ([30:05](ts:30:05)):

![LSTM](assets/l06-lstm.svg "Forget, input, and output gates guard the cell; the cell update is additive so gradients flow.")

- **Forget gate.** "Wrongly named," says the lecture ([30:13](ts:30:13)).
Think of it as a **remember gate**: how much of the old cell to keep.
- **Input gate.** How much of the new candidate to write.
- **Output gate.** How much of the cell to reveal as the hidden state.

cell = old x forget + candidate x input
hidden = output x tanh(cell)

The update is **additive**, not multiplicative. Gradients flow down the cell
unchanged when the forget gate is near 1. That is the whole trick.

Google shipped LSTMs in its 2014 keyboard ([28:00](ts:28:00)). They worked.

## Level 1: Bidirectional and stacked

One direction only sees the past. A **bidirectional LSTM** runs one LSTM
left-to-right and one right-to-left, then concatenates both ([51:55](ts:51:55)).
Each position gets past and future context.

**Stacked RNNs** pile cells in layers: deeper memory, more abstraction.

![Bidirectional](assets/l06-bilstm.svg "Left-to-right and right-to-left passes concatenate; stacking adds depth.")

## Level 2: Machine translation: rules to statistics to neural

"Machine translation is where NLP started" (1950s). Rule-based systems gave
way to statistical phrase-based systems, which "never really worked all that
great." The **2014 neural breakthrough** ([65:15](ts:65:15)) replaced decades
of engineering with two LSTMs.

![MT history](assets/l06-mt-history.svg "1950s rule-based, 1990s-2010s phrase-based, 2014 neural seq2seq.")

## Level 2: Seq2seq

**Seq2seq** = two LSTMs. The **encoder** reads the source sentence. Its final
hidden state seeds the **decoder**, which writes the target. A start token
begins generation. One end-to-end loss trains both.

![Seq2seq](assets/l06-seq2seq.svg "Encoder reads the source; its final state seeds the decoder; a start token begins generation.")

The lecture's example: "he hit me with a pie" ([67:52](ts:67:52)). The
encoder-decoder pattern generalizes: summarization, text-to-speech, any
sequence-to-sequence task.

> [!QA]
> Q: Why did seq2seq beat phrase-based translation?
> A: End-to-end learning. Phrase-based systems chained hand-built components: alignment, phrase tables, reordering rules. Each component's errors compounded. Seq2seq learned the whole mapping from data with one loss, so the components co-adapted.
> Follow-up: What is the bottleneck in seq2seq?
> A: The fixed-size final state. The entire source sentence must fit in one vector. Long sentences lose detail. Attention (L07) fixes exactly this bottleneck.

## Recap: the whole lesson on one screen

Eight ideas carry this lecture. Read each card. Say the core sentence out
loud. If you can, you own the lesson.

<div class="recap-grid">
<div class="recap-card">
<img src="assets/l06-vanish-explode.svg" alt="Vanishing versus exploding">
<div class="rc-body">
<strong>1. Eigenvalues decide the gradient's fate</strong>
<p>30 steps multiply the Jacobian 30 times. Below 1: vanish. Above 1:
explode. Vanishing is worse because it is silent.</p>
<p class="rc-num">Key: powers of W_h</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l06-clipping.svg" alt="Gradient clipping">
<div class="rc-body">
<strong>2. Clip the norm to stop explosions</strong>
<p>Cap ||g|| at 5, 10, or 20. Keep the direction. "A crude hack that works
really well." No help for vanishing.</p>
<p class="rc-num">Key: limit 5, 10, 20</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l06-lstm.svg" alt="LSTM">
<div class="rc-body">
<strong>3. The LSTM protects a cell</strong>
<p>Three gates as 0-to-1 probabilities. Forget is really "remember". The
cell update is additive, so gradients flow.</p>
<p class="rc-num">Key: Hochreiter and Schmidhuber, 1997</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l06-lstm.svg" alt="Cell update">
<div class="rc-body">
<strong>4. cell = old x forget + candidate x input</strong>
<p>hidden = output x tanh(cell). Additive beats multiplicative. Google
keyboard shipped it in 2014.</p>
<p class="rc-num">Key: additive update</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l06-bilstm.svg" alt="Bidirectional">
<div class="rc-body">
<strong>5. Bidirectional reads both ways</strong>
<p>Left-to-right plus right-to-left, concatenated. Each position sees past
and future. Stack for depth.</p>
<p class="rc-num">Key: concatenate both directions</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l06-mt-history.svg" alt="MT history">
<div class="rc-body">
<strong>6. MT went neural in 2014</strong>
<p>1950s rules, then phrase-based statistics ("never really worked all that
great"), then the seq2seq breakthrough.</p>
<p class="rc-num">Key: where NLP started</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l06-seq2seq.svg" alt="Seq2seq">
<div class="rc-body">
<strong>7. Seq2seq: encode, then decode</strong>
<p>Two LSTMs. The encoder's final state seeds the decoder. Start token
begins generation. One end-to-end loss.</p>
<p class="rc-num">Key: he hit me with a pie</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l06-seq2seq.svg" alt="Bottleneck">
<div class="rc-body">
<strong>8. The bottleneck is one fixed vector</strong>
<p>The whole source sentence must fit in the final state. Long sentences
lose detail. Attention (L07) is the fix.</p>
<p class="rc-num">Key: one vector is not enough</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- Lecture 6 video and transcript.
- Hochreiter and Schmidhuber (1997): the LSTM paper.
- Sutskever, Vinyals, Le (2014): seq2seq.

**Further reading:**
- Pascanu, Mikolov, Bengio (2013), "On the difficulty of training recurrent neural networks": the vanishing/exploding analysis the lecture cites.
- Bahdanau, Cho, Bengio (2015): attention, the fix for the seq2seq bottleneck. Covered in L07.

**Caveats from these sources.** Clipping thresholds (5, 10, 20) are heuristics, not derived values. "Never really worked all that great" is the lecture's verdict on phrase-based MT, not a measured claim.

## Connections to the other courses

- **This course:** L05 introduced the RNN's two problems. L07 fixes the bottleneck with attention. L08 replaces recurrence entirely.
- **CS336:** L09 scaling laws: LSTMs scaled with data and compute before transformers did.
- **CS229:** eigenvalues and matrix powers appear in the optimization theory.

> [!CHEAT]
> **LSTM cheatsheet.** Vanish: eigenvalues <1, 30 steps -> 0. Explode: >1 -> infinity. Vanishing worse (silent). Clip: norm cap 5/10/20, keep direction. LSTM 1997: cell + hidden, 3 gates (forget=remember, input, output). cell = old x forget + candidate x input. hidden = output x tanh(cell). Additive -> gradients flow. BiLSTM: concat both directions. Stacked: deeper. MT: rules -> phrase-based -> 2014 NMT. Seq2seq: encoder -> final state -> decoder + start token, one loss. Bottleneck: fixed vector.

> [!MEMORY]
> **Add, do not multiply.** The LSTM's cell adds. Addition preserves gradients. Multiplication destroys them. One design choice, one era of NLP.

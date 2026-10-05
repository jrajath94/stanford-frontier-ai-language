---
page_id: cs224n-l13
course_slug: cs224n
course_name: "CS224N: NLP with Deep Learning"
course_order: 4
order: 13
nav: "L13 · Speech BCI"
title: "Lecture 13: Speech Brain-Computer Interfaces"
summary: "Decoding speech from the brain: from Howard's letter board to implanted arrays, word error rates, decoding speed, and the frontier of inner speech."
instructor: "Chaofei Fan"
offering: "Spring 2024"
duration: "1:13:00"
video_id: tfVgHsKpRC8
video_title: "Lecture 13: Speech Brain-Computer Interfaces"
video_caption: "Guest lecture. Chaofei Fan (Stanford MPTL) covers speech decoding brain-computer interfaces."
concepts: [bci, speech-decoding, motor-cortex, microelectrode-array, word-error-rate, inner-speech]
sources:
  - tag: video
    label: "Lecture 13 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=tfVgHsKpRC8
  - tag: notes
    label: "Official subtitle transcript"
---

## How to read this lesson

This lesson has two levels. **Level 1 (Core)** tells the human story and the
2017 breakthrough. **Level 2 (Deep)** covers the implanted system, its error
rates and speed, and the frontier.

## Level 1: Howard's story

Howard was 21 when a severe stroke left him locked-in ([01:23](ts:01:23)):
a fully functioning brain, no way to move or speak. He communicated with a
**letter board**: someone watches his gaze, letter by letter. Minutes per
sentence.

![Howard](assets/l13-howard.svg "Stroke at 21, locked-in; letter board with gaze takes minutes per sentence; eye-tracking is tiring.")

**Eye-tracking** helped but tired him: staring at a screen all day is
exhausting, and moving the eyes is hard ([03:38](ts:03:38)). The motivation
for everything that follows: restore effortless communication.

> [!QA]
> Q: Why not just improve eye-tracking?
> A: The bottleneck is the body, not the device. Locked-in patients struggle to move their eyes precisely, and sustained gaze is tiring. A brain interface bypasses the broken output path entirely.
> Follow-up: What does "locked-in" mean exactly?
> A: Full consciousness with near-total paralysis. The brain works. The muscles do not respond. Communication must route around the body.

## Level 1: The 2017 breakthrough

In 2017, a participant **typed on a virtual keyboard with imagined
movement**. Peak: ~40 correct characters per minute. Average: ~20
([28:57](ts:28:57)).

![2017](assets/l13-2017.svg "Imagined movement drives a virtual keyboard: peak ~40 chars/min, average ~20.")

Slow. But it proved the channel: motor cortex signals decode into text.

## Level 2: T12 and the four arrays

Participant T12 (ALS, recruited 2022) received **four microelectrode arrays**
([41:56](ts:41:56)): two in **motor cortex**, two in **Broca's area**.

![T12](assets/l13-t12.svg "Two arrays in motor cortex carry phoneme and word information; two in Broca's area score near chance for this decoding.")

The motor cortex arrays carry phoneme and word information. The Broca's
arrays score not much above chance for this decoding. Lesson: **decode where
the signal lives**. The system is a real-time speech-to-text BCI.

## Level 2: Error rates and daily use

**Word error rate: ~25%.** For every 100 words the participant says, 25 are
wrong ([66:14](ts:66:14)). High, but the trajectory matters: UC Davis
collaborators reach **close to zero WER** after continuous training
([66:28](ts:66:28)). The participant uses the system daily with family
([65:52](ts:65:52)).

![WER](assets/l13-wer.svg "25% word error rate here; UC Davis near zero after continuous training; daily use with family.")

## Level 2: Speed

Maximum decoding speed: **60-70 words per minute**. Natural speech: **150 wpm**
([67:30](ts:67:30)). For context: handwriting runs 13-14 wpm, eye-tracking
about 5 ([35:44](ts:35:44)).

![Speed](assets/l13-speed.svg "BCI: 60-70 wpm. Natural speech: 150 wpm. Handwriting: 13-14 wpm. Eye-tracking: ~5 wpm.")

Half the speed of speech. Far ahead of every non-invasive alternative.

## Level 2: The frontier

Two frontiers:

![Frontier](assets/l13-frontier.svg "Decode inner speech: thought without attempted movement. UCSF: phonemes plus articulation drive a 3D avatar.")

1. **Inner speech.** Decode thought without attempted movement ([67:23](ts:67:23)).
Effortless, natural communication.
2. **Multimodal decoding (UCSF).** Decode phonemes **and** articulation to
drive a **3D avatar** ([65:28](ts:65:28)): speech you can see.

> [!QA]
> Q: What is the hardest part of inner speech decoding?
> A: Ground truth. Attempted speech has an intended output you can verify. Inner speech has no observable target, so training labels are hard to get. The signal is also weaker without motor execution.
> Follow-up: Why does this lecture belong in an NLP course?
> A: Decoding is language modeling over neural signals. Phonemes, words, and language models structure the output exactly as in speech recognition. The course's tools transfer to the brain.

## Recap: the whole lesson on one screen

Eight ideas carry this lecture. Read each card. Say the core sentence out
loud. If you can, you own the lesson.

<div class="recap-grid">
<div class="recap-card">
<img src="assets/l13-howard.svg" alt="Howard">
<div class="rc-body">
<strong>1. Locked-in means the brain works, the body does not</strong>
<p>Howard, 21, stroke. Letter board: minutes per sentence. Eye-tracking:
tiring. The motivation is effortless communication.</p>
<p class="rc-num">Key: bypass the broken output path</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l13-2017.svg" alt="2017">
<div class="rc-body">
<strong>2. 2017: type with imagined movement</strong>
<p>Virtual keyboard, ~40 chars/min peak, ~20 average. Slow. Proof that
motor signals decode into text.</p>
<p class="rc-num">Key: the channel exists</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l13-t12.svg" alt="T12">
<div class="rc-body">
<strong>3. Four arrays, two regions</strong>
<p>T12: two arrays in motor cortex, two in Broca's area. Motor cortex
carries the signal. Broca's scores near chance.</p>
<p class="rc-num">Key: decode where the signal lives</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l13-wer.svg" alt="WER">
<div class="rc-body">
<strong>4. 25% word error rate, falling</strong>
<p>25 of 100 words wrong. UC Davis: close to zero after continuous
training. Daily use with family.</p>
<p class="rc-num">Key: trajectory matters</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l13-speed.svg" alt="Speed">
<div class="rc-body">
<strong>5. 60-70 wpm versus 150 natural</strong>
<p>Half the speed of speech. Far ahead of handwriting (13-14) and
eye-tracking (~5).</p>
<p class="rc-num">Key: [67:30](ts:67:30)</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l13-frontier.svg" alt="Frontier">
<div class="rc-body">
<strong>6. Frontier: inner speech</strong>
<p>Decode thought without attempted movement. The hard part: no observable
ground truth for training.</p>
<p class="rc-num">Key: [67:23](ts:67:23)</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l13-frontier.svg" alt="Avatar">
<div class="rc-body">
<strong>7. UCSF: phonemes plus articulation to avatar</strong>
<p>Multimodal decoding drives a 3D avatar. Speech you can see, not just
read.</p>
<p class="rc-num">Key: [65:28](ts:65:28)</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l13-howard.svg" alt="Why it matters">
<div class="rc-body">
<strong>8. Why this is an NLP lecture</strong>
<p>Decoding is language modeling over neural signals. Phonemes, words,
language models: the course's tools, applied to the brain.</p>
<p class="rc-num">Key: same tools, new channel</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- Lecture 13 video and transcript.

**Further reading:**
- Willett et al. (2023), "A high-performance speech neuroprosthesis": the Stanford speech BCI results.
- Metzger et al. (2023, UCSF), "Generalizable spelling and speech decoding": the multimodal avatar work.

**Caveats from these sources.** WER and speed numbers are participant- and system-specific. They vary across labs and sessions. "Close to zero" is the lecture's report of UC Davis results, not a published benchmark in this lesson.

## Connections to the other courses

- **This course:** L05's language models structure BCI output. L09's pretraining intuition (predict from context) applies to neural signals.
- **CS336:** L17 multimodality: neural signals are another modality.
- **CS229:** signal processing and decoding theory underlie the implants.

> [!CHEAT]
> **Speech BCI cheatsheet.** Locked-in: brain works, body does not. Letter board: minutes/sentence. Eye-tracking: tiring. 2017: imagined movement, 40/20 chars/min. T12: 4 arrays (2 motor cortex, 2 Broca's). Motor cortex carries signal. WER: 25% (25/100 wrong); UC Davis near zero. Speed: 60-70 wpm vs 150 natural. Handwriting 13-14. Eye-track 5. Frontier: inner speech (no ground truth); UCSF avatar (phonemes + articulation).

> [!MEMORY]
> **Decode where the signal lives.** Motor cortex, not Broca's. Attempted speech first, inner speech next. The error rate falls. The channel is real.

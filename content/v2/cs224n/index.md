---
page_id: cs224n-index
course_slug: cs224n
course_name: "CS224N: NLP with Deep Learning"
course_order: 4
order: 0
nav: "CS224N · Overview"
title: "CS224N: NLP with Deep Learning"
summary: "From word vectors to alignment: the full NLP arc, Spring 2024, Christopher Manning."
instructor: "Christopher Manning"
offering: "Spring 2024"
---

From word vectors to alignment: fifteen lectures rebuilding NLP from its
foundations. Spring 2024, Christopher Manning, with guest lecturers.

The arc: word vectors (L01-L02), neural networks (L03-L04), RNNs (L05-L06),
attention (L07), transformers (L08), pretraining (L09), post-training
(L10), evaluation (L11), efficient training (L12), speech BCIs (L13),
reasoning and agents (L14), life after DPO (L15).

Lectures 8 through 12 and 15 are bridge lessons: they teach the CS224N
framing and link into CS336 for deep mechanics. The token chip belongs to
CS336 L01: this course reuses it by link and never redraws it.

Note: the Lecture 1 video was bot-blocked during archiving. L01 is built
from the official slide deck and flagged as such.

## Lectures and subchapters

Every lecture is deepened with subchapters that add one idea at a time
(0 → 0.1 → 0.12 → 1), each with its own figure plate, 6–8 interview Q&A,
a "what is used where" production map (Oct 2026), and curated video +
go-deeper links.

- **L01 — Introduction and word vectors.** CBOW · hierarchical softmax · Huffman-shaped trees · [watch and go deeper](l01-intro-word-vectors.html#watch-and-go-deeper)
- **L02 — Word vectors 2 and word senses.** Subsampling · 3/4 power negatives · FastText subwords
- **L03 — Neural networks and backprop.** ReLU, then GELU · forward-mode vs reverse-mode
- **L04 — Dependency parsing.** Arc-standard vs arc-eager vs hybrid · graph-based MST · beam search
- **L05 — Language modeling and RNNs.** Add-k · backoff · Kneser-Ney · temperature · top-p · perplexity
- **L06 — LSTMs and neural machine translation.** GRU · beam search decoding · teacher forcing and exposure bias
- **L07 — Attention.** Dot vs additive scores · scaled dot-product · Luong local vs global · attention is not explanation
- **L08 — Transformers.** MHA/MQA/GQA · RoPE · FlashAttention
- **L09 — Pretraining.** MLM · CLM · span corruption · RTD · full fine-tuning vs PEFT (adapters, prefix, LoRA)
- **L10 — Prompting, instruction tuning, RLHF.** Preference pairs · PPO · DPO-vs-PPO decision
- **L11 — Benchmarking and evaluation.** ROUGE · BERTScore · Elo math · contamination
- **L12 — Efficient neural network training.** Tensor parallelism · pipeline parallelism · gradient scaler · quantization
- **L13 — Speech brain-computer interfaces.** Signal chain (spikes → phonemes) · the LM in the decoder · invasive vs non-invasive
- **L14 — Reasoning and agents.** ReAct · tree of thought · Reflexion
- **L15 — Life after DPO.** The DPO loss term by term · iterative and online DPO · reward model evaluation

Plus the [one-page cheatsheet](cheatsheet.html) (every toy number, Oct 2026
production map) and the [30-minute crash course](crash-course.html)
(interview-speed review with links into the deep lessons).

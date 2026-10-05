---
title: "CS224N: NLP with Deep Learning"
course: cs224n
type: course-index
instructor: Chris Manning
term: Spring 2024
---

How deep learning learned language. From word vectors to agents, this is the course that defined NLP's deep learning era, now updated for the LLM age.

CS224N's early lessons are its unique contribution: word vectors, dependency parsing, RNNs, and the attention story from the NLP side. Its later lessons overlap with CS336, so those are built as bridges that link back rather than rewrite.

## Lessons

### Part I: Words and sequences
1. [Introduction and word vectors](l01-word-vectors-1.html) — why words as vectors, the distributional hypothesis
2. [Word vectors and language models](l02-word-vectors-2.html) — word2vec, GloVe, evaluation
3. [Neural networks for NLP](l03-neural-networks.html) — backprop refresher, applied to language
4. [Dependency parsing](l04-dependency-parsing.html) — syntactic structure, transition-based parsing

### Part II: Sequence models
5. [Recurrent neural networks](l05-rnns.html) — RNNs, LSTMs, vanishing gradients
6. [Seq2seq and attention](l06-seq2seq-attention.html) — machine translation, the attention breakthrough
7. [Attention and LLMs](l07-attention-llm-intro.html) — attention history, the road to LLMs

### Part III: Transformers and pretraining (bridges to CS336)
8. [Self-attention and transformers](l08-transformers.html) — the architecture from the NLP side
9. [Pretraining](l09-pretraining.html) — BERT, GPT, what pretraining learns
10. [Post-training](l10-post-training.html) — prompting, RLHF, instruction tuning

### Part IV: Evaluation and efficiency (bridges)
11. [Benchmarking](l11-benchmarking.html) — how NLP evaluates models
12. [Efficient training](l12-efficient-training.html) — the systems side of NLP

### Part V: Frontiers
13. [Brain-computer interfaces](l13-bci.html) — language and the brain
14. [Reasoning and agents](l14-reasoning-agents.html) — chain-of-thought, LM agents
15. [After DPO](l15-after-dpo.html) — Nathan Lambert on post-training beyond DPO

## Interview labs

- [Lab 1: Word2vec from scratch](lab1-word2vec.html) — implement skip-gram with negative sampling
- [Lab 2: Attention from scratch](lab2-attention.html) — single-head attention in NumPy

## Sources

- Videos: Stanford Online YouTube, Spring 2024
- Slides: official CS224N Spring 2024 decks (cs224n.1246)

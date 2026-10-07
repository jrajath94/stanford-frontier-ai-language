# Crash course , cs224n U01-U15

The whole course in one pass. Each unit: the idea, the number, the
trap. Numbers come from the unit compute scripts, all toy-scale and
labeled as such.

## U01 , NLP history, tasks, data

Idea: language is ambiguous, tasks need formulation (loss, metric,
data) before modeling. Number: the course starts from counting.
Trap: judging a task by its name.

## U02 , word vectors

Idea: the distributional hypothesis, skip-gram/CBOW, negative
sampling, GloVe, PMI. Number: analogy cosines near 1.0 on the toy.
Trap: reading cosine as probability.

## U03 , neural fundamentals and parsing

Idea: the computational graph, chain rule, embeddings as
parameters, shift/reduce parsing. Number: shift by max before
softmax. Trap: unshifted softmax NaNs.

## U04 , language models and recurrence

Idea: teacher forcing, perplexity, BPTT, vanishing gradients,
LSTM/GRU, decoding. Number: perplexity is per-token, compare only
within one tokenizer. Trap: teacher forcing vs free running.

## U05 , attention and transformers

Idea: Q/K/V, 1/sqrt(d_k), MHA, masks, residuals, positions, FFN,
KV cache. Number: the n^2 attention bill. Trap: the causal mask
goes before softmax.

## U06 , pretraining, scaling, systems, data

Idea: masked and autoregressive objectives, corpus curation,
scaling laws, leakage, transfer, model cards. Number: doubling
compute cuts loss about 3.4 percent on the toy curve. Trap:
scaling params without data.

## U07 , post-training and preferences

Idea: SFT, reward models, Bradley-Terry, RLHF/PPO, DPO, KL leash,
reward hacking, annotation bias. Number: reward gap 0.9 gives P
0.711. Trap: optimizing the proxy and calling it aligned.

## U08 , prompting and efficient adaptation

Idea: zero/few-shot, in-context learning, prompt sensitivity,
chain-of-thought (useful, not faithful), self-consistency,
adapters, LoRA, forgetting, validation. Number: LoRA trains 0.060
percent of params. Majority of 5 at p 0.6 gives 0.683. Trap:
conditioning is not learning.

## U09 , tools, agents, and retrieval

Idea: ReAct (Thought/Action/Observation), tool schemas, execution
feedback, RAG, retriever/generator split, chunking, hybrid/rerank,
citations, memory vs context, permissions, stopping, attribution.
Number: 7 steps, 92 tokens. Recall at k 0.000/0.333/0.667/1.000.
split rate 0.15 to 0.00 with overlap 50. Trap: the loop never
stops on its own.

## U10 , benchmarking and research methodology

Idea: benchmark families, coverage, judge calibration, contamination,
uncertainty, scoring formats, replication, slices, bias, acceptance
criteria, project choice, contribution accounting. Number: 78/100
is [0.689, 0.850]. 40 leaked items inflate 0.70 to 0.75. kappa
0.551. Trap: the bare number, the hidden slice.

## U11 , reasoning and test-time compute

Idea: reasoning supervision, verifiable rewards, outcome vs process
verification, repeated sampling, self-consistency, compute
allocation, speculative decoding, eval bias, failure analysis,
cost-quality, trace limits. Number: vote curve 0.600 to 0.826.
spec decode 2.94 tokens per pass. 10x10 beats 1x100 (3.94 vs
0.99). Trap: voting below p 0.5, trusting traces.

## U12 , tokenization and multilinguality

Idea: Unicode, BPE, vocab sharing, rare words, transfer,
multilingual space, cost inequality, low-resource data, parity,
morphology, code switching, fairness bets. Number: fertility 1.3
to 2.8. cost ratios to 2.42x. BPE 6 merges on the toy. Trap: the
same model at different prices.

## U13 , interpretability and social impacts

Idea: concept probes, behavioral vs mechanistic evidence, causal
interventions, vocabulary limits, steering, attribution ladder,
misinformation, bias, privacy, misuse, uncertainty, impact
reports. Number: disparity ratio 0.667. intervention effect 1.5.
ECE 0.425. extraction 0.185. Trap: "truth direction" headlines.

## U14 , multimodality and guest research

Idea: patch tokens, early/late fusion, mixed streams, joint
generation, diffusion/AR mix, modality experts, multimodal
retrieval, instruction tuning, reward eval, guest cards, claim
classes, gap ledger. Number: late fusion 5.03e7 params vs 0.
routing aux 1.04. image share 0.333. Trap: input format is not
multimodality. hearsay is not a claim.

## U15 , project tutorials and open questions

Idea: Python, PyTorch, Hugging Face, tiny GPT-2, downstream tasks,
the four gates, scoping, originality, reproducibility, limits,
open questions, defense. Number: 44,928 params. seeds mean
0.8045, std 0.0203. Trap: toy numbers cited as findings, one
seed, the cut control group.

## The course in five lines

Formulate before modeling. Condition is not learning. Report the
interval, not the number. Attribute before fixing. Every claim
carries its evidence and its limit.

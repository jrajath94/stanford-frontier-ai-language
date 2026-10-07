# Course map , cme295 (nine lectures)

Source: SRC-01 (official syllabus page, inspected 2026-10-06). Dates and
topic bullets are quoted from that page. Claim class per unit row:
OFFICIAL-SYLLABUS = topic named on the syllabus, REQUESTED-BRANCH =
requested extension not named on the syllabus.

| # | Date | Lecture | Topics (syllabus) | Units | Claim |
|---|------|---------|-------------------|-------|-------|
| L1 | Sep 25, 2026 | Transformers | NLP background and tasks, tokenization, embeddings, word2vec, RNN, LSTM, attention, transformer architecture, end-to-end example | U01 | OFFICIAL-SYLLABUS |
| L2 | Oct 2, 2026 | Large Language Models | Transformer model families, LLM definition and architecture, mixture of experts, MHA, MQA, GQA, position embeddings (RoPE and variants), context length, temperature, sampling strategies | U02, U03 | OFFICIAL-SYLLABUS |
| L3 | Oct 9, 2026 | LLM training | Pretraining, SFT, LoRA, preference tuning (RLHF, DPO), reasoning, on-policy distillation and variants, distillation to smaller models | U04, U05 | OFFICIAL-SYLLABUS |
| L4 | Oct 16, 2026 | Reinforcement learning with LLMs | Mathematical conventions, reward design, policy gradients, limitations, preference tuning with PPO (RLHF), reasoning with GRPO (RLVR), on-policy distillation | U05, U06 | OFFICIAL-SYLLABUS |
| - | Oct 23, 2026 | Midterm | (after baseline, no content claimed) | U09 | not available |
| L5 | Oct 30, 2026 | LLM systems | Distributed training, inference optimizations, KV caching, speculative decoding, efficient kernels, Flash Attention, hardware trade-offs | U04 (partial), U06 | OFFICIAL-SYLLABUS |
| L6 | Nov 6, 2026 | AI Agents | Tool calling, MCP, memory, retrieval, context compaction, agent loop optimization, coding agents, skills, plugins | U07 | OFFICIAL-SYLLABUS |
| L7 | Nov 13, 2026 | LLM evaluation | LLM-as-a-judge overview, best practices and benefits, biases and pitfalls, agent evaluation, benchmarks | U08 | OFFICIAL-SYLLABUS |
| L8 | Nov 20, 2026 | Diffusion LLMs | Continuous diffusion, discrete diffusion, masked diffusion, training, inference | U09 | OFFICIAL-SYLLABUS |
| L9 | Dec 4, 2026 | Trending topics | Recap, multimodality, closing thoughts | U09 | OFFICIAL-SYLLABUS |
| - | Dec 9, 2026 | Final | (after baseline, no content claimed) | U09 | not available |

## This build (first builder): U01-U05

- U01 anchors to L1 (all 12 concepts OFFICIAL-SYLLABUS).
- U02 anchors to L2: attention variants, MHA/MQA/GQA, RoPE and position
  variants, encoder/decoder layouts, BERT derivatives are
  OFFICIAL-SYLLABUS. Attention approximation, cache differences beyond
  naming, complexity analysis, and failure cases are REQUESTED-BRANCH
  extensions (KV caching is named in L5, detailed cache math is the
  branch).
- U03 anchors to L2: LLM definition, MoE, context length, temperature,
  sampling strategies are OFFICIAL-SYLLABUS. Prompting, ICL,
  demonstration selection, reasoning prompting, self-consistency, and
  evaluation limits are REQUESTED-BRANCH extensions.
- U04 anchors to L3: pretraining, SFT, LoRA, data/loss are
  OFFICIAL-SYLLABUS. Quantization, hardware efficiency, numerical
  precision, full-vs-PEFT comparison, regression tests, and resource
  budgets are REQUESTED-BRANCH (hardware trade-offs are named in L5).
- U05 anchors to L3/L4: RLHF, DPO, reward design, policy gradients, PPO
  are OFFICIAL-SYLLABUS. Pairwise-label protocol detail, policy ratios,
  KL estimators, advantages, PPO variants, DPO derivation assumptions,
  reference-model mechanics, optimization pathologies, and evaluation
  protocol are REQUESTED-BRANCH extensions.

## Second builder: U06-U09 (built 2026-10-07)

- U06 anchors to L3/L4/L5: reasoning models, verifiable outcomes
  (RLVR), GRPO are OFFICIAL-SYLLABUS (L4 "reasoning with GRPO
  (RLVR)"). Reward-hacking limitations are OFFICIAL-SYLLABUS (L4
  "limitations"). Group normalization mechanics, sampling
  diversity, train-time vs test-time scaling, scaling-data
  analysis, verifier quality, compute allocation, ablations, and
  generalization tests are REQUESTED-BRANCH extensions.
- U07 anchors to L6: retrieval pipeline, context construction
  (context compaction), function calling and tool results (tool
  calling), and agent state (memory) are OFFICIAL-SYLLABUS.
  Advanced retrieval, hybrid search, reranking, ReAct, stopping
  rules, permissions, and failure separation are
  REQUESTED-BRANCH extensions.
- U08 anchors to L7: judge rubrics, pointwise/pairwise judging,
  position/verbosity bias, human checks, appropriate metrics, and
  benchmark interpretation are OFFICIAL-SYLLABUS. Calibration,
  nondeterminism, blind evaluation, contamination, uncertainty,
  and cost/latency are REQUESTED-BRANCH extensions.
- U09 anchors to L8/L9 plus both exams: diffusion LLM topics
  (continuous, discrete, masked diffusion, training, inference)
  and the L9 recap are OFFICIAL-SYLLABUS. Exam objective mapping
  is original practice only (both exams fell after baseline, no
  exam content claimed). Dated update separation, original
  exam-style practice, cross-lecture dependencies, comparative
  architectures, end-to-end toy, research protocol, production
  bridge, and oral defense are REQUESTED-BRANCH.
- Consolidation: 2 capstones (research replication with honest
  negative result, applied/FDE with hypothetical numbers),
  transfer sets, oral defenses, crash course, cheatsheet.

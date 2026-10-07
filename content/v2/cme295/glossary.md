# Glossary , cme295 U01-U09

One meaning per term. First use in lessons matches these definitions.

- **Token**: integer id for a text piece, the model input alphabet.
- **Vocabulary**: the full set of token ids, size V.
- **Embedding**: dense vector for a token, row of the table E.
- **Logit**: unnormalized score per vocabulary item before softmax.
- **Perplexity**: exp of mean negative log-likelihood per token, lower
  is better, only comparable under one tokenizer.
- **Teacher forcing**: training with ground-truth previous tokens as
  inputs.
- **Causal mask**: mask that forbids attending to future positions.
- **KV cache**: stored keys and values of past tokens reused at decode.
- **MHA**: multi-head attention, each query head has a private KV head.
- **MQA**: multi-query attention, all query heads share one KV head.
- **GQA**: grouped-query attention, groups of query heads share KV heads.
- **RoPE**: rotary position embedding, rotates Q/K pairs by position.
- **Temperature**: divisor on logits before softmax, controls sharpness.
- **Top-k**: keep only the k highest-probability tokens, renormalize.
- **Top-p**: keep the smallest set with cumulative mass >= p, renormalize.
- **ICL**: in-context learning, task behavior from prompt examples only.
- **SFT**: supervised fine-tuning on instruction-response pairs.
- **LoRA**: low-rank adapter, frozen base plus trainable BA factors.
- **PEFT**: parameter-efficient fine-tuning, trains a small subset.
- **Reward model**: scalar head scoring response quality.
- **RLHF**: pipeline SFT -> reward model -> RL against the reward model.
- **PPO**: proximal policy optimization, clipped-surrogate RL.
- **DPO**: direct preference optimization, preference loss without RL.
- **KL penalty**: term pulling the policy toward a reference policy.
- **Advantage**: return minus baseline, centers the learning signal.
- **Reward hacking**: policy exploits flaws in the reward, not the task.
- **RLVR**: RL with verifiable rewards, program-computed not judged.
- **GRPO**: group relative policy optimization, critic-free RL on
  prompt groups.
- **Group normalization**: (r - mu)/sigma over one prompt group.
- **pass@k**: 1 - (1 - p)^k, success probability over k attempts.
- **Verifier**: program that scores outcomes, e.g. unit tests.
- **Reward hacking**: policy exploits verifier gaps, not the task.
- **RAG**: retrieval-augmented generation, index then read.
- **BM25**: keyword score from term statistics.
- **RRF**: reciprocal rank fusion, sum 1/(k + rank).
- **Reranker**: cross-encoder scoring (query, doc) pairs.
- **ReAct**: Thought -> Action -> Observation agent loop.
- **Tool tier**: green/yellow/red permission badge on a tool.
- **Judge rubric**: the scoring law an LLM judge follows.
- **Position bias**: judge favors the first-presented answer.
- **Verbosity bias**: judge favors the longer answer.
- **ECE**: expected calibration error, honesty about doubt.
- **Contamination**: benchmark items leaked into training.
- **Masked diffusion**: LM that unmasks tokens by confidence in
  parallel steps.
- **Keystone concept**: concept with the most downstream dependents.
- **Orphan objective**: exam objective with no lesson home.

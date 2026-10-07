# Glossary , cs224n U01-U15

One meaning per term. First-use definitions in lessons match these.

- ambiguity: one input admits two or more valid interpretations.
- analogy: a:b :: c:d tested by vector arithmetic on embeddings.
- attention: weighted mix of values by query-key similarity.
- autoregressive: each token predicted from the prefix only.
- batch: one set of examples processed together in an update.
- beam search: decoding that keeps the top-k partial hypotheses.
- Bradley-Terry: model of pairwise preference from scalar scores.
- CBOW: continuous bag of words, predicts a center word from context.
- chunking: splitting documents into retrieval-sized pieces.
- cooccurrence: two words appearing in one context window.
- cosine similarity: dot product divided by both norms, in [-1, 1].
- cross-entropy: negative log-likelihood of the true distribution.
- decoding: turning model scores into a token sequence.
- distributional hypothesis: words in similar contexts have similar
  meanings.
- DPO: direct preference optimization, preference loss without a
  separate reward model or RL loop.
- embedding: a learned vector that stands for a discrete token.
- encoder: a block that reads the full input, no causal mask.
- decoder: a block that generates left to right under a causal mask.
- exposure bias: train sees gold prefixes, test sees model prefixes.
- feedforward: position-wise two-layer MLP inside a transformer block.
- GloVe: count-based embedding method on a cooccurrence matrix.
- gradient clipping: rescale a gradient whose norm passes a cap.
- greedy decoding: always pick the highest-probability token.
- hierarchical softmax: softmax factorized over a binary tree.
- instruction tuning: supervised fine-tuning on instruction-response
  pairs.
- intrinsic evaluation: embedding quality judged by word tasks, not by
  downstream models.
- KL divergence: expected log ratio of p to q under p, in nats.
- KV cache: stored keys and values that avoid recompute in decoding.
- LoRA: low-rank additive update to a frozen weight matrix.
- masked LM: predict masked tokens from both directions of context.
- multi-head attention: parallel attention heads with separate
  projections, concatenated at the output.
- negative sampling: binary classification against sampled noise words.
- one-hot: vector with a single 1 and zeros elsewhere.
- padding: dummy tokens that fill a batch to one length.
- PEFT: parameter-efficient fine-tuning, update of a small subset.
- perplexity: exp of mean negative log-likelihood, in "choices".
- PMI: pointwise mutual information, log of observed over expected
  joint probability.
- policy: the model viewed as a mapping from context to action
  distribution (RL framing).
- polysemy: one word form with several distinct senses.
- post-training: training after pretraining: SFT, preferences, RL.
- pretraining: first-stage training on a broad corpus.
- PPO: proximal policy optimization, clipped policy-gradient method.
- prompt sensitivity: output changes under paraphrase of the prompt.
- reward hacking: a policy that raises the proxy reward without raising
  the true objective.
- reward model: a scorer trained on human preference pairs.
- RLHF: reinforcement learning from human feedback.
- scaling law: fitted relation of loss to parameters, data, compute.
- self-consistency: majority vote over sampled reasoning paths.
- SFT: supervised fine-tuning on demonstration data.
- skip-gram: predict context words from a center word.
- softmax: normalized exponential map from scores to probabilities.
- teacher forcing: train the next-token model on gold prefixes.
- token: the atomic text unit the model reads and writes.
- top-k sampling: sample from the k most likely tokens only.
- top-p sampling: sample from the smallest set with total mass p.
- vanishing gradient: gradient norm shrinks across long dependencies,
  so distant inputs stop moving the parameters.
- window: the fixed span of context around a center word.
- zero-shot: task solved with no examples in the prompt.
- acceptance criterion: pre-registered pass/fail contract for a
  decision, written before measuring.
- aux loss: extra training loss that balances expert routing.
- BPE: byte-pair encoding, merge frequent adjacent pairs.
- Brier score: mean squared error of stated probabilities.
- canary: planted string used to test memorization.
- causal intervention: change a component, measure the behavior
  change.
- citation: pointer from a claim to a supporting span.
- claim class: honesty label for a statement (official,
  requested-branch, restricted, edition tag).
- code point: Unicode number for a character.
- concept direction: vector in activation space that separates
  labeled examples.
- contamination: test items present in training data.
- context: the working token window of the current run.
- contribution: the built column of the project ledger.
- coverage matrix: benchmark tasks vs deployment capabilities.
- defense rubric: claim, evidence, limit, transfer, four points.
- disparity ratio: min group rate over max group rate.
- ECE: expected calibration error, mean |confidence - accuracy|.
- error attribution: labeling a failure R (retriever), T (tool),
  or P (policy).
- fertility: tokens per word in a language.
- guest card: title, source, inspection status, claims (none until
  inspected).
- hybrid retrieval: sparse plus dense scores fused by weight.
- impact report: scored checklist before shipping a system.
- kappa: chance-corrected agreement between judges.
- memory: stored state across turns (verbatim, summary, anchors).
- misinformation: fluent false claims at scale.
- modality expert: specialist sub-network per input sense.
- open question: dated unknown with a closing criterion.
- permission scope: the set of tools a run may call.
- RAG: retrieve top-k chunks, prepend, generate.
- ReAct: interleaved Thought / Action / Observation loop.
- reading card: title, claim, read status for a paper.
- red team: adversarial probing of a system's harms.
- rerank: heavy rescoring of a small retrieved pool.
- reproducibility receipt: seeds, versions, data, commands.
- seed sweep: run k seeds, report mean and std.
- slice: subgroup score with its delta from the mean.
- speculative decoding: draft with a small model, verify with the
  big one, same output distribution.
- split rate: fraction of answer spans cut by chunk boundaries.
- steering: add alpha times a direction to activations.
- step cap: max agent loop steps, the fire escape.
- tool schema: name, argument schema, return schema, side-effect
  class.
- trace: the model's step-by-step text, a story not a proof.
- transfer gap: source accuracy minus target accuracy.
- translationese: simplified language of translated text.
- UTF-8: variable-length byte encoding of code points.
- verifiable reward: reward a program checks (tests, exact match).
- Wilson interval: confidence interval for a proportion.

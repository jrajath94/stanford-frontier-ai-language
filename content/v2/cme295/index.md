# Index , cme295 learner navigation

Start at `prerequisites.md` and take the diagnostic. Then work units in
order, each unit lists its own prerequisites and a local remediation
block.

## Units (this build)

- `lessons/u01_nlp_foundations.md` , NLP tasks to the transformer
  block. Key: `keys/u01_answers.md`. Lab: `labs/u01_lab.md`
  (key: `labs/u01_lab_key.md`, runner: `labs/u01_lab_run.py`).
  Interview: `interview/u01_questions.md` (key:
  `interview/u01_key.md`).
- `lessons/u02_attention_variants.md` , MHA/MQA/GQA, positions, RoPE,
  layouts, caches, complexity. Key: `keys/u02_answers.md`. Lab:
  `labs/u02_lab.md`. Interview: `interview/u02_questions.md`.
- `lessons/u03_llm_generation.md` , LLM definition, MoE, context,
  temperature, top-k/top-p, prompting, ICL, self-consistency. Key:
  `keys/u03_answers.md`. Lab: `labs/u03_lab.md`. Interview:
  `interview/u03_questions.md`.
- `lessons/u04_training_adaptation.md` , pretraining, SFT, LoRA,
  quantization, precision, hardware, budgets. Key:
  `keys/u04_answers.md`. Lab: `labs/u04_lab.md`. Interview:
  `interview/u04_questions.md`.
- `lessons/u05_preference_optimization.md` , reward models, RLHF, PPO,
  DPO, KL, pathologies, evaluation. Key: `keys/u05_answers.md`. Lab:
  `labs/u05_lab.md`. Interview: `interview/u05_questions.md`.
- `lessons/u06_reasoning_scaling.md` , RLVR, GRPO, group
  normalization, reward hacking, test-time scaling, compute
  allocation, ablations, generalization. Key:
  `keys/u06_answers.md`. Lab: `labs/u06_lab.md`. Interview:
  `interview/u06_questions.md`.
- `lessons/u07_rag_agents.md` , retrieval, hybrid search, rerank,
  ReAct, agent state, stopping, permissions, failure separation.
  Key: `keys/u07_answers.md`. Lab: `labs/u07_lab.md`. Interview:
  `interview/u07_questions.md`.
- `lessons/u08_llm_evaluation.md` , judge rubrics, pairwise judging,
  bias correction, calibration, contamination, uncertainty,
  benchmark reading. Key: `keys/u08_answers.md`. Lab:
  `labs/u08_lab.md`. Interview: `interview/u08_questions.md`.
- `lessons/u09_trends_synthesis.md` , diffusion LMs, exam objective
  mapping, keystones, end-to-end toy, research protocol,
  production bridge, oral defense. Key: `keys/u09_answers.md`.
  Lab: `labs/u09_lab.md`. Interview: `interview/u09_questions.md`.

## Consolidation

- `capstones/research_grpo_extension.md` , executed replication:
  group normalization vs baseline REINFORCE on a synthetic task
  (runner `capstones/research_grpo_run.py`, figure
  `capstones/figures/capstone_research_fig01.png`). H2 rejected
  honestly.
- `capstones/applied_rag_support.md` , FDE-style RAG support agent
  for a fictional company, hypothetical numbers labeled
  (runner `capstones/applied_rag_run.py`, figure
  `capstones/figures/capstone_applied_fig01.png`).
- `interview/transfer-sets.md` (key: `interview/keys-transfer.md`),
  10 changed-scenario sets across units.
- `interview/oral-defenses.md` (key: `interview/keys-oral.md`),
  10 deep ladders x 8 follow-ups.
- `crash-course.md` , nine-unit synthesis. `cheatsheet.md` , one
  page per unit.

## Reference

- `notation_and_shapes.md` , symbols, shapes, units used everywhere.
- `glossary.md` , one-meaning definitions.
- `coverage_matrix.md` , the 108 rows and their evidence.
- `visuals/figures/` , 37 computed PNG plates referenced from lessons
  (20 from U01-U05, 17 from U06-U09), plus 2 capstone figures in
  `capstones/figures/`.
- `mastery_ledger.md` , your per-concept mastery state.
- `errors.md` , build log and classic learner traps.

## Fast path versus deep reference

The fast path reads each lesson's items 1-6 (question, model, objects,
mechanism, computed toy). The deep reference adds items 7-15
(implementation, checks, complexity, alternatives, failure, research,
assessment, lab, figures). The fast path is shorter navigation, not
omitted concepts.

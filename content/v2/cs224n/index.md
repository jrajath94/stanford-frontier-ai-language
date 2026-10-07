# Index , cs224n U01-U15

Start here. Read in this order.

1. `README.md` , scope, honesty rules, layout.
2. `prerequisites.md` , what you must know first, the diagnostic, and
   where each unit links for remediation.
3. `notation_and_shapes.md` , symbols and shapes used everywhere.
4. `glossary.md` , one-meaning definitions.
5. Unit lessons in `lessons/`, in order U01 through U15. Each lesson is
   self-contained: read its remediation block first if the diagnostic
   flagged it.
6. `labs/` , one lab per unit. Run `python3 labs/uNN_lab_run.py` before
   reading the lab key.
7. `interview/` , oral preparation after the lesson and lab.
8. `capstones/` , the two consolidation projects.
9. `crash-course.md` and `cheatsheet.md` , fast review after the units.
10. `currentness.md` , what is dated and when to refresh it.

## Unit order and dependency

U01 history, tasks, data (no course-internal dependency).
U02 word vectors (uses P03, P05, P08, U01 context).
U03 neural fundamentals and dependency parsing (uses P05, P11, P12, U02
embeddings as parameters).
U04 language models and recurrence (uses P11, P13, U03 backprop).
U05 attention and transformers (uses P12, P14, U04 decoding, perplexity).
U06 pretraining, scaling, systems, data (uses P10, P14, P15, U05
architecture).
U07 post-training and preferences (uses P08, P14, P17, U06 base models).
U08 prompting and efficient adaptation (uses P09, P14, U07 instruction
data, U06 compute).
U09 tools, agents, and retrieval (uses P19, P20, P21, U08 prompting,
U05 context).
U10 benchmarking and research methodology (uses P07, P10, P22, U01
metrics).
U11 reasoning and test-time compute (uses P14, P17, P22, U08 voting,
U10 eval).
U12 tokenization and multilinguality (uses P06, P13, U06 corpus).
U13 interpretability and social impacts (uses P10, P21, P22, U10 eval).
U14 multimodality and guest research (uses P11, P14, P22, U09
retrieval, U10 judges).
U15 project tutorials and open questions (uses P02, P12, P22, P24,
all units).

## Assessment files

- `keys/uNN_answers.md` , lesson assessment keys. Read after attempting
  the exercises.
- `keys/diagnostic_key.md` , prerequisite diagnostic key.
- `labs/uNN_lab_key.md` , lab keys with execution-verified outputs.
- `interview/uNN_key.md` , interview bank keys. Test mode only: do not
  read before a timed oral attempt.
- `interview/keys-transfer.md` , transfer set keys. Test mode only.
- `interview/keys-oral.md` , oral defense keys. Test mode only.

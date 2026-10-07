# CS224N , Natural Language Processing with Deep Learning (v2 pack)

Course builder scope: U01 through U08 plus the root identity set
(first builder, 2026-10-06). Second builder scope: U09 through U15
plus consolidation (2026-10-07). Coverage: 180/180 rows
TAUGHT+ASSESSED.
Independent audit: later, by a different agent. This build is not audited.

## Source-grounding note

This material is source-grounded on the Stanford CS224N Winter 2026
offering ("Natural Language Processing with Deep Learning", schedule at
`http://web.stanford.edu/class/cs224n/`, inspected 2026-10-06) plus
requested deep implementation branches. The build baseline is
October 6, 2026.

Honesty rules in force:

- Winter 2026 official schedule, assignments, tutorials and project
  descriptions anchor scope, 2024 public videos require separate edition
  tags. No claim of viewing restricted 2026 videos. Content derived from
  the 2024 public videos carries the tag `EDITION-2024`.
- Lecture videos for enrolled students sit behind Stanford Canvas login.
  I did not view them. Claim class `RESTRICTED-UNVIEWED` marks every
  leaf whose only source would be those videos.
- No leaf is called "covered" because it appears in a prompt or
  inventory. A leaf closes only when it is taught in a lesson, assessed
  in an exercise, lab, or interview bank, and recorded in
  `coverage_matrix.md` with evidence.
- Numbers in lessons are computed by committed scripts on this machine
  (numpy, matplotlib) or marked "Not in source". No benchmark, speedup,
  or test result is invented.
- Assignment learning objectives are extracted from official assignment
  titles and the course site. All practice is original equivalent work.
  No assessed work is solved.

## Directory layout

- `README.md` , this file.
- `state.md` , build state and RUN checkpoints.
- `source_manifest.md` , source register with inspection boundaries.
- `source_gaps.md` , unresolved sources and missing artifacts.
- `course_map.md` , Winter 2026 sessions mapped to units, with claim
  classes.
- `index.md` , learner navigation.
- `prerequisites.md` , unit prerequisites, local remediation, diagnostic.
- `notation_and_shapes.md` , shared symbols, shapes, units.
- `glossary.md` , terms with one-meaning definitions.
- `currentness.md` , baseline dates and update policy.
- `coverage_matrix.md` , 180 concept rows (U01-U15) with status and
  evidence.
- `visual_audit.md` , figure audit rows per unit (45 figures).
- `mastery_ledger.md` , per-concept learner mastery state (initial:
  untested).
- `errors.md` , error log for the build and for learner traps.
- `role_gap_map.md` , role rubrics versus unit coverage.
- `lessons/` , one lesson file per unit, 12 concepts each, 15-item
  contract per concept.
- `keys/` , answer keys for lesson assessments, kept separate from
  lessons. Also `keys/diagnostic_key.md`.
- `labs/` , one lab per unit: task file, numpy run script, lab key with
  execution-verified outputs.
- `interview/` , interview banks per unit, questions and keys separate.
  Also `interview/transfer-sets.md` + `keys-transfer.md` (10
  changed-scenario sets) and `interview/oral-defenses.md` +
  `keys-oral.md` (10 deep ladders x 8 follow-ups).
- `visuals/` , one compute/render script per unit plus computed PNG
  figures.
- `capstones/` , research replication + falsifiable extension (a) and
  applied/FDE capstone (b), each with runnable scripts and computed
  figures.
- `crash-course.md` , `cheatsheet.md` , synthesized from all 15 units.

## Shared prerequisite bridges

Prerequisite modules P01 through P24 live once at
`../shared/prerequisites/`. This course links them and never rebuilds
them. Each unit lesson carries its own local remediation block for the
diagnostic misses that matter most.

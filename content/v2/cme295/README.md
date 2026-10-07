# CME295 , Transformers and Large Language Models (v2 pack)

Course builder scope: U01 through U09 plus the root identity set and
consolidation (first builder: U01-U05 and identity, second builder:
U06-U09, capstones, transfer sets, oral defenses, crash course,
cheatsheet, root doc updates). Coverage of this build: 108/108 rows
TAUGHT+ASSESSED (U01-U09 x 12). Independent audit: later, by a
different agent. This build is not audited.

## Source-grounding note

This material is source-grounded on the Stanford CME295 Autumn 2026
offering ("Transformers and Large Language Models", nine lectures,
September 25 to December 9, 2026, midterm October 23, final December 9)
plus requested deep implementation branches. The build baseline is
October 6, 2026.

Honesty rules in force:

- The nine scheduled lectures map without omissions in `course_map.md`.
  Lecture titles and dates come from the official syllabus page inspected
  2026-10-06. Slide decks, video transcripts, and exams were NOT
  extracted, that extraction remains a subsequent artifact task. Every
  leaf therefore carries claim class OFFICIAL-SYLLABUS (topic named on
  the syllabus), REQUESTED-BRANCH (requested extension), or
  OFFICIAL-SOURCE (inspected artifact), and status PLANNED / SOURCE
  ATTRIBUTION PENDING until an inspected artifact verifies it.
- Exact Autumn 2026 nine-lecture syllabus anchors each unit, complete
  slides/videos/exam extraction remains a subsequent artifact task.
- No leaf is called "covered" because it appears in a prompt or
  inventory. A leaf closes only when it is taught in a lesson, assessed
  in an exercise, lab, or interview bank, and recorded in
  `coverage_matrix.md` with evidence.
- Numbers in lessons are computed by committed scripts on this machine
  (numpy, matplotlib) or marked "Not in source". No benchmark, speedup,
  or test result is invented.
- Assignment learning objectives are extracted from the public syllabus
  only. All practice is original equivalent work. No assessed work is
  solved. The midterm (Oct 23) and final (Dec 9) had not occurred at
  baseline, no exam content is claimed.

## Directory layout

- `README.md` , this file.
- `state.md` , build state and RUN checkpoints.
- `source_manifest.md` , source register with inspection boundaries.
- `source_gaps.md` , unresolved sources and missing artifacts.
- `course_map.md` , nine lectures mapped to units, with claim classes.
- `index.md` , learner navigation.
- `prerequisites.md` , unit prerequisites, local remediation, diagnostic.
- `notation_and_shapes.md` , shared symbols, shapes, units.
- `glossary.md` , terms with one-meaning definitions.
- `currentness.md` , baseline dates and update policy.
- `coverage_matrix.md` , 108 concept rows (U01-U09) with status and evidence.
- `visual_audit.md` , figure audit rows per unit.
- `mastery_ledger.md` , per-concept learner mastery state (initial: untested).
- `errors.md` , error log for the build and for learner traps.
- `role_gap_map.md` , role rubrics versus unit coverage.
- `lessons/` , one lesson file per unit, 12 concepts each, 15-item contract.
- `keys/` , answer keys for lesson assessments, kept separate from lessons.
- `labs/` , one lab per unit plus a lab key with execution-verified outputs.
- `interview/` , interview banks per unit (questions and keys separate),
  plus `transfer-sets.md` / `keys-transfer.md` and `oral-defenses.md` /
  `keys-oral.md` for cross-unit consolidation.
- `visuals/` , matplotlib render scripts and computed PNG figures.
- `capstones/` , research replication (executed, honest negative result)
  and applied/FDE capstone (hypothetical numbers labeled), each with a
  runnable script, a computed figure, and a render script.
- `crash-course.md` , nine-unit synthesis. `cheatsheet.md` , one page
  per unit.

## Shared prerequisite bridges

Prerequisite modules P01 through P24 live once at
`../shared/prerequisites/`. This course links them and never rebuilds them.
Each unit lesson carries a short local remediation block for its own
prerequisites.

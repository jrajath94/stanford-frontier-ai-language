# CS329H: Machine Learning from Human Preferences (v2 crash course)

Builder scope: U01-U05 (first builder) plus U06-U10 and consolidation (second builder, 2026-10-07). A separate agent audits the full course later (RUN 6).

## What this is

An independent crash course that teaches preference learning from first principles: choice data, random utility models, statistical estimation, experimental design, RLHF, DPO, active elicitation, dueling bandits, reward inversion, preference aggregation, and research practice. Each concept carries the full 15-item lesson contract, computed toys, original code, labs, figures, and interview banks with separated answer keys.

## Honesty note

Official Autumn 2026 eighteen-session/five-unit schedule verified. Only sessions dated through Oct 5 precede baseline. Later lectures are planned, independent theory may teach them with clear provenance. Sessions after Oct 5 carry the status PLANNED / SOURCE ATTRIBUTION PENDING until artifact-verified. Nothing in this course claims instructor authorship, and no unpublished course material is reproduced.

## Map

- `course_map.md`, session-to-unit map with claim classification and dates.
- `index.md`, every file in this course with one-line purpose.
- `lessons/`, U01-U10 lesson files, one per unit, 12 concepts each, full 15-item contract per concept.
- `answer_keys/`, lesson exercise keys, kept separate from lessons.
- `labs/`, one lab per unit plus `*_lab_keys.md` with execution-verified outputs.
- `interview/`, per-unit question banks and separate keys at the prompt quotas, plus transfer sets and oral defenses with separate keys.
- `visuals/`, committed matplotlib render scripts (`render_uXX_fNN.py`) and their PNG outputs.
- `prerequisites.md`, prerequisite graph, links to shared P01-P24 bridges, local remediation, diagnostic.
- `capstones/`, two runnable capstones with computed figures: (A) dueling TS replication + falsifiable extension, (B) annotation pipeline audit (HYPOTHETICAL numbers).
- `crash-course.md` and `cheatsheet.md`, synthesized from all ten units.

## Status

See `state.md` for the build checkpoint. See `coverage_matrix.md` for the 120 owned rows and their statuses (120/120 TAUGHT+ASSESSED). See `source_gaps.md` for unresolved gaps with evidence. See `errors.md` for known risks.

## Rules that bind this course

1. ASD-STE100: `~/workspace/skills/ste-lint/bin/ste_check.py`, zero hard fails on every deliverable including `.py` scripts.
2. `~/workspace/skills/watermarks-remover/bin/wm_clean.py` Layer A on every deliverable.
3. Visual system in `~/workspace/prompt-pack-v2/frontier_ai_prompt_pack_v2/visual_system_generic.md` binds every figure. Matplotlib-computed preferred. Every PNG is PIL-verified (IHDR/IDAT/IEND only) and metadata-stripped in the render script.
4. No invented numbers, timestamps, benchmarks, or test results. No exam dumps. No auth bypass.
5. Write only under `~/workspace/stanford-frontier-ai/v2-pack/cs329h/`.

# U07 answer key

Answers to the lesson exercises. Do not open before attempting.

## C01

- R1. Recall = 6/8 = 0.75. Precision = 6/20 = 0.30.
- R2. Recall rises to 7/8 = 0.875, precision falls to 7/40 =
  0.175. Wider net, more bycatch.
- R3. Splitting a fact across a chunk boundary, and
  overlapping chunks duplicating contradictory versions.

## C02

- R4. When the question needs one fact from one place. Hop 2
  then adds latency with no recall gain.
- R5. The hypothetical answer is fiction, its embedding
  points at fiction, retrieval follows. HyDE needs the
  model to be near the truth already.
- R6. 3 x (rewrite + retrieve) cost. Roughly 3x latency plus
  3x embedding cost versus single-hop.

## C03

- R7. A: 0.8*0.9 + 0.2*0.4 = 0.80. B: 0.8*0.3 + 0.2*0.8 =
  0.40. C: 0.8*0.5 + 0.2*0.5 = 0.50. A still wins, by more.
- R8. X: 1/61 + 1/64 = 0.0320. Y: 1/62 + 1/62 = 0.0323. Y
  wins by a hair, rank agreement beats one top rank.
- R9. When one signal dominates everywhere (pure part-number
  corpus), or the signals are near-identical. Then hybrid
  is two indexes for one opinion.

## C04

- R10. 0.90 x 0.80 = 0.72. The stages multiply, the chain is
  only as strong as its weakest stage.
- R11. 200 x 15 ms = 3000 ms = 3 s. Over budget for most
  interactive uses, shrink N.
- R12. When precision@5 is the business metric and the
  cross-encoder plateaued. The 10x cost must buy measured
  points.

## C05

- R13. Budget = 4000 - 500 - 500 = 3000. 3000 / 800 = 3.75,
  so 3 full docs fit.
- R14. The answer needs room. Without R the last doc
  crowds out the generation.
- R15. Put the key doc first regardless of rank, and
  compress long docs before packing.

## C06

- R16. {"name": "get_weather", "arguments": {"city":
  {"type": "string"}}} with city required.
- R17. A validation error naming the missing argument:
  "missing required argument: city". Nothing executes.
- R18. Execution can have side effects. Validation is the
  last cheap chance to reject a bad call.

## C07

- R19. The head 2000 tokens plus a truncation marker. The
  remaining 48000 stay in the store.
- R20. So the model can branch on the error type: retry on
  timeout, rephrase on bad arguments, escalate on denied.
- R21. That it has side effects and which tier it belongs
  to. The gate needs the truth to work.

## C08

- R22. Thought "need policy", Action search, Observation
  "found", Thought "summarize now", Action answer. Two
  tools max, here one.
- R23. The repeat guard or the step cap. Six thoughts with
  no action is the runaway signature.
- R24. Thoughts are cheap to fake. The actions are what
  touch the world, grade what matters.

## C09

- R25. 40 / 8 = 5 compactions, leaving 4 summaries plus the
  final window.
- R26. Codes, IDs, numbers, and exact quotes the task
  needs. Summaries paraphrase, paraphrase corrupts codes.
- R27. Explicit user facts ("allergic to penicillin") and
  repeated corrections. Both are durable.

## C10

- R28. Yes, 3% is under the 5% bar. The 3 are still worth
  reading, they may be a task class.
- R29. Whitelist the retry pattern: allow one identical
  retry after an error observation, then guard.
- R30. It runs per step forever. O(steps) would tax every
  long run, O(1) is invisible.

## C11

- R31. Red: irreversible, external, and hard to undo.
  send_email needs approval.
- R32. The string "denied" as a typed result. The model
  then explains or offers an alternative.
- R33. Each approval costs human minutes. Batching pays
  one interruption for many actions.

## C12

- R34. Retrieval bin. The doc was never in context, nothing
  downstream could work.
- R35. Retrieval first: 47 of 100 errors, the biggest
  lever. Then model, then tools.
- R36. Downstream stages inherit upstream failures. Fixing
  the model cannot recover a missing doc.

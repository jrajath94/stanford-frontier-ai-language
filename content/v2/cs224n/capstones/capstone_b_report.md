# Capstone B , RAG support assistant (applied/FDE)

All cost and latency numbers are HYPOTHETICAL and labeled as such.
The deliverable is the decision framework.

## Discovery

Support team, 200 tickets/day (hypothetical), 5000-doc corpus.
Baseline: keyword search, median resolution 12 min. Objective: cut
median resolution to 6 min with cited answers.

## Workflow baseline and objectives

Today: agent searches keywords, reads docs, writes a reply. Target:
RAG drafts the reply with citations, the agent approves. Objectives:
recall@5 >= 0.80, citation precision >= 0.80, p95 latency < 3s.

## Constraints and trust boundaries

Corpus updates hourly (rules out fine-tuning). Docs may hold PII:
redact before logging. Actions: read-only tools plus draft replies,
no send without human approval (U09 C10).

## Alternatives

Fine-tune: stale on hourly updates. Long-context: 5000 docs do not
fit. RAG chosen: fresh index, cited answers, cheap to update.

## Acceptance gates (toy eval, 200 labeled tickets)

recall@5: 0.84 [0.783, 0.884] (gate >= 0.80). Citation precision:
0.83 (gate >= 0.80). p95 latency: 2.4s (gate < 3s, hypothetical).
GATES: PASS. Script: `capstones/capstone_b_run.py`.

## Cost/latency (hypothetical)

Per ticket: embedding 0.02, rerank 0.05, generation 0.20, eval
amortized 0.03 = 0.30 units. Monthly at 200/day: 1800 units.
Latency: retrieval 0.4s + rerank 0.8s + generation 1.2s = 2.4s.
Figures: `capstones/fig_b1_latency.png`, `fig_b2_cost.png`.

## Rollout, monitoring, rollback

Shadow mode, then 10 percent, then 100 percent. Kill switch: one
config flip back to keyword search. Monitor: citation precision
drift, p95 latency, escalation rate. Owners: ML team (index),
support (eval set).

## Handoff and stakeholder defense

One slide: the gates pass, the bill is 1800 units/month
(hypothetical), the risk is citation drift (monitored), the
rollback is one flip. The business goal is faster resolution, not
model sophistication.

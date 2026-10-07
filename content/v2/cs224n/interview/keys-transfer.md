# Transfer set keys

## T1

Changes: nightly rebuild becomes incremental indexing with
freshness timestamps per chunk, plus a stale-chunk quarantine in
the retriever (U09 C04). Stays: the generator, the k, the citation
checker. New failure mode: partially updated indexes serve mixed
versions. The quarantine plus version stamps catch it.

## T2

Compute: Wilson on 33/40 vs 31/40 (0.82 vs 0.78). Intervals: about
[0.68, 0.91] and [0.63, 0.88]: heavy overlap, no win. Say: the
result is underpowered, the honest claim is "no evidence of a
difference", the next step is more items (n = 2401 for margin
0.02) or a cheaper directional test.

## T3

At p = 0.45 voting hurts more with n: majority of 11 is about
0.27, of 51 worse. "Vote harder" deepens the loss. Propose
instead: raise p first (better prompts, verification, C04), then
vote. Voting is a multiplier on p > 0.5, not a fix for p < 0.5.

## T4

Consequences: 4.5x token bill (C07), shattered rare words (C04),
shorter effective context. Mitigations: (1) byte fallback for the
script, cheapest. (2) continued pretraining on the script with a
rebalanced tokenizer, dearer. Measure fertility before and after.

## T5

Honest label: "a steerable correlate of confidence-like text"
(C04, C06). Next experiment: the intervention test, ablate the
direction and measure the behavior change (C03), plus the
mismatch set as the label's boundary.

## T6

Step by step: (1) watch with anchors (timestamps, quotes). (2)
re-teach C01-C09 from the video, upgrading claim classes to
official where supported. (3) fill C10's guest card with content.
(4) update the gap ledger (C12) and the coverage matrix notes.
Nothing upgrades on memory of the video: anchors or it stays.

## T7

Permissions: the send tool needs an explicit grant plus a
per-call approval gate (U09 C10). Retries: zero for send (no
retry on destructive actions). Stopping: the draft must validate
before the approval prompt appears. Refuse: fully automatic
sending with no human in the loop.

## T8

Biases: self-preference (the judge likes its own style), length
(the judge likes long), position (the judge likes first).
Controls: blind the judge to model identity, blind/truncate
length, swap positions and average. Report the blinded number.

## T9

Plan: sample 64, majority vote or self-consistency, with the p >
0.5 check first. Factual QA suits voting (verifiable answers).
Breaks: correlated errors (the model repeats one wrong answer),
latency (64 generations), and cost per query forever. The knee
(U11 C11): measure where the gains flatten and stop there.

## T10

It can prove: the plumbing works (shapes, loss falls, the
pipeline runs). It cannot prove: emergence, the scaling curve's
slope, or the 7B capability. Best predictor: the loss-vs-scale
trend on 2-3 small sizes (the shape of the curve), not any
single miniature number.

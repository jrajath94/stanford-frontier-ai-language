# Transfer sets , 10 changed-scenario sets

Each set changes one constraint from the taught units. Answer
closed-book, then check `interview/keys-transfer.md`. Test mode:
do not read the keys first.

## T1 , the corpus updates hourly (U09)

Your RAG index rebuilds nightly. The corpus now updates hourly and
answers go stale by noon. Redesign the pipeline. Name what changes,
what stays, and the new failure mode.

## T2 , the benchmark has 40 items (U10)

Your method beats the baseline 0.82 to 0.78 on 40 items. The
stakeholder wants to ship. What do you say, what do you compute,
and what is the smallest honest claim?

## T3 , p is below 0.5 (U11)

Self-consistency at p = 0.45, n = 11. A teammate proposes n = 51
to "vote harder". What happens, and what do you propose instead?

## T4 , a new script arrives (U12)

The deployment adds a language in a script the tokenizer never
saw. Fertility is 4.5. Name the three consequences and the two
mitigations, cheapest first.

## T5 , the probe fires on the proxy (U13)

Your "truth direction" probe hits 0.83, but a control shows it
fires on confident tone, not truth. What is the honest label,
and what is the next experiment?

## T6 , the guest video is released (U14)

The S20 guest talk video becomes public. Your U14 leaves are
requested-branch. What changes, step by step?

## T7 , the tool is destructive (U09)

The agent must send email, not just draft it. Redesign the loop:
permissions, approvals, retries, stopping. Name what you refuse
to automate.

## T8 , the judge is the model itself (U10)

You evaluate your model with itself as judge, win rate 0.68.
Name the three biases in play and the control for each.

## T9 , the budget is per query forever (U11)

Training budget is zero, but you may spend 64 samples per query
forever. The task is factual QA. Design the test-time plan and
name where it breaks.

## T10 , the miniature must convince (U15)

Your 44,928-param miniature must justify a 7B training run to a
skeptic. What can it prove, what can it not prove, and what is
the one experiment that best predicts the big run?

# Lab key , U07

Execution-verified outputs from `labs/u07_lab_run.py`, run 2026-10-06
(CPython, numpy 1.26.4, CPU). Re-run the script to confirm.

## T1 , Bradley-Terry

P = 0.7109, loss = 0.3412. A 0.9 gap is moderate evidence.

## T2 , DPO

Loss 0.6539. Near log 2: the 0.08 inside is uncertain, so the
gradient is large.

## T3 , KL

0.0253 nats. A small drift, the leash is slack.

## T4 , PPO

Unclipped 0.650, clipped 0.600. The trust region shaved 0.050 off
the step.

## T5 , reward model

Losses (0.3412, 0.7444, 0.2014), mean 0.429. The second pair costs
most: the model currently ranks it backwards.

## T6 , hacking

Proxy argmax 0, true argmax 1. The proxy's favorite is the truth's
runner-up.

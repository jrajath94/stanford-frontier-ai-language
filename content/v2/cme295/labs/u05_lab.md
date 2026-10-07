# U05 lab , preference optimization in code

Prerequisites: the U05 lesson. Runner: `u05_lab_run.py` (numpy, CPU,
deterministic). Work each task by hand first, then verify with the
runner. Answers and verified outputs: `u05_lab_key.md`.

## Task 1 , Bradley-Terry

1. Compute P and loss for (r_w, r_l) = (1.2, 0.4) and (0.5, 0.5).
2. State what the gap contributes.

## Task 2 , PPO objective

eps = 0.2.

1. Compute the clipped objective for (rho, A) = (2.0, 1.0),
   (0.5, -2.0), (1.0, 1.5).
2. State for each whether the clip binds.

## Task 3 , DPO margin

beta = 0.1.

1. Compute margin, P, loss for (logp_w, logp_l, logr_w, logr_l) =
   (-2.0, -3.5, -2.2, -2.3).
2. Repeat for (-1.0, -4.0, -2.0, -2.0). Explain the difference.

## Task 4 , KL

1. Compute KL([0.7, 0.3] || [0.5, 0.5]) in nats.
2. Compute KL([1, 0] || [0.5, 0.5]). Explain the jump.

## Task 5 , win rates

1. Compute both tie conventions for (W, L, T) = (62, 28, 10) and
   (70, 20, 10).
2. State why the convention must be reported.

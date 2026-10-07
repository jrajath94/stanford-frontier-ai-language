# Lab key , U04

Execution-verified outputs from `labs/u04_lab_run.py`, run 2026-10-06
(CPython, numpy 1.26.4, CPU). Re-run the script to confirm.

## T1 , hand unroll

h_1 = (0.762, 0.0), h_2 = (0.363, 0.762). The second coordinate of
h_1 is exactly 0 (tanh(0)), the first coordinate of h_2 is the
decayed trace 0.5 x 0.7616 passed through tanh.

## T2 , train RNN

Loss 2.8011 -> 0.2824, next-word accuracy 0.034 -> 0.862, training
perplexity 1.326. The accuracy starts near chance (1/14 = 0.071 for
uniform, 0.034 reflects the untrained bias) and ends at 0.862 by
memorizing six sentences.

## T3 , gradient norms

rho = 0.8: 0.8, 0.3277, 0.1074, 0.0115. rho = 1.2: 1.2, 2.4883,
6.1917, 38.3376. Exponential in T, both directions.

## T4 , forcing vs free

Forced input "sat", free input "on", KL(forced || free) = 2.9543.
One wrong word moves the state enough that the next-word
distributions barely overlap.

## T5 , clipping

Norm 29.025 -> 1.000 at cap 1.0, direction kept True. The rescale is
exact: new norm equals the cap.

## T6 , decoding

Greedy picks index 0. Top probability: T = 1.0 -> 0.575, T = 0.5 ->
0.828, T = 2.0 -> 0.406. Same logits, three risk levels.

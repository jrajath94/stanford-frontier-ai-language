# U01 answer keys

Keys for the lesson exercises. Keep separate from the lesson.

## C01

- E1: Pooled fraction = (0.69*200 + 0.54*200)/400 = 0.615. The naive preference estimate is 0.615, which mixes the two contexts.
- E2: With b = 0, P(y=1) = sigma(s) for every record, which is exactly the Bradley-Terry model on one pair. The log-likelihood and the MLE reduce to the C06 case.
- E3: With one context, s and b enter only as the sum s + b*c. Infinitely many (s, b) pairs give the same sum, so the split is arbitrary. You need at least two distinct context values.

## C02

- E1: sigma(0.3) = 1/(1+e^-0.3) = 0.574. sigma(0.8) = 0.690.
- E2: Design row for a comparison of a versus b with wording flag w: [1 at position a, -1 at position b, w at the wording slot], label y.
- E3: Randomize when you control the task and the context is a nuisance. Model when the data comes from the wild and the context varies beyond your control.

## C03

- E1: W = [[0,2,0],[1,0,3],[0,0,0]] for order A, B, C. Check: W[A,B]+W[B,A] = 3 = comparisons of that pair.
- E2: The log-likelihood is a sum over records of terms that depend only on the ordered pair and the outcome. Grouping terms by pair gives a sum over pairs of count-weighted terms. The record order never enters.
- E3: Add a third outcome code, e.g. y = 0.5 for a tie, with term 0.5*log p + 0.5*log(1-p). Or drop ties and report the drop rate.

## C04

- E1: sigma(0.8) = 0.690, sigma(3.0) = 0.953.
- E2: Any strictly increasing function keeps the order, e.g. s^3 or 2*s + 1. The predicted probabilities change but no pairwise order flips.
- E3: One comparison per pair identifies the sign of each gap (the order) but not its size. Cardinal gaps need repeated comparisons per pair.

## C05

- E1: l = sum y_i log sigma(d_i) + (1-y_i) log(1-sigma(d_i)) with d_i = s_a - s_b. d l/d s_a = sum (y_i - sigma(d_i)) over comparisons involving a, because d log sigma(z)/dz = 1 - sigma(z).
- E2: From zero, p = 0.5, gradient for A is 1 - 0.5 = 0.5. One step with lr 0.1 moves s_A to 0.05 and s_B to -0.05 before anchoring.
- E3: With all wins, y - p > 0 always, so every gradient step pushes the gap up with no opposing force. The likelihood has no finite maximizer. Fix with a prior or a cap.

## C06

- E1: l'(d) = 5(1-sigma(d)) - sigma(d) = 0 gives sigma(d) = 5/6, so d = log(5/1) = 1.609.
- E2: SE = sqrt(1/5 + 1/1) = sqrt(1.2) = 1.095. Wide because n = 6.
- E3: l(d) = 6 log sigma(d), increasing in d without bound. No finite maximizer exists. The data is separable.

## C07

- E1: P(a beats b | s + c) = sigma(s_a + c - s_b - c) = sigma(s_a - s_b). The c cancels in every pair term, so the full likelihood is unchanged.
- E2: Records only within {A,B} and only within {C,D}. Shift all of {A,B} by c1 and all of {C,D} by c2. Every observed pair keeps its gap. Two free constants, one per component.
- E3: Sum-zero gives s - mean(s), symmetric across items. Pin-first sets s_0 = 0, so other scores read as gaps versus the baseline item. Same probabilities both ways.

## C08

- E1: p_obs = P(true win and no flip) + P(true loss and flip) = (1-q) p + q (1-p).
- E2: p = sigma(1.5) = 0.818. p_obs = 0.9*0.818 + 0.1*0.182 = 0.754. Estimated gap = log(0.754/0.246) = 1.12, matching the simulated 1.11.
- E3: The observed win rate is one number. It equals (1-q) sigma(d) + q (1-sigma(d)), one equation in two unknowns (q, d). Without repeats or golds, a small gap with no noise looks identical to a big gap with noise.

## C09

- E1: At n = 20 the fit chases noise in the 20 training points, so train loss 0.423 is optimistic. Test loss 0.627 measures the same parameters on fresh data. The gap shows overfit.
- E2: log 2 = 0.693 nats is the loss of a coin-flip predictor. A model near 0.693 learned nothing.
- E3: Split by time: train on early records, test on later ones. Never let the same annotator-item pair appear on both sides when the goal is new-pair prediction.

## C10

- E1: Overhead = (100 repeats + 50 gold) / 1000 = 15%.
- E2: Drop below 0.85 gold accuracy, because golds have known answers and 0.85 leaves little margin above chance on hard items. Defend by showing held-out loss improves after the drop.
- E3: Gold accuracy high but self-consistency low means the annotator knows the gold answers (leaked or memorized) yet judges carelessly otherwise. Rotate golds and track consistency as the binding metric.

## C11

- E1: Careful judgments: P(short wins) = 0.60, gap = log(0.6/0.4) = 0.41 for short. Pressured: P(long wins) = 0.65, gap 0.62 for long. The sign flips. Training on pressured data inverts the careful preference.
- E2: Randomize annotators into pressure versus careful conditions, fit scores per batch, test the score difference against its standard error.
- E3: Match the deployment: if users decide under pressure, the pressured target is the honest one. If the product promises careful quality, use the careful target. Name the deployment first.

## C12

- E1: SE of the gap ≈ sqrt(0.87*0.13/600 + 0.59*0.41/600) = sqrt(0.000188 + 0.000403) = 0.024. The 0.28 gap is about 11 SE, far above noise.
- E2: Fit per-group scores. If the two score vectors agree within noise, the groups share preferences and the gap is noise (different q). If the vectors differ, it is genuine disagreement.
- E3: Group labels enable the audit but create a privacy surface: store them separately, aggregate before sharing, and never publish per-group rates that could re-identify members of small groups.

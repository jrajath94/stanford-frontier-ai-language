# cs329h capstones

Two capstones close the course. Both are runnable, and both carry computed figures. Provenance: PLANNED / SOURCE ATTRIBUTION PENDING.

## A. Research replication + falsifiable extension

Script: `replicate_dueling_ts.py`. Run: `python3 replicate_dueling_ts.py`.
Output: `results_cap_a.json`, `figures/cap_a_regret.png`.

Preregistered plan (frozen before the run): replicate dueling Thompson
sampling vs uniform dueling on a 4-arm Bradley-Terry bandit, then test an
extension (leader-focused dueling). Metric: mean cumulative strong regret
at T=200, seeds 0-19. Success: lower mean with non-overlapping 95 percent
CIs.

Results:
- Uniform: mean 17.24, CI [16.56, 17.92].
- Dueling TS: mean 3.11, CI [2.23, 3.99]. H1 SUPPORTED.
- Leader TS: mean 8.36, CI [6.86, 9.87]. H2 NOT SUPPORTED.

The extension failed: forcing the sampled leader to duel its least-tried
challenger adds strong regret (challenges are costly). This is an honest
negative result. The course uses it in U10 to teach negative-result
reporting: the failure is real data about the acquisition design, not a
wasted run.

## B. Applied / FDE capstone: annotation pipeline audit

Script: `annotation_pipeline_audit.py`. Run: `python3
annotation_pipeline_audit.py`. Output: `results_cap_b.json`,
`figures/cap_b_funnel.png`.

ALL NUMBERS ARE HYPOTHETICAL. Synthetic prompts, votes, and raters
(seed 7). No real vendor, no real annotators, no real costs. The script
models the audit procedure: funnel stages, bias metrics (position, length,
rater drift), four gates, and a SHIP / FIX / BLOCK decision.

Result: BLOCK. The drift gate failed (early accuracy 0.793, late 0.718, a drop of 0.075 above the 0.05 gate). Position bias 0.572 toward slot 0 is inside tolerance, agreement 0.72 passes, and size passes. The audit says fix
the rater-drift source before training.

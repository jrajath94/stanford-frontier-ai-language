# UU10 lab: research projects and oral mastery in code

Environment: Python 3 with NumPy and matplotlib. Capstone scripts at capstones/. Keys show verified numbers.

## Task 1: pre-analysis plan

Write the five-element plan for a new extension: 'duel the leader against a random challenger'. State the success and failure rules.

## Task 2: literature contract

Fill the reading contract (problem, theorem, proof sketch, setup) for dueling Thompson sampling.

## Task 3: hypothesis and consent

Write the extension hypothesis in I/O/D/K form. Then write the data-consent form for a hypothetical rater pool (purpose, voluntariness, withdrawal, data-use limits, retention).

## Task 4: method baselines

Implement uniform dueling, dueling TS, and the leader extension on the 4-arm BT bandit (strengths 1.2, 0.8, 0.3, -0.5), T=200, seeds 0-19. Report the three means.

## Task 5: uncertainty table

From the per-seed regrets, compute SD, SE, and 95 percent CIs for the three methods. Report the full table and state the H1/H2 verdicts.

## Task 6: reproducibility check

Run capstones/replicate_dueling_ts.py twice and diff the outputs. Report whether the JSON is identical.

## Task 7: contribution and reflection

Write the contribution statement (baseline, intervention, evidence, scope) and the integrity/reflection/impact paragraph for capstone A.

## Task 8: negative results and gaps

Write the four-part negative-result report for H2. Then write one open-source gap statement with a checkable 'done'.

# U02 lab: random utility in code

Environment: Python 3 with NumPy. Seed 0. Keep outputs. Keys show verified numbers.

## Task 1: stable softmax

Compute choice shares for utilities [1.0, 0.5, -0.5] with the max-subtraction trick. Verify the shares sum to 1 and survive adding a constant to all utilities.

## Task 2: Gumbel simulation

Simulate 20000 choices for utilities [1.0, 0.5, 0.0] with Gumbel shocks. Compare the histogram to the softmax formula. Report the max gap.

## Task 3: cycle detection

Build the majority tournament for A>B, B>C, C>A and detect the directed cycle with depth-first search. Report True/False.

## Task 4: five options

Compute shares for utilities [2.0, 1.0, 0.0, -1.0, -2.0]. Report all five.

## Task 5: truncated SVD

Build the 4x5 toy matrix from seed-0 factors (as in figure u02_f04). Reconstruct at rank 2. Report max absolute error.

## Task 6: fold-in

Using the task 5 item factors, fold in a new respondent with ratings 1.5 (item 0) and -0.5 (item 1) by least squares. Report r_new and the item-4 prediction. Diagnose the result.

## Task 7: heterogeneity screen

Simulate two groups of 500 with gaps +2.2 and -2.2. Report pooled and per-group win rates.

## Task 8: assumption checklist

On 300 simulated logit choices over 3 options, run: pair coverage (connectivity), cycle count on majority votes, and a heterogeneity screen (split-half win rates). Report pass/fail per clause.

## Task 9: counterexample battery

Run the naive single-score fitter on the 0.80 cycle data. Report the max residual. Confirm each of the four battery cases from U02-C12 fails naive and passes its repair.

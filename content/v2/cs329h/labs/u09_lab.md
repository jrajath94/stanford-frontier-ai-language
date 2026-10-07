# U09 lab: aggregation, impossibility, and fairness in code

Environment: Python 3 with NumPy and matplotlib. Seed 0 for the cycle simulation. Keep outputs. Keys show verified numbers.

## Task 1: rule tallies

Ballots A>B>C, B>C>A, C>A>B. Tally plurality, Borda (2/1/0), and the Condorcet winner. Report all three.

## Task 2: spoiler

Profile: 4x A>B>C, 3x B>A>C, 2x C>B>A. Report the plurality winner. Then add D: 2x D>A>B>C, 2x A>B>C>D, 3x B>A>C>D, 2x C>B>A>D. Report the new winner and the D-vs-B pairwise margin.

## Task 3: manipulation search

Exhaustively search 3-voter Borda profiles for a profitable misreport (tie-break: ballot order). Report the first found: profile, voter, misreport, outcomes.

## Task 4: margins and cycles

Compute the margin matrix for the 3x3 cycle profile. Then simulate cycle frequency under impartial culture: 20000 profiles, 11 voters, seed 0 with numpy legacy np.random.RandomState. Report both.

## Task 5: voting game

On the found manipulation profile, build the 2x2 game (voter 0: sincere/strategic, voter 1: sincere/bury-B). Report the outcome table and the Nash equilibrium.

## Task 6: jury numbers

Compute P(majority correct) for n=5 and n=101 voters at p=0.6. Report both.

## Task 7: subgroup slicing

Two groups of 500, accuracies 0.87 and 0.59. Compute pooled accuracy and disparity. Report all three.

## Task 8: single-peaked check

Candidates at 0, 0.5, 1.0, voter ideals 0.2, 0.4, 0.8. Build ballots by distance. Report the pairwise margins and the Condorcet winner.

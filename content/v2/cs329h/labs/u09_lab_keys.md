# U09 lab keys: execution-verified outputs

Seed 0 (cycle simulation). Pure Python tallies elsewhere.

## Task 1

Plurality {A:1, B:1, C:1} (tie). Borda {A:3, B:3, C:3} (tie). Condorcet: none.

## Task 2

Without D: A wins (4-3-2). With D: B wins (A2 D2 B3 C2). D vs B pairwise: 2-7 (D loses).

## Task 3

Profile (A>B>C, B>A>C, B>A>C). Voter 0 misreports A>C>B. Sincere Borda A4 B5 C0 -> B wins. Strategic A4 B4 C1 -> tie, ballot-order tie-break -> A wins. Profitable for voter 0 (A over B).

## Task 4

Margins: A>B +1, B>C +1, C>A +1 (cycle). Cycle frequency over 20000 profiles, 11 voters, seed 0 with numpy legacy np.random.RandomState: 0.081.

## Task 5

Outcomes: (sin,sin)->B (1,0), (sin,str)->A (0,1), (str,sin)->A (0,1), (str,str)->A (0,1). Nash: voter 0 strategic (dominant).

## Task 6

n=5: 0.6826. n=101: 0.9791.

## Task 7

Pooled 0.73. Group A 0.87, group B 0.59. Disparity 0.28.

## Task 8

Margins: B>A +1, B>C +1, A>C +1. Condorcet winner: B (median voter top).

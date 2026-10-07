# U09 interview bank: questions

## Breadth (6)

B1. Define a voting rule and name three.
B2. State Arrow's four conditions.
B3. State Gibbard-Satterthwaite in one sentence.
B4. What is a Condorcet cycle?
B5. Why can cardinal utilities not be summed across people without a common scale?
B6. State the Condorcet jury theorem's two key assumptions.

## Deep ladders (2 x 8)

### Ladder 1: Arrow

L1.1 Define: write the four conditions.
L1.2 Toy: show the spoiler flip on the plurality example.
L1.3 Derive (intuition): sketch the swing-voter argument.
L1.4 Implement and complexity: code the three tallies, state the cost.
L1.5 Compare: which axiom does each of plurality, Borda, Condorcet drop?
L1.6 Debug: someone cites Arrow to claim 'all voting is rigged'. Diagnose the error.
L1.7 Critique: what escapes the theorem's scope?
L1.8 Design: design the rule-selection memo for a real election.

### Ladder 2: strategic voting

L2.1 Define: what is a profitable misreport?
L2.2 Toy: walk the found Borda manipulation.
L2.3 Derive: why does the misreport help (show the totals)?
L2.4 Implement and complexity: code the manipulation search, state the cost.
L2.5 Compare: deterministic vs randomized rules on strategyproofness.
L2.6 Debug: the search finds no manipulation. Does that prove strategyproofness?
L2.7 Critique: the game assumes common knowledge. What breaks without it?
L2.8 Design: design the measurement of manipulation frequency over random profiles.

## Analytical exercises (2)

A1. Compute the Condorcet jury probabilities for n=5 and n=101 at p=0.6, and prove the direction reverses for p < 0.5.
A2. Work the interpersonal-comparison flip: raw sums vs normalized sums on the toy, and explain which step is meaning-preserving.

## Implementation / debug (1)

D1. The aggregation below elects A. A losing candidate C drops out, and suddenly B wins though no ballot changed. The code is correct. Name the violated axiom and the smallest principled fix.

```python
winner = plurality(ballots)
```

## Changed-constraint scenarios (2)

S1. Voters arrive online (not a fixed electorate). What breaks in the profile model?
S2. The election must be auditable by non-experts. Which rules survive and what is lost?

## Research critique (1)

R1. A paper proposes a new voting rule and proves it satisfies 'all Arrow axioms'. List three objections and state what you would check first.

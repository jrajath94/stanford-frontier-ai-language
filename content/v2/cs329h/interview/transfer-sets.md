# Transfer sets: cs329h

Ten changed-scenario sets. Each set changes one constraint from a unit and asks what the mechanism does under the change. Full keys in keys-transfer.md. Provenance: PLANNED / SOURCE ATTRIBUTION PENDING throughout.

## T1. Expensive duels (U06, U07)

Scenario: each duel costs 1 unit of budget. You have 50 duels, not 200 rounds. The dueling TS loop is unchanged otherwise.

Q1. How do you adapt the acquisition to the hard budget? State the new objective.
Q2. What breaks in the regret analysis when duels cost budget instead of time?

## T2. Two candidates (U09)

Scenario: the election has exactly 2 candidates. Arrow's theorem assumed 3 or more outcomes.

Q1. Which voting rule now satisfies all four Arrow conditions? Prove it in one paragraph.
Q2. A third candidate enters mid-election (the T2-to-T1 direction). What is the minimal repair to the rule choice?

## T3. Biased annotators (U01, U08)

Scenario: annotators pick the first-listed response 60 percent of the time regardless of content. You fit Bradley-Terry on 1000 pairs.

Q1. What happens to the estimated score gap? Derive the direction of the bias.
Q2. Name two fixes: one at data-collection time, one at modeling time. State what each assumes.

## T4. Adversarial human (U06)

Scenario: in the assistance game, the human's objective is the opposite of the agent's. The agent still observes the human's actions.

Q1. What is the value of observing the human now? Work the tea/coffee toy with reversed objectives.
Q2. What is the minimal change to the assistance formulation that handles adversarial humans?

## T5. Off-policy reward data (U05, U08)

Scenario: the preference pairs were collected under an old policy. You train the reward model and optimize a new policy against it.

Q1. Where does the off-policy gap enter the RLHF pipeline? Name the two failure points.
Q2. Propose one guardrail that detects the failure before deployment. State what it measures.

## T6. Correlated jury (U09)

Scenario: 101 voters each correct with probability 0.6, but their errors are correlated (they all watch the same misleading broadcast).

Q1. What happens to the jury theorem's conclusion? Compute nothing. Argue from the assumptions.
Q2. What is the minimal repair to the aggregation that accounts for correlation?

## T7. Three-query budget (U06, U07)

Scenario: the user answers at most 3 elicitation queries, then leaves. You must pick all 3 up front (no adaptation).

Q1. How do you choose the 3 queries? State the objective and the algorithm.
Q2. What is lost relative to the adaptive policy? Name the loss precisely.

## T8. Bad reference policy (U05)

Scenario: in DPO, the reference policy is terrible (near-uniform over bad responses). The preference data is clean.

Q1. What happens to the DPO optimum? Trace the KL term's effect.
Q2. What is the minimal repair: fix the reference, the data, or the loss? Justify the choice.

## T9. Unknown scales (U09)

Scenario: voters report cardinal scores, but each voter's scale is unknown (one uses 0-10, another uses 0-100). You must aggregate.

Q1. Show the aggregation failure on a two-voter toy. What changes under rescaling?
Q2. Name two escapes and state which Arrow-style assumption each drops.

## T10. The extension wins (U10)

Scenario: you re-run capstone A and the leader-focused extension beats vanilla dueling TS (mean 2.10 vs 3.11, non-overlapping CIs).

Q1. Write the updated contribution statement (baseline, intervention, evidence, scope).
Q2. What follow-up experiment distinguishes "challenges are cheap here" from "the focus rule is genuinely better"? Design it.

---
page_id: cme295-l08
course_slug: cme295
course_name: "CME295: Transformers and Large Language Models"
course_order: 5
order: 8
nav: "L08 · Evaluation"
title: "Lecture 8: Evaluating LLMs"
summary: "Why free-form output resists measurement, told as a ladder: human ratings with the chance-agreement arithmetic that breaks them, rule metrics on a worked BLEU toy, LLM-as-a-judge with its three biases demonstrated, factuality checked fact by fact to 0.6, and benchmarks read like an adult."
date: "2025-11-21"
instructor: "Afshine Amidi, Shervine Amidi"
offering: "Autumn 2025"
duration: "1:49:14"
video_id: 8fNP4N46RRo
video_title: "CME295 Lecture 8, Autumn 2025"
video_caption: "Original lecture. Human ratings, rule-based metrics, LLM-as-a-judge, benchmarks, and agent evaluation."
sources:
  - tag: video
    label: "Lecture 8 slides (PDF), CME295 Autumn 2025"
  - tag: paper
    label: "Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (2023)"
    url: https://arxiv.org/abs/2306.05685
  - tag: paper
    label: "Jimenez et al., SWE-bench: Can Language Models Resolve Real-World GitHub Issues? (2023)"
    url: https://arxiv.org/abs/2310.06770
concepts: [evaluation, human-ratings, inter-rater-agreement, cohens-kappa, fleiss-kappa, meteor, bleu, rouge, llm-as-judge, judge-biases, position-bias, verbosity-bias, self-enhancement-bias, structured-output, factuality, factscore, agent-evaluation, mmlu, aime, piqa, swe-bench, harmbench, tau-bench, pass-hat-at-k, pareto-frontier, goodharts-law, contamination]
---

## The problem: what does "evaluate the LLM" mean

"Evaluate the LLM" can mean latency, price, or uptime. This lecture
means one thing: **output quality**. How good is the actual
response?

The difficulty is structural. The LLM outputs free-form text:
natural language, code, math. No universal metric covers all of it.
The lecture climbs a ladder of approximations. Each rung fixes the
previous rung's flaw and introduces a new one.

![Eval scope](assets/l08-eval-scope.svg "This lecture: output quality. Three rungs: humans, rules, judges. Stanford Frontier AI.")

## The key question

Text has no single right answer. How do you score a model when the
output space is infinite and every answer differs?

## Rung 1: human ratings, the unaffordable ideal

The ideal: a human rates every output. Collect the ratings, quantify
the model. Two problems kill it.

**Subjectivity.** "What birthday gift should I get?" answered with "a
teddy bear is almost always a sweet gift" splits raters: one calls it
useful, another calls it vague
([08:22](https://www.youtube.com/watch?v=8fNP4N46RRo&t=502s)). The fix
is **inter-rater agreement**: measure how consistently raters judge,
with clear guidelines and alignment sessions when they drift.

**Cost and speed.** Rating a thousand outputs takes days and real
money. The ideal does not scale.

The naive agreement metric is the **agreement rate**: the fraction of
times two raters give the same verdict
([09:18](https://www.youtube.com/watch?v=8fNP4N46RRo&t=558s)). Watch
it lie. If rater A says "good" with probability P_A and rater B with
P_B, independently, chance agreement is:

**P(agree) = P_A * P_B + (1 - P_A) * (1 - P_B)**

Two scenarios on the same observed 85% agreement:

```ascii
scenario 1: P_A = P_B = 0.5 (balanced)
  chance agreement = 0.25 + 0.25 = 0.50
  observed 0.85 vs chance 0.50: real signal.

scenario 2: P_A = P_B = 0.9 (almost everything is "good")
  chance agreement = 0.81 + 0.01 = 0.82
  observed 0.85 vs chance 0.82: barely above random guessing.
```

Same 85%, opposite conclusions. A raw percentage cannot tell them
apart. The fix is chance-corrected metrics: **Cohen's kappa** for two
raters
([14:56](https://www.youtube.com/watch?v=8fNP4N46RRo&t=896s)),
**Fleiss's kappa** for many, **Krippendorff's alpha** when data is
missing ([16:30](https://www.youtube.com/watch?v=8fNP4N46RRo&t=990s)).
Cohen's kappa is (observed - chance) / (1 - chance): scenario 1 gives
(0.85 - 0.50) / 0.50 = 0.70. Scenario 2 gives (0.85 - 0.82) / 0.18 =
0.17. Positive means better than chance. That is the health metric to
track. Low kappa usually means the rubric is ambiguous, not that the
raters are careless: hold an agreement session and fix the rubric.

![Human ratings](assets/l08-human.svg "Ideal but slow, subjective, drifty. Stanford Frontier AI.")
![Kappa](assets/l08-kappa.svg "Agreement rate lies. Chance-corrected metrics tell the truth. Stanford Frontier AI.")

> [!QA]
> Q: Why is chance agreement not zero?
> A: Because agreement has two paths: both say yes or both say no.
> Even coin-flip raters land on the same side half the time. Any
> agreement metric must subtract this baseline, or it rewards raters
> for the base rate instead of their judgment.
> Follow-up: What do you do when kappa is low?
> A: Hold an agreement session: raters discuss disagreements and
> align on the guidelines. Low kappa usually means the rubric is
> ambiguous, not that the raters are careless. Fix the rubric.

## Rung 2: rule-based metrics

Humans write reference outputs once. A metric compares model outputs
to the reference forever. The metric should tolerate paraphrase: the
same content in different words should still score well. Three
classics:

- **BLEU**
  ([25:16](https://www.youtube.com/watch?v=8fNP4N46RRo&t=1516s),
  bilingual evaluation understudy): precision over n-gram matches,
  with a brevity penalty so tiny outputs cannot game it.
  Translation's workhorse.
- **ROUGE**
  ([26:17](https://www.youtube.com/watch?v=8fNP4N46RRo&t=1577s)):
  recall-flavored overlap, many variants. Summarization's workhorse.
- **METEOR**
  ([21:06](https://www.youtube.com/watch?v=8fNP4N46RRo&t=1266s)): an
  F-score over unigram matches times (1 - penalty), where the
  penalty punishes word-order differences via the ratio of
  contiguous matched chunks to matched unigrams. Handles synonyms
  and stems. Carries arbitrary hyperparameters.

Work BLEU on a toy. Reference: "the teddy bear is cute and soft".
Candidate: "the bear is cute".

```ascii
unigram precision: "the", "bear", "is", "cute" all in reference = 4/4 = 1.0
bigram precision:  "the bear" (no), "bear is" (yes), "is cute" (yes) = 2/3 = 0.67
brevity penalty:   candidate length 4 < reference length 7
                   BP = exp(1 - 7/4) = exp(-0.75) = 0.47
BLEU-ish score:    0.47 * (1.0 * 0.67)^(1/2) = 0.47 * 0.82 = 0.39
```

The candidate dropped "teddy" and "soft" and the brevity penalty
punishes it: 0.39 despite perfect unigram precision. Now the real
problem: a candidate with identical meaning but different words
("the stuffed animal is adorable and cuddly") scores near zero on
n-gram overlap. All three metrics share the flaws: paraphrases score
badly, correlation with human judgment is weak, and you still need
human-written references to start. The teddy-bear comfort sentence
in three different wordings defeats all of them.

![Rule metrics](assets/l08-rule-metrics.svg "BLEU, ROUGE, METEOR. Compare to a reference. Punish paraphrase by accident. Stanford Frontier AI.")

## Rung 3: LLM-as-a-judge

The lecture's key method
([28:08](https://www.youtube.com/watch?v=8fNP4N46RRo&t=1688s)): use an
LLM to grade outputs. Inputs: the prompt, the model response, and the
grading criteria. Outputs: a **rationale** and a **score**. Two
design choices matter:

- **Rationale before score**
  ([32:15](https://www.youtube.com/watch?v=8fNP4N46RRo&t=1935s)).
  Like chain-of-thought, externalizing the reasoning first
  empirically improves the judgment.
- **Binary scale**
  ([29:36](https://www.youtube.com/watch?v=8fNP4N46RRo&t=1776s)).
  Pass/fail beats fine-grained scales: easier for the judge, easier
  for humans to agree on, less noise.

No reference text is needed: the judge brings pretraining knowledge.
Parsing is guaranteed with **structured output**
([34:40](https://www.youtube.com/watch?v=8fNP4N46RRo&t=2080s)):
constrained decoding (Lecture 3) forces the rationale-plus-score
shape. Two modes exist: single-response grading (good or bad?) and
pairwise (A better or B?), the latter generating synthetic
preference labels for reward-model training (Lecture 5's data, made
by machines).

![Judge](assets/l08-judge.svg "Prompt plus response plus criteria in. Rationale, then score, out. Stanford Frontier AI.")

Three biases to watch, none exhaustive. Each demonstrated:

- **Position bias**
  ([38:48](https://www.youtube.com/watch?v=8fNP4N46RRo&t=2328s)).
  The judge prefers whichever answer came first. The toy: A and B
  graded as (A, B) gives "A wins". Graded as (B, A) gives "B wins".
  Same answers, opposite verdicts. Fix: ask both orders, take the
  majority.
- **Verbosity bias**
  ([40:31](https://www.youtube.com/watch?v=8fNP4N46RRo&t=2431s)).
  The judge prefers longer answers. The toy: a 40-word correct-ish
  answer beats a 10-word correct answer, because length reads as
  thoroughness. Fix: say so in the guidelines, show counter-examples,
  or penalize length.
- **Self-enhancement bias**
  ([42:29](https://www.youtube.com/watch?v=8fNP4N46RRo&t=2549s)).
  The judge prefers its own outputs: they look like what it would
  produce, which it already believes is good. Fix: use a different
  model as judge, ideally a bigger one with strong reasoning.

Best practices: crisp guidelines, binary scale, rationale first,
bias mitigations, low temperature (0.1-0.2) for reproducibility
([51:24](https://www.youtube.com/watch?v=8fNP4N46RRo&t=3084s)), and
calibration against human ratings. The judge is a proxy for human
judgment: do not overoptimize the proxy until it diverges from the
truth.

![Biases](assets/l08-biases.svg "Position, verbosity, self-enhancement. Each has a fix. Stanford Frontier AI.")

> [!QA]
> Q: Why must the judge differ from the generator?
> A: Self-enhancement bias: a model rates its own outputs higher
> because they look like what it would produce, which it already
> believes is good. A separate judge breaks the loop. Bigger and
> stronger-reasoning judges resist being fooled by fluent-but-wrong
> answers.
> Follow-up: If the judge is imperfect, why not skip to human
> ratings?
> A: Cost. The judge screens everything cheaply. Humans audit a
> sample and calibrate the judge. The pipeline is judge-at-scale
> plus human-at-the-margin, not either alone.

## Factuality, fact by fact

Factuality resists binary grading: a paragraph can be mostly right
with two errors. The lecture's method: decompose, check, aggregate.

1. **Extract.** One LLM call turns the text into a list of atomic
   facts
   ([56:23](https://www.youtube.com/watch?v=8fNP4N46RRo&t=3383s)).
2. **Check each.** Binary verdict per fact, using RAG or web search
   against a knowledge base (Lecture 7's machinery, reused).
3. **Aggregate.** Weighted mean over facts. Weights alpha_i capture
   importance.

The example
([54:23](https://www.youtube.com/watch?v=8fNP4N46RRo&t=3263s)):
"Teddy bears, first created in the 1920s, were named after President
Theodore Roosevelt after he proudly wanted to shoot a captured bear."
Work it:

```ascii
fact 1: teddy bears were first created in the 1920s        -> FALSE (it was the 1900s)
fact 2: named after President Theodore Roosevelt           -> TRUE
fact 3: after an incident with a captured bear             -> TRUE
fact 4: he proudly wanted to shoot it                      -> FALSE (he refused)

weights: [0.2, 0.3, 0.2, 0.3]
score = (0*0.2 + 1*0.3 + 1*0.2 + 0*0.3) / (0.2+0.3+0.2+0.3) = 0.5 / 1.0 = 0.5
```

The lecture's worked score on this passage is 0.6 with its own
weighting. The mechanism is the same either way: nuance without
hand-waving, two of four facts wrong, caught individually.

![Factuality](assets/l08-factuality.svg "Extract facts, check each, weighted aggregate. The example scores 0.6. Stanford Frontier AI.")

## Evaluating agents

Agents fail at three stages, seven ways (Lecture 7's taxonomy,
applied as an evaluation lens):

**Prediction:** the punt (needed a tool, did not call one,
[63:19](https://www.youtube.com/watch?v=8fNP4N46RRo&t=3799s)), tool
hallucination (called `find_bear` instead of `find_teddy_bear`,
[66:37](https://www.youtube.com/watch?v=8fNP4N46RRo&t=3997s)), wrong
tool, wrong arguments. **Execution:** buggy tool output, or no output
at all: for actions, silence invites false confirmation, so return
even an empty JSON
([77:57](https://www.youtube.com/watch?v=8fNP4N46RRo&t=4677s)) rather
than nothing. **Synthesis:** the model fails to use a good tool
result.

The evaluation lesson: categorize failures methodically and fix them
in groups. One-off debugging does not scale. Taxonomies do.

![Failure taxonomy](assets/l08-agent-failures.svg "Seven ways agents fail. Categorize in groups, fix in groups. Stanford Frontier AI.")

## The benchmark families

Five families cover what benchmarks actually test:

- **Knowledge: MMLU**
  ([85:12](https://www.youtube.com/watch?v=8fNP4N46RRo&t=5112s),
  Massive Multitask Language Understanding). ~60 tasks across law,
  medicine, and everyday topics, four-choice questions. Measures how
  well pretraining stuck. Answer extraction is hard-coded: output the
  letter.
- **Reasoning: AIME and PIQA.** AIME
  ([89:41](https://www.youtube.com/watch?v=8fNP4N46RRo&t=5381s)):
  hard high-school math with integer answers, LLM-friendly by
  construction. PIQA
  ([91:06](https://www.youtube.com/watch?v=8fNP4N46RRo&t=5466s),
  Physical Interaction QA): 20k two-choice common-sense questions
  about the physical world (the hairnet vacuum example).
- **Coding: SWE-bench**
  ([94:12](https://www.youtube.com/watch?v=8fNP4N46RRo&t=5652s)).
  Real GitHub issues from popular Python repos, with the tests from
  the fixing PR. The model writes a patch. Tests before and after
  decide. Test-driven development as a benchmark.
- **Safety: HarmBench**
  ([98:07](https://www.youtube.com/watch?v=8fNP4N46RRo&t=5887s)).
  Standard, copyright, contextual, and multimodal harm categories. A
  trained classifier judges attempts, and an attempt counts if the
  model tries, even if it fails from low capability. Provider
  policies differ, so safety numbers do not compare across labs the
  way accuracy does.
- **Agents: tau-bench**
  ([101:11](https://www.youtube.com/watch?v=8fNP4N46RRo&t=6071s),
  tool-agent-user). Airline and retail domains with tools, policies,
  and tasks. An LLM simulates the user. Success is measured on
  database state. The metric is **pass-hat@k**
  ([103:46](https://www.youtube.com/watch?v=8fNP4N46RRo&t=6226s)):
  the probability that all k attempts succeed, because automation
  needs reliability, not luck.

![Benchmarks](assets/l08-benchmarks.svg "Knowledge, reasoning, coding, safety, agents. Five families, five designs. Stanford Frontier AI.")

Constrained formats dominate: multiple choice, integer answers, test
suites. Free-form plus LLM-judge adds a second error layer, so
benchmark designers avoid it where they can.

> [!QA]
> Q: Why do benchmarks use constrained formats instead of free-form
> grading?
> A: To remove the judge as an error source. An LLM judge is itself
> imperfect and biased, so free-form benchmarks stack two
> uncertainties: the model's and the judge's. Multiple choice,
> exact integers, and test suites give hard-coded extraction and
> deterministic scoring.
> Follow-up: Does that make benchmarks less realistic?
> A: Yes, deliberately. Benchmarks trade realism for comparability.
> That is why the lecture pairs them with Chatbot Arena (realistic,
> noisy) and personal testing (realistic, specific). One leg never
> stands.

## Reading results like an adult

Benchmarks profile a model. They do not crown one. Four cautions:

- **Pareto frontier**
  ([107:36](https://www.youtube.com/watch?v=8fNP4N46RRo&t=6456s)).
  Plot performance against price (or safety, or context length). The
  toy: model A scores 80 at $1 per million tokens. Model B scores 76
  at $0.10 per million. B is 10x cheaper for 95% of the quality: B
  sits on the frontier, A does not, for cost-sensitive use. The
  frontier is the set of best-per-dollar models. The lecturer's
  personal picks (Sonnet for code, Gemini Flash for cheap-and-fast)
  are personal, not universal.
- **Contamination**
  ([107:49](https://www.youtube.com/watch?v=8fNP4N46RRo&t=6469s)).
  Benchmarks leak into training data. Defenses: hashes, blocklists,
  fresh tests the model cannot have seen.
- **Goodhart's law**
  ([108:26](https://www.youtube.com/watch?v=8fNP4N46RRo&t=6506s)).
  "When a measure becomes a target, it ceases to be a good measure."
  Optimized benchmarks stop measuring.
- **Try it yourself.** Chatbot Arena adds real-usage signal, but the
  final test is your own tasks on your own data.

![Pareto](assets/l08-pareto.svg "Pareto, contamination, Goodhart. Then try the models yourself. Stanford Frontier AI.")

## Mapping back: each rung answers the previous rung's flaw

| Rung | Fixes | Introduces |
|---|---|---|
| Human ratings | Ground truth, the ideal | Subjectivity and cost. Agreement rate lies (0.85 can be 0.70 or 0.17 kappa) |
| Rule metrics | Cheap, deterministic | Paraphrase blindness: the BLEU toy scores 0.39 on a good answer. Perfect paraphrases score near zero |
| LLM-as-a-judge | Scales, no references | Position, verbosity, self-enhancement biases. Each demonstrated, each with a fix |
| Factuality decomposition | Binary grading is too blunt | Extraction and checking are themselves LLM steps with error |
| Benchmark families | Comparability | Constrained formats trade realism for determinism |
| Adult reading | Numbers that crown | Pareto, contamination, Goodhart: profile, do not crown |

## The honest price

Every rung is a proxy, and every proxy has a price. Humans are
truthful and unaffordable. Rules are cheap and blind. Judges scale
and are biased. Benchmarks compare and distort. The lecture's
honest position: use all of them, trust none of them alone, and keep
a human at the margin calibrating the judge. The moment you
overoptimize any proxy, Goodhart collects.

## Recap: the whole lesson on one screen

The story in eight steps. Each step answers the one before it.

1. **Evaluation means output quality.** Free-form text resists
   measurement. The lecture climbs a ladder of approximations.
2. **Humans are ideal but unaffordable.** Subjective and slow. The
   agreement rate lies: observed 0.85 is kappa 0.70 at balanced
   base rates and 0.17 at 90/10. Chance-correct with kappa, Fleiss,
   Krippendorff. Fix the rubric when kappa drops.
3. **Rule metrics compare to a reference.** BLEU: precision plus
   brevity penalty (the toy scores 0.39). ROUGE: recall-flavored.
   METEOR: F-score with ordering penalty. All punish paraphrase and
   correlate weakly with humans.
4. **Let an LLM grade.** Prompt plus response plus criteria in.
   rationale then score out. Binary pass/fail. Structured output
   guarantees the parse. Pairwise mode makes synthetic preference
   labels.
5. **Judges are biased. Correct them.** Position: both orders, take
   the majority. Verbosity: guidelines, counter-examples, length
   penalty. Self-enhancement: a different, bigger judge. Low
   temperature (0.1-0.2). Calibrate against humans.
6. **Check factuality fact by fact.** Extract atomic facts, verify
   each binary with RAG or search, aggregate with importance
   weights. Two of four facts wrong, caught individually.
7. **Five benchmark families.** MMLU for knowledge, AIME/PIQA for
   reasoning, SWE-bench for coding, HarmBench for safety, tau-bench
   for agents with pass-hat@k (all k must succeed).
8. **Read numbers like an adult.** Pareto: best per dollar. 10x
   cheaper at 95% quality sits on the frontier. Contamination:
   hashes, blocklists, fresh tests. Goodhart: optimized measures
   stop measuring. Then try the models yourself.

## Official sources and further reading

**Official:**
- Lecture 8 recording (YouTube): timestamped above.
- Lecture 8 slides (PDF), CME295 Autumn 2025.
- Zheng et al., "LLM-as-a-Judge" (2023):
  https://arxiv.org/abs/2306.05685 — MT-Bench and Chatbot Arena.
- Jimenez et al., "SWE-bench" (2023):
  https://arxiv.org/abs/2310.06770 — issues plus tests.

**Further reading:**
- Banerjee and Lavie, "METEOR" (2005).
- Papineni et al., "BLEU"
  (2002). Lin, "ROUGE" (2004).
- Cohen (1960): kappa.
- Fleiss (1971)
- Krippendorff (2004).
- Mazeika et al., "HarmBench" (2024).
- Tau-bench authors (2024).
- Min et al., "FActScore" (2023): atomic-fact factuality checking.

**Caveats from these sources.** Judge biases listed are not
exhaustive. The lecture says so explicitly. HarmBench's classifier
judge can itself err. Safety benchmarks encode provider policies, so
cross-lab comparison is limited. tau-bench's simulated user is
itself an LLM.

## Connections to the other courses

- **CS336 L12:** evaluation: perplexity, benchmarks, and
  contamination at systems depth.
- **CS224N L11:** evaluation from the NLP side.
- **CME295 L05:** pairwise judging as synthetic preference data.
- **CME295 L06:** reasoning benchmarks (AIME, HumanEval) as
  verifiable rewards.
- **CME295 L07:** agent failure modes meet their evaluation here.

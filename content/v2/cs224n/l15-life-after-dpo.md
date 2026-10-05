---
page_id: cs224n-l15
course_slug: cs224n
course_name: "CS224N: NLP with Deep Learning"
course_order: 4
order: 15
nav: "L15 · Life After DPO"
title: "Lecture 15: Life After DPO (Bridge)"
summary: "The alignment moment after DPO: lab-scale preference data, online versus offline, self-rewarding models, beyond-pairwise methods, and the open questions. Deep mechanics live in CS336."
instructor: "Nathan Lambert"
offering: "Spring 2024"
duration: "1:09:00"
video_id: dnF463_Ar9I
video_title: "Lecture 15: Life After DPO"
video_caption: "Invited talk. Nathan Lambert (AI2) on the state of alignment research after DPO."
concepts: [dpo, alignment, preference-data, online-rlhf, offline-rlhf, self-rewarding, kto, reward-model]
sources:
  - tag: video
    label: "Lecture 15 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=dnF463_Ar9I
  - tag: notes
    label: "Official subtitle transcript"
  - tag: paper
    label: "Rafailov et al., Direct Preference Optimization (2023)"
    url: https://arxiv.org/abs/2305.18290
---

> [!NOTE]
> **Bridge lesson.** This invited talk frames the alignment moment after
> DPO. For deep mechanics, follow the links:
> [CS336 L15](../cs336/l15-post-training.html) (post-training systems),
> [CS336 L16](../cs336/l16-rlvr.html) (RL with verifiable rewards). This
> lesson never re-explains what those cover.

## How to read this lesson

This lesson has two levels. **Level 1 (Core)** sets the scene: the speaker,
the data scale, and the online/offline distinction. **Level 2 (Deep)** covers
self-rewarding models, beyond-pairwise methods, and the open questions.

## Level 1: The moment after DPO

**Nathan Lambert**: PhD at UC Berkeley, then HuggingFace, now AI2
([00:26](ts:00:26)). RL background: robots first, language models after.
His talk: "Life after DPO." DPO was "the story of last year" (2023). The
question: what is the field interested in now?

![Speaker](assets/l15-speaker.svg "Nathan Lambert: Berkeley PhD, HuggingFace, AI2. DPO was the 2023 story; the talk asks what comes next.")

The premise: more and more of the labs' action moved from pretraining to
**post-training**. Alignment research is now the main event.

## Level 1: Preference data at lab scale

![Data scale](assets/l15-data-scale.svg "Chatbot Arena: 800,000 public data points. Meta bought ~1.5M comparisons for LLaMA 2. Labs buy far more.")

- **Chatbot Arena**: ~800,000 public data points ([02:49](ts:02:49)).
- **Meta's LLaMA 2 paper**: ~1.5M comparisons bought ([02:53](ts:02:53)).
Both "years outdated." OpenAI and Anthropic buy far more.

Researchers cannot match that scale. The talk's challenge: what can research
do without lab resources?

> [!QA]
> Q: Why does preference data scale matter so much?
> A: Post-training is preference learning. More comparisons mean better reward models and better policies. Labs treat preference data as a moat: a great open reward model "would help people catch up in alignment."
> Follow-up: What can researchers do instead?
> A: Methods that need less data: DPO variants, self-rewarding loops, synthetic preferences, and better use of small high-quality sets. The talk's research agenda is efficiency, not scale.

## Level 1: Online versus offline

The central distinction ([47:47](ts:47:47)):

![Online versus offline](assets/l15-online.svg "Offline (DPO): static dataset, fixed labels. Online (PPO): fresh data from the policy, refreshed labels.")

- **Offline (DPO).** A static dataset. Labels fixed once. UltraFeedback
distills generations from many models (Alpaca, Vicuna, GPT-3.5, GPT-4,
LLaMA) into one policy.
- **Online (PPO).** Fresh data generated from the current policy. Labels
refreshed over time. The distribution shifts as the model improves.

Two axes: **what the text is** and **when the label was given**. RL's
on/off-policy distinction is related but "more definitional" ([47:21](ts:47:21)).
alignment cares about fresh data and fresh labels. April-May 2024 papers
agree: **online is important**.

Note: DPO "still has a reward model" in the math ([13:17](ts:13:17)): the
reference policy plays the reward's role. Skipping the explicit reward model
does not skip the concept.

## Level 2: Self-rewarding models

**Self-Rewarding LMs** (Meta): the DPO model judges its own generations as
an LLM judge, relabels the data, and iterates DPO ([49:55](ts:49:55)).

![Self-rewarding](assets/l15-selfreward.svg "The DPO model judges its own generations, relabels, and iterates; strong scores across rounds.")

Strong scores across rounds. Variations: batched DPO with data refreshes,
**discriminator-guided DPO** (reward models plus DPO training). The pattern:
close the loop between generation and judgment.

## Level 2: Beyond pairwise

Pairwise preferences are not the only signal ([61:21](ts:61:21)):

![Beyond pairwise](assets/l15-beyond-pairwise.svg "KTO: one-sided yes/no. Starling: k-wise ranking over 5-9 answers. SteerLM: fine-grained conciseness, helpfulness, honesty.")

- **KTO** (Stanford): **one-sided** preferences. Customer apps already
collect "was this helpful: yes/no." Different loss, same idea.
- **Starling**: **k-wise** preferences. Five or nine answers per prompt, a
ranking loss. One of the few open models that "broke through."
- **SteerLM** (Nvidia): **fine-grained** labels per completion:
conciseness, helpfulness, honesty.

"All of social choice needs to get condensed into these things"
([62:50](ts:62:50)). Preference learning is voting theory with gradients.

> [!QA]
> Q: When is one-sided feedback better than pairwise?
> A: When pairwise data does not exist. Products collect thumbs up/down, not A/B comparisons. KTO uses the abundant signal instead of demanding the scarce one. Use pairwise where you can afford it. Use one-sided where you must.
> Follow-up: Why fine-grained labels?
> A: One reward conflates qualities: a long answer can be helpful but not concise. Separate labels let you steer each axis independently. The cost is annotation complexity.

## Level 2: Open questions

![Open questions](assets/l15-frontier-q.svg "Is the reward model mostly distribution matching? Can search plus synthetic data exceed humans? Are good reward models a moat?")

1. **What do reward models actually learn?** "Mostly distribution matching,"
the speaker guesses ([60:08](ts:60:08)). Grading PPO answers with a reward
model trained on the same prompts is circular.
2. **How do we exceed human performance?** Old CS ideas return: **search**
as exploration in RL, generating new data ([63:41](ts:63:41)). Humans get it
"across the line" where search cannot.
3. **Why are good reward models closed?** Because they are a moat. Opening
one would help everyone catch up.

## Recap: the whole lesson on one screen

Eight ideas carry this lecture. Read each card. Say the core sentence out
loud. If you can, you own the lesson.

<div class="recap-grid">
<div class="recap-card">
<img src="assets/l15-speaker.svg" alt="Speaker">
<div class="rc-body">
<strong>1. DPO was the 2023 story</strong>
<p>Lambert (Berkeley, HF, AI2): post-training is where the action moved.
The question is what comes after DPO.</p>
<p class="rc-num">Key: alignment is the main event</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l15-data-scale.svg" alt="Data scale">
<div class="rc-body">
<strong>2. Labs buy preference data at scale</strong>
<p>Arena: 800k public. Meta: 1.5M for LLaMA 2. Both outdated. Researchers
cannot match it. Research must be efficient.</p>
<p class="rc-num">Key: data is a moat</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l15-online.svg" alt="Online versus offline">
<div class="rc-body">
<strong>3. Online versus offline</strong>
<p>Offline: static data, fixed labels. Online: fresh data from the policy,
refreshed labels. Two axes: text and label timing.</p>
<p class="rc-num">Key: [47:47](ts:47:47)</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l15-selfreward.svg" alt="Self-rewarding">
<div class="rc-body">
<strong>4. Judge your own generations</strong>
<p>Self-Rewarding LMs: LLM-as-judge relabels, iterate DPO. Close the loop
between generation and judgment.</p>
<p class="rc-num">Key: [49:55](ts:49:55)</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l15-beyond-pairwise.svg" alt="Beyond pairwise">
<div class="rc-body">
<strong>5. Beyond pairwise</strong>
<p>KTO: one-sided yes/no. Starling: k-wise ranking. SteerLM: fine-grained
axes. Social choice with gradients.</p>
<p class="rc-num">Key: [61:21](ts:61:21)</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l15-frontier-q.svg" alt="Open questions">
<div class="rc-body">
<strong>6. Reward models may just match distributions</strong>
<p>Grading with a model trained on the same prompts is circular. What do
reward models really learn?</p>
<p class="rc-num">Key: [60:08](ts:60:08)</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l15-frontier-q.svg" alt="Exceeding humans">
<div class="rc-body">
<strong>7. Search plus synthetic data to exceed humans</strong>
<p>Old CS ideas return: search as exploration. Humans get it across the
line where search cannot.</p>
<p class="rc-num">Key: [63:41](ts:63:41)</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l15-bridge.svg" alt="Bridge">
<div class="rc-body">
<strong>8. Depth lives in CS336</strong>
<p>Post-training systems (L15), RLVR (L16). This talk: the framing of the
moment after DPO.</p>
<p class="rc-num">Key: bridge, not duplicate</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- Lecture 15 video and transcript.
- Rafailov et al. (2023): DPO.

**Further reading:**
- [CS336 L15](../cs336/l15-post-training.html): post-training systems in depth.
- [CS336 L16](../cs336/l16-rlvr.html): RL with verifiable rewards.
- Yuan et al. (2024), "Self-Rewarding Language Models": the Meta paper.
- Ethayarajh et al. (2024), KTO: the one-sided method.

**Caveats from these sources.** Data counts (800k, 1.5M) are the talk's 2024 figures. "Online is important" summarizes April-May 2024 papers, not a settled theorem.

## Connections to the other courses

- **This course:** L10 built RLHF and DPO. This talk asks what follows. L11 evaluates the results.
- **CS336:** L15 (post-training), L16 (RLVR) carry the deep mechanics.
- **CS329H:** social choice theory is the formal home of preference aggregation.

> [!CHEAT]
> **Life after DPO cheatsheet.** Speaker: Lambert, Berkeley/HF/AI2, RL background. Moment: post-training is the action. Data: Arena 800k, Meta 1.5M, labs more. Researchers priced out. Online vs offline: fresh data + fresh labels vs static. UltraFeedback: many models distilled. Self-rewarding: judge own data, iterate DPO. Beyond pairwise: KTO one-sided, Starling k-wise, SteerLM fine-grained. Open Qs: distribution matching? search + synthetic to exceed humans? reward models as moat?

> [!MEMORY]
> **Fresh data, fresh labels.** Static datasets go stale as the policy improves. Online methods close the loop. The frontier is efficiency: do more with less preference data.

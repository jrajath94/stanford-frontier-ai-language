# Course map , cs224n Winter 2026 to units

Source: official schedule table at `http://web.stanford.edu/class/cs224n/`,
fetched 2026-10-06. Disclaimer carried verbatim: the schedule is
tentative and subject to change. U01-U08 were built by the first
builder, U09-U15 by the second builder. Every session below is mapped
to a unit. No session is orphaned.

| # | Date (2026) | Session | Unit | Claim class | Note |
|---|-------------|---------|------|-------------|------|
| S01 | Tue Jan 6 | History of NLP | U01 | OFFICIAL-SOURCE (SRC-02) | A1 out: Introduction to word vectors |
| S02 | Thu Jan 8 | Word Vectors | U02 | OFFICIAL-SOURCE | Readings: word2vec, negative sampling, GloVe papers (titles only) |
| S03 | Fri Jan 9 | Python Review Session | U01/U15 | OFFICIAL-SOURCE | Bridge session, maps to P02 remediation |
| S04 | Tue Jan 13 | Backpropagation and Neural Network Basics | U03 | OFFICIAL-SOURCE | A2 out (neural nets, tensor derivatives, dependency parsing), A1 due |
| S05 | Thu Jan 15 | Language Models and RNNs | U04 | OFFICIAL-SOURCE | Readings: vanishing gradient papers, Attention Is All You Need (titles) |
| S06 | Fri Jan 16 | PyTorch Tutorial Session | U03/U04 | OFFICIAL-SOURCE | Maps to P12 remediation |
| S07 | Tue Jan 20 | Transformers | U05 | OFFICIAL-SOURCE | Readings: Attention paper, Illustrated Transformer, LayerNorm (titles) |
| S08 | Thu Jan 22 | Final Projects: Custom and Default, Practical Tips | U15 | OFFICIAL-SOURCE | A3 out (self-attention and transformers), A2 due |
| S09 | Tue Jan 27 | Pretraining (Scaling, Systems, Data) | U06 | OFFICIAL-SOURCE | Readings: BERT, ELMo, Llama 3 (titles) |
| S10 | Thu Jan 29 | Post-training (RLHF, SFT, DPO) | U07 | OFFICIAL-SOURCE | Readings: InstructGPT, FLAN, AlpacaFarm, DPO (titles), project proposal out |
| S11 | Tue Feb 3 | Efficient Adaptation (Prompting + PEFT) | U08 | OFFICIAL-SOURCE | Readings: few-shot learners, CoT, lottery ticket, LoRA, adapters (titles) |
| S12 | Thu Feb 5 | Agents, Tool Use, and RAG | U09 | OFFICIAL-SOURCE | A4 out (LLM benchmarking), A3 due |
| S13 | Fri Feb 6 | Hugging Face Transformers Tutorial | U09/U15 | OFFICIAL-SOURCE | Tutorial: tool use and HF plumbing |
| S14 | Tue Feb 10 | Benchmarking and Evaluation | U10 | OFFICIAL-SOURCE | Readings: MMLU, HELM, AlpacaEval (titles) |
| S15 | Thu Feb 12 | Reasoning 1 | U11 | OFFICIAL-SOURCE | Readings: self-consistency, DeepSeek-R1, DAPO (titles) |
| S16 | Tue Feb 17 | Reasoning 2 | U11 | OFFICIAL-SOURCE | Readings: verification, speculative decoding, test-time compute, RoFormer (titles) |
| S17 | Thu Feb 19 | Guest: Tokenization and Multilinguality (Julie Kallini) | U12 | OFFICIAL-SOURCE | A4 due |
| S18 | Tue Feb 24 | Guest: Interpretability (Been Kim) | U13 | OFFICIAL-SOURCE | Guest content: title only in lessons (C10-C12 method leaves) |
| S19 | Thu Feb 26 | Social and Broader Impacts of NLP (Risks) | U13 | OFFICIAL-SOURCE | Impact half of U13 |
| S20 | Tue Mar 3 | Guest: Multimodality (Luke Zettlemoyer) | U14 | OFFICIAL-SOURCE | Guest content: title only. U14 C10-C12 are method leaves on attribution |

## Mismatches flagged

- Dependency parsing: A2's official title includes "dependency parsing",
  but no scheduled lecture carries that name. U03-C07 (parsing
  state/actions) and U03-C08 (supervised objective) are mapped to A2 plus
  the W2 neural-net sessions. Claim class: OFFICIAL-SOURCE for the A2
  title, REQUESTED-BRANCH for lecture-level attribution.
- "Efficient Adaptation (Prompting + PEFT)" reads as one session, the
  unit U08 splits prompting (C01-C06) from adaptation (C07-C12). The split
  is a teaching choice, not an official division.
- S12 "Agents, Tool Use, and RAG" and the tutorial sessions are
  built in U09 (agents) and U15 (tutorials). They were mapped early so
  no session was orphaned.

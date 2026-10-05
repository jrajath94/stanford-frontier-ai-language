# Stanford Frontier AI Learning System — Language

Private study repo. Reconstructed from official Stanford course material (latest offerings as of October 2026).

## Contents

- `site/` — the built static website. Open `site/language/cs224n/index.html` in a browser. Fully offline-capable (videos need internet). `site/foundations/` carries the Foundations repo's site so cross-course links resolve.
- `content/` — lesson sources in Markdown. Rebuild with `python3 build/build.py content/language site/language "Stanford Frontier AI"`.
- `build/` — the static site generator, templates, and design system.
- `cs224n-site.zip` — the built site as a single zip (CS224N complete).
- `cme295-site.zip` — the built site as a single zip (CS224N + CME295 complete).
- `cs329h-site.zip` — the built site as a single zip (all three courses complete).

## Courses in this repo

1. **CS224N: NLP with Deep Learning** (Spring 2024, Chris Manning) — complete. 15 lessons. Early lessons (word vectors, dependency parsing, RNNs, seq2seq, attention) are taught in full; transformer/pretraining/post-training lessons are bridges that link to the Foundations repo's CS336 rather than rewriting it. Guest lectures: Anna Goldie (transformers), Archit Sharma (post-training), Yann Dubois (benchmarking), Shikhar Murty (efficient training), Chaofei Fan (brain-computer interfaces), Nathan Lambert (after DPO).
2. **CME295: Transformers and Large Language Models** (Autumn 2025, Afshine Amidi and Shervine Amidi) — complete. 9 lessons. Bridge lessons throughout: they link to CS336 for deep mechanics and teach the Amidi framing, practical insights, and exam-style material. Primary source is Autumn 2025 (complete); Autumn 2026 is running now with a restructured syllabus, differences noted.
3. **CS329H: Machine Learning from Human Preferences** (Sanmi Koyejo) — complete. 10 lessons. Built from the course textbook (Truong, Haupt, Koyejo, 2025) and lecture videos. Unique material in this system: choice theory, Bradley-Terry/Plackett-Luce, bandits, social choice, mechanism design. The RLHF lesson bridges to CS336.

## Source fidelity

Every lesson cites its sources: the official lecture video (with timestamps), official slide decks, and the official subtitle transcript. Nothing is invented. Uncertain points are labeled `[uncertain]`. One caveat: the Lecture 1 video transcript was unreachable at build time (YouTube bot check); Lesson 1 is reconstructed from the official slide deck and flags this.

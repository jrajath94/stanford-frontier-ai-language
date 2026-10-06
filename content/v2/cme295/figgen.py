#!/usr/bin/env python3
"""Hand-authored flat SVG figures for CME295, in the VISUAL_SYSTEM.md binding style.
Background #F7F4EE, ink #1B2838, 8px grid, 1.5px stroke, radius table,
type stack Inter -> Source Sans 3 -> IBM Plex Sans (body 450, title 600, label 500).
Run: python3 figgen.py  (writes into content/v2/cme295/assets/)
"""
import os

OUT = "/home/hatch/workspace/stanford-frontier-ai/content/v2/cme295/assets"

INK = "#1B2838"
MUTED = "#5C6B7A"
LINE = "#D9D3C7"
PAPER = "#F7F4EE"
PANEL = "#FFFDF8"
COUNT = "#E7F1F8"
NEWTOK = "#E7F4EF"
ACTIVE = "#F4E6D4"
CHIP = "#E6E2DA"
TEAL = "#1F7A72"
ORANGE = "#C46B2C"
FOCUS = "#1E4D8C"

FONT = "Inter,'Source Sans 3','IBM Plex Sans',sans-serif"
MONO = "'IBM Plex Mono',ui-monospace,monospace"
SERIF = "'Source Serif 4',Newsreader,serif"


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Fig:
    def __init__(self, name, w, h, title, subtitle=None):
        self.name = name
        self.w = w
        self.h = h
        self.parts = []
        self.parts.append(
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="{PAPER}"/>')
        self.defs = (
            '<defs>'
            f'<marker id="ah-{name}" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">'
            f'<path d="M0,0 L8,3 L0,6 Z" fill="{INK}"/></marker>'
            f'<marker id="aht-{name}" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">'
            f'<path d="M0,0 L8,3 L0,6 Z" fill="{TEAL}"/></marker>'
            '</defs>')
        self.parts.append(
            f'<text x="24" y="44" font-family="{FONT}" font-size="27" font-weight="600" fill="{INK}">{_esc(title)}</text>')
        if subtitle:
            self.parts.append(
                f'<text x="24" y="68" font-family="{FONT}" font-size="15" font-weight="450" fill="{MUTED}">{_esc(subtitle)}</text>')

    def src(self, label="Original figure \u00b7 CME295"):
        self.parts.append(
            f'<text x="{self.w - 16}" y="{self.h - 14}" text-anchor="end" font-family="{FONT}" font-size="12" font-weight="500" fill="{MUTED}">{_esc(label)}</text>')

    def box(self, x, y, w, h, fill=PANEL, rx=12, stroke=INK, sw=1.5):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self, x, y, s, size=16, weight=450, fill=INK, anchor="middle", font=FONT, lh=None):
        lines = s.split("\n")
        step = int(size * 1.35) if lh is None else lh
        y0 = y - (len(lines) - 1) * step / 2
        for i, ln in enumerate(lines):
            self.parts.append(
                f'<text x="{x}" y="{y0 + i * step:.1f}" text-anchor="{anchor}" font-family="{font}" font-size="{size}" font-weight="{weight}" fill="{fill}" dominant-baseline="middle">{_esc(ln)}</text>')

    def label(self, x, y, s, size=14, fill=INK, anchor="middle", weight=500):
        self.text(x, y, s, size=size, weight=weight, fill=fill, anchor=anchor)

    def arrow(self, x1, y1, x2, y2, s=None, color=INK, dashed=False, teal=False):
        m = f"aht-{self.name}" if teal else f"ah-{self.name}"
        c = TEAL if teal else color
        d = ' stroke-dasharray="6,5"' if dashed else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="1.5"{d} marker-end="url(#{m})"/>')
        if s:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            self.on_line(mx, my, s)

    def on_line(self, x, y, s, size=13):
        w = len(s) * size * 0.58 + 16
        self.parts.append(
            f'<rect x="{x - w / 2:.1f}" y="{y - size * 0.85:.1f}" width="{w:.1f}" height="{size * 1.7:.1f}" rx="8" fill="{PAPER}" stroke="{LINE}" stroke-width="1"/>')
        self.text(x, y, s, size=size, weight=500, fill=MUTED)

    def pill(self, x, y, w, h, s, fill=CHIP, mono=True, size=14):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="999" fill="{fill}" stroke="{INK}" stroke-width="1.5"/>')
        self.text(x + w / 2, y + h / 2, s, size=size, weight=500,
                  font=MONO if mono else FONT)

    def bar(self, x, y, w, h, frac, fill=TEAL):
        self.box(x, y, w, h, fill="#FFFFFF", rx=4)
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w * frac:.1f}" height="{h}" rx="4" fill="{fill}"/>')

    def save(self):
        body = "\n".join(self.parts)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}" role="img">\n{self.defs}\n{body}\n</svg>')
        with open(os.path.join(OUT, self.name + ".svg"), "w") as f:
            f.write(svg)
        return self.name


FIGS = []


def fig(fn):
    FIGS.append(fn)
    return fn


# ---------------------------------------------------------------- L01
@fig
def l01_nlp_tasks():
    f = Fig("l01-nlp-tasks", 960, 560, "NLP tasks: three families",
            "From CME295 Lecture 1 slides. Classification, multi-classification, generation.")
    cols = [("Classification", "Input text -> Model -> label",
             ["Sentiment extraction", "Intent detection", "Language detection", "Topic modeling"]),
            ("Multi-classification", "Input text -> Model -> per-token tags",
             ["POS tagging", "Named entity recognition", "Dependency parsing", "Constituency parsing"]),
            ("Generation", "Input text -> Model -> out text",
             ["Machine translation", "Question answering", "Summarization", "Text generation"])]
    for i, (t, flow, items) in enumerate(cols):
        x = 24 + i * 304
        f.box(x, 88, 288, 420, rx=12)
        f.text(x + 144, 124, t, size=20, weight=600)
        f.text(x + 144, 152, flow, size=13, fill=MUTED)
        f.parts.append(f'<line x1="{x + 16}" y1="176" x2="{x + 272}" y2="176" stroke="{LINE}" stroke-width="1.5"/>')
        for j, it in enumerate(items):
            f.pill(x + 40, 200 + j * 72, 208, 48, it, mono=False, size=14)
    f.src()
    f.save()


@fig
def l01_token_levels():
    f = Fig("l01-token-levels", 960, 600, "Tokenization: three granularities",
            "From CME295 Lecture 1 slides. Same sentence, three ways to split it.")
    f.text(24, 108, "A cute teddy bear is reading.", size=18, weight=600, anchor="start")
    rows = [("Word-level", ["A", "cute", "teddy", "bear", "is", "reading", "."]),
            ("Subword-level", ["A", "cute", "ted", "##dy", "bear", "is", "read", "##ing", "."]),
            ("Char-level", ["A", "c", "u", "t", "e", "cute", "ted", "##dy"])]
    for r, (name, toks) in enumerate(rows):
        y = 140 + r * 120
        f.label(90, y + 24, name, size=15)
        x = 210
        for t in toks[:9]:
            w = max(48, len(t) * 13 + 28)
            if x + w > 930:
                break
            f.pill(x, y, w, 48, t, size=14)
            x += w + 8
    notes = [("Word", "Simple, interpretable. Risk of OOV. Ignores roots."),
             ("Subword", "Reuses prefixes and suffixes. Small OOV risk. Learned from data."),
             ("Char", "Robust to casing and typos. Slow. Embeddings not interpretable.")]
    for r, (n, d) in enumerate(notes):
        y = 140 + r * 120
        f.text(24, y + 92, n + ": " + d, size=13, fill=MUTED, anchor="start", weight=450)
    f.src()
    f.save()


@fig
def l01_word2vec():
    f = Fig("l01-word2vec", 960, 560, "Word2vec: embeddings from a proxy task",
            "From CME295 Lecture 1 slides. Neural network over billions of words. Mikolov et al., 2013.")
    f.box(60, 140, 180, 320, fill=COUNT, rx=8)
    f.text(150, 180, "Input", size=18, weight=600)
    f.text(150, 212, "one-hot\nsize V", size=15, fill=MUTED)
    f.pill(85, 260, 130, 48, "[1,0,0,...]", size=13)
    f.box(390, 140, 180, 320, fill=ACTIVE, rx=8)
    f.text(480, 180, "Hidden", size=18, weight=600)
    f.text(480, 212, "embedding\nsize d", size=15, fill=MUTED)
    f.pill(415, 260, 130, 48, "[0.2, 0.9]", size=13)
    f.box(720, 140, 180, 320, fill=NEWTOK, rx=8)
    f.text(810, 180, "Output", size=18, weight=600)
    f.text(810, 212, "softmax\nsize V", size=15, fill=MUTED)
    f.pill(745, 260, 130, 48, "[0.2,0.4,...]", size=13)
    f.arrow(240, 300, 390, 300, "W_in")
    f.arrow(570, 300, 720, 300, "W_out")
    f.text(480, 500, "Proxy tasks: CBOW predicts the center word from context.\nSkip-gram predicts context words from the center word.",
           size=15, fill=MUTED)
    f.src()
    f.save()


@fig
def l01_rnn_fig():
    f = Fig("l01-rnn", 960, 560, "RNNs: one token at a time",
            "From CME295 Lecture 1 slides. Temporal sequence, internal state. 1980s. LSTM: 1997.")
    toks = ["A", "cute", "teddy bear", "is", "reading", "."]
    for i, t in enumerate(toks):
        x = 60 + i * 146
        f.pill(x, 160, 128, 48, t, mono=False, size=14)
        f.box(x + 14, 280, 100, 96, fill=ACTIVE, rx=12)
        f.text(x + 64, 328, f"h{i + 1}", size=18, weight=600, font=MONO)
        f.arrow(x + 64, 208, x + 64, 280)
        if i > 0:
            f.arrow(60 + (i - 1) * 146 + 114, 328, x + 14, 328)
    f.text(60, 450, "Limit: long-range dependencies fade. Tokens encoded far in the past are hard to keep.",
           size=15, fill=MUTED, anchor="start")
    f.text(60, 480, "Heads: classification (sentiment), multi-classification (tags), generation (translation).",
           size=15, fill=MUTED, anchor="start")
    f.src()
    f.save()


@fig
def l01_qkv_attention():
    # Numbers are l01's own worked toy: 3 tokens, 2-D vectors, identity projections.
    import math
    keys = ["counselor", "helped", "frame"]
    sc = [1, 1, 2]
    exps = [round(math.e**s, 2) for s in sc]      # [2.72, 2.72, 7.39]
    tot = round(sum(exps), 2)                       # 12.83
    wt = [round(e / tot, 2) for e in exps]           # [0.21, 0.21, 0.58]
    assert exps == [2.72, 2.72, 7.39] and tot == 12.83 and wt == [0.21, 0.21, 0.58]
    out = [round(wt[0]*1 + wt[1]*0 + wt[2]*1, 2), round(wt[0]*0 + wt[1]*1 + wt[2]*1, 2)]
    assert out == [0.79, 0.79] and abs(sum(wt) - 1.0) < 0.01
    f = Fig("l01-qkv-attention", 960, 660, "Self-attention: query, key, value",
            "The canonical attention arrow for this course. Softmax(Q K^T / sqrt(d_k)) V.")
    f.text(24, 116, 'Example: the query token is "frame". Scores are dot products.',
           size=16, weight=500, anchor="start")
    f.pill(80, 150, 140, 48, 'query: "frame"', fill=NEWTOK, mono=False, size=15)
    for i, (k, s, w) in enumerate(zip(keys, sc, wt)):
        y = 140 + i * 90
        f.pill(400, y, 130, 48, k, mono=False, size=14)
        f.bar(560, y + 8, 180, 32, w, fill=TEAL if w > 0.5 else FOCUS)
        f.text(752, y + 24, f"score {s} -> {w:.2f}", size=14, font=MONO, anchor="start")
        f.arrow(220, 174, 400, y + 24)
    f.box(60, 430, 840, 150, fill=COUNT, rx=8)
    f.text(480, 465, f"softmax: e^1={exps[0]:.2f}, e^1={exps[1]:.2f}, e^2={exps[2]:.2f}, total={tot:.2f}",
           size=15, weight=500, font=MONO)
    f.text(480, 505, f"weights = [{wt[0]:.2f}, {wt[1]:.2f}, {wt[2]:.2f}] (sum to 1)",
           size=15, weight=500, font=MONO)
    f.text(480, 545, f"output = {wt[0]:.2f}*[1,0] + {wt[1]:.2f}*[0,1] + {wt[2]:.2f}*[1,1] = [{out[0]:.2f}, {out[1]:.2f}]",
           size=15, weight=600, font=MONO)
    f.src()
    f.save()


@fig
def l01_transformer_arch():
    f = Fig("l01-transformer-arch", 960, 680, "The transformer, 2017",
            'From CME295 Lecture 1 slides. Vaswani et al., "Attention Is All You Need". Post-norm.')
    f.box(60, 120, 380, 480, rx=12)
    f.text(250, 156, "ENCODER", size=22, weight=600)
    f.text(250, 184, "Nx: self-attention -> add+norm -> FFN -> add+norm", size=13, fill=MUTED)
    f.pill(110, 220, 280, 48, "input embeddings + pos. encoding", mono=False, size=14)
    f.box(110, 300, 280, 120, fill=COUNT, rx=12)
    f.text(250, 340, "Multi-head\nself-attention", size=15, weight=500)
    f.text(250, 396, "add + norm", size=13, fill=MUTED)
    f.box(110, 444, 280, 120, fill=COUNT, rx=12)
    f.text(250, 484, "Feed-forward\nnetwork", size=15, weight=500)
    f.text(250, 540, "add + norm", size=13, fill=MUTED)
    f.box(520, 120, 380, 480, rx=12)
    f.text(710, 156, "DECODER", size=22, weight=600)
    f.text(710, 184, "Nx: masked self-attn -> cross-attn -> FFN", size=13, fill=MUTED)
    f.pill(570, 220, 280, 48, "output embeddings, shifted right", mono=False, size=14)
    f.box(570, 300, 280, 84, fill=COUNT, rx=12)
    f.text(710, 342, "Masked multi-head self-attention", size=14, weight=500)
    f.box(570, 400, 280, 84, fill=ACTIVE, rx=12)
    f.text(710, 442, "Encoder-decoder attention", size=14, weight=500)
    f.box(570, 500, 280, 64, fill=COUNT, rx=12)
    f.text(710, 532, "FFN + linear + softmax", size=14, weight=500)
    f.arrow(440, 360, 520, 442, "encoder output", teal=True)
    f.text(480, 640, "Parameters: N layers, h heads, d_model, d_FF, d_key, d_value, V vocab size.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l01_positional_encoding():
    f = Fig("l01-positional-encoding", 960, 560, "Positional encoding: order matters",
            "From CME295 Lecture 1 slides. Add position info to token embeddings.")
    f.pill(80, 160, 200, 56, "token embedding", mono=False, size=15, fill=CHIP)
    f.text(330, 188, "+", size=28, weight=600)
    f.pill(380, 160, 200, 56, "position embedding", mono=False, size=15, fill=COUNT)
    f.text(630, 188, "=", size=28, weight=600)
    f.pill(680, 160, 200, 56, "position-aware", mono=False, size=15, fill=NEWTOK)
    f.box(80, 280, 380, 180, rx=12)
    f.text(270, 316, "Learned", size=18, weight=600)
    f.text(270, 352, "Each position gets its own\ntrainable vector.", size=14, fill=MUTED)
    f.box(500, 280, 380, 180, rx=12)
    f.text(690, 316, "Hard-coded", size=18, weight=600)
    f.text(690, 352, "Fixed sin/cos waves.\nNo parameters to learn.", size=14, fill=MUTED)
    for i in range(12):
        x = 540 + i * 28
        hgt = 20 + 40 * abs(((i * 37) % 11) / 11 - 0.5) * 2
        f.parts.append(f'<rect x="{x}" y="{420 - hgt}" width="16" height="{hgt:.0f}" fill="{FOCUS}" opacity="0.7"/>')
    f.text(480, 510, "Goal: let the model understand relative position, not just absolute index.",
           size=15, fill=MUTED)
    f.src()
    f.save()


@fig
def l01_multihead():
    f = Fig("l01-multihead", 960, 560, "Multi-head attention: h views in parallel",
            "From CME295 Lecture 1 slides. Like multiple filters of a convolutional layer.")
    f.text(480, 120, "Q, K, V", size=20, weight=600)
    for i in range(4):
        x = 60 + i * 222
        f.box(x, 160, 198, 200, fill=COUNT, rx=12)
        f.text(x + 99, 200, f"Head {i + 1}", size=17, weight=600)
        f.text(x + 99, 244, "its own QKV\nprojections", size=14, fill=MUTED)
        f.text(x + 99, 310, "attends differently", size=13, fill=TEAL, weight=500)
    f.arrow(480, 360, 480, 400)
    f.box(330, 400, 300, 72, fill=ACTIVE, rx=12)
    f.text(480, 436, "Concat heads, project with Wo", size=16, weight=500)
    f.text(480, 510, "Benefit: capture different attention features at the same time.",
           size=15, fill=MUTED)
    f.src()
    f.save()


@fig
def l01_end_to_end():
    f = Fig("l01-end-to-end", 960, 640, "End to end: English to French",
            "From CME295 Lecture 1 slides. The full pipeline on one sentence.")
    f.text(24, 116, "Input: A cute teddy bear is reading.", size=16, weight=500, anchor="start")
    steps = [("[BOS] A cute teddy bear is reading . [EOS]", "tokenize"),
             ("embeddings + position encoding", "embed"),
             ("ENCODER: context-aware embeddings", "encode"),
             ("DECODER starts from [BOS]", "decode")]
    y = 150
    for s, op in steps:
        f.box(120, y, 720, 64, rx=12, fill=PANEL)
        f.text(480, y + 32, s, size=15, weight=500)
        if y > 150:
            f.arrow(480, y - 8, 480, y, None)
        y += 96
    f.box(120, y, 720, 80, rx=12, fill=NEWTOK)
    f.text(480, y + 40, "Un ours en peluche mignon lit . [EOS]", size=17, weight=600)
    f.arrow(480, y - 8, 480, y, "softmax over vocab")
    f.text(480, 610, "Each step: linear + softmax over V classes, where each class is a word.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l01_label_smoothing():
    f = Fig("l01-label-smoothing", 960, 520, "Label smoothing: fight overconfidence",
            "From CME295 Lecture 1 slides. 2015 vision idea. Prevents overfitting, improves BLEU.")
    f.box(80, 140, 360, 240, rx=12)
    f.text(260, 176, "Hard targets", size=18, weight=600)
    f.text(260, 216, 'correct word: 1.0\neverything else: 0.0', size=15, font=MONO, fill=MUTED)
    f.text(260, 300, "Model becomes\noverconfident.", size=15, fill=ORANGE, weight=500)
    f.box(520, 140, 360, 240, rx=12)
    f.text(700, 176, "Smoothed targets", size=18, weight=600)
    f.text(700, 216, 'correct word: 1 - eps\neverything else: eps / (V-1)', size=15, font=MONO, fill=MUTED)
    f.text(700, 300, "Noise in true labels.\nBetter accuracy and BLEU.", size=15, fill=TEAL, weight=500)
    f.arrow(440, 260, 520, 260)
    f.text(480, 440, "General technique. Small change, real gains on translation quality.",
           size=15, fill=MUTED)
    f.src()
    f.save()


# ---------------------------------------------------------------- L02
@fig
def l02_attention_map():
    f = Fig("l02-attention-map", 960, 600, "Attention maps: what the model looks at",
            "From CME295 Lecture 2. Anaphora: the pronoun points back to its noun.")
    f.text(24, 116, '"The animal did not cross the street because it was too tired."',
           size=17, weight=500, anchor="start")
    toks = ["The", "animal", "did", "not", "cross", "street", "it"]
    strong = {1: 0.72, 6: 0.15}  # 0.72 = the lesson toy weight on "animal" (l02)
    for i, t in enumerate(toks):
        x = 60 + i * 122
        f.pill(x, 160, 110, 48, t, mono=False, size=15,
               fill=NEWTOK if i == 6 else CHIP)
    f.text(480, 260, 'query: "it" -> keys', size=14, fill=MUTED)
    for i, t in enumerate(toks):
        if i == 6:
            continue
        s = strong.get(i, 0.04)
        x = 60 + i * 122 + 55
        y2 = 300 + (1 - s) * 160
        f.arrow(x, 208, x, y2, f"{s:.2f}" if s > 0.1 else None, teal=(s > 0.5))
    f.box(60, 480, 840, 64, fill=COUNT, rx=8)
    f.text(480, 512, '"it" attends most strongly to "animal". The map resolves the pronoun.',
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l02_pos_embeddings():
    f = Fig("l02-pos-embeddings", 960, 600, "Position embeddings: learned vs sinusoidal",
            "From CME295 Lecture 2. Dot product of two positions = function of relative distance.")
    f.box(60, 120, 400, 200, rx=12)
    f.text(260, 156, "Learned", size=19, weight=600)
    f.text(260, 196, "One vector per position.\nAdded to the token embedding.", size=14, fill=MUTED)
    f.text(260, 272, "BERT style [uncertain: hard-coded?]", size=13, fill=ORANGE, weight=500)
    f.box(500, 120, 400, 200, rx=12)
    f.text(700, 156, "Sinusoidal", size=19, weight=600)
    f.text(700, 196, "Fixed sin/cos waves.\nNo parameters.", size=14, fill=MUTED)
    f.text(700, 272, "Original transformer", size=13, fill=MUTED, weight=500)
    for i in range(24):
        x = 528 + i * 15
        h1 = 24 + 18 * abs(((i * 53) % 17) / 17 - 0.5) * 2
        f.parts.append(f'<rect x="{x}" y="{300 - h1}" width="10" height="{h1:.0f}" fill="{FOCUS}" opacity="0.75"/>')
    f.box(60, 380, 840, 150, fill=COUNT, rx=8)
    f.text(480, 420, "Key property: PE(i) dot PE(j) depends only on (i - j).", size=16, weight=600, font=MONO)
    f.text(480, 462, "High-frequency dims track local order. Low-frequency dims track long range.",
           size=15, fill=MUTED)
    f.text(480, 494, "Relative distance is what attention actually needs.", size=15, fill=MUTED)
    f.src()
    f.save()


@fig
def l02_rope():
    f = Fig("l02-rope", 960, 600, "RoPE: rotate queries and keys",
            "From CME295 Lecture 2. Rotary position embeddings. The default choice today.")
    f.box(80, 140, 360, 320, rx=12)
    f.text(260, 176, "Before", size=18, weight=600)
    f.parts.append(f'<line x1="160" y1="380" x2="160" y2="240" stroke="{INK}" stroke-width="1.5" marker-end="url(#ah-l02-rope)"/>')
    f.parts.append(f'<line x1="360" y1="380" x2="360" y2="240" stroke="{INK}" stroke-width="1.5" marker-end="url(#ah-l02-rope)"/>')
    f.text(160, 410, "q", size=18, weight=600, font=MONO)
    f.text(360, 410, "k", size=18, weight=600, font=MONO)
    f.text(260, 300, "no rotation", size=14, fill=MUTED)
    f.box(520, 140, 360, 320, rx=12)
    f.text(700, 176, "RoPE", size=18, weight=600)
    f.parts.append(f'<line x1="600" y1="380" x2="660" y2="250" stroke="{TEAL}" stroke-width="1.5" marker-end="url(#aht-l02-rope)"/>')
    f.parts.append(f'<line x1="800" y1="380" x2="740" y2="250" stroke="{TEAL}" stroke-width="1.5" marker-end="url(#aht-l02-rope)"/>')
    f.text(600, 410, "q", size=18, weight=600, font=MONO)
    f.text(800, 410, "k", size=18, weight=600, font=MONO)
    f.text(700, 300, "rotate by angle(m)", size=14, fill=TEAL, weight=500)
    f.arrow(440, 300, 520, 300)
    f.box(80, 500, 800, 56, fill=NEWTOK, rx=8)
    f.text(480, 528, "q_m dot k_n then depends only on (m - n). Relative position, inside attention.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l02_norm():
    f = Fig("l02-norm", 960, 600, "Normalization moves before the sublayer",
            "From CME295 Lecture 2. Post-norm (2017) -> pre-norm. RMSNorm drops the mean.")
    f.box(60, 140, 400, 240, rx=12)
    f.text(260, 176, "Post-norm (2017)", size=18, weight=600)
    f.box(110, 210, 300, 56, fill=COUNT, rx=8)
    f.text(260, 238, "sublayer: attn or FFN", size=14, weight=500)
    f.arrow(260, 266, 260, 296)
    f.box(110, 296, 300, 56, fill=ACTIVE, rx=8)
    f.text(260, 324, "layer norm", size=14, weight=500)
    f.box(500, 140, 400, 240, rx=12)
    f.text(700, 176, "Pre-norm (today)", size=18, weight=600)
    f.box(550, 210, 300, 56, fill=ACTIVE, rx=8)
    f.text(700, 238, "RMSNorm", size=14, weight=500)
    f.arrow(700, 266, 700, 296)
    f.box(550, 296, 300, 56, fill=COUNT, rx=8)
    f.text(700, 324, "sublayer: attn or FFN", size=14, weight=500)
    f.arrow(460, 260, 500, 260)
    f.box(60, 430, 840, 110, fill=COUNT, rx=8)
    f.text(480, 468, "Why: gradients flow cleanly through the residual stream. Training is stable at depth.",
           size=15, weight=500)
    f.text(480, 504, "RMSNorm: normalize by root-mean-square only. Fewer parameters than layer norm.",
           size=15, fill=MUTED)
    f.src()
    f.save()


@fig
def l02_attention_approx():
    f = Fig("l02-attention-approx", 960, 600, "Cheaper attention: look at less",
            "From CME295 Lecture 2. Full attention is O(n^2). Sliding windows cut the cost.")
    f.box(60, 120, 260, 280, rx=12)
    f.text(190, 156, "Full", size=18, weight=600)
    f.text(190, 196, "Every token attends\nto every token.", size=14, fill=MUTED)
    for r in range(6):
        for c in range(6):
            f.parts.append(f'<rect x="{100 + c * 30}" y="{240 + r * 22}" width="26" height="18" fill="{FOCUS}" opacity="0.85"/>')
    f.box(350, 120, 260, 280, rx=12)
    f.text(480, 156, "Sliding window", size=18, weight=600)
    f.text(480, 196, "Each token attends\nto w neighbors.", size=14, fill=MUTED)
    for r in range(6):
        for c in range(6):
            on = abs(r - c) <= 1
            f.parts.append(f'<rect x="{390 + c * 30}" y="{240 + r * 22}" width="26" height="18" fill="{TEAL if on else CHIP}" opacity="0.85"/>')
    f.box(640, 120, 260, 280, rx=12)
    f.text(770, 156, "Mistral 7B", size=18, weight=600)
    f.text(770, 196, "Stacked windows:\nreceptive field grows.", size=14, fill=MUTED)
    for L in range(4):
        wdt = 40 + L * 30
        f.parts.append(f'<rect x="{770 - wdt / 2}" y="{250 + L * 34}" width="{wdt}" height="24" rx="6" fill="{ORANGE}" opacity="0.7"/>')
        f.text(770, 262 + L * 34, f"layer {L + 1}", size=11, fill=INK)
    f.text(480, 470, "Longformer: sliding window + a few global tokens. Cost drops from n^2 to n * w.",
           size=15, fill=MUTED)
    f.src()
    f.save()


@fig
def l02_mqa_gqa():
    f = Fig("l02-mqa-gqa", 960, 620, "MHA, GQA, MQA: share the keys and values",
            "From CME295 Lecture 2. Fewer KV heads = smaller KV cache = faster inference.")
    heads = [("MHA", "h query heads\nh KV heads", 8, TEAL),
             ("GQA", "h query heads\ng KV groups", 4, FOCUS),
             ("MQA", "h query heads\n1 KV head", 1, ORANGE)]
    for i, (name, sub, kv, col) in enumerate(heads):
        x = 40 + i * 300
        f.box(x, 120, 280, 380, rx=12)
        f.text(x + 140, 156, name, size=20, weight=600)
        f.text(x + 140, 190, sub, size=14, fill=MUTED)
        for q in range(4):
            f.pill(x + 30 + q * 62, 240, 54, 40, "Q", mono=False, size=13)
        for k in range(kv):
            fx = x + 30 + k * (220 / max(kv - 1, 1)) - 27 if kv > 1 else x + 113
            f.pill(fx, 330, 54, 40, "KV", mono=False, size=13, fill=col)
        f.text(x + 140, 420, "KV cache size", size=13, fill=MUTED)
        f.bar(x + 60, 440, 160, 24, [1.0, 0.5, 0.2][i], fill=col)
    f.box(40, 540, 880, 48, fill=COUNT, rx=8)
    f.text(480, 564, "Inference keeps every key and value. Sharing them shrinks memory and speeds decoding.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l02_t5():
    f = Fig("l02-t5", 960, 600, "T5: span corruption with sentinel tokens",
            "From CME295 Lecture 2. Encoder-decoder. Text-to-text for every task.")
    f.text(24, 130, "Input (spans masked):", size=16, weight=600, anchor="start")
    toks = ["The", "cute", "<X>", "is", "reading", "<Y>", "."]
    x = 24
    for t in toks:
        w = 96 if t.startswith("<") else max(64, len(t) * 12 + 24)
        f.pill(x, 150, w, 48, t, mono=False, size=14, fill=ACTIVE if t.startswith("<") else CHIP)
        x += w + 8
    f.arrow(480, 210, 480, 250, "encoder-decoder")
    f.text(24, 300, "Target (fill the sentinels):", size=16, weight=600, anchor="start")
    toks2 = ["<X>", "teddy bear", "<Y>", "a book"]
    x = 24
    for t in toks2:
        w = 96 if t.startswith("<") else max(64, len(t) * 12 + 24)
        f.pill(x, 320, w, 48, t, mono=False, size=14, fill=NEWTOK if not t.startswith("<") else ACTIVE)
        x += w + 8
    f.box(24, 420, 912, 120, fill=COUNT, rx=8)
    f.text(480, 458, "Every task becomes text-in, text-out. Same loss, same model, many tasks.",
           size=15, weight=500)
    f.text(480, 496, "Relative position bias: learned bias per distance bucket, added to attention scores.",
           size=15, fill=MUTED)
    f.src()
    f.save()


@fig
def l02_bert():
    f = Fig("l02-bert", 960, 660, "BERT: the encoder-only workhorse",
            "From CME295 Lecture 2. WordPiece ~30k. Notation: L layers, H hidden size, A heads.")
    f.text(24, 120, "Input:", size=16, weight=600, anchor="start")
    toks = ["[CLS]", "A", "cute", "bear", "[SEP]", "is", "reading", "[SEP]"]
    seg = [0, 0, 0, 0, 0, 1, 1, 1]
    x = 24
    for t, s in zip(toks, seg):
        w = max(64, len(t) * 12 + 24)
        f.pill(x, 140, w, 48, t, mono=False, size=14,
               fill=FOCUS if t in ("[CLS]", "[SEP]") else (COUNT if s == 0 else NEWTOK))
        x += w + 8
    f.text(24, 230, "Segment A = sentence 1 (blue). Segment B = sentence 2 (green). Position added to each.",
           size=14, fill=MUTED, anchor="start")
    f.box(24, 270, 440, 170, rx=12)
    f.text(244, 306, "MLM: mask 15%", size=17, weight=600)
    f.text(244, 344, "80% -> [MASK]\n10% -> random token\n10% -> unchanged", size=14, fill=MUTED)
    f.text(244, 414, "Predict the original.", size=14, fill=TEAL, weight=500)
    f.box(496, 270, 440, 170, rx=12)
    f.text(716, 306, "NSP: next sentence?", size=17, weight=600)
    f.text(716, 344, "50%: true next sentence\n50%: random sentence", size=14, fill=MUTED)
    f.text(716, 414, "[CLS] embedding decides.", size=14, fill=TEAL, weight=500)
    f.box(24, 480, 912, 110, fill=COUNT, rx=8)
    f.text(480, 518, "Fine-tuning: add one head on [CLS] for classification, per-token heads for tagging.",
           size=15, weight=500)
    f.text(480, 552, "RoBERTa drops NSP, masks dynamically, trains longer on more data. DistilBERT distills via KL.",
           size=14, fill=MUTED)
    f.src()
    f.save()


# ---------------------------------------------------------------- L03
@fig
def l03_llm_def():
    f = Fig("l03-llm-def", 960, 600, "What makes a language model large",
            "From CME295 Lecture 3. A large LM = next-token probabilities at scale.")
    f.box(60, 120, 840, 110, fill=COUNT, rx=8)
    f.text(480, 160, "P(x_t | x_1 ... x_{t-1})", size=24, weight=600, font=SERIF)
    f.text(480, 200, "The model outputs a probability for every possible next token.", size=15, fill=MUTED)
    rows = [("Parameters", "hundreds of billions", "More weights, more capacity."),
            ("Training tokens", "hundreds of billions to tens of trillions", "More text seen, more knowledge."),
            ("Architecture", ">90% decoder-only transformers", "The GPT pattern won.")]
    for i, (k, v, d) in enumerate(rows):
        y = 260 + i * 100
        f.box(60, y, 840, 84, rx=12)
        f.text(200, y + 42, k, size=18, weight=600)
        f.text(520, y + 30, v, size=16, weight=500, fill=TEAL)
        f.text(520, y + 58, d, size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l03_moe():
    f = Fig("l03-moe", 960, 640, "Mixture of experts: a room of specialists",
            "From CME295 Lecture 3. y-hat = sum of g_i * E_i(x). Experts are FFNs.")
    f.pill(400, 120, 160, 48, "token x", mono=False, size=16, fill=CHIP)
    f.arrow(480, 168, 480, 200, "router")
    f.box(380, 200, 200, 72, fill=ACTIVE, rx=12)
    f.text(480, 236, "gating g(x)", size=16, weight=600)
    for i in range(4):
        x = 60 + i * 222
        on = i in (1, 2)
        f.box(x, 300, 198, 130, fill=NEWTOK if on else PANEL, rx=12,
              stroke=TEAL if on else INK, sw=2.5 if on else 1.5)
        f.text(x + 99, 340, f"Expert {i + 1}", size=16, weight=600)
        f.text(x + 99, 376, "FFN", size=14, fill=MUTED)
        if on:
            f.text(x + 99, 406, "ACTIVE", size=13, weight=600, fill=TEAL)
        f.arrow(480, 272, x + 99, 300, None)
    f.box(60, 470, 840, 110, fill=COUNT, rx=8)
    f.text(480, 506, "Sparse: top-1 or top-2 experts per token, per layer. Dense would run them all.",
           size=15, weight=500)
    f.text(480, 544, "Risk: routing collapse. Fix: auxiliary loss + noisy gating. Switch Transformer ~1.6T params.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l03_decoding():
    f = Fig("l03-decoding", 960, 640, "Decoding: from probabilities to tokens",
            "From CME295 Lecture 3. Greedy, beam, sampling, top-K, top-P.")
    rows = [("Greedy", "take argmax each step", "Fast. Can loop or go bland.", TEAL),
            ("Beam search", "keep B hypotheses, sum log-probs + length norm", "Better. Costs B times more.", FOCUS),
            ("Sampling", "draw from the distribution", "Diverse. Can surprise you.", ORANGE),
            ("Top-K", "sample from the K best tokens", "Cuts the weird tail.", TEAL),
            ("Top-P (nucleus)", "sample from smallest set with mass P", "Adapts to sharp or flat.", FOCUS)]
    for i, (name, how, note, col) in enumerate(rows):
        y = 110 + i * 100
        f.box(60, y, 220, 84, fill=col, rx=12)
        f.text(170, y + 42, name, size=17, weight=600, fill="#FFFFFF")
        f.box(296, y, 604, 84, rx=12)
        f.text(330, y + 30, how, size=15, weight=500, anchor="start")
        f.text(330, y + 60, note, size=14, fill=MUTED, anchor="start")
    f.src()
    f.save()


@fig
def l03_temperature():
    f = Fig("l03-temperature", 960, 600, "Temperature: sharpen or flatten",
            "From CME295 Lecture 3. p_i proportional to exp(z_i / T).")
    cases = [("T -> 0", "spiky", "argmax wins.\nDeterministic in theory.", TEAL, 0.95),
             ("T = 1", "normal", "the trained\ndistribution.", FOCUS, 0.6),
             ("T -> inf", "flat", "uniform.\nPure noise.", ORANGE, 0.25)]
    for i, (t, kind, d, col, pk) in enumerate(cases):
        x = 60 + i * 280
        f.box(x, 120, 260, 320, rx=12)
        f.text(x + 130, 156, t, size=20, weight=600, font=MONO)
        f.text(x + 130, 190, kind, size=15, fill=col, weight=500)
        bars = [pk, pk * 0.55, pk * 0.3]
        for b, v in enumerate(bars):
            f.parts.append(f'<rect x="{x + 40 + b * 70}" y="{380 - v * 180}" width="48" height="{v * 180:.0f}" rx="6" fill="{col}" opacity="0.8"/>')
        f.text(x + 130, 420, d, size=14, fill=MUTED)
    f.box(60, 480, 840, 64, fill=COUNT, rx=8)
    f.text(480, 512, "Nothing in the transformer is probabilistic except sampling. T=0 can still vary on GPUs.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l03_guided_decoding():
    f = Fig("l03-guided-decoding", 960, 560, "Guided decoding: force the shape",
            "From CME295 Lecture 3. Constrain sampling to valid tokens: JSON, FSM, grammar.")
    f.pill(60, 200, 220, 56, '"name": "', mono=False, size=15, fill=CHIP)
    f.arrow(280, 228, 340, 228)
    f.box(340, 140, 280, 176, fill=NEWTOK, rx=12)
    f.text(480, 176, "Allowed next tokens", size=16, weight=600)
    f.text(480, 216, '"Alice", "Bob", ...\n(not "}" , not "42")', size=14, font=MONO, fill=MUTED)
    f.text(480, 280, "mask the rest to zero", size=14, fill=TEAL, weight=500)
    f.arrow(620, 228, 680, 228)
    f.pill(680, 200, 220, 56, '"name": "Alice"', mono=False, size=15, fill=NEWTOK)
    f.box(60, 380, 840, 110, fill=COUNT, rx=8)
    f.text(480, 416, "The model can only sample tokens the grammar allows. Output always parses.",
           size=15, weight=500)
    f.text(480, 452, "Use it for JSON APIs, code skeletons, form filling.", size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l03_prompting():
    f = Fig("l03-prompting", 960, 680, "Prompting: program the model with text",
            "From CME295 Lecture 3. Anatomy plus the main techniques.")
    f.box(60, 110, 840, 150, rx=12)
    f.text(480, 142, "Prompt anatomy", size=19, weight=600)
    parts = [("Context", "background facts"), ("Instructions", "what to do"),
             ("Inputs", "the data"), ("Constraints", "format, length, tone")]
    for i, (k, v) in enumerate(parts):
        x = 90 + i * 205
        f.pill(x, 160, 190, 48, k, mono=False, size=14, fill=COUNT)
        f.text(x + 95, 228, v, size=13, fill=MUTED)
    techs = [("Zero-shot", "no examples", "Works when the task is standard."),
             ("Few-shot", "2-5 examples in context", "Shows the pattern to copy."),
             ("Chain-of-thought", '"think step by step"', "More tokens = more compute."),
             ("Self-consistency", "sample N, majority vote", "Noise cancels out.")]
    for i, (k, v, d) in enumerate(techs):
        y = 290 + i * 88
        f.box(60, y, 260, 72, fill=ACTIVE, rx=12)
        f.text(190, y + 36, k, size=16, weight=600)
        f.text(360, y + 24, v, size=15, weight=500, anchor="start", font=MONO)
        f.text(360, y + 52, d, size=14, fill=MUTED, anchor="start")
    f.src()
    f.save()


@fig
def l03_kv_paged():
    f = Fig("l03-kv-paged", 960, 640, "KV cache and PagedAttention",
            "From CME295 Lecture 3. Cache keys and values. Page them like OS memory.")
    f.text(24, 120, "Without cache: recompute all K, V every step. With cache: append one row.",
           size=15, weight=500, anchor="start")
    for i in range(6):
        f.box(60 + i * 70, 160, 60, 90, fill=COUNT if i < 5 else NEWTOK, rx=8)
        f.text(90 + i * 70, 205, f"t{i + 1}", size=13, font=MONO)
    f.text(540, 205, "+ new token", size=14, fill=TEAL, weight=500)
    f.box(60, 300, 400, 160, rx=12)
    f.text(260, 336, "PagedAttention (vLLM)", size=18, weight=600)
    f.text(260, 374, "Split cache into blocks\n(block size ~16 [uncertain]).", size=14, fill=MUTED)
    f.text(260, 430, "Kills fragmentation.", size=14, fill=TEAL, weight=500)
    f.box(500, 300, 400, 160, rx=12)
    f.text(700, 336, "MLA (DeepSeek V2)", size=18, weight=600)
    f.text(700, 374, "Compress K, V into\nlatent vectors.", size=14, fill=MUTED)
    f.text(700, 430, "Less memory per token.", size=14, fill=TEAL, weight=500)
    f.box(60, 510, 840, 70, fill=COUNT, rx=8)
    f.text(480, 545, "Decoding is memory-bound: moving the cache costs more than the math.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l03_speculative():
    f = Fig("l03-speculative", 960, 600, "Speculative decoding: draft, then verify",
            "From CME295 Lecture 3. Small model drafts. Big model accepts or rejects.")
    f.box(60, 140, 380, 130, fill=COUNT, rx=12)
    f.text(250, 176, "Draft model (small)", size=18, weight=600)
    f.text(250, 214, "proposes k tokens fast", size=14, fill=MUTED)
    f.pill(110, 236, 60, 40, "a", size=14)
    f.pill(180, 236, 60, 40, "b", size=14)
    f.pill(250, 236, 60, 40, "c", size=14)
    f.pill(320, 236, 60, 40, "d", size=14)
    f.arrow(440, 205, 520, 205, "propose")
    f.box(520, 140, 380, 130, fill=ACTIVE, rx=12)
    f.text(710, 176, "Target model (large)", size=18, weight=600)
    f.text(710, 214, "verifies all k in one pass", size=14, fill=MUTED)
    f.pill(570, 236, 60, 40, "a", size=14, fill=NEWTOK)
    f.pill(640, 236, 60, 40, "b", size=14, fill=NEWTOK)
    f.pill(710, 236, 60, 40, "c", size=14, fill="#F3D4D8")
    f.pill(780, 236, 60, 40, "x", size=14, fill="#F3D4D8")
    f.text(480, 330, "Accept a, b. Reject c, d. Resample from the target at c. Repeat.",
           size=15, fill=MUTED)
    f.box(60, 390, 840, 140, fill=COUNT, rx=8)
    f.text(480, 428, "Why it wins: verification is one memory pass for k tokens.", size=15, weight=500)
    f.text(480, 466, "Multi-token prediction: train the model to emit k tokens per step. Same idea, learned.",
           size=14, fill=MUTED)
    f.src()
    f.save()


# ---------------------------------------------------------------- L04
@fig
def l04_pretrain_scale():
    f = Fig("l04-pretrain-scale", 960, 600, "Pre-training: the most expensive step",
            "From CME295 Lecture 4. Next-token prediction on the internet.")
    rows = [("Common Crawl", "~3B pages / month", "raw web text", FOCUS),
            ("GPT-3", "300B tokens", "2020 scale", TEAL),
            ("Llama 3", "15T tokens", "2024 scale, 50x GPT-3", ORANGE)]
    for i, (k, v, d, col) in enumerate(rows):
        y = 110 + i * 100
        f.box(60, y, 240, 84, fill=col, rx=12)
        f.text(180, y + 42, k, size=17, weight=600, fill="#FFFFFF")
        f.text(340, y + 30, v, size=16, weight=600, font=MONO, anchor="start")
        f.text(340, y + 60, d, size=14, fill=MUTED, anchor="start")
    f.box(60, 440, 840, 100, fill=COUNT, rx=8)
    f.text(480, 476, "Cost: millions to hundreds of millions of dollars per run.", size=15, weight=500)
    f.text(480, 510, "Side effects: knowledge cutoff. Editing knowledge later is hard. Plagiarism risk.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l04_flops():
    f = Fig("l04-flops", 960, 520, "FLOPs vs FLOPS: work vs speed",
            "From CME295 Lecture 4. Capital S matters.")
    f.box(80, 140, 380, 220, rx=12, fill=COUNT)
    f.text(270, 180, "FLOPs", size=26, weight=600, font=MONO)
    f.text(270, 224, "floating-point operations", size=15, fill=MUTED)
    f.text(270, 268, "Total work.\nHow big is the job?", size=15, weight=500)
    f.box(500, 140, 380, 220, rx=12, fill=ACTIVE)
    f.text(690, 180, "FLOPS", size=26, weight=600, font=MONO)
    f.text(690, 224, "operations per second", size=15, fill=MUTED)
    f.text(690, 268, "Rate.\nHow fast is the machine?", size=15, weight=500)
    f.text(480, 430, "H100 80GB: ~34 TFLOPS in FP64. Training = FLOPs / FLOPS, times efficiency.",
           size=15, fill=MUTED)
    f.src()
    f.save()


@fig
def l04_scaling_laws():
    f = Fig("l04-scaling-laws", 960, 620, "Scaling laws: bigger is better, then Chinchilla",
            "From CME295 Lecture 4. Kaplan 2020. Chinchilla: 20x tokens per parameter.")
    f.box(60, 110, 400, 240, rx=12)
    f.text(260, 146, "Kaplan et al., 2020", size=18, weight=600)
    f.text(260, 186, "Loss falls as a power law in\nparams, data, and compute.", size=15, fill=MUTED)
    f.text(260, 260, "Bigger model = better loss.\nNo sign of stopping.", size=15, fill=TEAL, weight=500)
    f.box(500, 110, 400, 240, rx=12)
    f.text(700, 146, "Chinchilla, 2022", size=18, weight=600)
    f.text(700, 186, "Fix compute. Split it between\nparams and tokens.", size=15, fill=MUTED)
    f.text(700, 260, "Optimal: ~20 tokens per param.\nGPT-3 was undertrained.", size=15, fill=ORANGE, weight=500)
    f.box(60, 400, 840, 150, fill=COUNT, rx=8)
    f.text(480, 438, "Rule of thumb: tokens >= 20 x parameters.", size=17, weight=600, font=MONO)
    f.text(480, 478, "100B params -> train on at least 2T tokens.", size=15, fill=MUTED)
    f.text(480, 512, "Llama 3 (15T tokens) followed this. GPT-3 (300B) did not.", size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l04_parallelism():
    f = Fig("l04-parallelism", 960, 640, "Parallelism: split the work",
            "From CME295 Lecture 4. Data, ZeRO, tensor, pipeline.")
    rows = [("Data parallel", "each GPU: full model, different batch", "simple, needs full model per GPU"),
            ("ZeRO 1/2/3", "shard optimizer, gradients, params", "fits bigger models, more comms"),
            ("Tensor parallel", "split layers across GPUs", "one forward pass, many GPUs"),
            ("Pipeline parallel", "split layers into stages", "bubbles, but scales deep")]
    for i, (k, v, d) in enumerate(rows):
        y = 110 + i * 118
        f.box(60, y, 260, 100, fill=COUNT, rx=12)
        f.text(190, y + 50, k, size=16, weight=600)
        f.text(360, y + 36, v, size=15, weight=500, anchor="start", font=MONO)
        f.text(360, y + 68, d, size=14, fill=MUTED, anchor="start")
    f.src()
    f.save()


@fig
def l04_flashattention():
    f = Fig("l04-flashattention", 960, 640, "FlashAttention: respect the memory hierarchy",
            "From CME295 Lecture 4. Stanford, 2022. ~10x fewer HBM accesses. Exact, not approximate.")
    f.box(60, 130, 400, 200, fill=COUNT, rx=12)
    f.text(260, 166, "HBM", size=22, weight=600)
    f.text(260, 204, "big, slow\nall of Q, K, V live here", size=15, fill=MUTED)
    f.box(500, 130, 400, 200, fill=NEWTOK, rx=12)
    f.text(700, 166, "SRAM", size=22, weight=600)
    f.text(700, 204, "small, fast\non-chip, per block", size=15, fill=MUTED)
    f.arrow(460, 230, 500, 230, "tiles")
    steps = [("1. Load a tile of Q, K, V to SRAM.", "2. Block softmax trick: running max + sum, no full matrix."),
             ("3. Write the output tile back.", "4. Recompute, do not store: extra math beats extra memory trips.")]
    for i, (a, b) in enumerate(steps):
        y = 370 + i * 56
        f.text(60, y, a, size=15, weight=500, anchor="start")
        f.text(480, y, b, size=15, fill=MUTED, anchor="start")
    f.box(60, 510, 840, 70, fill=COUNT, rx=8)
    f.text(480, 545, "Insight: memory moves dominate. Recomputing is cheaper than re-reading.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l04_quant():
    f = Fig("l04-quant", 960, 600, "Quantization: fewer bits per number",
            "From CME295 Lecture 4. Mixed precision: FP32 weights, FP16 compute.")
    rows = [("FP64", "64", "reference, slow", MUTED),
            ("FP32", "32", "weights in mixed precision", FOCUS),
            ("FP16 / BF16", "16", "compute in mixed precision", TEAL),
            ("INT8", "8", "inference, needs care", ORANGE)]
    for i, (k, b, d, col) in enumerate(rows):
        y = 110 + i * 92
        f.box(60, y, 200, 76, fill=col, rx=8)
        f.text(160, y + 38, k, size=17, weight=600, fill="#FFFFFF", font=MONO)
        f.text(300, y + 26, b + " bits", size=15, weight=500, anchor="start", font=MONO)
        f.text(300, y + 54, d, size=14, fill=MUTED, anchor="start")
    f.box(60, 500, 840, 56, fill=COUNT, rx=8)
    f.text(480, 528, "Mixed precision: keep weights in FP32, do the math in FP16. Speed without drift.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l04_sft():
    f = Fig("l04-sft", 960, 640, "SFT: teach the format, not the facts",
            "From CME295 Lecture 4. Loss on output tokens only. Instruction tuning.")
    f.box(60, 120, 840, 120, rx=12)
    f.text(480, 152, "Prompt (no loss):  How do I fix my washer?", size=16, weight=500)
    f.text(480, 196, "Answer (loss here):  First unplug it, then ...", size=16, weight=600, fill=TEAL)
    f.box(60, 280, 400, 180, rx=12)
    f.text(260, 316, "Data mixture", size=18, weight=600)
    f.text(260, 356, "instructions\nsafety + hedging\nmulti-turn chat", size=14, fill=MUTED)
    f.box(500, 280, 400, 180, rx=12)
    f.text(700, 316, "Scale of SFT", size=18, weight=600)
    f.text(700, 356, "GPT-3 era: ~13K examples\nLlama 3 era: ~10M examples", size=14, fill=MUTED)
    f.text(700, 424, "Quality beats quantity.", size=14, fill=TEAL, weight=500)
    f.box(60, 510, 840, 70, fill=COUNT, rx=8)
    f.text(480, 545, "SFT teaches behavior: follow instructions, be helpful, refuse safely. Facts come from pre-training.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l04_eval_challenges():
    f = Fig("l04-eval-challenges", 960, 600, "Evaluation is hard: three traps",
            "From CME295 Lecture 4. Benchmarks measure, but they can mislead.")
    rows = [("MMLU", "~50 academic tasks", "broad knowledge, multiple choice"),
            ("GSM-8K", "grade-school math", "reasoning, exact answers"),
            ("Chatbot Arena", "pairwise human votes", "vibes, but riggable and brittle")]
    for i, (k, v, d) in enumerate(rows):
        y = 110 + i * 110
        f.box(60, y, 240, 94, fill=COUNT, rx=12)
        f.text(180, y + 47, k, size=17, weight=600)
        f.text(340, y + 32, v, size=15, weight=500, anchor="start", font=MONO)
        f.text(340, y + 62, d, size=14, fill=MUTED, anchor="start")
    f.box(60, 470, 840, 70, fill="#F3D4D8", rx=8)
    f.text(480, 505, 'Trap: "training on the test task". Optimizing the benchmark is not improving the model.',
           size=15, weight=500, fill=ORANGE)
    f.src()
    f.save()


@fig
def l04_lora():
    f = Fig("l04-lora", 960, 600, "LoRA: freeze the giant, train the delta",
            "From CME295 Lecture 4. W = W0 + BA. QLoRA: NF4 + double quantization, ~16x VRAM cut.")
    f.box(60, 140, 240, 140, fill=CHIP, rx=8)
    f.text(180, 180, "W0", size=22, weight=600, font=MONO)
    f.text(180, 216, "frozen", size=14, fill=MUTED)
    f.text(340, 210, "+", size=28, weight=600)
    f.box(380, 140, 180, 140, fill=NEWTOK, rx=8)
    f.text(470, 180, "B x A", size=22, weight=600, font=MONO)
    f.text(470, 216, "rank r ~ 4", size=14, fill=TEAL, weight=500)
    f.text(610, 210, "=", size=28, weight=600)
    f.box(650, 140, 240, 140, fill=ACTIVE, rx=8)
    f.text(770, 180, "W", size=22, weight=600, font=MONO)
    f.text(770, 216, "adapted", size=14, fill=MUTED)
    f.box(60, 340, 840, 190, fill=COUNT, rx=8)
    f.text(480, 378, "Practice: 10x the learning rate vs full fine-tuning. FFN blocks benefit most.",
           size=15, weight=500)
    f.text(480, 420, "QLoRA: 4-bit NormalFloat weights, BF16 adapters, double quantization.",
           size=15, fill=MUTED)
    f.text(480, 460, "~16x less VRAM. Fine-tune a 65B-class model on one GPU.", size=15, fill=MUTED)
    f.src()
    f.save()


# ---------------------------------------------------------------- L05
@fig
def l05_why_pref():
    f = Fig("l05-why-pref", 960, 600, "Why preference tuning exists",
            "From CME295 Lecture 5. SFT shows what to do. It never shows what not to do.")
    f.box(60, 140, 400, 200, rx=12)
    f.text(260, 176, "SFT gives", size=18, weight=600)
    f.text(260, 216, "good examples only\npositive signal", size=15, fill=MUTED)
    f.text(260, 290, "the model imitates", size=15, fill=TEAL, weight=500)
    f.box(500, 140, 400, 200, rx=12)
    f.text(700, 176, "SFT cannot give", size=18, weight=600)
    f.text(700, 216, "bad examples\nnegative signal", size=15, fill=MUTED)
    f.text(700, 290, "no way to say: not this", size=15, fill=ORANGE, weight=500)
    f.box(60, 400, 840, 130, fill=COUNT, rx=8)
    f.text(480, 438, "Preference tuning: show pairs, say which is better. Cheaper than rewriting SFT data.",
           size=15, weight=500)
    f.text(480, 478, "It does not teach new facts. It teaches tone, safety, and taste.",
           size=15, fill=MUTED)
    f.src()
    f.save()


@fig
def l05_pairs():
    f = Fig("l05-pairs", 960, 620, "Preference pairs: the washer example",
            "From CME295 Lecture 5. Pairwise is the standard. Pointwise and listwise exist.")
    f.box(60, 120, 840, 110, rx=12)
    f.text(480, 152, "Prompt: my washer is broken, what do I do?", size=16, weight=500)
    f.text(480, 192, "Two answers. A human picks the better one.", size=14, fill=MUTED)
    f.box(60, 260, 400, 170, fill="#F3D4D8", rx=12)
    f.text(260, 296, "Rejected", size=17, weight=600, fill=ORANGE)
    f.text(260, 336, "Factually correct\nbut rough and blunt.", size=14, fill=MUTED)
    f.box(500, 260, 400, 170, fill=NEWTOK, rx=12)
    f.text(700, 296, "Chosen", size=17, weight=600, fill=TEAL)
    f.text(700, 336, "Same facts.\nGentler, clearer, safer.", size=14, fill=MUTED)
    f.arrow(460, 345, 500, 345, "prefer")
    f.box(60, 480, 840, 80, fill=COUNT, rx=8)
    f.text(480, 520, "Pairs come from human logs or rewrites. Binary choice keeps annotation simple.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l05_rl_frame():
    f = Fig("l05-rl-frame", 960, 600, "RL framing: the LLM is an agent",
            "From CME295 Lecture 5. State, action, policy, sparse reward.")
    rows = [("Agent", "the LLM itself", "it acts in the world of tokens"),
            ("State", "input tokens so far", "prompt + generated prefix"),
            ("Action", "the next token", "one step, one token"),
            ("Policy", "pi_theta", "the model weights"),
            ("Reward", "one signal per completion", "sparse: only at the end")]
    for i, (k, v, d) in enumerate(rows):
        y = 110 + i * 88
        f.box(60, y, 220, 72, fill=COUNT, rx=12)
        f.text(170, y + 36, k, size=16, weight=600)
        f.text(320, y + 24, v, size=15, weight=500, anchor="start", font=MONO)
        f.text(320, y + 52, d, size=14, fill=MUTED, anchor="start")
    f.src()
    f.save()


@fig
def l05_bradley_terry():
    f = Fig("l05-bradley-terry", 960, 620, "Reward model: Bradley-Terry",
            "From CME295 Lecture 5. Train pairwise, use pointwise.")
    f.box(60, 120, 840, 120, fill=COUNT, rx=8)
    f.text(480, 158, "P(y_i succeeds y_j) = sigma(r_i - r_j)", size=24, weight=600, font=SERIF)
    f.text(480, 200, "Probability i is preferred = sigmoid of the score gap.", size=15, fill=MUTED)
    f.box(60, 280, 400, 160, rx=12)
    f.text(260, 316, "Training", size=18, weight=600)
    f.text(260, 354, "pairwise: (chosen, rejected)\nloss = -E[log sigma(r_w - r_l)]", size=14, font=MONO, fill=MUTED)
    f.box(500, 280, 400, 160, rx=12)
    f.text(700, 316, "Inference", size=18, weight=600)
    f.text(700, 354, "pointwise: one text in\none score out", size=14, fill=MUTED)
    f.text(700, 410, "subtle and important", size=14, fill=TEAL, weight=500)
    f.box(60, 490, 840, 70, fill=COUNT, rx=8)
    f.text(480, 525, "Tens of thousands of pairs. RewardBench measures reward models. Dims: useful, friendly, safe.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l05_ppo():
    f = Fig("l05-ppo", 960, 640, "PPO: move, but not too far",
            "From CME295 Lecture 5. Maximize the clipped objective. r = pi_theta / pi_old is a ratio.")
    f.box(60, 120, 840, 110, fill=COUNT, rx=8)
    f.text(480, 158, "L = min( r * A , clip(r, 1-eps, 1+eps) * A )", size=22, weight=600, font=SERIF)
    f.text(480, 198, "maximize this. r is a probability ratio, not a reward.", size=15, fill=MUTED)
    f.box(60, 270, 400, 180, rx=12)
    f.text(260, 306, "Clip", size=18, weight=600)
    f.text(260, 344, "If A > 0: cap the upside.\nIf A < 0: floor the downside.", size=14, fill=MUTED)
    f.text(260, 414, "no giant steps", size=14, fill=TEAL, weight=500)
    f.box(500, 270, 400, 180, rx=12)
    f.text(700, 306, "KL penalty", size=18, weight=600)
    f.text(700, 344, "beta * KL(pi || pi_ref)\nstay near the SFT model", size=14, fill=MUTED)
    f.text(700, 414, "fights reward hacking", size=14, fill=TEAL, weight=500)
    f.text(480, 510, "Four models in memory: policy, reference, reward (frozen), value. ~100k+ rollouts.",
           size=15, fill=MUTED)
    f.src()
    f.save()


@fig
def l05_advantage():
    f = Fig("l05-advantage", 960, 600, "Advantage: reward minus baseline",
            "From CME295 Lecture 5. GAE blends multi-step estimates. Value head predicts per token.")
    f.box(60, 140, 840, 110, fill=COUNT, rx=8)
    f.text(480, 178, "A_t = reward - baseline", size=24, weight=600, font=SERIF)
    f.text(480, 216, "How much better was this action than average?", size=15, fill=MUTED)
    f.box(60, 300, 400, 180, rx=12)
    f.text(260, 336, "GAE", size=18, weight=600)
    f.text(260, 374, "Generalized advantage estimation.\nMixes 1-step and n-step returns.", size=14, fill=MUTED)
    f.box(500, 300, 400, 180, rx=12)
    f.text(700, 336, "Value function", size=18, weight=600)
    f.text(700, 374, "Regression head on the model.\nPredicts reward per token.", size=14, fill=MUTED)
    f.text(700, 440, "trained jointly", size=14, fill=TEAL, weight=500)
    f.src()
    f.save()


@fig
def l05_dpo():
    f = Fig("l05-dpo", 960, 620, "DPO: the reward model is secretly the policy",
            "From CME295 Lecture 5. Two models, no RL loop. beta ~ 0.1.")
    steps = [("1. Write the RLHF objective", "maximize reward - beta * KL"),
             ("2. Solve for the optimal policy", "pi* in closed form"),
             ("3. Plug pi* into Bradley-Terry", "reward cancels out"),
             ("4. Train pi directly on pairs", "plain supervised loss")]
    for i, (a, b) in enumerate(steps):
        y = 110 + i * 100
        f.box(60, y, 840, 84, rx=12, fill=NEWTOK if i == 3 else PANEL)
        f.text(110, y + 42, str(i + 1), size=20, weight=600, fill=TEAL)
        f.text(170, y + 30, a, size=16, weight=600, anchor="start")
        f.text(170, y + 58, b, size=14, fill=MUTED, anchor="start")
    f.box(60, 530, 840, 56, fill=COUNT, rx=8)
    f.text(480, 558, "Cheaper than PPO. Caveat: pairs come from another policy, distribution shift bites.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l05_compare():
    f = Fig("l05-compare", 960, 600, "PPO vs DPO vs best-of-N",
            "From CME295 Lecture 5. Pick by budget and goal.")
    rows = [("PPO", "4 models, RL loop", "best reported results", "heavy, unstable, tuning-hungry"),
            ("DPO", "2 models, no RL", "cheap alignment", "weaker, shift-sensitive"),
            ("Best-of-N", "N samples, pick best", "no training at all", "N times the inference cost")]
    for i, (k, v, w, d) in enumerate(rows):
        y = 110 + i * 130
        f.box(60, y, 200, 114, fill=COUNT, rx=12)
        f.text(160, y + 57, k, size=19, weight=600)
        f.text(300, y + 32, v, size=15, weight=500, anchor="start", font=MONO)
        f.text(300, y + 62, "+ " + w, size=14, fill=TEAL, anchor="start")
        f.text(300, y + 90, "- " + d, size=14, fill=ORANGE, anchor="start")
    f.src()
    f.save()


# ---------------------------------------------------------------- L06
@fig
def l06_weaknesses():
    f = Fig("l06-weaknesses", 960, 600, "Vanilla LLMs: four weaknesses",
            "From CME295 Lecture 6. Reasoning models attack the first one.")
    items = [("Limited reasoning", "one-shot answers fail\non math and code", ORANGE),
             ("Static knowledge", "frozen at the cutoff date", FOCUS),
             ("No action", "all talk, no tools", TEAL),
             ("Hard to evaluate", "BLEU and ROUGE miss\nfree-form quality", MUTED)]
    for i, (k, v, col) in enumerate(items):
        x = 60 + i * 216
        f.box(x, 140, 200, 280, rx=12)
        f.parts.append(f'<rect x="{x + 80}" y="170" width="40" height="40" rx="20" fill="{col}"/>')
        f.text(x + 100, 250, k, size=15, weight=600)
        f.text(x + 100, 300, v, size=13, fill=MUTED)
    f.box(60, 470, 840, 70, fill=COUNT, rx=8)
    f.text(480, 505, "Lecture 6 fixes reasoning. Lecture 7 fixes knowledge and action. Lecture 8 fixes evaluation.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l06_reasoning_model():
    f = Fig("l06-reasoning-model", 960, 620, "Reasoning model: think, then answer",
            "From CME295 Lecture 6. Chains are usually hidden. You pay for them as output tokens.")
    f.pill(60, 200, 200, 56, "prompt", mono=False, size=16, fill=CHIP)
    f.arrow(260, 228, 320, 228)
    f.box(320, 140, 300, 176, fill=ACTIVE, rx=12)
    f.text(470, 176, "reasoning chain", size=17, weight=600)
    f.text(470, 216, "hidden from user\nthought summary shown", size=14, fill=MUTED)
    f.text(470, 280, "billed as output", size=14, fill=ORANGE, weight=500)
    f.arrow(620, 228, 680, 228)
    f.pill(680, 200, 220, 56, "final answer", mono=False, size=16, fill=NEWTOK)
    f.box(60, 380, 840, 170, fill=COUNT, rx=8)
    f.text(480, 416, "Timeline: o1 preview Sep 2024 -> Gemini 2.0 Flash Thinking Dec 2024 -> DeepSeek R1 Jan 2025.",
           size=15, weight=500)
    f.text(480, 456, "Intuition: hard problems split into tractable subproblems. More tokens = more compute.",
           size=14, fill=MUTED)
    f.text(480, 494, "Summaries, not raw chains: raw chains confuse users, run long, and train competitors.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l06_passk():
    f = Fig("l06-passk", 960, 620, "pass@k: at least one success in k tries",
            "From CME295 Lecture 6. Estimate from n samples with c successes. Sample without replacement.")
    f.box(60, 120, 840, 120, fill=COUNT, rx=8)
    f.text(480, 158, "pass@k = 1 - C(n-c, k) / C(n, k)", size=24, weight=600, font=SERIF)
    f.text(480, 200, "1 minus the chance that all k draws miss. pass@1 = c / n.", size=15, fill=MUTED)
    f.box(60, 280, 840, 170, rx=12)
    f.text(480, 316, "Temperature vs pass@k", size=18, weight=600)
    f.text(200, 356, "T = 0: flat, no diversity", size=14, fill=MUTED, anchor="start")
    f.text(200, 388, "T = 0.2-0.8: sweet spot", size=14, fill=TEAL, weight=500, anchor="start")
    f.text(200, 420, "T = 1.2: too diverse, hurts", size=14, fill=ORANGE, anchor="start")
    f.text(620, 380, "consensus@k:\nmajority vote\n(self-consistency)", size=14, fill=MUTED)
    f.box(60, 490, 840, 70, fill=COUNT, rx=8)
    f.text(480, 525, "Papers must report temperature. Without it, pass@k numbers do not compare.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l06_benchmarks():
    f = Fig("l06-benchmarks", 960, 620, "Reasoning benchmarks: answers you can check",
            "From CME295 Lecture 6. Code runs tests. Math parses an answer.")
    rows = [("HumanEval", "~100+ hand-written problems", "code, pass tests"),
            ("Codeforces / SWE-bench", "contests, real GitHub issues", "code, hard"),
            ("AIME", "US math olympiad qualifier", "math, 3-digit answer"),
            ("GSM-8K", "grade-school word problems", "math, exact match")]
    for i, (k, v, d) in enumerate(rows):
        y = 110 + i * 110
        f.box(60, y, 260, 94, fill=COUNT, rx=12)
        f.text(190, y + 47, k, size=16, weight=600)
        f.text(360, y + 32, v, size=15, weight=500, anchor="start")
        f.text(360, y + 62, d, size=14, fill=MUTED, anchor="start")
    f.box(60, 560 - 40, 840, 56, fill=NEWTOK, rx=8)
    f.text(480, 548, "Verifiable rewards: the checker is free, so RL can run without humans.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l06_grpo():
    f = Fig("l06-grpo", 960, 640, "GRPO: advantage from the group",
            "From CME295 Lecture 6. Group Relative Policy Optimization. No value function.")
    f.pill(380, 110, 200, 48, "one prompt", mono=False, size=16, fill=CHIP)
    for i in range(4):
        x = 90 + i * 200
        f.box(x, 200, 180, 120, rx=12, fill=PANEL)
        f.text(x + 90, 236, f"completion {i + 1}", size=14, weight=600)
        f.text(x + 90, 272, f"reward r{i + 1}", size=14, font=MONO, fill=MUTED)
        f.arrow(480, 158, x + 90, 200)
    f.box(60, 380, 840, 110, fill=COUNT, rx=8)
    f.text(480, 416, "A_i = (r_i - mean(r)) / std(r)", size=24, weight=600, font=SERIF)
    f.text(480, 458, "Z-score inside the group. Hard problems upweight automatically.", size=15, fill=MUTED)
    f.box(60, 530, 840, 56, fill=ACTIVE, rx=8)
    f.text(480, 558, "Only the policy trains. KL to the reference model keeps it close.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l06_ppo_grpo():
    f = Fig("l06-ppo-grpo", 960, 620, "PPO vs GRPO: same goals, different machinery",
            "From CME295 Lecture 6. Both use the ratio and clipping.")
    f.box(60, 120, 840, 90, fill=COUNT, rx=8)
    f.text(480, 165, "Same: ratio pi / pi_old. Same: clipping keeps updates small.", size=16, weight=500)
    rows = [("Advantage", "reward - value model (GAE)", "(r - mean) / std over group"),
            ("KL", "folded into the advantage", "explicit in the objective"),
            ("Trains", "policy + value model", "policy only")]
    for i, (k, p, g) in enumerate(rows):
        y = 240 + i * 110
        f.box(60, y, 200, 94, fill=CHIP, rx=8)
        f.text(160, y + 47, k, size=16, weight=600)
        f.box(276, y, 296, 94, rx=8)
        f.text(424, y + 26, "PPO", size=14, weight=600, fill=MUTED)
        f.text(424, y + 58, p, size=14, fill=MUTED)
        f.box(588, y, 312, 94, rx=8, fill=NEWTOK)
        f.text(744, y + 26, "GRPO", size=14, weight=600, fill=TEAL)
        f.text(744, y + 58, g, size=14, fill=INK)
    f.src()
    f.save()


@fig
def l06_length_bias():
    f = Fig("l06-length-bias", 960, 620, "Length bias: GRPO rewards long failures",
            "From CME295 Lecture 6. The 1/|o_i| term. Fixed by DAPO and Dr. GRPO.")
    f.box(60, 120, 840, 110, fill=COUNT, rx=8)
    f.text(480, 158, "token weight ~ 1 / |output length|", size=22, weight=600, font=SERIF)
    f.text(480, 198, "Tokens in short outputs count more than tokens in long ones.", size=15, fill=MUTED)
    f.box(60, 270, 400, 150, fill="#F3D4D8", rx=12)
    f.text(260, 306, "Negative advantage", size=17, weight=600)
    f.text(260, 346, "short bad output:\ndownweighted MORE", size=14, fill=MUTED)
    f.text(260, 396, "model prefers long bad outputs", size=14, fill=ORANGE, weight=500)
    f.box(500, 270, 400, 150, fill=NEWTOK, rx=12)
    f.text(700, 306, "Fixes", size=17, weight=600)
    f.text(700, 346, "DAPO: equalize token weights\nDr. GRPO: drop the term", size=14, fill=MUTED)
    f.text(700, 396, "wrong answers get shorter", size=14, fill=TEAL, weight=500)
    f.text(480, 480, "Also: clip-higher uses asymmetric epsilon. Low-prob tokens need room to grow.",
           size=15, fill=MUTED)
    f.src()
    f.save()


@fig
def l06_r1_pipeline():
    f = Fig("l06-r1-pipeline", 960, 700, "DeepSeek R1: the full recipe",
            "From CME295 Lecture 6. R1-Zero proves RL works. R1 makes it usable.")
    steps = [("V3 base", ["pretrained MoE + MLA"], COUNT),
             ("Cold-start SFT", ["small, human-rewritten", "CoTs"], ACTIVE),
             ("RL", ["accuracy + format +", "language rewards"], NEWTOK),
             ("Big SFT", ["rejection sampling, 3:1", "200k non-reasoning"], ACTIVE),
             ("Final RL", ["reasoning + helpful +", "harmless"], NEWTOK)]
    for i, (k, v, col) in enumerate(steps):
        x = 36 + i * 180
        f.box(x, 130, 168, 170, fill=col, rx=12)
        f.text(x + 84, 176, k, size=14, weight=600)
        f.text(x + 84, 224, "\n".join(v), size=12, fill=MUTED)
        if i < 4:
            f.arrow(x + 168, 215, x + 180, 215)
    f.box(36, 350, 888, 110, fill=COUNT, rx=8)
    f.text(480, 388, "R1-Zero: no SFT at all. AIME accuracy climbs with RL steps. But chains mix languages.",
           size=15, weight=500)
    f.text(480, 428, "Fixes: cold-start SFT for format, language-consistency reward, harmlessness on think tokens.",
           size=14, fill=MUTED)
    f.box(36, 500, 888, 130, fill=ACTIVE, rx=8)
    f.text(480, 538, "Distillation: R1 writes answers with thoughts. Small model fits the sequences. Beats RL from scratch.",
           size=15, weight=500)
    f.text(480, 578, "Result: competitive with o1-mini at small sizes.", size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l06_thinking_control():
    f = Fig("l06-thinking-control", 960, 600, "Thinking control: budget the thoughts",
            "From CME295 Lecture 6. Reasoning tokens cost money and context.")
    rows = [("Dynamic budget", "classifier picks high/low thinking", "easy questions stay cheap"),
            ("Budget forcing (s1)", '"wait" to continue, "time is up" to stop', "steer length at inference"),
            ("Continuous thoughts", "reason in hidden states, not tokens", "research stage")]
    for i, (k, v, d) in enumerate(rows):
        y = 120 + i * 130
        f.box(60, y, 260, 114, fill=ACTIVE, rx=12)
        f.text(190, y + 57, k, size=16, weight=600)
        f.text(360, y + 40, v, size=15, weight=500, anchor="start", font=MONO)
        f.text(360, y + 74, d, size=14, fill=MUTED, anchor="start")
    f.box(60, 530, 840, 56, fill=COUNT, rx=8)
    f.text(480, 558, "Incentive: users pay per reasoning token. Shorter thoughts, same score, wins.",
           size=15, weight=500)
    f.src()
    f.save()


# ---------------------------------------------------------------- L07
@fig
def l07_cutoff():
    f = Fig("l07-cutoff", 960, 620, "The knowledge cutoff problem",
            "From CME295 Lecture 7. The model only knows its training data.")
    f.box(60, 130, 840, 100, rx=12)
    f.text(480, 166, '"Who won the election two weeks ago?"', size=19, weight=600)
    f.text(480, 204, "Cutoff was a month ago. The model guesses or refuses.", size=15, fill=MUTED)
    rows = [("Retrain?", "tricky: regressions, maintenance per use case", "avoid"),
            ("Dump everything in context?", "limited ctx, needle-in-haystack, $/token", "avoid"),
            ("Retrieve only what matters", "RAG: relevant chunks in the prompt", "do this")]
    for i, (k, v, d) in enumerate(rows):
        y = 270 + i * 100
        f.box(60, y, 200, 84, fill=NEWTOK if i == 2 else PANEL, rx=12)
        f.text(160, y + 42, k, size=16, weight=600)
        f.text(300, y + 30, v, size=14, fill=MUTED, anchor="start")
        f.text(300, y + 58, d, size=15, weight=600, fill=TEAL if i == 2 else ORANGE, anchor="start")
    f.src()
    f.save()


@fig
def l07_rag_pipeline():
    f = Fig("l07-rag-pipeline", 960, 600, "RAG: retrieve, augment, generate",
            "From CME295 Lecture 7. Retrieval Augmented Generation. Relevant is the whole game.")
    steps = [("RETRIEVE", "find relevant\nchunks", COUNT),
             ("AUGMENT", "paste them\ninto the prompt", ACTIVE),
             ("GENERATE", "LLM answers\nwith context", NEWTOK)]
    for i, (k, v, col) in enumerate(steps):
        x = 60 + i * 280
        f.box(x, 160, 240, 170, fill=col, rx=12)
        f.text(x + 120, 210, k, size=20, weight=600)
        f.text(x + 120, 260, v, size=15, fill=MUTED)
        if i < 2:
            f.arrow(x + 240, 245, x + 280, 245)
    f.pill(120, 400, 720, 56, '"Who won the local election?" + [retrieved: results article]', mono=False, size=14)
    f.text(480, 510, "The prompt now contains the answer. The LLM just has to read it.",
           size=15, fill=MUTED)
    f.src()
    f.save()


@fig
def l07_kb_chunks():
    f = Fig("l07-kb-chunks", 960, 620, "Knowledge base: chunk and embed",
            "From CME295 Lecture 7. Three hyperparameters to tune.")
    f.pill(60, 150, 200, 56, "documents", mono=False, size=15, fill=CHIP)
    f.arrow(260, 178, 320, 178, "chunk")
    for i in range(4):
        f.box(330 + i * 150, 140, 136, 76, fill=COUNT, rx=8)
        f.text(398 + i * 150, 178, f"chunk {i + 1}", size=14, weight=500)
    f.arrow(480, 216, 480, 260, "embed")
    f.box(330, 260, 300, 72, fill=ACTIVE, rx=8)
    f.text(480, 296, "vector index", size=16, weight=600)
    rows = [("Chunk size", "~500 tokens", "too small: out of context. too big: blurry embedding."),
            ("Overlap", "low hundreds of tokens", "carry context across the cut."),
            ("Embedding size", "~1500 dims", "bigger: nuanced. smaller: cheap.")]
    for i, (k, v, d) in enumerate(rows):
        y = 370 + i * 72
        f.text(60, y, k + ":", size=15, weight=600, anchor="start")
        f.text(240, y, v, size=15, font=MONO, anchor="start")
        f.text(440, y, d, size=14, fill=MUTED, anchor="start")
    f.src()
    f.save()


@fig
def l07_two_stage():
    f = Fig("l07-two-stage", 960, 620, "Retrieval in two stages",
            "From CME295 Lecture 7. Borrowed from search and recommenders.")
    f.box(60, 130, 400, 260, rx=12)
    f.text(260, 166, "1. Candidate retrieval", size=19, weight=600)
    f.text(260, 206, "bi-encoder: embed query,\nembed chunks, cosine sim", size=14, fill=MUTED)
    f.text(260, 280, "millions -> ~100", size=16, font=MONO, fill=TEAL, weight=600)
    f.text(260, 316, "goal: recall. ANN index.", size=14, fill=MUTED)
    f.box(500, 130, 400, 260, rx=12)
    f.text(700, 166, "2. Rerank", size=19, weight=600)
    f.text(700, 206, "cross-encoder: query + chunk\ntogether, one score", size=14, fill=MUTED)
    f.text(700, 280, "~100 -> top k", size=16, font=MONO, fill=TEAL, weight=600)
    f.text(700, 316, "goal: precision. costs more.", size=14, fill=MUTED)
    f.arrow(460, 260, 500, 260)
    f.box(60, 440, 840, 110, fill=COUNT, rx=8)
    f.text(480, 478, "Bi-encoder: fast, no interaction. Cross-encoder: slow, full attention between query and chunk.",
           size=15, weight=500)
    f.text(480, 514, "Sentence-BERT: the recommended read for training bi-encoders.", size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l07_semantic_bm25():
    f = Fig("l07-semantic-bm25", 960, 620, "Semantic vs keyword search",
            "From CME295 Lecture 7. The teddy bear test: where is Cuddly?")
    f.box(60, 130, 400, 220, rx=12)
    f.text(260, 166, "Embeddings", size=18, weight=600)
    f.text(260, 206, "same meaning, different words\nno keyword guarantee", size=14, fill=MUTED)
    f.text(260, 280, '"Where is Cuddly?"\nmay return Huggy docs', size=14, fill=ORANGE, weight=500)
    f.box(500, 130, 400, 220, rx=12)
    f.text(700, 166, "BM25", size=18, weight=600)
    f.text(700, 206, "heuristic keyword overlap\nmust share words", size=14, fill=MUTED)
    f.text(700, 280, '"Where is Cuddly?"\nreturns docs with "Cuddly"', size=14, fill=TEAL, weight=500)
    f.box(60, 410, 840, 130, fill=NEWTOK, rx=8)
    f.text(480, 448, "Hybrid: combine both scores. Keyword precision plus semantic recall.",
           size=16, weight=600)
    f.text(480, 488, "Choose by use case: exact names and codes favor BM25.", size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l07_extensions():
    f = Fig("l07-extensions", 960, 640, "RAG extensions: three upgrades",
            "From CME295 Lecture 7. HyDE, contextual retrieval, prompt caching.")
    rows = [("HyDE", "LLM writes a fake answer doc, embed that", "query and docs finally look alike"),
            ("Contextual retrieval", "LLM writes a short context per chunk", "chunks make sense alone"),
            ("Prompt caching", "same prefix computed once", "~1/10 price on cached tokens")]
    for i, (k, v, d) in enumerate(rows):
        y = 110 + i * 150
        f.box(60, y, 240, 134, fill=ACTIVE, rx=12)
        f.text(180, y + 67, k, size=17, weight=600)
        f.text(340, y + 50, v, size=15, weight=500, anchor="start")
        f.text(340, y + 84, d, size=14, fill=MUTED, anchor="start")
    f.box(60, 580 - 40, 840, 56, fill=COUNT, rx=8)
    f.text(480, 568, "All three trade extra LLM calls for better retrieval. Cache the repeated prefixes.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l07_metrics():
    f = Fig("l07-metrics", 960, 640, "Retrieval metrics: is the ranker good?",
            "From CME295 Lecture 7. Same as search. MTEB is the benchmark.")
    rows = [("NDCG", "relevant docs ranked high score more", "normalized by the ideal ranking"),
            ("Reciprocal rank", "1 / rank of first relevant doc", "simple, correlates well"),
            ("Precision@k", "of the top k, how many relevant", "quality of what you show"),
            ("Recall@k", "of all relevant, how many in top k", "coverage of what exists")]
    for i, (k, v, d) in enumerate(rows):
        y = 110 + i * 110
        f.box(60, y, 240, 94, fill=COUNT, rx=12)
        f.text(180, y + 47, k, size=16, weight=600)
        f.text(340, y + 32, v, size=15, weight=500, anchor="start")
        f.text(340, y + 62, d, size=14, fill=MUTED, anchor="start")
    f.box(60, 570 - 40, 840, 56, fill=COUNT, rx=8)
    f.text(480, 558, "Labels: which chunks are truly relevant. Everything compares against that.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l07_toolcall():
    f = Fig("l07-toolcall", 960, 680, "Tool calling: three stages",
            "From CME295 Lecture 7. IBM definition: complete tasks via external resources.")
    f.box(60, 120, 840, 100, fill=COUNT, rx=8)
    f.text(480, 152, '"Find a teddy bear near me."', size=18, weight=600)
    f.text(480, 190, "The model sees the API + docstring. Not the implementation.", size=14, fill=MUTED)
    steps = [("1. Predict", "query + API -> arguments\nfind_teddy_bear(loc=Stanford)", COUNT),
             ("2. Execute", "run the function\n(no LLM involved)", ACTIVE),
             ("3. Respond", "structured result ->\nnatural language answer", NEWTOK)]
    for i, (k, v, col) in enumerate(steps):
        x = 60 + i * 280
        f.box(x, 270, 240, 190, fill=col, rx=12)
        f.text(x + 120, 310, k, size=19, weight=600)
        f.text(x + 120, 360, v, size=14, fill=MUTED)
        if i < 2:
            f.arrow(x + 240, 365, x + 280, 365)
    f.box(60, 520, 840, 100, fill=COUNT, rx=8)
    f.text(480, 556, "Train with 2 SFT pair types: tool prediction, and history-to-final-answer.",
           size=15, weight=500)
    f.text(480, 590, "Or skip SFT: strong models follow a well-written explanation, tuned offline on an eval set.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l07_router_mcp():
    f = Fig("l07-router-mcp", 960, 640, "Many tools: route, then standardize",
            "From CME295 Lecture 7. Tool selector. MCP from Anthropic.")
    f.box(60, 120, 400, 220, rx=12)
    f.text(260, 156, "Tool selector", size=18, weight=600)
    f.text(260, 196, "query + tool names\n(one-line descriptions)", size=14, fill=MUTED)
    f.text(260, 270, "LLM picks the few\nrelevant tools", size=14, fill=TEAL, weight=500)
    f.text(260, 314, "fixes the haystack problem", size=13, fill=MUTED)
    f.box(500, 120, 400, 220, rx=12)
    f.text(700, 156, "MCP", size=18, weight=600)
    f.text(700, 196, "Model Context Protocol\nservers serve tools", size=14, fill=MUTED)
    f.text(700, 270, "tools, prompts, resources\nclient <-> server 1:1", size=14, fill=TEAL, weight=500)
    f.text(700, 314, "write once, use everywhere", size=13, fill=MUTED)
    f.box(60, 400, 840, 170, fill=COUNT, rx=8)
    f.text(480, 438, "Categories: informational (search APIs), computation (code execution), actions (send email).",
           size=15, weight=500)
    f.text(480, 478, "Too many tools in context: lost APIs, conflicting APIs, finite window.",
           size=14, fill=MUTED)
    f.text(480, 514, "You cannot fit every user's tools. Route first.", size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l07_react():
    f = Fig("l07-react", 960, 700, "ReAct: observe, plan, act, repeat",
            "From CME295 Lecture 7. Agents = tool loops with reasoning. Thermostat example.")
    f.pill(60, 150, 260, 56, '"My teddy bear is cold."', mono=False, size=14, fill=CHIP)
    f.arrow(320, 178, 380, 178)
    steps = [("OBSERVE", "cold -> temperature?\nunknown, must measure", COUNT),
             ("PLAN", "determine room temp\ntool: get_temp()", ACTIVE),
             ("ACT", "temp = 65F\ncolder than expected", NEWTOK),
             ("PLAN", "increase by 5 degrees\ntool: set_temp()", ACTIVE),
             ("OBSERVE", "temp correct\n-> exit loop", COUNT)]
    x = 380
    for i, (k, v, col) in enumerate(steps):
        f.box(x, 120, 96, 116, fill=col, rx=8)
        f.text(x + 48, 152, k, size=11, weight=600)
        f.text(x + 48, 196, v, size=10, fill=MUTED)
        if i < 4:
            f.arrow(x + 96, 178, x + 108, 178)
        x += 108
    f.box(60, 330, 840, 120, fill=COUNT, rx=8)
    f.text(480, 368, "Each loop: check the goal. Reached it? Answer. Not yet? Reason and act again.",
           size=15, weight=500)
    f.text(480, 406, "A2A protocol (Google): standardize agent-to-agent skills, status, cancel.",
           size=14, fill=MUTED)
    f.box(60, 500, 840, 140, fill="#F3D4D8", rx=8)
    f.text(480, 536, "7 failure modes: punt, tool hallucination, wrong tool, wrong args, bad output, no output, bad synthesis.",
           size=15, weight=500, fill=ORANGE)
    f.text(480, 574, "Safety: data exfiltration via tools. Train harmlessness. Add inference classifiers.",
           size=14, fill=MUTED)
    f.src()
    f.save()


# ---------------------------------------------------------------- L08
@fig
def l08_eval_scope():
    f = Fig("l08-eval-scope", 960, 560, "Evaluation: this lecture is about output quality",
            "From CME295 Lecture 8. Not latency, not price. How good is the response?")
    f.box(60, 140, 840, 120, rx=12)
    f.text(480, 176, "The LLM outputs free-form text.", size=19, weight=600)
    f.text(480, 216, "Natural language, code, math. No universal metric exists.", size=15, fill=MUTED)
    f.box(60, 310, 260, 150, fill=COUNT, rx=12)
    f.text(190, 350, "Human ratings", size=17, weight=600)
    f.text(190, 392, "ideal but slow\nand expensive", size=14, fill=MUTED)
    f.box(350, 310, 260, 150, fill=ACTIVE, rx=12)
    f.text(480, 350, "Rule metrics", size=17, weight=600)
    f.text(480, 392, "compare to reference\nBLEU, ROUGE, METEOR", size=14, fill=MUTED)
    f.box(640, 310, 260, 150, fill=NEWTOK, rx=12)
    f.text(770, 350, "LLM-as-judge", size=17, weight=600)
    f.text(770, 392, "LLM grades the output\nscore + rationale", size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l08_human():
    f = Fig("l08-human", 960, 600, "Human ratings: the ideal you cannot afford",
            "From CME295 Lecture 8. Subjective tasks break simple agreement.")
    f.box(60, 130, 840, 110, rx=12)
    f.text(480, 166, '"What birthday gift should I get?"', size=18, weight=600)
    f.text(480, 204, '"A teddy bear is almost always a sweet gift." Useful? Two raters disagree.', size=15, fill=MUTED)
    rows = [("Slow", "rating 1000 outputs takes days", "expensive"),
            ("Subjective", "usefulness is in the eye of the rater", "needs guidelines"),
            ("Drifty", "raters disagree with each other", "measure agreement")]
    for i, (k, v, d) in enumerate(rows):
        y = 280 + i * 96
        f.box(60, y, 200, 80, fill=COUNT, rx=12)
        f.text(160, y + 40, k, size=16, weight=600)
        f.text(300, y + 28, v, size=15, weight=500, anchor="start")
        f.text(300, y + 56, d, size=14, fill=ORANGE, anchor="start")
    f.src()
    f.save()


@fig
def l08_kappa():
    f = Fig("l08-kappa", 960, 620, "Agreement rate lies. Kappa corrects it.",
            "From CME295 Lecture 8. Random raters already agree 50% of the time.")
    f.box(60, 120, 840, 120, fill=COUNT, rx=8)
    f.text(480, 158, "P(agree) = P_A * P_B + (1-P_A) * (1-P_B)", size=22, weight=600, font=SERIF)
    f.text(480, 200, "If both rate randomly at 0.5, agreement is 0.25 + 0.25 = 0.5. By chance alone.",
           size=15, fill=MUTED)
    f.box(60, 280, 400, 200, rx=12)
    f.text(260, 316, "Cohen's kappa", size=18, weight=600)
    f.text(260, 354, "2 raters\n(observed - chance) /\n(1 - chance)", size=14, fill=MUTED)
    f.text(260, 440, "positive = better than chance", size=14, fill=TEAL, weight=500)
    f.box(500, 280, 400, 200, rx=12)
    f.text(700, 316, "Extensions", size=18, weight=600)
    f.text(700, 354, "Fleiss kappa: many raters\nKrippendorff alpha:\nmissing data", size=14, fill=MUTED)
    f.box(60, 520, 840, 56, fill=COUNT, rx=8)
    f.text(480, 548, "Practice: track agreement as a health metric. Hold alignment sessions when it drops.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l08_rule_metrics():
    f = Fig("l08-rule-metrics", 960, 640, "Rule-based metrics: compare to a reference",
            "From CME295 Lecture 8. Humans write the reference once. Then compare forever.")
    rows = [("BLEU", "precision over n-grams + brevity penalty", "translation", "gaming: output little"),
            ("ROUGE", "recall-flavored overlap", "summarization", "many variants"),
            ("METEOR", "F-score x (1 - ordering penalty)", "translation", "synonyms, stems, arbitrary knobs")]
    for i, (k, v, d, n) in enumerate(rows):
        y = 110 + i * 130
        f.box(60, y, 200, 114, fill=COUNT, rx=12)
        f.text(160, y + 40, k, size=19, weight=600)
        f.text(160, y + 74, d, size=14, fill=MUTED)
        f.text(300, y + 40, v, size=15, weight=500, anchor="start", font=MONO)
        f.text(300, y + 74, "limit: " + n, size=14, fill=ORANGE, anchor="start")
    f.box(60, 520, 840, 70, fill="#F3D4D8", rx=8)
    f.text(480, 555, "Core flaw: paraphrases score badly. Correlation with humans is weak.",
           size=15, weight=500, fill=ORANGE)
    f.src()
    f.save()


@fig
def l08_judge():
    f = Fig("l08-judge", 960, 640, "LLM-as-a-judge: grade with a model",
            "From CME295 Lecture 8. Input: prompt + response + criteria. Output: rationale, then score.")
    f.box(60, 120, 840, 110, rx=12)
    f.text(480, 156, "Inputs", size=18, weight=600)
    f.text(480, 194, "the prompt, the model response, the grading criteria", size=15, fill=MUTED)
    f.arrow(480, 230, 480, 270)
    f.box(60, 270, 400, 160, fill=ACTIVE, rx=12)
    f.text(260, 306, "Rationale FIRST", size=18, weight=600)
    f.text(260, 346, "explain what is good\nor bad, then score", size=14, fill=MUTED)
    f.text(260, 404, "empirically better", size=14, fill=TEAL, weight=500)
    f.box(500, 270, 400, 160, fill=NEWTOK, rx=12)
    f.text(700, 306, "Score", size=18, weight=600)
    f.text(700, 346, "binary: pass / fail\neasier for models\nand humans", size=14, fill=MUTED)
    f.box(60, 480, 840, 100, fill=COUNT, rx=8)
    f.text(480, 516, "No reference text needed. Parse with structured output (constrained decoding).",
           size=15, weight=500)
    f.text(480, 552, "Pairwise mode generates synthetic preference labels for reward models.", size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l08_biases():
    f = Fig("l08-biases", 960, 640, "Judge biases: three to watch",
            "From CME295 Lecture 8. None of these lists is exhaustive.")
    rows = [("Position bias", "prefers whichever answer came first", "ask both orders, majority vote"),
            ("Verbosity bias", "prefers longer answers", "guidelines, examples, length penalty"),
            ("Self-enhancement", "prefers its own outputs", "different judge, ideally bigger")]
    for i, (k, v, d) in enumerate(rows):
        y = 110 + i * 150
        f.box(60, y, 260, 134, fill="#F3D4D8", rx=12)
        f.text(190, y + 67, k, size=17, weight=600, fill=ORANGE)
        f.text(360, y + 50, "symptom: " + v, size=15, weight=500, anchor="start")
        f.text(360, y + 84, "fix: " + d, size=14, fill=TEAL, anchor="start")
    f.box(60, 580 - 40, 840, 56, fill=COUNT, rx=8)
    f.text(480, 568, "Best practice: low temperature (0.1-0.2) for reproducibility. Calibrate against human ratings.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l08_factuality():
    f = Fig("l08-factuality", 960, 640, "Factuality: check fact by fact",
            "From CME295 Lecture 8. Binary per fact, weighted aggregate. Example scores 0.6.")
    steps = [("1. Extract", "LLM splits text\ninto facts", COUNT),
             ("2. Check each", "RAG / web search\ncorrect or not", ACTIVE),
             ("3. Aggregate", "weighted mean\nof binaries", NEWTOK)]
    for i, (k, v, col) in enumerate(steps):
        x = 60 + i * 280
        f.box(x, 150, 240, 180, fill=col, rx=12)
        f.text(x + 120, 196, k, size=19, weight=600)
        f.text(x + 120, 246, v, size=14, fill=MUTED)
        if i < 2:
            f.arrow(x + 240, 240, x + 280, 240)
    f.box(60, 390, 840, 110, fill=COUNT, rx=8)
    f.text(480, 428, '"Teddy bears, first created in the 1920s..." -> 4 facts, 2 wrong.', size=15, weight=500)
    f.text(480, 466, "Wrong: 1920s (it was 1900s). Wrong: the president proudly wanted to shoot (he refused).",
           size=14, fill=MUTED)
    f.box(60, 540, 840, 56, fill=COUNT, rx=8)
    f.text(480, 568, "Weights alpha_i let important facts count more. Nuance without hand-waving.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l08_agent_failures():
    f = Fig("l08-agent-failures", 960, 780, "Seven ways agents fail",
            "From CME295 Lecture 8. Prediction, execution, synthesis. Fix failures in groups.")
    stages = [("Prediction", [( "the punt", "needed a tool, did not call one"),
                               ("tool hallucination", "find_bear, not find_teddy_bear"),
                               ("wrong tool", "misread the docs"),
                               ("wrong args", "bad parameters")]),
              ("Execution", [("bad tool output", "bugs in the result"),
                              ("no output", "silence invites false ok.\nempty JSON beats none.")]),
              ("Synthesis", [("bad synthesis", "good tool result, unused.")])]
    y = 110
    for stage, items in stages:
        f.text(90, y, stage, size=17, weight=600, anchor="start")
        yy = y + 16
        for name, desc in items:
            f.box(90, yy, 780, 64, fill="#F3D4D8", rx=8)
            f.text(116, yy + 26, name, size=15, weight=600, fill=ORANGE, anchor="start")
            f.text(116, yy + 48, desc, size=14, fill=MUTED, anchor="start")
            yy += 76
        y = yy + 24
    f.src()
    f.save()


@fig
def l08_benchmarks():
    f = Fig("l08-benchmarks", 960, 680, "Benchmark taxonomy: five families",
            "From CME295 Lecture 8. Knowledge, reasoning, coding, safety, agents.")
    rows = [("MMLU", "~60 tasks, 4-choice", "knowledge. measures pretraining.", TEAL),
            ("AIME / PIQA", "3-digit math, 2-choice common sense", "reasoning. 20k PIQA examples.", FOCUS),
            ("SWE-bench", "real GitHub issues + tests", "coding. patch must pass.", TEAL),
            ("HarmBench", "classifier judges attempts", "safety. provider policies differ.", ORANGE),
            ("tau-bench", "airline + retail, simulated user", "agents. pass-hat-k: all k succeed.", FOCUS)]
    for i, (k, v, d, col) in enumerate(rows):
        y = 100 + i * 104
        f.box(60, y, 220, 88, fill=col, rx=12)
        f.text(170, y + 44, k, size=16, weight=600, fill="#FFFFFF")
        f.text(320, y + 30, v, size=15, weight=500, anchor="start", font=MONO)
        f.text(320, y + 60, d, size=14, fill=MUTED, anchor="start")
    f.src()
    f.save()


@fig
def l08_pareto():
    f = Fig("l08-pareto", 960, 620, "Read benchmarks like an adult",
            "From CME295 Lecture 8. Pareto, contamination, Goodhart.")
    f.box(60, 120, 840, 130, rx=12)
    f.text(480, 156, "Pareto frontier", size=19, weight=600)
    f.text(480, 196, "Best model per dollar. Also per safety level, per context length.", size=15, fill=MUTED)
    f.text(480, 226, "Sonnet for code, Gemini Flash for cheap-and-fast: personal, not universal.", size=14, fill=MUTED)
    f.box(60, 290, 400, 130, fill="#F3D4D8", rx=12)
    f.text(260, 326, "Contamination", size=18, weight=600, fill=ORANGE)
    f.text(260, 366, "benchmarks leak into training.\nhashes, blocklists, fresh tests.", size=14, fill=MUTED)
    f.box(500, 290, 400, 130, fill="#F3D4D8", rx=12)
    f.text(700, 326, "Goodhart's law", size=18, weight=600, fill=ORANGE)
    f.text(700, 366, '"when a measure becomes\na target, it ceases to\nbe a good measure."', size=14, fill=MUTED)
    f.box(60, 470, 840, 90, fill=COUNT, rx=8)
    f.text(480, 505, "Chatbot Arena balances with real usage. Final test: try the models yourself.",
           size=15, weight=500)
    f.src()
    f.save()


# ---------------------------------------------------------------- L09
@fig
def l09_arc():
    f = Fig("l09-arc", 960, 600, "The quarter in one map",
            "From CME295 Lecture 9. Final covers lectures 5-8. Midterm covered 1-4.")
    items = [("L1", "transformer"), ("L2", "attention+"), ("L3", "LLMs"),
             ("L4", "training"), ("L5", "pref tuning"), ("L6", "reasoning"),
             ("L7", "RAG/agents"), ("L8", "evaluation")]
    for i, (k, v) in enumerate(items):
        x = 40 + i * 112
        mid = i == 3
        f.box(x, 180, 100, 150, fill=ACTIVE if mid else COUNT, rx=12)
        f.text(x + 50, 224, k, size=18, weight=600)
        f.text(x + 50, 262, v, size=12, fill=MUTED)
        if i < 7:
            f.arrow(x + 100, 255, x + 112, 255)
    f.box(40, 400, 420, 120, fill=COUNT, rx=8)
    f.text(250, 438, "Midterm", size=17, weight=600)
    f.text(250, 474, "lectures 1-4", size=14, fill=MUTED)
    f.box(500, 400, 420, 120, fill=NEWTOK, rx=8)
    f.text(710, 438, "Final", size=17, weight=600)
    f.text(710, 474, "lectures 5-8", size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l09_vit():
    f = Fig("l09-vit", 960, 640, "Vision Transformer: patches are tokens",
            "From CME295 Lecture 9. ViT, 2020. Enough data beats inductive bias.")
    f.box(80, 150, 180, 180, fill=CHIP, rx=8)
    f.text(170, 240, "image", size=16, weight=600)
    for r in range(3):
        for c in range(3):
            f.parts.append(f'<rect x="{110 + c * 40}" y="{180 + r * 40}" width="36" height="36" fill="{PANEL}" stroke="{INK}" stroke-width="1"/>')
    f.arrow(260, 240, 320, 240, "patches")
    for i in range(3):
        f.box(330 + i * 90, 200, 80, 80, fill=COUNT, rx=8)
        f.text(370 + i * 90, 240, f"p{i + 1}", size=13, font=MONO)
    f.pill(610, 205, 90, 70, "[CLS]", size=14, fill=FOCUS)
    f.arrow(700, 240, 760, 240)
    f.box(760, 150, 140, 180, fill=ACTIVE, rx=12)
    f.text(830, 220, "encoder", size=16, weight=600)
    f.text(830, 260, "self-\nattention", size=13, fill=MUTED)
    f.arrow(610, 330, 610, 380)
    f.box(460, 380, 300, 110, fill=NEWTOK, rx=12)
    f.text(610, 418, "[CLS] -> FFN -> class", size=16, weight=600)
    f.text(610, 456, '"teddy bear"', size=14, fill=MUTED)
    f.box(80, 540, 800, 56, fill=COUNT, rx=8)
    f.text(480, 568, "CNNs slide a window (strong bias). ViT lets every patch attend to every patch (weak bias).",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l09_vlm():
    f = Fig("l09-vlm", 960, 620, "Vision-language models: two wirings",
            "From CME295 Lecture 9. LLaVA concatenates. Llama 3 uses cross-attention.")
    f.box(60, 130, 400, 220, rx=12)
    f.text(260, 166, "Method 1: concatenate", size=18, weight=600)
    f.text(260, 206, "image tokens + text tokens\n-> decoder-only LLM", size=14, fill=MUTED)
    f.text(260, 280, "LLaVA. Most common.", size=14, fill=TEAL, weight=500)
    f.box(500, 130, 400, 220, rx=12)
    f.text(700, 166, "Method 2: cross-attention", size=18, weight=600)
    f.text(700, 206, "text in the stream,\nimages at cross-attn layers", size=14, fill=MUTED)
    f.text(700, 280, "Llama 3. Less common.", size=14, fill=MUTED, weight=500)
    f.box(60, 410, 840, 140, fill=COUNT, rx=8)
    f.text(480, 448, "Same transformer idea, other direction: diffusion transformers (DiT) generate images.",
           size=15, weight=500)
    f.text(480, 488, "Also used in recommendation, speech, and more. The architecture travels.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l09_diffusion():
    f = Fig("l09-diffusion", 960, 640, "Diffusion: carve the image out of noise",
            "From CME295 Lecture 9. Start from Gaussian noise. Learn to denoise.")
    f.box(80, 160, 200, 160, fill=CHIP, rx=8)
    f.text(180, 240, "noise", size=18, weight=600)
    f.text(180, 272, "Gaussian\n easy to sample", size=13, fill=MUTED)
    f.arrow(280, 240, 360, 240, "denoise x N")
    f.box(360, 160, 200, 160, fill=ACTIVE, rx=8)
    f.text(460, 240, "less noise", size=18, weight=600)
    f.arrow(560, 240, 640, 240)
    f.box(640, 160, 200, 160, fill=NEWTOK, rx=8)
    f.text(740, 240, "image", size=18, weight=600)
    f.box(80, 390, 800, 170, fill=COUNT, rx=8)
    f.text(480, 428, "Sculptor analogy (Michelangelo): the statue is in the marble. Remove what is not statue.",
           size=15, weight=500)
    f.text(480, 468, "Forward process: add noise gradually. Reverse: predict the noise to remove.",
           size=14, fill=MUTED)
    f.text(480, 504, "Why noise: models well, samples easily, adds the randomness you want.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l09_mdm():
    f = Fig("l09-mdm", 960, 660, "Masked diffusion LMs: noise = [MASK]",
            "From CME295 Lecture 9. MDM / DLLM. LLaDA, early 2025. Draft-to-refine, not left-to-right.")
    f.text(60, 140, "Forward: mask more and more.", size=15, weight=500, anchor="start")
    toks = ["The", "bear", "is", "cute"]
    for i, t in enumerate(toks):
        f.pill(60 + i * 120, 160, 108, 44, t, mono=False, size=13)
    f.arrow(560, 182, 620, 182)
    for i in range(4):
        f.pill(630 + i * 70, 160, 60, 44, "[M]", size=13, fill=CHIP)
    f.text(60, 260, "Reverse: unmask all at once, conditioned on the prompt. Repeat for N steps.",
           size=15, weight=500, anchor="start")
    f.box(60, 310, 840, 120, fill=NEWTOK, rx=8)
    f.text(480, 348, "Fewer forward passes: N steps, not one per token. ~10x faster on long outputs.",
           size=16, weight=600)
    f.text(480, 388, "N controls quality. N is far smaller than output length.", size=14, fill=MUTED)
    f.box(60, 480, 840, 120, fill=COUNT, rx=8)
    f.text(480, 516, "Wins: speed, and fill-in-the-middle coding (sees both directions).",
           size=15, weight=500)
    f.text(480, 552, "Catching up to autoregressive, not at the frontier yet. Reasoning for diffusion: open work.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l09_pollination():
    f = Fig("l09-pollination", 960, 620, "Ideas cross-pollinate both ways",
            "From CME295 Lecture 9. Text borrows from vision. Vision borrows from text.")
    f.box(60, 130, 400, 200, rx=12)
    f.text(260, 166, "Vision -> text", size=18, weight=600)
    f.text(260, 206, "diffusion for images\n-> masked diffusion LMs", size=14, fill=MUTED)
    f.text(260, 286, "lower latency generation", size=14, fill=TEAL, weight=500)
    f.box(500, 130, 400, 200, rx=12)
    f.text(700, 166, "Text -> vision", size=18, weight=600)
    f.text(700, 206, "transformers replace\nconvolutions (DiT)", size=14, fill=MUTED)
    f.text(700, 286, "RoPE reformulated in 2D", size=14, fill=TEAL, weight=500)
    f.box(60, 390, 840, 160, fill=COUNT, rx=8)
    f.text(480, 428, "DeepSeek OCR: image patches as tokens carry text meaning with very few tokens.",
           size=15, weight=500)
    f.text(480, 466, "Tokenizers may not be the best tool. Patches already encode emojis and layout.",
           size=14, fill=MUTED)
    f.text(480, 504, "Lesson: keep an open mind for non-text transformer applications.", size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l09_design():
    f = Fig("l09-design", 960, 640, "The design space is still open",
            "From CME295 Lecture 9. Every choice below is under active debate.")
    rows = [("Optimizer", "Adam challenged. Muon / MuonClip (Kimi K2).", "may become the standard"),
            ("Normalization", "post -> pre. layer -> RMSNorm.", "location and type both move"),
            ("Attention", "per-layer variants, not one design", "every paper picks its own"),
            ("Activations", "ReLU -> GELU and friends", "new ones still appear"),
            ("Scale", "MoE or dense? heads? FFN size?", "nothing is settled")]
    for i, (k, v, d) in enumerate(rows):
        y = 110 + i * 96
        f.box(60, y, 220, 80, fill=ACTIVE, rx=12)
        f.text(170, y + 40, k, size=16, weight=600)
        f.text(320, y + 28, v, size=14, weight=500, anchor="start")
        f.text(320, y + 56, d, size=14, fill=MUTED, anchor="start")
    f.src()
    f.save()


@fig
def l09_data():
    f = Fig("l09-data", 960, 620, "Data: the internet is now synthetic",
            "From CME295 Lecture 9. Model collapse is real. Curation is the answer.")
    f.box(60, 130, 400, 200, rx=12)
    f.text(260, 166, "Then", size=18, weight=600)
    f.text(260, 206, "scrape the web\nhuman-written text\ntrain next-token", size=14, fill=MUTED)
    f.box(500, 130, 400, 200, fill="#F3D4D8", rx=12)
    f.text(700, 166, "Now", size=18, weight=600, fill=ORANGE)
    f.text(700, 206, "~80% of search results\nLLM-generated [uncertain]\nless diverse", size=14, fill=MUTED)
    f.box(60, 390, 840, 160, fill=COUNT, rx=8)
    f.text(480, 428, "Model collapse: training on synthetic text narrows the distribution. Learning degrades.",
           size=15, weight=500)
    f.text(480, 466, "Response: data curation companies. Mid-training: large but higher-quality corpus.",
           size=14, fill=MUTED)
    f.text(480, 504, "Pipeline grows: pretrain -> mid-train -> fine-tune.", size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l09_frontier():
    f = Fig("l09-frontier", 960, 660, "What comes next",
            "From CME295 Lecture 9. Smaller, cheaper, weirder hardware. Stay current.")
    rows = [("Small LMs", "the second Pareto border: quality per dollar", "providers lose money on top tiers"),
            ("Hardware", "GPUs love matmul; attention wants more", "analog compute: physics does the math"),
            ("Agents for all", "Atlas-style browsing assistants", "security first: prompt injection, exfiltration"),
            ("Open problems", "continuous learning, hallucinations, personalization", "weights are frozen; RAG is a patch")]
    for i, (k, v, d) in enumerate(rows):
        y = 110 + i * 110
        f.box(60, y, 220, 94, fill=NEWTOK, rx=12)
        f.text(170, y + 47, k, size=16, weight=600)
        f.text(320, y + 32, v, size=15, weight=500, anchor="start")
        f.text(320, y + 62, d, size=14, fill=MUTED, anchor="start")
    f.box(60, 570, 840, 56, fill=COUNT, rx=8)
    f.text(480, 598, "Stay current: arXiv, HF trending papers, X community, Karpathy and Kilcher videos, company blogs.",
           size=15, weight=500)
    f.src()
    f.save()


# ---------------------------------------------------------------- L01 new plates
@fig
def l01_onehot():
    f = Fig("l01-onehot", 960, 620, "One-hot vs embedding: sparse index to dense meaning",
            "From CME295 Lecture 1. One-hot carries no meaning. Embeddings learn it.")
    f.box(60, 120, 400, 220, fill=PANEL, rx=12)
    f.text(260, 150, "one-hot: \"cat\"", size=16, weight=600)
    f.text(260, 190, "50,000 dims, one 1", size=14, fill=MUTED)
    f.text(260, 230, "[0, 0, ..., 1, ..., 0]", size=14, font=MONO)
    f.text(260, 270, "cat vs dog: distance = sqrt(2)", size=14, fill=MUTED)
    f.text(260, 305, "every word equally far", size=14, fill=ORANGE, weight=500)
    f.box(500, 120, 400, 220, fill=NEWTOK, rx=12)
    f.text(700, 150, "embedding: \"cat\"", size=16, weight=600)
    f.text(700, 190, "a few hundred dims, all dense", size=14, fill=MUTED)
    f.text(700, 230, "[0.2, -1.1, ..., 0.7]", size=14, font=MONO)
    f.text(700, 270, "cat near dog, far from car", size=14, fill=MUTED)
    f.text(700, 305, "distance carries meaning", size=14, fill=TEAL, weight=500)
    f.arrow(460, 230, 500, 230, teal=True)
    f.on_line(480, 200, "learn")
    f.box(60, 380, 840, 160, fill=COUNT, rx=8)
    f.text(480, 418, "One-hot is an index. The embedding matrix turns indices into geometry.",
           size=15, weight=500)
    f.text(480, 458, "Geometry is what attention compares: similar vectors score high.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l01_bpe():
    f = Fig("l01-bpe", 960, 640, "BPE: merge the most frequent pair, repeat",
            "From CME295 Lecture 1. Worked on a toy corpus: low low lower.")
    f.text(480, 120, "corpus: l o w | l o w | l o w e r", size=16, font=MONO)
    steps = [("merge 1: e+r -> er", "\"low\", \"lower\" -> \"low\", \"low er\""),
             ("merge 2: er -> ?", "keep merging until the vocab budget fills"),
             ("result: \"lower\" = low + er", "common words stay whole; rare words split")]
    for i, (k, v) in enumerate(steps):
        y = 170 + i * 110
        f.box(120, y, 720, 94, fill=PANEL if i < 2 else NEWTOK, rx=12)
        f.text(200, y + 47, k, size=15, weight=600, font=MONO)
        f.text(560, y + 47, v, size=14, fill=MUTED)
    f.box(60, 510, 840, 70, fill=COUNT, rx=8)
    f.text(480, 545, "Frequent strings become tokens. Rare strings compose from pieces. No word is ever unknown.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l01_causal_mask():
    f = Fig("l01-causal-mask", 960, 640, "The causal mask: -inf above the diagonal",
            "From CME295 Lecture 1. Training parallelizes. Inference does not.")
    f.text(240, 120, "scores QK^T", size=16, weight=600)
    f.text(720, 120, "masked, then softmax", size=16, weight=600)
    toks = ["the", "cat", "sat", "down"]
    cs = 64
    for mi, mx in enumerate([140, 620]):
        for i in range(4):
            f.text(mx - 40, 190 + i * cs + 20, toks[i], size=13, fill=MUTED, anchor="end")
            f.text(mx + 20 + i * cs, 165, toks[i], size=13, fill=MUTED)
            for j in range(4):
                x, y = mx + j * cs, 180 + i * cs
                masked = j > i
                f.parts.append(
                    f'<rect x="{x}" y="{y}" width="60" height="60" fill="{"#E8E2D5" if masked else "#E7F4EF"}" stroke="{INK}" stroke-width="1"/>')
                f.text(x + 30, y + 32, "-inf" if masked else "s", size=13, font=MONO,
                       fill=MUTED if masked else TEAL)
    f.box(60, 480, 840, 90, fill=COUNT, rx=8)
    f.text(480, 512, "Token 3 sees tokens 1-3, never token 4. One forward pass scores every position at once.",
           size=15, weight=500)
    f.text(480, 544, "At inference the future does not exist yet: decode one token per pass.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l01_model_map():
    f = Fig("l01-model-map", 960, 680, "What is used where: October 2026",
            "From CME295 Lecture 1. Public facts only. Closed labs do not publish internals.")
    rows = [("DeepSeek V4.1 Flash", "causal encoder-decoder MoE, 8B/16B active", "open weights, MIT"),
            ("Llama 4 Maverick", "decoder-only MoE, 17B active, 128 experts", "open weights"),
            ("BERT (2018)", "encoder-only, bidirectional", "search, classification, embeddings"),
            ("T5 (2019)", "encoder-decoder", "input and output differ in kind"),
            ("GPT-6 / Gemini 3.8 Flash", "decoder-only, causal", "closed weights")]
    for i, (m, a, n) in enumerate(rows):
        y = 110 + i * 100
        f.box(60, y, 260, 84, fill=NEWTOK, rx=12)
        f.text(190, y + 42, m, size=15, weight=600)
        f.text(470, y + 30, a, size=14, weight=500, anchor="start")
        f.text(470, y + 58, n, size=13, fill=MUTED, anchor="start")
    f.box(60, 620, 840, 44, fill=COUNT, rx=8)
    f.text(480, 642, "Decoder-only won the LLM era. The other shapes survive where generation is not needed.",
           size=14, weight=500)
    f.src()
    f.save()


# ---------------------------------------------------------------- L02 new plates
@fig
def l02_sinusoid_toy():
    f = Fig("l02-sinusoid-toy", 960, 620, "Sinusoids: the dot product keeps only distance",
            "From CME295 Lecture 2. PE(p) . PE(q) is a function of p - q.")
    f.box(60, 120, 840, 200, fill=COUNT, rx=8)
    f.text(480, 170, "PE(pos, 2i) = sin(pos / 10000^(2i/d))", size=20, weight=600, font=SERIF)
    f.text(480, 210, "PE(pos, 2i+1) = cos(pos / 10000^(2i/d))", size=20, weight=600, font=SERIF)
    f.text(480, 255, "dot(PE(3), PE(5)) = dot(PE(10), PE(12)): only the gap 2 matters", size=15, fill=MUTED)
    f.box(60, 360, 400, 180, fill=PANEL, rx=12)
    f.text(260, 395, "why it works", size=16, weight=600)
    f.text(260, 435, "sin/cos of (p-q) expand into\nproducts of sin/cos of p and q", size=14, fill=MUTED)
    f.box(500, 360, 400, 180, fill=NEWTOK, rx=12)
    f.text(700, 395, "why it extrapolates", size=16, weight=600)
    f.text(700, 435, "the formula works at any p:\nno learned table to run out", size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l02_kv_bytes():
    # Numbers computed from the lesson toy (l02, KV cache byte math).
    layers, kv_heads, d_head, nbytes = 80, 8, 128, 2
    per_layer = 2 * kv_heads * d_head * nbytes          # 4,096
    per_token = per_layer * layers                       # 327,680 = 320 KiB
    tokens = 32768
    total_bytes = per_token * tokens                     # 10,737,418,240
    gib = total_bytes / 1024**3                          # exactly 10.0
    assert per_layer == 4096 and per_token == 327680 and gib == 10.0
    mha_gib = gib * (64 / kv_heads)                      # 80.0
    mqa_gib = gib / kv_heads                             # 1.25
    f = Fig("l02-kv-bytes", 960, 640, "KV cache bytes: GQA's 4x saving, counted",
            "From CME295 Lecture 2. Llama-3-style 70B, 80 layers, fp16, GQA with 8 KV heads.")
    f.box(60, 120, 840, 250, fill=COUNT, rx=8)
    f.text(480, 160, f"per token per layer: 2 (K,V) x {kv_heads} KV heads x {d_head} dim x {nbytes} bytes = {per_layer:,} bytes",
           size=16, weight=600, font=MONO)
    f.text(480, 205, f"per token, {layers} layers: {per_token:,} bytes = 320 KB", size=16, weight=500)
    f.text(480, 250, f"{tokens:,} tokens x 320 KB = {gib:.1f} GiB per request (GQA, {kv_heads} KV heads)",
           size=16, weight=600)
    f.text(480, 295, f"MHA with 64 KV heads: 8x the cache = {mha_gib:.0f} GiB per request",
           size=16, fill=ORANGE, weight=500)
    f.text(480, 335, f"MQA with 1 KV head: 8x smaller = {mqa_gib:.2f} GiB per request",
           size=16, fill=TEAL, weight=500)
    f.box(60, 410, 840, 150, fill=NEWTOK, rx=8)
    f.text(480, 450, "Decision rule: the cache, not the weights, sets max batch size.",
           size=16, weight=600)
    f.text(480, 490, "100 concurrent 32K requests need 1 TiB with GQA. Halve to 4 heads and it is 5.0 GiB.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l02_bert_family():
    f = Fig("l02-bert-family", 960, 680, "The BERT family: what each variant changed",
            "From CME295 Lecture 2. One mechanism per variant, not a catalog.")
    rows = [("BERT (2018)", "MLM 15% + NSP 50/50", "bidirectional encoder baseline"),
            ("RoBERTa (2019)", "drop NSP, more data, longer", "NSP was the weak link"),
            ("DistilBERT (2019)", "KL-distill to 6 layers", "60% faster, 97% of the quality"),
            ("ALBERT (2019)", "share params across layers", "same depth, far fewer params"),
            ("T5 (2019)", "span corruption + sentinels", "everything is text-to-text")]
    for i, (m, c, w) in enumerate(rows):
        y = 110 + i * 100
        f.box(60, y, 220, 84, fill=NEWTOK, rx=12)
        f.text(170, y + 42, m, size=15, weight=600)
        f.text(420, y + 30, c, size=14, weight=500, anchor="start")
        f.text(420, y + 58, w, size=13, fill=MUTED, anchor="start")
    f.box(60, 620, 840, 44, fill=COUNT, rx=8)
    f.text(480, 642, "Interview line: name the one change and the one reason.",
           size=14, weight=500)
    f.src()
    f.save()


# ---------------------------------------------------------------- L03 new plates
@fig
def l03_lineup():
    f = Fig("l03-lineup", 960, 700, "The October 2026 lineup: who runs what",
            "From CME295 Lecture 3. Public model cards only. Closed labs are unknown.")
    rows = [("DeepSeek V4.1 Flash", "causal encoder-decoder MoE, 552B, 8B/16B active", "sparse is the open default"),
            ("Llama 4 Maverick", "decoder-only MoE, 17B active, 128 experts", "Meta's sparse turn"),
            ("Kimi K3 (Moonshot)", "decoder-only MoE, 2.8T, 104B active\n896 experts (16+2 shared per token)", "Kimi Delta Attention, 1M context"),
            ("GPT-6 / Gemini 3 / Claude", "decoder-only, internals unknown", "closed: mark unknown"),
            ("Small fast models", "dense, latency-predictable", "MoE's overhead is not worth it")]
    for i, (m, a, n) in enumerate(rows):
        y = 110 + i * 104
        f.box(60, y, 280, 88, fill=NEWTOK, rx=12)
        f.text(200, y + 44, m, size=15, weight=600)
        f.text(480, y + 32, a, size=14, weight=500, anchor="start")
        f.text(480, y + 60, n, size=13, fill=MUTED, anchor="start")
    f.box(60, 640, 840, 44, fill=COUNT, rx=8)
    f.text(480, 662, "Decoder-only won the architecture war. MoE won the capacity war.",
           size=14, weight=500)
    f.src()
    f.save()


@fig
def l03_beam():
    f = Fig("l03-beam", 960, 660, "Beam search, worked: B = 2 on the toy",
            "From CME295 Lecture 3. Extend, score by summed log-probs, keep the best two.")
    f.box(60, 110, 840, 90, fill=COUNT, rx=8)
    f.text(480, 145, "step 1: keep \"lit\" (-0.69) and \"read\" (-1.20)", size=15, weight=500)
    f.text(480, 175, "step 2: extend both, score four, keep \"lit well\" (-1.19) and \"read books\" (-1.60)",
           size=15, weight=500)
    f.box(60, 240, 400, 150, fill=PANEL, rx=12)
    f.text(260, 275, "length normalization", size=16, weight=600)
    f.text(260, 315, "score / length^alpha, alpha 0.6-1.0\nlog-probs are negative:\nlonger is always worse raw",
           size=14, fill=MUTED)
    f.box(500, 240, 400, 150, fill=ACTIVE, rx=12)
    f.text(700, 275, "beam vs sampling", size=16, weight=600)
    f.text(700, 315, "beam: stable, deterministic-ish\nhigh-probability paths\nsampling: diverse, human",
           size=14, fill=MUTED)
    f.box(60, 430, 840, 150, fill=NEWTOK, rx=8)
    f.text(480, 470, "Decision rule: beam for translation and code (correctness), sampling for chat (variety).",
           size=16, weight=600)
    f.text(480, 510, "B = 4-8 is the practical range. Larger B burns compute for little gain.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l03_topp():
    f = Fig("l03-topp", 960, 640, "Top-P adapts. Top-K cannot.",
            "From CME295 Lecture 3. Toy: lit 0.50, read 0.30, slept 0.12, ate 0.08.")
    f.box(60, 120, 400, 220, fill=NEWTOK, rx=12)
    f.text(260, 155, "sharp: lit 0.95, rest 0.05", size=15, weight=600)
    f.text(260, 200, "top-P 0.9 -> {lit} alone", size=14, fill=TEAL, weight=500)
    f.text(260, 235, "top-K 2 -> {lit, +junk}", size=14, fill=MUTED)
    f.text(260, 280, "top-P follows the entropy", size=14, fill=MUTED)
    f.box(500, 120, 400, 220, fill=ACTIVE, rx=12)
    f.text(700, 155, "flat: the toy distribution", size=15, weight=600)
    f.text(700, 200, "top-P 0.9 -> {lit, read, slept}", size=14, fill=TEAL, weight=500)
    f.text(700, 235, "top-K 2 -> {lit, read}", size=14, fill=MUTED)
    f.text(700, 280, "top-K is blind to shape", size=14, fill=MUTED)
    f.box(60, 380, 840, 180, fill=COUNT, rx=8)
    f.text(480, 420, "Production default: top-P 0.9-0.95 with a top-K cap of a few hundred.",
           size=16, weight=600)
    f.text(480, 460, "Top-P for quality, the cap for the flat-distribution worst case.",
           size=14, fill=MUTED)
    f.text(480, 500, "Greedy for facts. Sampling for variety. Always with the cap.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l03_amortize():
    f = Fig("l03-amortize", 960, 640, "Decoding is memory-bound: the proof",
            "From CME295 Lecture 3. 1 FLOP per byte vs ~300 needed. The GPU waits on memory.")
    f.box(60, 120, 840, 200, fill=COUNT, rx=8)
    f.text(480, 165, "70B decode step: ~140 GFLOP of math, ~140 GB of weights moved",
           size=16, weight=600)
    f.text(480, 210, "arithmetic intensity: 1 FLOP per byte", size=18, weight=600, fill=ORANGE)
    f.text(480, 255, "H100 needs ~300 FLOPs/byte to stay compute-bound. It gets 1. 300x underfed.",
           size=15, fill=MUTED)
    f.box(60, 360, 400, 200, fill=NEWTOK, rx=12)
    f.text(260, 395, "move fewer bytes", size=16, weight=600)
    f.text(260, 435, "quantization: 140 GB -> 35 GB\nGQA/MLA: smaller KV cache", size=14, fill=MUTED)
    f.box(500, 360, 400, 200, fill=PANEL, rx=12)
    f.text(700, 395, "amortize one move", size=16, weight=600)
    f.text(700, 435, "batching: one move, B tokens\nspeculation: one move, k tokens", size=14, fill=MUTED)
    f.src()
    f.save()


# ---------------------------------------------------------------- L04 new plates
@fig
def l04_6nd():
    f = Fig("l04-6nd", 960, 620, "C = 6ND: where the 6 comes from",
            "From CME295 Lecture 4. 2 forward + 4 backward, per parameter per token.")
    f.box(60, 120, 400, 200, fill=NEWTOK, rx=12)
    f.text(260, 160, "forward: 2 FLOPs", size=18, weight=600)
    f.text(260, 205, "one multiply-add\nper parameter", size=14, fill=MUTED)
    f.box(500, 120, 400, 200, fill=ACTIVE, rx=12)
    f.text(700, 160, "backward: 4 FLOPs", size=18, weight=600)
    f.text(700, 205, "gradients for activations\n+ gradients for weights", size=14, fill=MUTED)
    f.box(60, 360, 840, 180, fill=COUNT, rx=8)
    f.text(480, 400, "GPT-3: 6 x 175e9 x 300e9 = 3.15e23 FLOPs", size=18, weight=600, font=MONO)
    f.text(480, 445, "10.7 days on 10,000 H100s at fantasy 100% MFU. ~21 days at real 50%.",
           size=15, fill=MUTED)
    f.text(480, 485, "Ignores the attention quadratic term: fine at 2K context, not at 128K.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l04_chinchilla_table():
    f = Fig("l04-chinchilla-table", 960, 660, "Chinchilla applied: 20 tokens per parameter",
            "From CME295 Lecture 4. Everyone since overtrains on purpose.")
    rows = [("GPT-3", "175B", "300B", "3.5T", "11.7x undertrained"),
            ("Llama 3 70B", "70B", "15T", "1.4T", "10.7x OVERTRAINED"),
            ("DeepSeek V4.1", "552B", "45T", "11T", "4x overtrained"),
            ("Llama 4 Maverick", "17B act", "~22T corpus", "340B", "far past optimal")]
    f.text(170, 100, "model", size=13, fill=MUTED, weight=600)
    f.text(400, 100, "params", size=13, fill=MUTED, weight=600)
    f.text(550, 100, "tokens", size=13, fill=MUTED, weight=600)
    f.text(700, 100, "optimal", size=13, fill=MUTED, weight=600)
    f.text(850, 100, "verdict", size=13, fill=MUTED, weight=600)
    for i, (m, p, t, o, v) in enumerate(rows):
        y = 120 + i * 100
        f.box(60, y, 840, 88, fill=NEWTOK if i % 2 == 0 else PANEL, rx=12)
        f.text(170, y + 44, m, size=15, weight=600)
        f.text(400, y + 44, p, size=14, font=MONO)
        f.text(550, y + 44, t, size=14, font=MONO)
        f.text(700, y + 44, o, size=14, font=MONO)
        f.text(850, y + 44, v, size=14, weight=500, fill=ORANGE if "under" in v else TEAL)
    f.box(60, 540, 840, 60, fill=COUNT, rx=8)
    f.text(480, 570, "Overtraining is deliberate: inference cost, not training cost, dominates the lifetime bill.",
           size=15, weight=500)
    f.src()
    f.save()


@fig
def l04_flash_numbers():
    f = Fig("l04-flash-numbers", 960, 640, "FlashAttention: the memory hierarchy is the argument",
            "From CME295 Lecture 4. FLOPs are free. Bytes are expensive.")
    f.box(60, 120, 400, 200, fill=ACTIVE, rx=12)
    f.text(260, 160, "HBM: 80 GB", size=18, weight=600)
    f.text(260, 200, "~3.35 TB/s\n~hundreds of cycles", size=14, fill=MUTED)
    f.box(500, 120, 400, 200, fill=NEWTOK, rx=12)
    f.text(700, 160, "SRAM: ~50 MB", size=18, weight=600)
    f.text(700, 200, "~19 TB/s\n~tens of cycles", size=14, fill=MUTED)
    f.box(60, 360, 840, 200, fill=COUNT, rx=8)
    f.text(480, 400, "Naive: write the N x N score matrix to HBM, read it back. Per head, per layer.",
           size=15, weight=500)
    f.text(480, 440, "Flash: stream blocks through SRAM, running softmax, write only the output.",
           size=15, weight=500)
    f.text(480, 480, "~10x fewer HBM accesses. Exact, not approximate. Every fast kernel is a byte-saving kernel.",
           size=15, fill=TEAL, weight=600)
    f.src()
    f.save()


@fig
def l04_lora_family():
    f = Fig("l04-lora-family", 960, 680, "The LoRA family: one idea, four budgets",
            "From CME295 Lecture 4. The update lives in a small subspace.")
    rows = [("LoRA (2021)", "W = W0 + BA, rank 4", "merge after: zero inference cost"),
            ("QLoRA (2023)", "NF4 base, BF16 adapters", "65B-class on one GPU"),
            ("DoRA (2024)", "magnitude + direction", "closer to full fine-tuning"),
            ("rsLoRA", "scale by 1/sqrt(r)", "stable at rank > 16")]
    for i, (m, c, w) in enumerate(rows):
        y = 110 + i * 104
        f.box(60, y, 220, 88, fill=NEWTOK, rx=12)
        f.text(170, y + 44, m, size=15, weight=600)
        f.text(430, y + 32, c, size=14, weight=500, anchor="start", font=MONO)
        f.text(430, y + 60, w, size=13, fill=MUTED, anchor="start")
    f.box(60, 540, 840, 90, fill=COUNT, rx=8)
    f.text(480, 575, "Rank is the budget knob: r = 4-16 for instruction tuning, higher for harder tasks.",
           size=15, weight=500)
    f.text(480, 605, "Pick by budget: QLoRA for one GPU, LoRA for a few, DoRA when quality matters most.",
           size=14, fill=MUTED)
    f.src()
    f.save()


# ---------------------------------------------------------------- L05 new plates
@fig
def l05_kl():
    f = Fig("l05-kl", 960, 640, "The KL leash: beta sets the exchange rate",
            "From CME295 Lecture 5. The policy spends KL where reward is highest.")
    f.box(60, 120, 840, 200, fill=COUNT, rx=8)
    f.text(480, 165, "penalty = beta x KL(pi || pi_ref), paid per token", size=18, weight=600, font=SERIF)
    f.text(480, 210, "toy: policy 0.5 vs reference 0.4 -> token KL 0.11 -> penalty 0.011 at beta = 0.1",
           size=15, fill=MUTED)
    f.text(480, 255, "small per token, everywhere per sequence: drift everywhere, pay everywhere",
           size=15, fill=MUTED)
    f.box(60, 360, 400, 200, fill=NEWTOK, rx=12)
    f.text(260, 400, "beta = 0.1", size=18, weight=600)
    f.text(260, 440, "lenient\npolicy can improve", size=14, fill=MUTED)
    f.box(500, 360, 400, 200, fill=ACTIVE, rx=12)
    f.text(700, 400, "beta = 0.5", size=18, weight=600)
    f.text(700, 440, "strict\npolicy cannot hack\nbut cannot improve either", size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l05_gae():
    f = Fig("l05-gae", 960, 640, "GAE: lambda blends the horizons",
            "From CME295 Lecture 5. Lambda = 0 trusts the value head. Lambda = 1 trusts the raw return.")
    f.box(60, 120, 840, 120, fill=COUNT, rx=8)
    f.text(480, 160, "A_t = delta_t + lambda x delta_t+1 + lambda^2 x delta_t+2 + ...",
           size=20, weight=600, font=SERIF)
    f.text(480, 200, "delta_t = r_t + V_t+1 - V_t: the one-step surprise", size=15, fill=MUTED)
    f.box(60, 280, 260, 200, fill=ACTIVE, rx=12)
    f.text(190, 320, "lambda = 0", size=18, weight=600)
    f.text(190, 360, "one-step only\nlow variance\nhigh bias", size=14, fill=MUTED)
    f.box(350, 280, 260, 200, fill=NEWTOK, rx=12)
    f.text(480, 320, "lambda = 0.95", size=18, weight=600, fill=TEAL)
    f.text(480, 360, "the standard\ncompromise", size=14, fill=MUTED)
    f.box(640, 280, 260, 200, fill=PANEL, rx=12)
    f.text(770, 320, "lambda = 1", size=18, weight=600)
    f.text(770, 360, "full return\nlow bias\nhigh variance", size=14, fill=MUTED)
    f.box(60, 520, 840, 60, fill=COUNT, rx=8)
    f.text(480, 550, "Never-confuse: gamma discounts the future (how much). Lambda blends estimators (how far to trust).",
           size=14, weight=500)
    f.src()
    f.save()


@fig
def l05_dpo_variants():
    f = Fig("l05-dpo-variants", 960, 680, "The DPO family: same core, different data",
            "From CME295 Lecture 5. No RL loop anywhere in the family.")
    rows = [("DPO", "chosen/rejected pairs", "the original: logistic loss on pairs"),
            ("IPO", "small pair sets", "squared loss stops the gap at a target"),
            ("KTO", "thumbs up/down labels", "binary signals, no pairs needed"),
            ("SimPO", "tight memory", "drops the reference, normalizes length")]
    for i, (m, c, w) in enumerate(rows):
        y = 110 + i * 104
        f.box(60, y, 220, 88, fill=NEWTOK, rx=12)
        f.text(170, y + 44, m, size=16, weight=600)
        f.text(430, y + 32, c, size=14, weight=500, anchor="start")
        f.text(430, y + 60, w, size=13, fill=MUTED, anchor="start")
    f.box(60, 540, 840, 90, fill=COUNT, rx=8)
    f.text(480, 575, "Pick by data: pairs (DPO), small pairs (IPO), binary labels (KTO), tight memory (SimPO).",
           size=15, weight=500)
    f.text(480, 605, "All share the closed-form trick: the reward is the log-ratio, the RL loop is gone.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l05_ppo_memory():
    f = Fig("l05-ppo-memory", 960, 640, "PPO's memory bill: four models",
            "From CME295 Lecture 5. 7B model in fp16: ~14 GB per model.")
    items = [("policy", "14 GB", "trains", TEAL),
             ("reference", "14 GB", "the KL anchor", FOCUS),
             ("reward model", "~14 GB", "frozen grader", ORANGE),
             ("value function", "14 GB", "the baseline", MUTED)]
    for i, (k, v, d, col) in enumerate(items):
        x = 60 + i * 216
        f.box(x, 140, 200, 220, rx=12)
        f.parts.append(f'<rect x="{x + 80}" y="165" width="40" height="40" rx="20" fill="{col}"/>')
        f.text(x + 100, 245, k, size=15, weight=600)
        f.text(x + 100, 280, v, size=16, font=MONO)
        f.text(x + 100, 315, d, size=13, fill=MUTED)
    f.box(60, 400, 840, 160, fill=COUNT, rx=8)
    f.text(480, 440, "Total: ~42-56 GB before optimizer states and activations.",
           size=17, weight=600)
    f.text(480, 480, "DPO holds two models (~28 GB). That is who can afford preference tuning.",
           size=15, fill=MUTED)
    f.text(480, 515, "Decision rule: PPO with GPUs and quality needs, DPO without.",
           size=14, fill=MUTED)
    f.src()
    f.save()


# ---------------------------------------------------------------- L06 new plates
@fig
def l06_timeline():
    f = Fig("l06-timeline", 960, 620, "Reasoning: o1 proved it, R1-Zero explored, R1 shipped",
            "From CME295 Lecture 6. Length and accuracy rise together: verification gets rewarded.")
    steps = [("o1 preview\nSep 2024", "RL on reasoning\ntraces works", TEAL),
             ("R1-Zero\nJan 2025", "pure RL from base\nmessy but smart", ORANGE),
             ("R1\nJan 2025", "cold-start + staged RL\nreadable and smart", FOCUS),
             ("distill", "teacher's tokens\nto small models", MUTED)]
    for i, (t, d, col) in enumerate(steps):
        x = 60 + i * 216
        f.box(x, 140, 200, 240, rx=12)
        f.parts.append(f'<rect x="{x + 80}" y="165" width="40" height="40" rx="20" fill="{col}"/>')
        f.text(x + 100, 250, t, size=15, weight=600)
        f.text(x + 100, 310, d, size=13, fill=MUTED)
        if i < 3:
            f.arrow(x + 200, 260, x + 216, 260, teal=(i == 0))
    f.box(60, 430, 840, 120, fill=COUNT, rx=8)
    f.text(480, 468, "The curve to memorize: response length and accuracy climb together.",
           size=16, weight=600)
    f.text(480, 508, "Length is the symptom. Learned verification is the cause.",
           size=14, fill=MUTED)
    f.src()
    f.save()


# ---------------------------------------------------------------- L07 new plates
@fig
def l07_ann():
    f = Fig("l07-ann", 960, 680, "ANN indexes: IVF clusters, HNSW graphs, PQ compresses",
            "From CME295 Lecture 7. Linear scan over 1M chunks is 1.5B multiply-adds per query.")
    rows = [("IVF", "cluster into ~1K cells, search ~10", "100x fewer compares; misses borders"),
            ("HNSW", "graph, greedy walk coarse-to-fine", "default: fast, accurate, RAM-hungry"),
            ("PQ", "compress vectors to ~100 bytes", "15x less memory, approximate")]
    for i, (m, c, w) in enumerate(rows):
        y = 110 + i * 110
        f.box(60, y, 200, 94, fill=NEWTOK, rx=12)
        f.text(160, y + 47, m, size=17, weight=600, font=MONO)
        f.text(400, y + 32, c, size=14, weight=500, anchor="start")
        f.text(400, y + 62, w, size=13, fill=MUTED, anchor="start")
    f.box(60, 460, 840, 150, fill=COUNT, rx=8)
    f.text(480, 500, "Decision rule: HNSW under ~100M vectors when RAM allows, IVF-PQ past that.",
           size=16, weight=600)
    f.text(480, 540, "Tune recall on your own labels: nprobe (IVF) or ef (HNSW) trades latency for recall.",
           size=14, fill=MUTED)
    f.text(480, 575, "50M chunks x 1500 dims: 300 GB raw, ~5 GB with PQ. That is the whole argument.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l07_mcp_arch():
    f = Fig("l07-mcp-arch", 960, 640, "MCP: host, client, server",
            "From CME295 Lecture 7. MCP is USB for model tools. 1:1 connections isolate failures.")
    f.box(60, 140, 240, 200, fill=NEWTOK, rx=12)
    f.text(180, 175, "MCP host", size=17, weight=600)
    f.text(180, 215, "the app:\nClaude Desktop, an IDE", size=14, fill=MUTED)
    f.box(360, 140, 240, 200, fill=PANEL, rx=12)
    f.text(480, 175, "MCP client", size=17, weight=600)
    f.text(480, 215, "1:1 connection\nper server", size=14, fill=MUTED)
    f.box(660, 140, 240, 200, fill=ACTIVE, rx=12)
    f.text(780, 175, "MCP server", size=17, weight=600)
    f.text(780, 215, "tools + resources\n+ prompts", size=14, fill=MUTED)
    f.arrow(300, 240, 360, 240, teal=True)
    f.arrow(600, 240, 660, 240, teal=True)
    f.box(60, 400, 840, 160, fill=COUNT, rx=8)
    f.text(480, 440, "The server advertises capabilities at connect time. The client exposes them as tool schemas.",
           size=15, weight=500)
    f.text(480, 480, "Write the book tools once. Every MCP client uses them. The 2026 standard.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l07_debug_tree():
    f = Fig("l07-debug-tree", 960, 700, "Agent debugging: read the chain, branch by stage",
            "From CME295 Lecture 7. The cheapest fix wins. Fix tools before blaming the model.")
    f.box(230, 100, 500, 70, fill=COUNT, rx=12)
    f.text(480, 135, "agent failed -> read the reasoning chain", size=16, weight=600)
    branches = [("never called a tool", "PREDICT", "fix router recall\nor SFT/prompt", ORANGE),
                ("called the wrong thing", "PREDICT", "rename APIs\n disambiguate scopes", FOCUS),
                ("tool misbehaved", "EXECUTION", "fix implementation\nreturn something always", TEAL),
                ("ignored the result", "SYNTHESIS", "trim outputs\nmeaningful objects", MUTED)]
    for i, (s, st, fx, col) in enumerate(branches):
        x = 60 + i * 216
        f.box(x, 230, 200, 280, rx=12)
        f.parts.append(f'<rect x="{x + 80}" y="250" width="40" height="40" rx="20" fill="{col}"/>')
        f.text(x + 100, 330, s, size=14, weight=600)
        f.text(x + 100, 375, st, size=13, font=MONO, fill=MUTED)
        f.text(x + 100, 430, fx, size=13, fill=MUTED)
        f.arrow(480, 170, x + 100, 230, dashed=True)
    f.box(60, 560, 840, 70, fill=NEWTOK, rx=8)
    f.text(480, 595, "Most mysterious failures trace to silent, bloated, or raw-error tool outputs.",
           size=15, weight=500)
    f.src()
    f.save()


# ---------------------------------------------------------------- L08 new plates
@fig
def l08_elo():
    f = Fig("l08-elo", 960, 640, "Elo, worked: 200 points means 76% expected wins",
            "From CME295 Lecture 8. Bradley-Terry fits the ratings by maximum likelihood.")
    f.box(60, 120, 840, 200, fill=COUNT, rx=8)
    f.text(480, 165, "E[A beats B] = 1 / (1 + 10^((R_B - R_A)/400))", size=20, weight=600, font=SERIF)
    f.text(480, 210, "A = 1200, B = 1000: 1 / (1 + 10^(-0.5)) = 0.76", size=17, font=MONO)
    f.text(480, 255, "A wins: 1200 + 32 x (1 - 0.76) = 1207.7. B falls symmetrically.",
           size=15, fill=MUTED)
    f.box(60, 360, 400, 200, fill=NEWTOK, rx=12)
    f.text(260, 400, "Elo", size=17, weight=600)
    f.text(260, 440, "update per match\nonline, simple", size=14, fill=MUTED)
    f.box(500, 360, 400, 200, fill=PANEL, rx=12)
    f.text(700, 400, "Bradley-Terry", size=17, weight=600)
    f.text(700, 440, "P(A beats B) = sigma(r_A - r_B)\nfit all ratings at once", size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l08_swebench():
    f = Fig("l08-swebench", 960, 640, "SWE-bench, worked: the patch must pass both suites",
            "From CME295 Lecture 8. Test-driven development as a benchmark.")
    f.box(60, 120, 400, 200, fill=NEWTOK, rx=12)
    f.text(260, 160, "FAIL_TO_PASS", size=17, weight=600, font=MONO)
    f.text(260, 200, "tests that failed before\nmust pass after", size=14, fill=MUTED)
    f.text(260, 255, "the issue is fixed", size=14, fill=TEAL, weight=500)
    f.box(500, 120, 400, 200, fill=ACTIVE, rx=12)
    f.text(700, 160, "PASS_TO_PASS", size=17, weight=600, font=MONO)
    f.text(700, 200, "tests that passed before\nmust still pass", size=14, fill=MUTED)
    f.text(700, 255, "nothing broke", size=14, fill=TEAL, weight=500)
    f.box(60, 360, 840, 200, fill=COUNT, rx=8)
    f.text(480, 400, "Toy: issue #452, divide by zero on empty input. Patch adds a guard.",
           size=15, weight=500)
    f.text(480, 440, "test_empty_input passes now. The other 47 still pass. Score: 1.",
           size=15, fill=MUTED)
    f.text(480, 480, "2026: frontier models hit 60-80% on Verified. Harder variants carry the signal now.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l08_judge_pipeline():
    f = Fig("l08-judge-pipeline", 960, 680, "The judge pipeline: rubric to calibrated score",
            "From CME295 Lecture 8. An uncalibrated judge is a random number generator with good grammar.")
    steps = [("rubric", "crisp guidelines\nworked examples", TEAL),
             ("prompt", "rationale first\nbinary, structured", FOCUS),
             ("sample", "temp 0.1-0.2\nmultiple for ties", ORANGE),
             ("de-bias", "both orders\nlength rules", MUTED)]
    for i, (t, d, col) in enumerate(steps):
        x = 60 + i * 216
        f.box(x, 140, 200, 220, rx=12)
        f.parts.append(f'<rect x="{x + 80}" y="160" width="40" height="40" rx="20" fill="{col}"/>')
        f.text(x + 100, 245, t, size=16, weight=600)
        f.text(x + 100, 295, d, size=13, fill=MUTED)
        if i < 3:
            f.arrow(x + 200, 250, x + 216, 250)
    f.box(60, 410, 840, 200, fill=COUNT, rx=8)
    f.text(480, 450, "Calibrate: humans grade 200-500 items too. Judge-human kappa must clear ~0.7.",
           size=16, weight=600)
    f.text(480, 490, "Monitor: re-calibrate on a rolling sample. Judges drift as models update. Version them.",
           size=14, fill=MUTED)
    f.text(480, 530, "The pipeline is judge-at-scale plus human-at-the-margin. The human sample is not optional.",
           size=14, fill=MUTED)
    f.src()
    f.save()


# ---------------------------------------------------------------- L09 new plates
@fig
def l09_collapse_math():
    f = Fig("l09-collapse-math", 960, 640, "Model collapse: tails decay geometrically",
            "From CME295 Lecture 9. Sampling concentrates. Each generation trains on the last one's sample.")
    gens = [("human text", "100 words\nZipf spread", TEAL),
            ("gen 1", "top 20 dominate", ORANGE),
            ("gen 2", "top 10 dominate", ORANGE),
            ("gen 3", "top 5 dominate", "#A33B2E")]
    for i, (t, d, col) in enumerate(gens):
        x = 60 + i * 216
        f.box(x, 140, 200, 220, rx=12)
        f.parts.append(f'<rect x="{x + 80}" y="160" width="40" height="40" rx="20" fill="{col}"/>')
        f.text(x + 100, 245, t, size=16, weight=600)
        f.text(x + 100, 295, d, size=13, fill=MUTED)
        if i < 3:
            f.arrow(x + 200, 250, x + 216, 250)
    f.box(60, 410, 840, 150, fill=COUNT, rx=8)
    f.text(480, 450, "tail_G3 ~ tail_G1 x c^2, c < 1: rare words and rare facts vanish first.",
           size=16, weight=600, font=MONO)
    f.text(480, 490, "A diversity catastrophe, not a quality dip. Re-seed the tail: provenance, fresh human data, mid-training.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l09_exam_map():
    f = Fig("l09-exam-map", 960, 680, "The final covers L05-L08: each lecture's testable core",
            "From CME295 Lecture 9. L01-L04 are background vocabulary.")
    rows = [("L05 preference", "derive BT, explain the clip, PPO vs DPO", "data, objective, cost"),
            ("L06 reasoning", "GRPO z-score, length bias, R1 stages", "work the toy numbers"),
            ("L07 RAG/agents", "funnel, BM25, ReAct, 7 failures", "design, then debug"),
            ("L08 evaluation", "kappa, judge biases, benchmarks", "read skeptically")]
    for i, (m, c, w) in enumerate(rows):
        y = 110 + i * 104
        f.box(60, y, 220, 88, fill=NEWTOK, rx=12)
        f.text(170, y + 44, m, size=15, weight=600)
        f.text(430, y + 32, c, size=14, weight=500, anchor="start")
        f.text(430, y + 60, w, size=13, fill=MUTED, anchor="start")
    f.box(60, 540, 840, 90, fill=COUNT, rx=8)
    f.text(480, 575, "Highest yield: the pipeline end to end (pre-train, SFT, preference, reasoning RL).",
           size=15, weight=500)
    f.text(480, 605, "Every other topic attaches to it. Exam questions ask how the stages differ.",
           size=14, fill=MUTED)
    f.src()
    f.save()


@fig
def l09_block_diffusion():
    f = Fig("l09-block-diffusion", 960, 640, "Block diffusion: the compromise the field converged on",
            "From CME295 Lecture 9. Left to right between blocks, parallel inside.")
    f.box(60, 120, 840, 200, fill=COUNT, rx=8)
    f.text(480, 165, "1,000 tokens in 10 blocks of 100, 8 steps per block: 80 forward passes, not 1,000",
           size=16, weight=600)
    f.text(480, 210, "pure diffusion: 32 passes, no cross-position conditioning inside a step", size=15, fill=MUTED)
    f.text(480, 250, "autoregressive: 1,000 passes, full conditioning. Blocks split the difference.",
           size=15, fill=MUTED)
    for i in range(5):
        x = 120 + i * 160
        f.box(x, 380, 140, 100, fill=NEWTOK if i % 2 == 0 else PANEL, rx=12)
        f.text(x + 70, 420, f"block {i + 1}", size=14, weight=600, font=MONO)
        f.text(x + 70, 450, "diffuse inside", size=12, fill=MUTED)
        if i < 4:
            f.arrow(x + 140, 430, x + 160, 430, teal=True)
    f.text(480, 530, "condition left to right between blocks", size=14, fill=TEAL, weight=500)
    f.src()
    f.save()


def main():
    os.makedirs(OUT, exist_ok=True)
    for fn in FIGS:
        fn()
    print(f"wrote {len(FIGS)} figures to {OUT}")


if __name__ == "__main__":
    main()

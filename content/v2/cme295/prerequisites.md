# Prerequisites , cme295

Shared bridges P01-P24 live at `../shared/prerequisites/` and are
linked, never rebuilt. Each unit lesson opens with a local remediation
block for its own prerequisites. Read the bridge file first if the
diagnostic item for it is not full marks.

## Per-unit prerequisites

- U01 (NLP to transformer foundations): P03 vectors, P05 calculus, P06
  probability, P11 neural nets, P13 language modelling. Bridges:
  `../shared/prerequisites/p03_vectors.md`,
  `../shared/prerequisites/p05_calculus.md`,
  `../shared/prerequisites/p06_probability.md`,
  `../shared/prerequisites/p11_neural_nets.md`,
  `../shared/prerequisites/p13_language.md`.
- U02 (attention variants, positions): P04 spectral/numerical linear
  algebra, P12 PyTorch and stability, P14 transformer mechanics. Bridges:
  `../shared/prerequisites/p04_spectral.md`,
  `../shared/prerequisites/p12_pytorch.md`,
  `../shared/prerequisites/p14_transformer.md`.
- U03 (LLM architecture and generation): P13 language modelling, P14
  transformer mechanics, P15 hardware. Bridges:
  `../shared/prerequisites/p13_language.md`,
  `../shared/prerequisites/p14_transformer.md`,
  `../shared/prerequisites/p15_hardware.md`.
- U04 (training and adaptation): P10 ML foundations, P11 neural nets,
  P12 PyTorch and stability, P15 hardware. Bridges:
  `../shared/prerequisites/p10_ml_foundations.md`,
  `../shared/prerequisites/p11_neural_nets.md`,
  `../shared/prerequisites/p12_pytorch.md`,
  `../shared/prerequisites/p15_hardware.md`.
- U05 (preference tuning, policy optimization): P08 information theory,
  P17 reinforcement learning, P18 Bayesian inference and sampling.
  Bridges: `../shared/prerequisites/p08_information.md`,
  `../shared/prerequisites/p17_rl.md`,
  `../shared/prerequisites/p18_bayesian.md`.

## Local remediation

Each lesson file opens with a remediation block: 3-5 worked micro
checks that repair the exact prerequisite skills the unit needs
(matrix shapes, softmax arithmetic, log rules, KL definition, policy
gradient intuition). Do the block before the concepts if any diagnostic
item below scores below full marks.

## Diagnostic (closed-book, 15 minutes)

Answer in `keys/diagnostic_key.md` (separate file). Score each 0/1/2.

- D1. Compute the dot product of [1, 2, 3] and [4, -1, 0], and the norm
  of each vector.
- D2. Softmax of [2, 1, 0] by hand to 3 decimals.
- D3. d/dx of log(1 + exp(x)) at x = 0.
- D4. A fair coin lands heads. State P(heads) as a probability, then as
  surprisal in bits.
- D5. Write the chain rule for P(A, B, C) as a product of conditionals.
- D6. One linear layer: x in R^4, W in R^{4x3}, b in R^3. State the
  shape of Wx + b and count its parameters.
- D7. Cross-entropy between true [1, 0] and predicted [0.7, 0.3].
- D8. KL(P||Q) for P = [0.5, 0.5], Q = [0.9, 0.1]. Which direction did
  you compute, and why does direction matter?
- D9. A policy picks action A with probability 0.6 and gets reward 10,
  action B with probability 0.4 and gets reward 0. State the expected
  return.
- D10. 7B parameters in fp16. How many bytes is the weight file, and why?

Remediation routing: D1-D2 miss -> P03/P06 bridges, D3 -> P05, D4, D7,
D8 -> P08, D5 -> P06, D6 -> P11, D9 -> P17, D10 -> P15.

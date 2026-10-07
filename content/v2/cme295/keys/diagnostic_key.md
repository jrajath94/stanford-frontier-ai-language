# Diagnostic key , cme295

Closed-book diagnostic from `prerequisites.md`. Score 0/1/2 per item.

- D1. Dot: 1*4 + 2*(-1) + 3*0 = 2. Norms: sqrt(1+4+9) = sqrt(14) ~=
  3.74, sqrt(16+1+0) = sqrt(17) ~= 4.12.
- D2. exp: [7.389, 2.718, 1.000], sum 11.107, softmax [0.665, 0.245,
  0.090].
- D3. d/dx log(1+e^x) = e^x/(1+e^x) = sigmoid(x). At 0: 0.5.
- D4. P = 0.5. Surprisal = -log2(0.5) = 1 bit.
- D5. P(A,B,C) = P(A) P(B|A) P(C|A,B).
- D6. Shape (3,). Parameters: 4*3 + 3 = 15.
- D7. -log(0.7) ~= 0.357 nats.
- D8. KL = 0.5*log(0.5/0.9) + 0.5*log(0.5/0.1) ~= 0.511 nats.
  Direction: P||Q. It matters because KL is asymmetric, the penalty
  for Q missing P's mass differs from the reverse.
- D9. Expected return: 0.6*10 + 0.4*0 = 6.
- D10. 7e9 * 2 bytes = 14 GB (13.0 GiB). Each fp16 value is 2 bytes.

Remediation routing is in `prerequisites.md`.

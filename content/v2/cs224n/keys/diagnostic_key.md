# Diagnostic key , cs224n prerequisites

D1. U+20AC in binary is 0010 0000 1010 1100 (14 bits). UTF-8 three-byte
form: 1110xxxx 10xxxxxx 10xxxxxx. Fill: 0010 | 000010 | 101100. Bytes:
0xE2=226, 0x82=130, 0xAC=172. Byte count: 3.

D2. Cross-entropy = -(0.5 log 0.5 + 0.5 log 0.5) = -log 0.5 = 0.693
nats. For a fair coin with a perfect model this equals the entropy.

D3. Dot product = 1*3 + 2*(-1) = 1. Positive, so the angle is acute
(under 90 degrees).

D4. d/dx log(1+exp(x)) = exp(x)/(1+exp(x)) = sigma(x), the logistic
sigmoid.

D5. Shift by -1002: (-2, -1, 0). Softmax = (e^-2, e^-1, 1)/(e^-2 +
e^-1 + 1) = (0.135, 0.368, 1)/1.503 = (0.090, 0.245, 0.665). Largest
entry: 0.665 at index 2. The shift changes nothing, it prevents
overflow.

D6. Logits: (4, 7, 50). Targets: (4, 7) token ids. For loss, drop the
last logit position: predict tokens 2..7 from prefixes 1..6, so the
used logit block is (4, 6, 50) against targets (4, 6).

D7. Perplexity is exp of the mean negative log-likelihood: the model's
effective number of equally likely choices. At p=0.25 each step,
perplexity = exp(-log 0.25) = 4.

D8. Q, K, V per head: (2, 6, 8). Scores per head: (2, 6, 6). With heads
kept: (2, h, 6, 6).

D9. One multiply-add per (input, output) pair per batch element: 2 x 32
x 1024 x 4096 = 268,435,456 FLOPs, about 0.27 GFLOPs. Bias add is
negligible next to it.

D10. P(A wins) = exp(rA) / (exp(rA) + exp(rB)) = sigma(rA - rB).

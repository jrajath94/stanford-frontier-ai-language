# Notation and shapes: cs329h

One symbol table for U01-U05. A symbol keeps one meaning across units. First-use page in parentheses.

## Choice data

| Symbol | Meaning | Shape / units | First use |
| --- | --- | --- | --- |
| D | preference dataset | set of n records | U01 |
| (a, b, y) | one comparison: item a, item b, outcome y | a, b in [m]. Y in {0,1} | U01 |
| y = 1 | a chosen over b | dimensionless | U01 |
| n | number of comparisons | positive integer | U01 |
| m | number of items | positive integer | U01 |

## Scores and utilities

| Symbol | Meaning | Shape / units | First use |
| --- | --- | --- | --- |
| s_i, theta_i | latent score of item i | real scalar, utils | U01 |
| U_i | random utility of item i | real scalar | U02 |
| u_i | deterministic part of utility | real scalar | U02 |
| epsilon_i | random shock | real scalar, same units as u | U02 |
| Delta | score gap s_a - s_b | real scalar | U01 |

## Models

| Symbol | Meaning | Shape / units | First use |
| --- | --- | --- | --- |
| sigma(z) | logistic function 1/(1+exp(-z)) | scalar in (0,1) | U01 |
| P(a > b) | probability a beats b | scalar in [0,1] | U01 |
| L(theta) | likelihood | positive scalar | U01 |
| l(theta) | log-likelihood | real scalar | U01 |
| pi | policy, distribution over outputs | function or vector | U05 |
| pi_ref | reference policy | same shape as pi | U05 |
| r(x, y) | reward of response y to prompt x | real scalar | U05 |
| beta | KL penalty strength | positive scalar | U05 |

## Statistics

| Symbol | Meaning | Shape / units | First use |
| --- | --- | --- | --- |
| s(theta) | score function, gradient of log-likelihood | vector, dim(theta) | U04 |
| I(theta) | Fisher information matrix | dim(theta) by dim(theta) | U04 |
| Var_hat | asymptotic variance estimate | scalar or matrix | U04 |
| p(theta | D) | posterior | density over theta | U03 |
| p(y_new | D) | posterior predictive | scalar in [0,1] for binary y | U03 |

## Embeddings

| Symbol | Meaning | Shape / units | First use |
| --- | --- | --- | --- |
| w_j | respondent j embedding | real vector in R^d | U02 |
| v_i | item i embedding | real vector in R^d | U02 |
| d | embedding dimension | positive integer | U02 |

## Conventions

- Vectors are column vectors. Probabilities are dimensionless and live in [0,1].
- Log means natural log unless stated.
- "utils" marks an arbitrary latent scale. Only gaps and ratios of gaps are meaningful.
- Seed, dtype, and shape are stated for every computed example: float64, fixed seed 0 unless noted.

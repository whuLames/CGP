# ForkGraph GR Input Order Validation

Dataset: `cit-Patents`

Source: `3014863`

The locally generated partition files preserve original vertex IDs. Two input
orders were tested with a validation-only ForkGraph build that reports the
number of reachable vertices, the sum of BFS levels, and a vertex-weighted
level sum after the timed region.

| Input order | Reached | Level sum | Weighted sum |
|---|---:|---:|---:|
| `intra.gr`, `inter.gr` | 3,764,117 | 35,297,251 | 71,729,043,750,206 |
| `inter.gr`, `intra.gr` | 3,731,866 | 37,289,499 | 75,873,354,726,164 |
| PuerCGP reference | 3,764,117 | 35,297,251 | 71,729,043,750,206 |

Only the `intra.gr`, `inter.gr` order exactly matches PuerCGP on all three
summaries. The external-baseline runner therefore uses that order. The
validation instrumentation is guarded by `VALIDATE_SUMMARY`; the production
ForkGraph binary and formal timing runs do not enable it.

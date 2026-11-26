# Does morphing hurt a kept neuron's own job? From saved outputs (`code/oc_vs_ownjob_saved.py`)

h = KL (nats/photo) that swapping j back REMOVES on j's own photos. h > 0 means the morph hurt j's own photos.

## 1. Which saved photo set stands in for a unit's own photos? (cut 64, the 20 smoke units)

| Set | same sign as smoke own h (pooled 4 builds) | rank corr with smoke own h (pooled) | passes (>= .8 and >= .6) |
|---|---|---|---|
| top50-in-U | 0.71 (n 80) | +0.58 | no |
| rand-top | 0.53 (n 80) | +0.02 | no |
| clean-top | 0.57 (n 80) | -0.08 | no |

Smoke own h for scale (true own photos, cut 64): median over 20 units and 4 builds -0.00053 nats; own-job size median 0.01611.

## 2. Every kept unit (sets that passed section 1 only)

### Cut 11: 501 kept units (193 widened). True top-50 photos inside U: median 4 of 50

| Build | Set | h median, widened / control (nats) | units with h > 0 | median h / build KL on the same photos | read / write side h, median (L, R4C) |
|---|---|---|---|---|---|

| Build | own-write share a: median / 10th pct | write cosine with A's: median | read cosine with A's: median |
|---|---|---|---|
| C | 1.002 / 1.000 | 0.998 | 1.000 |
| WC | 1.003 / 1.000 | 0.997 | 1.000 |
| L | 1.000 / 1.000 | 0.999 | 1.000 |
| R4C | 1.003 / 1.000 | 0.998 | 1.000 |

### Cut 64: 448 kept units (353 widened). True top-50 photos inside U: median 13 of 50

| Build | Set | h median, widened / control (nats) | units with h > 0 | median h / build KL on the same photos | read / write side h, median (L, R4C) |
|---|---|---|---|---|---|

| Build | own-write share a: median / 10th pct | write cosine with A's: median | read cosine with A's: median |
|---|---|---|---|
| C | 1.022 / 1.008 | 0.984 | 1.000 |
| WC | 1.025 / 1.010 | 0.976 | 1.000 |
| L | 1.004 / 1.001 | 0.991 | 0.996 |
| R4C | 1.027 / 1.014 | 0.977 | 0.996 |

### Cut 192: 320 kept units (276 widened). True top-50 photos inside U: median 27 of 50

| Build | Set | h median, widened / control (nats) | units with h > 0 | median h / build KL on the same photos | read / write side h, median (L, R4C) |
|---|---|---|---|---|---|

| Build | own-write share a: median / 10th pct | write cosine with A's: median | read cosine with A's: median |
|---|---|---|---|
| C | 1.084 / 1.051 | 0.936 | 1.000 |
| WC | 1.103 / 1.062 | 0.888 | 1.000 |
| L | 1.035 / 1.017 | 0.936 | 0.994 |
| R4C | 1.115 / 1.082 | 0.895 | 0.993 |

### Cut 384: 128 kept units (125 widened). True top-50 photos inside U: median 38 of 50

| Build | Set | h median, widened / control (nats) | units with h > 0 | median h / build KL on the same photos | read / write side h, median (L, R4C) |
|---|---|---|---|---|---|

| Build | own-write share a: median / 10th pct | write cosine with A's: median | read cosine with A's: median |
|---|---|---|---|
| C | 1.226 / 1.151 | 0.815 | 1.000 |
| WC | 1.198 / 1.112 | 0.735 | 1.000 |
| L | 0.965 / 0.937 | 0.861 | 0.983 |
| R4C | 1.184 / 1.117 | 0.771 | 0.985 |

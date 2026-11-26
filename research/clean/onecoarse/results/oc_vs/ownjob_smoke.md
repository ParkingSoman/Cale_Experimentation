# Does morphing hurt a kept neuron's own job? (oc_vs_ownjob.py, read by oc_vs_ownjob_read.py)

h = share of j's own job lost (KL that swapping j back removes, over KL of deleting j in A), on j's own top 50 test photos. clean = the same, on photos where D is quiet. t = KL that j's morph removes on its partner's top 50 photos (nats). Widened = section 8.5 rule in L.

## Cut 64: 20 kept units, 18 widened

D's summed firing on own photos / clean photos, as a multiple of its median over test photos: 1.06 / 0.87. Own-job size KL(A, j deleted) on own photos: median 1.61e-02, 90th 2.68e-02.

| Build | Group | h own: median / 90th | h clean: median | h read only / write only (median) | pooled h own (sum/sum) | t on partner photos: sum (nats/photo) | own-job loss: sum (nats/photo) |
|---|---|---|---|---|---|---|---|
| C | widened (18) | -0.050 / 0.059 | -0.008 | 0.000 / -0.050 | -0.055 | +0.0202 | -0.0179 |
| C | control (2) | 0.054 / 0.101 | -0.035 | 0.000 / 0.054 | 0.038 | -0.0001 | +0.0010 |
| C | widened, big job (9) | -0.019 / 0.016 | -0.006 | -0.000 / -0.019 | -0.051 | +0.0151 | -0.0120 |
| C | control, big job (1) | -0.004 / -0.004 | -0.034 | 0.000 / -0.004 | -0.004 | -0.0005 | -0.0001 |
| WC | widened (18) | -0.030 / 0.010 | -0.019 | -0.000 / -0.030 | -0.052 | +0.0157 | -0.0170 |
| WC | control (2) | 0.056 / 0.060 | -0.037 | -0.000 / 0.056 | 0.057 | +0.0000 | +0.0015 |
| WC | widened, big job (9) | -0.030 / 0.009 | -0.020 | -0.000 / -0.030 | -0.049 | +0.0130 | -0.0116 |
| WC | control, big job (1) | 0.060 / 0.060 | 0.000 | 0.000 / 0.060 | 0.060 | -0.0004 | +0.0010 |
| L | widened (18) | -0.033 / 0.007 | -0.011 | -0.002 / -0.030 | -0.036 | +0.0198 | -0.0116 |
| L | control (2) | -0.019 / 0.003 | -0.031 | -0.016 / -0.004 | -0.011 | -0.0004 | -0.0003 |
| L | widened, big job (9) | -0.033 / -0.014 | -0.016 | -0.002 / -0.027 | -0.035 | +0.0192 | -0.0083 |
| L | control, big job (1) | 0.008 / 0.008 | -0.001 | 0.007 / 0.001 | 0.008 | -0.0003 | +0.0001 |
| R4C | widened (18) | -0.039 / 0.004 | -0.007 | -0.002 / -0.033 | -0.042 | +0.0195 | -0.0138 |
| R4C | control (2) | -0.002 / 0.005 | -0.035 | -0.022 / 0.020 | 0.001 | -0.0007 | +0.0000 |
| R4C | widened, big job (9) | -0.033 / 0.002 | -0.004 | -0.002 / -0.031 | -0.039 | +0.0176 | -0.0093 |
| R4C | control, big job (1) | 0.007 / 0.007 | -0.016 | -0.011 / 0.019 | 0.007 | -0.0006 | +0.0001 |

| Build | widened - control, median h [95% CI] | Verdict |
|---|---|---|
| C | -0.105 [-0.191, +0.003] | free |
| WC | -0.086 [-0.144, -0.070] | free |
| L | -0.014 [-0.048, +0.028] | free |
| R4C | -0.037 [-0.057, -0.013] | free |

Named units:

| Unit -> partner | Build | widened | a | h own | h read / write | h clean | t (nats/photo) | own-job size |
|---|---|---|---|---|---|---|---|---|
| n339 -> n476 | C | True | 1.01 | -0.048 | -0.000 / -0.048 | -0.002 | +0.0047 | 1.93e-02 |
| n339 -> n476 | WC | True | 1.01 | -0.019 | -0.000 / -0.019 | -0.018 | +0.0034 | 1.93e-02 |
| n339 -> n476 | L | True | 1.01 | -0.083 | -0.045 / -0.046 | -0.016 | +0.0095 | 1.93e-02 |
| n339 -> n476 | R4C | True | 1.01 | -0.068 | -0.018 / -0.048 | -0.005 | +0.0074 | 1.93e-02 |

Most harmed units in L (own job at least the median size), own write read with vecread (s4b0):

- n11: h 0.008, widened False, partner n404; own write: tank, court, swimming pool vs pane, mirror (leftover 0.84)
- n12: h 0.002, widened True, partner n352; own write: cow, sheep, tail vs blue-c, pink-c (leftover 0.95)
- n7: h -0.019, widened True, partner n393; own write: chequered, zigzagged, striped vs flecked, marbled (leftover 0.69)

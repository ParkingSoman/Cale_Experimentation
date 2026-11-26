# vecread: read a vector in ResNet-18 the way word vectors are read

Built 2026-11-23 (Shonu: "a good way to read a vector ... so subsequent sessions can build off of it").

A vector in a network layer is read against known concept vectors, the way a word vector is read against known words.
No neuron analysis and no hand-written description is needed. The known concepts come from Broden (the Network
Dissection data set: colors, textures, materials, parts, objects, scenes, labeled by region or by photo).

## Read a vector in three commands

```python
import vecread as vr                      # run from research/clean/vecread with curves/.venv
dic, held = vr.Dictionary("s4b0"), vr.Held()
card = vr.read_vector(dic, v, kind="write", held=held)          # v: 512 numbers
```

| Your vector | `kind` | Space |
|---|---|---|
| a write into the stream (a neuron's write column summed over 3 x 3, a fold's write, a difference of writes) | `"write"` | `s4b0` (stage-4 stream after block 0) or `s4b1` (final stream) |
| a read (a neuron's read row, a probe, a head row) | `"read"` | the space it reads |
| a stage-4 block-0 read kernel (256 x 3 x 3) | `"read"` | `s3k`, pass `kernel.reshape(-1)` |
| a per-cell direction in stage 3 | either | `s3` |

`vr.compare(dic, a, b, kind, held)` reads `a`, `b` and the unit difference `a - b`, plus their cosine.
`vr.card_md(title, card)` turns a reading into a Markdown table. Example: `cards_n101.py`.

## What a card says

| Field | Meaning |
|---|---|
| `up` / `down` | the concepts the vector points toward / away from. Write: raw cosine. Read: z-score of the whitened cosine against the same concept under random reads (hub correction) |
| `decomposition` | the vector as a signed mix of 5 concept directions (fixed number of terms) |
| `leftover` | the share of the vector those 5 terms do not explain. Compare with the floor below: near the floor = the vector is no more "conceptual" than a random one |
| `examples` | concepts over-represented on the held-out cells where the vector's projection is highest (IoU). This is the data-side reading; it should agree with `up` |

## How it works

1. Concept direction (per space, from 44,404 Broden training photos): mean activation on the concept's cells minus the
   mean on the other cells of the same source data set (Broden merges ADE20K, Pascal, OpenSurfaces and DTD; materials,
   textures and scenes each come from one source, so a cross-source contrast would partly encode the source).
2. Metric. Reads: whitened (chosen by K4, an independent known-answer test, confirmed on untouched photos). Writes: raw
   cosine is the WORKING CHOICE, not a verified one. Reason: a concept direction is the shift the concept causes in the
   stream, and a write is a shift. Support: K8, a consistency check (geometry against the data side), where raw beat
   whitened and shrunk-whitened in all three write spaces; K8 leans toward raw by construction. An independent write
   test is still open (idea: use each A neuron's own firing cells as the data side).
3. Read-like vectors are first turned into their pattern, covariance x r: the activation change the read responds to
   (Haufe et al. 2014, "On the interpretation of weight vectors of linear models").
4. Dictionary hygiene, decided on the two halves of the build photos only: labels need >= 50 cells and >= 10 photos
   (scene, texture: 20); label twins that the network cannot tell apart are merged ("door frame / doorframe"); labels
   whose half-1 direction does not pick itself out among the half-2 directions are dropped (`dic.dropped`).
5. Kept entries passed K1 (real against label shuffles on held-out photos).

## Spaces

| Name | Object | Per cell | Grid |
|---|---|---|---|
| `s3` | stage-3 output h | 256 | 14 x 14 |
| `s3k` | stage-3 3 x 3 patches as stage-4 block 0 reads them (stride 2, padding 1), labeled by the centre cell | 2304 | 7 x 7 |
| `s4b0` | stage-4 stream after block 0, before the ReLU (r + skip) | 512 | 7 x 7 |
| `s4b1` | final stream before the ReLU | 512 | 7 x 7 |

Network: torchvision ResNet-18 IMAGENET1K_V1 = network A (weights checked equal to `w_A.npz`). B128aug_s1 shares
A's stages 1-3, skip and head, so its vectors live in these spaces too.

## Verification (`python vr_check.py`, results in `results/vr_check.json`)

Bars were written before the first run and never changed. Development set = held set 1 (8,000 Broden val photos).
Confirmation = held set 2 (8,000 other val photos). K1-K7 ran on it once with the tool frozen at v2.5. After that the
write metric changed (v2.6) and K8 was added and run on it once; the raw-metric K2 and K3 numbers come from runs that
also scored raw. Held set 2 is now spent; about 2,900 untouched Broden val photos remain.

| Test | Bar | s4b0 dev / confirm | s3k dev / confirm | s4b1 dev / confirm | s3 dev / confirm |
|---|---|---|---|---|---|
| Entries kept (K1: real against 100 label shuffles) | per entry | 387 | 376 | 380 | 492 |
| K1 kept entries still real | (report) | .99 / .99 | 1.0 / .99 | 1.0 / .99 | .99 / .99 |
| K2 a direction finds itself on other photos (top 1), whitened (reads) | .80 | .88 / .89 | .84 / .84 | .86 / .88 | .85 / .85 |
| K2 the same, raw (writes) | .80 | .92 / .91 | .89 / .91 | .86 / .88 | .86 / .85 |
| K3 a .6/.4 mix decodes to its two parts, in order (with noise), whitened / raw | .80 (.60) | 1.0 (1.0) / .97 (.96) | 1.0 (1.0) / .98 (.98) | 1.0 (1.0) / .98 (.98) | 1.0 (1.0) / .97 (.95) |
| K4 a cross-validated logistic probe for c reads back as c | .80 | .90 / .99 | .90 / .94 | .91 / .94 | .88 / .89 |
| K6 geometry agrees with data (IoU above shuffles) | .80 | .98 / .97 | .96 / .96 | .97 / .98 | .96 / .96 |
| K8 real writes: top-5 geometry agrees with top-5 data side (mean shared; mismatched-pair 99th) | above | 1.32 (.23) / 1.26 (.21) | (no writes) | 1.15 (.10) / 1.19 (.11) | .75 (.16) / .73 (.16) |
| K7 5 nearest entries share a category (shuffled 99th) | above | .63 (.31) | .64 (.31) | .55 (.32) | .64 (.34) |

Floors (K5, random vectors; a real reading must beat these):

| Floor | s4b0 | s3k | s4b1 | s3 |
|---|---|---|---|---|
| Write (raw): top cosine, median / 95th | .13 / .16 | .06 / .08 | .13 / .16 | .18 / .23 |
| Write (raw): leftover share, median / 5th | .95 / .93 | .99 / .98 | .94 / .92 | .88 / .84 |
| Read: top z, median / 95th | 2.6 / 3.4 | 2.6 / 3.4 | 2.7 / 3.4 | 2.7 / 3.4 |
| Read: leftover share, median | .79 | .91 | .82 | .79 |

Real-vector reference (K5b, `vr_real_floor.py`): all of A's own vectors read by the same tool. Use percentiles in
these spreads, not only the random floors: real vectors are far more conceptual than random ones.

| Reference | Top score, 5th / median / 95th | Leftover, 5th / median / 95th | Top data-side IoU, median (random: median) | Hubs (share of vectors with the entry in their top 5) |
|---|---|---|---|---|
| A's 512 block-0 writes, `s4b0`, raw | .15 / .25 / .41 | .69 / .83 / .90 | .044 (.027) | zigzagged .11, dog .08, muzzle .07, neck .07 |
| A's 512 block-0 read kernels, `s3k`, hub z | 2.1 / 3.8 / 5.8 | .65 / .80 / .93 | .026 (.020) | potholed .12, stained .11, crystalline .10, marbled .10 |

Caveat on K8: its data side ranks cells by a raw projection, so it leans toward the raw metric by construction. Raw is
kept for writes also for the definitional reason in "How it works".

Face validity (`demo_head.py`, 8 head rows drawn with a fixed seed, read on `s4b1`, final tool; top concept, z):
bullet train -> platform 5.4; deerhound -> matted 4.7; guenon -> tail 4.2; sloth -> matted 4.1; syringe -> bottle 3.9;
below the floor (3.4): mailbox -> road 2.8, centipede -> cracked 2.8, newt -> scaly 2.6.

Known issue: `probe_cv_auc_median` in `vr_check.json` is nan (wrong key in the CV score table); `C_median` shows the
chosen regularization (.001-.003).

## Limits

- The dictionary holds only Broden's concepts (about 400-500 per space). A vector whose concept is not in Broden
  shows a large `leftover`. That is a named gap, not a reading.
- Scene and texture labels cover whole photos, so they describe a cell's context, not its own content.
- Head rows read on `s4b1` ignore the final ReLU (approximate; `demo_head.py`).
- The dictionary and covariance come from network A. A vector from another network (B's n101) is read with A's
  statistics; the fold, built from A's own writes, has a home advantage. Control: read all of that network's vectors
  the same way and compare percentiles.
- Write hubs among real vectors (zigzagged) are not corrected; read hubs are (against random reads).

## Files

| File | Job |
|---|---|
| `get_broden.sh`, `get_broden_fast.sh` | download Broden (Internet Archive copy; the MIT server timed out on 2026-11-23). The fast one resumes a partial download over 12 connections |
| `broden.py` | Broden reader: index, labels, cell labels per grid |
| `build_dict.py` | one forward pass: build sums (two halves) and held-out maps (8,000 photos). About 4 minutes. `held2` = the confirmation set |
| `vecread.py` | the library |
| `vr_check.py` | tests K1-K7 (`--reuse-k1` reruns K2, K4, K5 only; `--confirm` = the one-time run on held set 2) |
| `vr_k8.py` | test K8 on all real writes of A (`--confirm` = held set 2) |
| `vr_real_floor.py` | real-vector reference K5b: A's 512 writes and 512 read kernels as percentile spreads |
| `vr_diag.py`, `vr_diag_k4.py`, `vr_hub.py`, `vr_diag_write.py` | diagnostics used while building (description only) |
| `demo_head.py` | face-validity demo on 8 random head rows |
| `cards_n101.py` | n101 against the simple average fold (`results/n101_vs_fold.md`) |

## Sources (claims checked in the papers, 2026-11-23)

| Paper | Used for |
|---|---|
| Bau et al. 2017, Network Dissection | Broden; IoU data-side reading |
| Kim et al. 2018, TCAV | concept vectors from example photos; random-concept significance |
| Fong & Vedaldi 2018, Net2Vec | concept vectors are embeddings, with word-vector arithmetic ("grass - green" ~ "sky - blue") |
| Zhou et al. 2018, Interpretable Basis Decomposition | vector = weighted sum of concept vectors + "uninterpreted residual" |
| Olah et al. 2017, Feature Visualization | "random directions often seem interpretable, but at a lower rate than basis directions" |
| Haufe et al. 2014 | read weights must be turned into patterns before they are interpreted |
| Conneau et al. 2018 (CSLS) | hub correction for nearest-neighbour readings |

## History

Each change came from a failed test. The bars stayed fixed.

| Version | Change | Why (the failure that showed it) |
|---|---|---|
| v1 | difference of means per label, raw cosine, effect size for reads | first run: K2 .63-.71, K4 .23-.32 (stage 4) |
| v2 | support minimums (photos, not only cells); twin merge; reads as patterns (Haufe); `s3k` patch space | K2 failures were label twins (door frame / doorframe, autobus / bus) and scenes with 2 photos; reads favoured big concepts for any vector (random reads: effect size ~2) |
| v2.1 | twins = mutual confusion across build halves (not a cosine threshold) | the threshold rule chained 661 labels into one entry |
| v2.2 | tests scored in the tool metric (whitened, chosen by K3) | raw cosine let the largest-variance directions dominate (random reads matched concepts at .5-.6) |
| v2.3 | reliability: an entry must pick itself out across the two build halves, else dropped | K2 / K4 at .70-.80, misses on labels measured from too few photos |
| v2.4 | K4 probes: cross-validated regularization, folds split by photo | probes were overfit (train AUC 1.00, new photos .79), so not valid known answers |
| v2.5 | read hub correction: z-score per entry against 2,000 random reads | sky / wall / painted were top-1 for 3-8% of random reads (15-31x uniform) |
| confirm | frozen tool on held set 2 | all bars pass (table above) |
| v2.6 | writes use the raw metric; new test K8 (real writes, geometry against data side); confirmed on held set 2 | n101's write read as nothing in the whitened metric while its data side said wheel: 84% of its whitened length sat in the stream's low-variance half (concept directions: 21%), which whitening inflates. K3 could not choose the write metric (both passed) |

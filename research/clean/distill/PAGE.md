# PAGE: the tree, one page (the build does exactly this, nothing more)

Version: **v1** (2026-11-26 late night). Ledger of loops: `LOOPS.md`. Code: `tree/build.py`.
A loop changes ONE line of this page, or fixes the code to match it. Every change names the essay passage and the look row behind it.

**A smaller network is the big network's tree of distinctions, cut higher, in every layer at once.**
Size = how high the cut goes. 95% cuts near the leaves. 5% cuts near the root (Shonu, 2026-11-26).

| # | Part | Line (both networks) |
|---|---|---|
| 1 | Node | CLIP: an MLP neuron, or a whole head. ResNet-18: an inner channel of a block (conv1 -> bn1 -> ReLU -> conv2) |
| 2 | Write of a node | the direction it adds to the stream. CLIP MLP: its `pr_w` column. CLIP head: its 64 `o_w` columns. ResNet: its conv2 column (bn2 scale), mapped through later downsample convs. Roots: CLIP embed dims, ResNet stem channels |
| 3 | Parent (R1) | the upstream write the node reads most: score = read . write x the parent's spread (standard deviation over tokens) on the generic input. A constant offset is not a distinction. Signed for one-sided parents (MLP neurons, ResNet channels: children read the parent's ON side). Size only (no sign) for two-sided parents (CLIP heads, embed dims) and for head readers |
| 4 | Tree | following main parents up gives each node its chain of ancestors. A level = one step up that chain |
| 5 | Cut | merges go bottom up. Nodes of one block merge only inside groups that share an ancestor within L steps. L rises from 1 until the block reaches its size. The last level merges only part of the way |
| 6 | Order | inside a level, the weakest group merges first, into the sibling group with the closest read (cosine of the two groups' strongest members' reads). Strength of a node = its spread on the generic input x the size of its write as the next readers read it (MLP / ResNet node: the next block's readers; last block: the class readout; CLIP head: its own block's MLP). Strength of a group = the sum over members |
| 7 | Read | coarse pre-activation = alpha x (sum over members of z_i / s_i  -  t). s_i = a member's strong-firing level (mean of its top 1% on the generic input; its standard deviation if that mean is not positive) |
| 8 | Gate | t is set so the coarse node fires on the union rate: the share of generic-input tokens where ANY member fires. Of the shifts that give this rate, t is the one nearest 0 (the smallest move of the sum's own zero). No crossing seen (rate 0 or 1): t = 0 |
| 9 | Scale | alpha sets the strong-firing level of the read (before the gate shift t) equal to the members' largest s_i |
| 10 | Write of a coarse node | sum over members of (member's write) x E[member's activation \| coarse node fires] / E[coarse node's activation \| it fires]. If it fires on under 1% of tokens, "fires" = its top 1% |
| 11 | Heads (CLIP) | lines 3-6 on whole heads (read = V). Coarse head = the strongest member's Q and K (where it looks), and the members' summed V-O map cut to 64 dims by SVD (what it moves). V biases move into the block's output bias |
| 12 | Groups of one | the big network's own node, exactly |
| 13 | Stream | full width (loop 1). Every row states its true parameter count |
| 14 | Statistics | on the big network's own stream, generic input only (`fanout/cache/{clip,r18}_self_n256_k32_s1.pt`). Spreads on all tokens (CLIP) or cells (ResNet); pre-activations for lines 7-10 on 8,192 sampled tokens or cells per block |

Why line 7: one unit cannot take a maximum of several reads. A plain sum lets the loudest member own the union (CALC 3.5). Dividing by each member's strong-firing level makes every member's strong firings count the same.

Leave-one-out rows (each sets ONE line back to plain deletion, inside the whole build):

| Row | Line changed |
|---|---|
| `fire` | lines 7-9: the coarse node = its strongest member's own read and gate. Heads unchanged (line 11 already uses the strongest head's Q and K) |
| `write` | line 10 with only the strongest member's term (its write, at line 10's scale). Heads: line 11's summed V-O map -> the strongest head's own head (its V-O and V bias) |
| `rand` | lines 3-6: random groups of the same sizes (same-form null) |

## Page history

| Version | Change | Essay | Look row behind it |
|---|---|---|---|
| v1 | first page | 223-241, 251 | none (first build) |
| v1 (fix before loop 1) | lines 3 and 6: spread in place of RMS | 223 (a neuron is a distinction) | step 0 CLIP: one block-0 head was the main parent of 1,565-2,363 of 3,072 MLP neurons in blocks 4-10 and of every head. CLIP early heads write large constant values. Code bug (the code measured a constant as a distinction) |
| v1 (fix before loop 1) | lines 8 and 10: edge cases for units that never fire | none | planted check CLIP KL 10.4 (ResNet passed): twins that never fire on the generic input got a threshold above all tokens and a write divided by about 0. Code bug |
| v1 (fix before loop 1) | line 9: alpha is measured on the read before the gate shift. Line 10 keeps the sign of a GELU mean | none | planted check, debug: every merged group was an exact twin pair (cut correct), but coarse pre-activations were off by up to .49 and one write coefficient was -1e6. Code bug |
| v1 (fix before loop 1) | line 8: t = the shift nearest 0 that gives the union rate (was: midpoint of two samples) | none | planted check CLIP KL 2.1e-4 after the line 9-10 fixes; the midpoint moved exact twins off their own zero. Code bug |
| v1 (critic, before loop 1 scores) | lines 6, 7, 14 and the `fire` / `write` rows now state what the code already did | none | critic agent: no result-changing bug; 3 unstated choices (group read = strongest member's read, head strength via its block's MLP, `write` row on heads). Wording only. Open cosmetic: a sample tie can set the rate slightly above the union rate |

# PAGE: the tree, one page (the build does exactly this, nothing more)

Version: **v2** (2026-11-27) = **strongest member kept + branch write**. In effect this is pruning chosen by the tree, with a write fold. It is not yet a coarse node. v1 = union firing (now the row `union`). Ledger of loops: `LOOPS.md`. Code: `tree/build.py`.
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
| 7 | Read and gate (v2) | the coarse node reads and fires as its branch's STRONGEST member (line 6 strength): that member's own read, bias and gate, unchanged. It does not fire on the union of the branch |
| 8 | (v1 only, row `union`) | v1 read: alpha x (sum over members of z_i / s_i - t), s_i = strong-firing level (top-1% mean, never below the spread); t set so the coarse node fires at the union rate (any member fires); all-firing union: fires everywhere |
| 9 | (v1 only, row `union`) | alpha: strong-firing level of the v1 read = the members' largest s_i |
| 10 | Write of a coarse node | sum over members of (member's write) x w_i, w_i = E[member's activation \| coarse node's strong firings] / E[coarse node's activation \| same], clipped to [0, 1]. Strong firings = its top 1% of tokens or cells |
| 11 | Heads (CLIP) | lines 3-6 on whole heads (read = V). Coarse head = the strongest member's Q and K (where it looks), and the members' summed V-O map cut to 64 dims by SVD (what it moves). V biases move into the block's output bias |
| 12 | Groups of one | the big network's own node, exactly |
| 13 | Stream | full width (loop 1). Every row states its true parameter count |
| 14 | Statistics | on the big network's own stream, generic input only (`fanout/cache/{clip,r18}_self_n256_k32_s1.pt`). Spreads on all tokens (CLIP) or cells (ResNet); pre-activations for lines 7-10 on 8,192 sampled tokens or cells per block |

Why line 7 changed (v2): the v1 union read did fire on the union (overlap look: rate = union rate, 85-99% of members' strong firings caught on ResNet). It still lost to the strongest member's own read on BOTH networks (85%: ResNet 34.7 vs 41.6, CLIP 14.9 vs 41.6 agree). The trained-from-random students fire only a little more broadly than one big-network node at 50% (s4b0 .24 vs .19), far below the union. On ResNet the union was realized and still lost. On CLIP it was realized only in part (strong recall .38-.58 in blocks 6-10), so the CLIP loss may be the read, not the concept. Open: THEORY section 1 says both 'read = root of taxi+ambulance' (diagram) and 'fires on the union' (table). Essay 223-241: siblings carry each other on their negative poles, and the small model's neuron sits at the parent's distinction.

Leave-one-out rows (each sets ONE line back to plain deletion, inside the whole build):

| Row | Line changed |
|---|---|
| `union` | lines 7-9 back to v1 (union read and gate). `zero`: v1 read with no gate shift (t = 0) |
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
| v1 (fix, 2026-11-27) | line 7: s_i never below the member's spread | none | CLIP read weights up to 1,820: a member that almost never fires had a top-1% mean near 0. Code bug |
| v1 (fix, 2026-11-27) | line 10: w on the coarse node's top 1% (was: every token with z > 0) | none | CLIP write weights up to 1.15e4: weak firings set the denominator near 0. Code bug |
| v1 (fix, 2026-11-27) | line 8: a union that fires everywhere -> the coarse node fires everywhere (was t = 0) | none | ResNet 5%: groups of 30-163 were silent, write weights 7e8, KL 2e22. Code bug |
| v1 (fix, 2026-11-27) | line 10: w clipped to [0, 1] | none | CLIP block 1 pair 928+1524 (rates .1-.2% on the generic input): w = 181, 215 from two tiny negative GELU means. This one pair alone sent CLIP to chance (bisection). Code bug |
| **v2** (2026-11-27) | **line 7: the coarse node fires as its branch's strongest member, not on the union** | 251 (a crushed layer stays interpretable, "coarser") | `fire` beat the union on both networks; overlap look showed the union was realized; trained students fire near one node's breadth at 50% (`LOOPS.md` loop 1-2) |


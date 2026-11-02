# Tree model, $0 track: tasks Z1, Z3, Z4

Source plan: `~/.claude/plans/explain-again-to-me-misty-river.md` (approved 2026-10-18). Spec = that file. Z2 (cut vs merge, `research/tree/z2.py`) already exists and is run by the controller.

## Global Constraints

- No training of any model. No rented GPU. No network use after the first load (the model is cached at `research/tree/hf`). Run python with `HF_HOME=<repo>/research/tree/hf`.
- Work only inside `research/tree/` and write result files to `results/tree-<name>.json` (and `.md` where stated). Do not touch any other file in the repo. Do not run `git commit`, `git add`, `git push` or any git write. The repo has many unrelated uncommitted changes.
- Reuse `research/tree/tree_lib.py` (read it first). Add new helpers to a NEW file `research/tree/tree_lib2.py` if needed. Do not edit `tree_lib.py` or `z2.py` (z2.py may be running).
- Memory on this Mac is very tight (swap almost full). Never hold more than two copies of the model. Free big arrays when done. Run each script once in the foreground with a timeout, never leave background processes behind.
- Do not run any script until `pgrep -f z2.py` prints nothing (Z2 is still using memory). Write code first, then wait: `until ! pgrep -f z2.py >/dev/null; do sleep 5; done`.
- Model: SmolLM2-135M, 30 layers, hidden 576, MLP width 1536 (SwiGLU), 9 query heads, 3 key/value heads, head dim 64. Sizes for the MLP: k = 768, 384, 192 (50%, 25%, 12.5%).
- Calibration text = `cal = token_blocks(tok, "train", 32)`. Held-out text = `held = token_blocks(tok, "test", 100)`. Never fit anything on held-out text. Baseline held-out CE = 3.0180 nats/token.
- Report every number with its source. If a check fails or a result looks too good, say so. Do not tune on held-out text.
- Merge rule already fixed in `tree_lib.py`: sign-align neurons first, average gate and up rows, sum down columns ("average parent"). Optional least-squares refit of the down weights is `ls_down`.

## Task 1: Z1, do natural groups exist, and are parents broader than children?

Question: do neurons that a Ward tree puts together share meaning, and does a merged parent respond to a broader range of tokens than its children? (Essay claim: with less room, a neuron covers a coarser idea.)

Create `research/tree/z1.py`. For every layer (0 to 29), using calibration text only for the groups, held-out blocks (`held[:40]`) for the readings:

1. Build the Ward tree on weight features and on function features (`neuron_features`, `ward_tree`, `cut_tree`) at k = 768 and k = 192, plus a random partition of the same sizes as control (equal sizes, seed 0).
2. Quantitative, for every layer and each of the 3 partition types at each k:
   - mean pairwise Pearson correlation of the sign-aligned neuron activations (`hidden(...) * sign`) inside groups of size 2 or more, measured on held-out text.
   - breadth: for each neuron, take its 300 highest sign-aligned activations over the held-out tokens and count distinct token ids among the tokens at those positions. Breadth of a group's parent = the same count for the merged neuron (`merge_groups` result applied to held-out inputs, top 300 positions). Report mean child breadth vs parent breadth, per layer, for groups of size 2 or more.
3. Readable samples: for layers 3, 15 and 28, k = 192, weight-feature Ward: write `results/tree-z1-groups.md` with 8 groups of size 3 to 8 per layer (pick the groups with the highest within-group correlation, then 4 more at random of size 3+). For each member neuron list the 12 highest and the 12 lowest raw activations of `hidden` (not sign-aligned, so both poles show) as decoded token with 3 tokens of left context. Add the same layout for 2 random-control groups per layer.
4. Save numbers to `results/tree-z1.json`. Print a summary table: per layer group, mean within-group correlation for weights/function/random, and mean breadth ratio parent/child.

Done when: `results/tree-z1.json` and `results/tree-z1-groups.md` exist, the script ran once without error, and the printed summary shows numbers for all 30 layers.

## Task 2: Z3, do the merges nest, and what does stacking cost?

Question: if the 384-neuron slice is made by merging the 768-neuron parents (parents of parents), is it as good as merging straight from the 1536 neurons? And does refitting at every level change that?

Create `research/tree/z3.py`. Use the weight-feature Ward tree (`ward_tree`, `cut_tree`) at k = 768, 384, 192. Held-out CE and KL (to an untouched teacher copy) as in `z2.py` (`eval_loss`).

1. Check by code that the Ward partitions are nested (every k=192 group is a union of k=384 groups, every k=384 group a union of k=768 groups) for all 30 layers. Print the count of violations (expect 0).
2. Straight: merge from the original 1536 neurons straight to each size (average parent). Stacked: build k=384 by merging the k=768 parent neurons (use `merge_groups` on the 768 parents with their own sign alignment, weighting by member count so that a parent of parents equals the plain mean over all leaves). Print the max abs weight difference between straight and stacked. Expect about 0. If not about 0, say why.
3. Four variants at each size, each with and without the least-squares refit of the down weights fitted against the ORIGINAL layer output on the calibration text:
   - straight, no refit; straight, refit.
   - chain-with-refit: level 768 = straight merge + refit; level 384 = merge the refit 768 parents, then refit again against the original output; level 192 = same from 384.
   Report CE and KL for each, and the per-layer relative error on held-out MLP inputs like `z2.py`.
4. Storage note, computed not guessed: count the numbers stored for (a) the 1536 original neurons, (b) the tree "average parent" form, (c) the "trained parent" form (extra down matrix per level). Print the three counts for one layer.
5. Save `results/tree-z3.json`. Print a table of CE and KL.

Done when: the file exists, the script ran once without error, the nesting check printed 0 violations or an explained number, and the table prints for all 3 sizes.

## Task 3: Z4, does the same idea work on query heads and on depth?

Question (hypothesis, not a claim): can groups of query heads, or neighbor layers, be merged with less loss than cutting them? Main width is NOT in this task (it needs RMSNorm folding; the controller defers it).

Create `research/tree/z4.py`. Same held-out CE/KL as `z2.py`.

Heads. Query heads 0-2 share key/value head 0, heads 3-5 share key/value head 1, heads 6-8 share key/value head 2 (check this in the mlx_lm Llama attention code; fix the grouping if it differs). Only merge or cut heads inside one key/value group. Sizes: 6 heads (one removed per group) and 3 heads (one left per group). To keep the model code unchanged, emulate removal: a removed head keeps its slot but its `o_proj` columns are set to 0.
   - cut: remove the head(s) with the smallest activity (mean norm of the head output times norm of its `o_proj` columns on calibration text).
   - merge: within each group, merge heads by averaging their `q_proj` rows (the kept head gets the mean). Choose the pairs by highest cosine similarity of the heads' outputs on calibration text.
   - Each with and without a least-squares refit of the kept heads' `o_proj` columns. The refit target is the ORIGINAL attention block output (the `o_proj` output of the intact model on calibration text). The refit input is the kept heads' outputs with the NEW q (so the layer must be modified first and its `o_proj` input captured). Do this one layer at a time with all other layers intact (original upstream inputs), restore, then apply all layers and evaluate.
   - control: merge random heads inside the group.

Depth. Sizes: 28, 26 and 22 layers out of 30.
   - cut: drop layers (choose the least harmful by single-layer-removal held-out... NO: choose on calibration text only), never layer 0 or 29.
   - merge: replace two adjacent layers by one layer whose weights are the average of the two (all attention and MLP weights and the two norm weights). Choose adjacent pairs by smallest relative weight distance. Make a runnable model for evaluation (a new list of layers; mlx_lm Llama uses `model.model.layers`).
   - control: merge random adjacent pairs.

Save `results/tree-z4.json`. Print one table with rows = axis, method, refit yes/no, size, CE, KL. Say clearly which parts you could not finish.

Done when: the file exists, the script ran once without error (or the failing part is named and the rest finished), and the table prints.

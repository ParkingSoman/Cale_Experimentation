# GRIFFIN and MoEfication: what to borrow (2026-10-19)

Shallow clones: `src/griffin/` and `src/moefication/`. Not installed. Nothing was run.
Warning: `src/griffin` has committed `.pyc` files (`src/griffin/src/lm_eval/**/__pycache__`). Never import them.

## GRIFFIN (hdong920/GRIFFIN): picks FFN neurons from the prompt, no training

Where (file `src/griffin/src/griffin/llama.py`, same pattern in `gemma.py`, `mistral.py`, `opt.py`):

| What | Lines |
|---|---|
| Swap each layer's MLP for the GRIFFIN MLP: `get_llama_griffin(model, k_schedule)` | 8-30 |
| Static "magnitude" variant: score = row-norm(gate) x row-norm(up), top-k | 18-26 |
| The intermediate state `act_fn(gate(x)) * up(x)` | 83 (gen), 102 (class) |
| Neuron statistic from the prompt | 88 (gen), 105 (class) |
| Selection call | 89 (gen), 106 (class) |
| Build slim weights (rows of gate and up, columns of down) | `prepare_reduced_weights`, 49-59 |
| Decode with slim weights | 98 |
| `select_neurons(neuron_stat, method, k)`: `topk`, `topk_sample`, `sample`, `random` | `utils.py` 4-21 |

Plain description:
1. Run the prompt through the full model (prefill). Each layer keeps its FFN intermediate state `z = silu(gate x) * (up x)`, shape `[batch, tokens, d_ff]`.
2. Divide each token's `z` by its own L2 norm over neurons. This makes every token count the same.
3. For each neuron, take the L2 norm of that normalised column over the prompt tokens. This is the neuron's "flocking" score (line 88).
4. Keep the top `k = d_ff * k_factor` neurons (line 87, 89). `k_factor` comes from `k_schedule[layer]`.
5. Cut the gate, up and down weights to those neurons (lines 52-59).
6. For every new token (decode), run only the cut FFN (line 98).
7. The choice is made once per prompt and per layer. It needs no training and no labels.
8. "class" mode (lines 100-115) tests the last prompt token only, and masks it with the set found from earlier tokens.
9. A layer with `k_factor == 0` is dropped (lines 95-96).
10. Claim of the idea: the same neurons stay strong across the tokens of one sequence ("flocking").

SwiGLU: yes. It is written for SwiGLU (Llama, Gemma, Mistral). Stat uses the product, so it works on `act_fn(gate)*up` = our raw neuron. Gaps: it uses the L2 size of each neuron, so sign is lost. It is built for old `transformers` (uses `config.pretraining_tp`, `config.mode`, `config.selection_method`), so port the 10 lines, do not import the package. Batch size 1 in `prepare_reduced_weights` (line 51).

## MoEfication (thunlp/MoEfication): groups neurons into experts, routes without a trained router

Where:

| What | File and lines |
|---|---|
| Co-activation graph: `relu(x W^T)` per token, outer product, clamp at 1, sum over tokens | `moefication/adj.py` 41-43 |
| Keep edges above a percentile threshold, write a METIS graph file | `adj.py` 53-83 |
| Cut the graph into experts (external `gpmetis`, see README) | README "Expert Construction" |
| Even out expert sizes after METIS | `moefication/trans_gp.py` 19-51 |
| Training-free grouping by weights: balanced k-means on normalised first-layer rows | `moefication/utils.py` `ParamSplit.split`, 87-93 |
| Fast GPU version of that k-means | `faster_moefication/moefication.py` `kmeans_balanced` (63), `split_ckpt` (112) |
| Training-free routing ("similarity"): expert centre = mean of its normalised weight rows; score = cosine of the input to each centre; take top experts | `utils.py` `ParamCenter.cal_center` 142-186 (route step at 177-178); driver `similarity_select_example.py` |
| Trained router (we do not want it) | `utils.py` `MLPCenter` 188-303 |

Plain description (co-activation graph plus similarity routing):
1. Run text through the model. Save each layer's FFN input `x`.
2. For each token, compute `a = relu(W x)`, one value per neuron.
3. Two neurons share an edge weight when both are above 0 on the same token (`a a^T`, clamped at 1, summed over tokens).
4. Make the graph sparse: take a percentile of the weights as a threshold (lines 55-61), keep only edges above it.
5. Give the graph to METIS. It cuts it into `num_expert` parts that have few cut edges.
6. Move neurons between parts so each part has the same size (`trans_gp.py`).
7. For each expert, average the (unit-length) weight rows of its neurons. This is the expert's centre.
8. At run time, normalise `x`, take the distance to each centre, and run only the nearest experts (top 25 of 96 in the code, line 174-178).
9. This router has no training. The "MLP" router (`MLPCenter`) is trained; skip it.
10. Quality is measured as overlap with the true top experts (`acc`, lines 180-186).

SwiGLU: no, not as written.
- The activation test is `torch.relu(x @ W)` (`adj.py` 41, `utils.py` 172). SwiGLU has no hard zero. The graph would need a threshold on `|silu(gate) * up|` instead.
- It uses only the first linear layer (`wi`). A SwiGLU layer has two (gate and up) plus down. Grouping must tie all three rows per neuron.
- `adj.py` hard-codes `reshape(-1, 4096)` (line 33) and CUDA devices. The examples are stale: `similarity_select_example.py` calls `ParamCenter(config, filename)` but the class takes `(config, filename, layer)`, and `cal_center` calls `load_ffn_weight` with a different signature than its definition. Treat the code as a recipe to rewrite, not to run.
- Old stack: torch 1.6, transformers 4.20.1, CUDA, `k_means_constrained`, `gpmetis`, `balanced_assignment` C++.
- Useful idea that does transfer: step 3 (graph from co-firing) can run on our own saved raw-neuron activations with a magnitude threshold; step 7 (centre from weight rows) needs no data.

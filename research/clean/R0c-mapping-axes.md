# R0c: Which axis should we use to map new neurons to old neurons?

Date: 2026-10-19. Web only. "NOT VERIFIED" means I did not confirm the claim on a primary page. Several fetches returned only abstracts or search snippets. I say so in each row.

## 1. What the essay does (Question 1)

Source: https://already-clean.goodfire.com/essay.md (166k characters). The fetch tool read all of it in two passes through a small summary model. Quotes below come from those summaries. Check the wording on the page before you cite it.

| Item | Finding |
|---|---|
| Result | After retraining layer 3b of InceptionV1 from 480 to 10 neurons, the neurons are coarser. Example: "one 'curves & loops' neuron instead of ten curve detectors." |
| How they show it | They show each new neuron's five strongest natural image crops. They report accuracy (58.35% top-1, 81.14% top-5). |
| How they look for the "children" | They do NOT list the new neurons' children among the old ones. They go the other way. They train a sparse autoencoder (SAE) on the crushed layer. They then ask if the old curve detectors can be pulled out. Quote: "If they were placed in some high dimensional combination of neurons, we should be able to extract them. So we can train an SAE on this crushed layer and compare them to the originals." |
| Result of that test | The SAE features are "broad and bunched together, not ten curves." The figure caption says it shows "the autoencoder feature closest to" each original curve detector. |
| Match measure | The essay does not define "closest." NOT VERIFIED which axis it used. |
| Axes compared | Only activation examples (natural crops) and SAE feature closeness appear in this test. I found no weight, circuit or feature-visualization comparison for it. |
| Per-layer differences | None given. One layer (3b) was tested in vision. A second test cut the final layer of a medium ConvNeXt from 1,024 to 13 neurons: "We see the same pattern of it broadening." |
| Caveat | The summary model said it could not find any narrowed-layer text in the second half. The first-half summary is the only source. Treat details as NOT VERIFIED until you read the page. |

What this means for us. The essay gives no recipe for finding children. It shows the claim on one early-to-middle vision layer and one late ConvNeXt layer. It does not show a layer where the claim fails. Our own coverage test is therefore not a copy of the essay. It is our own design.

## 2. Does the best axis depend on depth? (Question 2)

| Source | What it says | Axis it supports | Verified? |
|---|---|---|---|
| Geva et al. 2021, https://arxiv.org/abs/2012.14913 | "lower layers tend to capture shallow patterns, while upper layers learn more semantic ones." Values "induce output distributions", "especially" in upper layers. | Input side early. Output side late. | Abstract read. |
| Geva et al. 2022, https://arxiv.org/abs/2203.14680 | Each FFN sub-update promotes concepts in vocabulary space. Often human-interpretable. | Output side (vocabulary space). | Abstract read. The page I read said nothing on depth. Depth claim NOT VERIFIED. |
| Gur-Arieh et al. 2025, https://arxiv.org/abs/2501.08319 | Input-based descriptions "fail to capture the causal effect of the feature on outputs." Output-centric descriptions (unembedding projection, or token change when amplified) do better on output tests. Combining both does best. | Use both axes. | Abstract read. A per-layer result is NOT VERIFIED. The full PDF was too big to fetch. |
| Gurnee et al. 2024, https://arxiv.org/abs/2401.12181 | Neurons with clear roles in predicting or suppressing a token set exist. A search summary says suppression neurons sit in later layers than prediction neurons. They are found from output weights times the unembedding. | Output side, late layers. | The abstract did not mention layers. The late-layer claim comes from a search snippet. NOT VERIFIED on the paper body. |
| Voita et al. 2023, https://arxiv.org/abs/2309.04827 | In OPT, early layers hold many dead neurons and token or n-gram detectors. The second half is "fully alive." Later layers learn higher-level semantics. Detector neurons also remove their own trigger token from the residual stream. | Input side early (token triggers). Later layers are not simple. | HTML page read through a summary model. |
| Lad et al. 2024, https://arxiv.org/abs/2406.19384 | Four stages: detokenization, feature engineering, prediction ensembling, residual sharpening. Early and last layers are the most sensitive to edits. | Gives a stage map. It does not name an axis. | Abstract only. |

Summary of the evidence. Three sources point the same way. Early layers read tokens, so inputs describe them. Late layers write predictions, so outputs describe them. Gur-Arieh says outputs add information that inputs miss, at any depth. No source I found states a clean rule such as "use inputs below layer N." The depth rule is a pattern, not a proven law.

Caution for our prior. Voita shows that a token detector also writes "remove this token." So even early neurons have an output job. Our prior ("every layer should show strong children") is still a hypothesis. The literature does not promise it. A weak layer could also be a real result.

## 3. Methods that map units one-to-many (Question 3)

| Method | Source | What it matches on | One-to-many? | Verified? |
|---|---|---|---|---|
| Bipartite matching | Li et al. 2015, https://arxiv.org/abs/1511.07543 | Activation correlation (page did not say; NOT VERIFIED) | No. One-to-one. | Abstract read. |
| Sparse prediction | Li et al. 2015 | Predict one network's unit from many units of the other | Yes. One-to-many. | Abstract read. |
| Spectral clustering | Li et al. 2015 | Group units | Yes. Many-to-many. | Abstract read. |
| Feature splitting by decoder similarity | Anthropic, https://transformer-circuits.pub/2023/monosemantic-features/ | Decoder (output) direction cosine. A secondary post also used UMAP of cosines. | Yes. One small-SAE feature has several large-SAE features. | Search snippets only. The exact procedure is NOT VERIFIED. |
| Absorption | Chanin et al., https://arxiv.org/html/2409.14507v3 | Activation. A child latent fires and the parent latent stays silent. It follows from sparsity with co-occurring features. | Yes. Parent to child. | Search snippets only. |
| Matryoshka SAE | Bussmann et al. 2025, https://arxiv.org/html/2503.17547v1 | Nested dictionaries. Small ones learn general features. Large ones learn specific features. | Yes by design. | Search snippets only. |
| Feature hedging | https://arxiv.org/html/2505.11756v2 | Narrow SAEs merge correlated features. | Warning: a narrow level can blend features. | Search snippet only. |

Is there one general method? No. I found none that works across all layers. Li et al. matched by activations. The SAE work mixes decoder cosine (output side) with activation behaviour. The field does not map per layer by a fixed rule. It picks one signal and checks it by hand. Also note the Matryoshka warning: a narrow level can blend features. That is close to our case, a 10-neuron layer.

## 4. Middle layers (Question 4)

| Approach | Source | Axis | Verified? |
|---|---|---|---|
| Replace MLPs with cross-layer transcoders, then trace attribution graphs | Anthropic, "On the Biology of a Large Language Model" and "Circuit Tracing" (transformer-circuits.pub) | Effect on later layers and on the output, by causal attribution. They avoid raw neurons because neurons are polysemantic. | Secondary summaries only. NOT VERIFIED on the primary page. |
| Virtual weights (weight links between features across layers) | Searched. No source found. | Weights read across layers. | NOT VERIFIED. |
| Attention-head reading of MLP output | Searched. No source found. | Output into later components. | NOT VERIFIED. |

So for middle layers the common practice is the downstream effect. It is not a bare input list. I found no paper that settles a neuron-level axis for mid-depth MLPs.

## Recommended axis per depth for our test

| Depth | Our layer | Primary axis | Second axis (to compare) | Rests on |
|---|---|---|---|---|
| Early | L3 | Input: where the neuron fires (keep what already works) | Output: unembedding projection of the neuron's write vector | Geva 2021 abstract. Voita 2023 on token detectors. |
| Middle | L15 | Both, scored side by side. Add a downstream effect check: ablate the child set and see if the new neuron's effect on the later layers is explained | Output projection | Gur-Arieh 2025 (both axes best). Anthropic attribution work (downstream effect), secondary only. |
| Late | L28 | Output: each neuron's write vector projected through the unembedding. Compare the new neuron's write direction with its children's write directions (cosine), and compare top promoted tokens | Input coverage as a control | Geva 2021 and Gurnee 2024 (late neurons act on outputs). Gur-Arieh 2025. |

Rule for reading results. If L28 jumps from 45% to near L3 levels on the output axis, the "wrong axis" prior is supported. If it stays low on both axes, the prior is not supported at L28. Do not tune the axis until it passes. Decide the axes first and run the same nulls on both.

## Recommended child-finding method

1. For each new neuron, take its input pattern (top tokens) and its output pattern (write vector). Take the same two things for every old neuron.
2. Score each old neuron against the new one on both axes. Use the same shuffled-neuron nulls as now.
3. Call an old neuron a child if it passes on the axis for that depth. Count units with 2 or more children, as now.
4. Add one sparse-prediction check from Li et al. 2015: predict the new neuron's activation from a small set of old neurons (a sparse regression). Report how many old neurons it needs. This is the one-to-many match built for this question. The fit uses activation co-occurrence. Add a second fit that predicts the new neuron's write vector from old write vectors. This is the decoder-cosine idea from the SAE feature-splitting work.
5. Do not use an SAE on the narrow layer as the main test. The essay used one only to show that it cannot recover the originals. Feature hedging (arXiv 2505.11756) warns that narrow dictionaries blend features.

Rests on: Li et al. 2015 abstract (sparse prediction finds one-to-many mappings), Anthropic feature splitting (decoder cosine, secondary snippets), Gur-Arieh 2025 (use both axes). The essay does not supply a finding method. Our design is our own and needs its own nulls.

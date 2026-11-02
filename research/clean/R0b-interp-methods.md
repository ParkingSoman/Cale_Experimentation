# R0b: Which interpretability methods should we use?

Date: 2026-10-19. Written from web searches and page fetches. "NOT VERIFIED" marks anything I did not confirm on a primary page.

## 1. What the essay says

I fetched the essay text (https://already-clean.goodfire.com/essay.md). It is about 166,000 characters. Tool summaries covered all of it. The quotes below come from those summaries. Check the exact wording against the page before you cite it.

### Methods it used to read neurons

| Method | What the essay does with it |
|---|---|
| Top activations, both poles | Looks at the strongest firings and the negative side. The negative side "tells us what distinction matters at this point in the circuit." |
| Steering and ablation | Turns a neuron up to see what the model writes. Removes neurons to see if the output changes. Example: removing Santa neurons drops Santa's probability. |
| Attribution graphs and circuit tracing | Follows upstream and downstream links to test an explanation. |
| Blind maximum test | One agent writes an explanation. A second agent sees only that text and writes an example to fire the neuron. |
| Bundle clustering | Groups upstream flow into bundles. Ablates the other bundles to check for several jobs. |
| Sign split for SwiGLU | Splits gate and value products into four sign cases. This matters for us, because SmolLM2 uses SwiGLU. |
| Training-time watching | Watches neurons grow during training ("recursive generalization"). |
| Narrow-layer retraining | Shrinks a layer (480 to 10 neurons). The essay reports coarser neurons, not mixed ones. This is the same shape as our experiment. |

### Methods it calls misleading or limited

| Method | Why, in the essay's words or close to them |
|---|---|
| Top-k labels that ignore the negative side | Can mislabel a Santa neuron as "capital cities." |
| Noun-list descriptions ("cats and cars") | Can hide one shared pattern. |
| Sparse autoencoders | "An SAE trained on the layer fractures it the same way, splitting the V into a car feature and a cat feature." On curve neurons, the SAE used 84k of 100k dictionary items. The author warns this "can make it look like there are more building blocks than there really are." |
| Probes | "can only answer a question we already had." |
| Simulation scores | "Simulation scores often end up measuring where attention puts information." The author also says he no longer thinks it is a great metric for explanations. |
| Free feature visualization | Finds routes that real data rarely uses. |
| Percentile thresholds | "Flatten" a neuron's natural activation shape. |
| Too little context or data | Short contexts made long-context neurons look polysemantic. |
| Superposition as the default | Should be "the last hypothesis," not the null. |
| Counting top-k matches to a label | Called "a reasonable first step." But it would score superposition and recursive generalization alike. |

The summaries said the essay does not discuss regressing neurons on other neurons. I did not read every line myself.

### What this means for us

- Our hand-rolled top-token eyeballing is only a first step by the essay's own words.
- The essay does not argue that SAEs are the fix. It says SAEs can fracture clean neurons.
- So the plan should be: look at the raw neurons first, with more data, both poles, and causal tests. Use SAEs only as a cross-check.

## 2. Standard method for each of our questions

| Question | Standard method | Paper (id) | Tool | Works for our models? | Cost on a Mac |
|---|---|---|---|---|---|
| 2a. Describe a unit, with a score | LLM writes an explanation. A second step scores it. Scores: detection, fuzzing, simulation, intervention. | Bills et al. 2023 (OpenAI page, NOT VERIFIED this session). Paulo et al. arXiv 2410.13928 (verified). | EleutherAI `delphi` (github.com/EleutherAI/delphi). Neuronpedia for browsing. | The scoring code is model-agnostic. It reads cached activations from SAEs or sparse coders. Raw MLP neurons need a small adapter. NOT VERIFIED. | Activation caching is cheap on CPU. The explainer is the cost. |
| 2a. Can Claude be the explainer? | `delphi` ships two clients. One is local vLLM. The other is OpenRouter, an OpenAI-compatible API. Default explainer is Llama 3.1 70B. The README does not mention Anthropic directly. | Same | `delphi` | Claude via OpenRouter is likely. NOT VERIFIED. A custom client is the safe plan. | API fees. Amount NOT VERIFIED. |
| 2b. Causal role | Ablation (zero, mean, resample). Direct logit attribution. Activation patching. Attribution graphs with transcoders. | Circuit tracing paper at BlackboxNLP 2025 (aclanthology 2025.blackboxnlp-1.14, verified via repo page). Transcoders: Dunefsky et al. arXiv 2406.11944 (verified). | `circuit-tracer` (github.com/decoderesearch/circuit-tracer, MIT). TransformerLens or nnsight for ablation. | `circuit-tracer` lists Gemma-2 2B, Llama-3.2 1B, Qwen3 0.6B to 14B, Gemma-3 270M+. It lists no SmolLM2 and no GPT-2. Its default backend needs TransformerLens support. An nnsight backend covers most HF models. Ready-made transcoders are the limit. | Ablation and DLA: minutes on CPU. Transcoder training: see section 3. |
| 2c. Compare units across layers or models | SAE or transcoder feature matching. Pair by decoder cosine, with the Hungarian algorithm. Crosscoders for two models at once. CKA or SVCCA only give one similarity number, not a unit-to-unit map. | Paulo and Belrose arXiv 2501.16615 (verified). Crosscoders: Anthropic, transformer-circuits.pub/2024/crosscoders (verified). CKA: Kornblith 2019 arXiv 1905.00414 (NOT VERIFIED). | Own code for cosine matching. Crosscoder code exists in public repos (NOT VERIFIED which). | Cosine matching: yes, any model. For our case (10, 48, 192 neurons vs 1536), use weights directly. Compare the small layer's output directions with the original neurons' output directions. Also compare activations on shared text. | Cheap. |
| 2d. Parent/child structure | Train dictionaries of several sizes. Watch big features split from small ones. Matryoshka SAEs train nested sizes at once, so coarse units come first. Absorption metric checks that a parent still fires where children fire. | Matryoshka: Bussmann et al. arXiv 2503.17547 (verified). Absorption: Chanin et al. arXiv 2409.14507 (verified). | `SAELens` (Matryoshka and absorption tools are inside, NOT VERIFIED which version). | Training is possible on any model. No ready SmolLM2 Matryoshka SAE found. | See section 3. |
| 2e. What a smaller model lacks | Crosscoder or SAE-feature diffing. Exclusive features of one model. | Anthropic crosscoder diffing, transformer-circuits.pub/2025/crosscoder-diffing (verified to exist). | Same as 2c. | Warning: Anthropic's update says model-exclusive features tend to be polysemantic and dense. A third-party post says crosscoders can create fake chat-only latents. So use diffing as a lead, not as proof. | Training a crosscoder on two models: NOT VERIFIED. |

## 3. Availability

| Item | Finding | Status |
|---|---|---|
| SmolLM2-135M SAEs, EleutherAI | Repo `EleutherAI/sae-SmolLM2-135M-64x`. Folders `layers.{0,3,6,9,12,15,18,21,24,27}.mlp`. 1.7 GB. 64x expansion from the name. Hookpoints are MLP. The README is 176 bytes. k is not stated. | Verified listing. Format likely `sparsify` (EleutherAI). Loading NOT VERIFIED. |
| SmolLM2-135M SAEs, community | `juiceb0xc0de/smollm2-135m-SAE`. One Top-K SAE per residual layer, 18,432 features, k=50, reported explained variance 0.949. The card says no downstream validation was done. | Verified card. Treat as unvalidated. |
| SmolLM2-1.7B | `EleutherAI/sae-SmolLM2-1.7B-layer17-32x`, loaded with `Sae.load_from_hub`. | Verified. |
| SmolLM2-360M | Nothing found. | NOT VERIFIED (one search only). |
| SmolLM2 transcoders | Nothing found. | NOT VERIFIED. |
| SAELens format for SmolLM2 | Nothing found. | NOT VERIFIED. |
| GPT-2 small SAEs | SAELens release `gpt2-small-res-jb` (residual stream, e.g. `blocks.8.hook_resid_pre`). Neuronpedia hosts it. | Verified through secondary docs. |
| GPT-2 small transcoders | `pchlenski/gpt2-transcoders` and a repack at `jacobdunefsky/gpt2small-transcoders`. Also `EleutherAI/gpt2-mntss-transcoder-relu-sp6-skip`. | Names seen in search. Contents NOT VERIFIED. |
| Training time on a Mac | No tool doc gave a number for a Mac. `sparsify` says a CUDA GPU is "highly recommended." | NOT VERIFIED. I give no estimate. Measure a 10-minute pilot first. |

The EleutherAI 64x SAE on layer MLPs is the closest thing to a "wide original layer" dictionary for SmolLM2-135M. It is cheap to test.

## 4. Pitfalls the field warns about

| Pitfall | Source | What to do |
|---|---|---|
| A neuron looks like one concept on a narrow text set but is more complex | Bolukbasi et al. 2021, arXiv 2104.07143 (verified). It also holds for linear combinations. | Test every claim on several text sets. |
| Top-activating examples cannot tell polysemantic from coarse-with-fine-detail | Essay, quoted above | Add negative pole, mid-range examples, and causal tests. |
| Simulation score may track attention patterns, not the unit | Essay | Prefer detection and fuzzing, plus ablation. |
| SAEs can fracture clean units, and dictionary size inflates the count of "building blocks" | Essay | Never count SAE features as ground truth. |
| Feature splitting is not clean. Parents can fail to fire where children fire (absorption) | Chanin et al. 2409.14507 | Run the absorption check before you call a unit a parent. |
| SAE features differ across seeds. About 30% shared for the largest model tested. Smaller models and SAEs share more. | arXiv 2501.16615 | Train two seeds. Keep only matches. |
| High decoder cosine does not prove same computation | LessWrong post and a 2026 preprint (alphaxiv 2609.12591), both second-hand | Also compare what units do to the output. |
| Model-exclusive features in diffing are often dense and polysemantic | Anthropic crosscoder diffing update | Filter dense and dead features before you read any. |
| Dead and dense features | General SAE practice. Not tied to one paper here. | Report the dead and dense share. |
| Short contexts hide long-context neurons | Essay | Use long text. |

## Recommended methods for R1 and R3

**R1 step: what does each small-layer unit do?**
Use the essay's own recipe first. Look at both poles of each unit, on long text, on at least two text sets. Then run a causal test: zero the unit and measure the change in loss and in the logits. For scoring, use `delphi` detection and fuzzing with a Claude explainer through a custom client or OpenRouter. Skip simulation scoring. The essay doubts it. Eyeballing stays as a first look only.

**R1 step: is a small unit a parent of several original neurons?**
Do not regress neurons on neurons. Use two checks. First, compare the unit's output weights with the original neurons' output weights (cosine). Second, ablate the small unit and the matched original neurons and compare the effects on logits. If the same logits move, the unit is a merge. This stays in the raw neuron basis, which the essay supports. Then use the existing EleutherAI 64x SmolLM2-135M SAE on the same layer as a cross-check, not as ground truth. Expect the SAE to split things more than the neurons do.

**R1 step: how do the layers relate?**
Use decoder-cosine matching with the Hungarian algorithm, across K = 10, 48 and 192. Run two seeds, as the field's seed-stability paper advises. Add the absorption check from Chanin et al. before you call any unit a parent. A Matryoshka-style nested design is the field's tool for coarse-to-fine, and our K ladder is a cheap version of it.

**R3 step: do small models lose details or whole concepts?**
Start with simple, causal measures that need no training: compare per-token loss between small and big models, then group the tokens by family. For the feature view, use SAE features at a matched layer. Pair features by decoder cosine. A big-model feature with a close small-model partner but a weaker score means a lost detail. A big-model feature with no partner means a lost concept. Treat the unpaired ones as leads. Anthropic's update warns that exclusive features are often dense and polysemantic. Do not train crosscoders until the cheap version shows a signal.

**Before any of this:** time a 10-minute SAE or transcoder pilot on the Mac. No source gave a Mac number.

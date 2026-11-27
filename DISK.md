# Disk ledger

The disk is shared with another project and is nearly full. Every large file
in this project is listed here with the reason it exists. Remove anything that
is not listed.

## Rules

1. At most one 8-bit and one 4-bit nest build exist at a time. Build, score, then delete or replace.
2. Never keep two copies of the same weights in different formats "just in case".
3. Dataset downloads go to a cache that is deleted after the samples are written to `data/`.
4. Check `df -h ~` before any step that writes more than 5 GB. Stop if free space would drop below 15 GB.
5. Never delete files outside this folder, except HF cache entries this project created.

## Current large files

| Path | Size | Why it exists | Delete when |
|---|---|---|---|
| `~/Library/Application Support/Cale/models/gemma-3n-e4b/` | 5.5 GB | Cale's own copy, downloaded from Hugging Face by the install test on 2026-10-09 (byte-identical to `models/gemma-3n-e4b-4bit`, checked by SHA-256). It is a second copy, not a link. | `./cale/uninstall.sh`. To save 5.5 GB, reinstall with `CALE_MODEL_SRC=models/gemma-3n-e4b-4bit` (hard links). |
| `models/gemma-3n-e4b-4bit` | 5.5 GB | Phase 3 main model: Gemma 3n E4B, 4-bit MLX. E2B and the in-between sizes are nested inside it (MatFormer). | Replaced by a newer build |
| `apps/open-webui/.venv` | 1.8 GB | App 1 of the demo: Open WebUI 0.11.4 in its own venv (not a source clone) | When the demo is retired |
| `apps/aider/.venv` | 0.6 GB | App 2 of the demo: Aider 0.86.2 in its own venv | When the demo is retired |
| `research/tree/hf/` | 0.27 GB | SmolLM2-135M weights for the tree research ($0 track, Z1-Z6). Downloaded 2026-10-18. | When the tree research ends |
| `research/clean/hf/` | 4.7 GB | SmolLM2-360M, SmolLM2-1.7B, GPT-2, DistilGPT2 for R3 (what small models lose). Downloaded 2026-10-19. | After R3. The 1.7B (3.4 GB) goes first. |
| `research/clean/hf/pythia/` | 5.1 GB | Lineage track (2026-10-21): Pythia-160M final weights (0.36 GB) + `lineage-data/` (4.7 GB: MLP weights and dev-layer activations of 19 checkpoints). Checkpoint downloads were deleted after use. | After the lineage report is read. `lineage-data/` first. |
| `research/clean/curves/cache/map18/` | 1.6 GB | ResNet-18 map evidence (2026-11-03): pooled firing, peak positions and feeder tables for reading and sealed photos. | When the ResNet-18 map is finished or dropped. Rebuilt by `map_agents/resnet18/prepare_store.py` and `prepare_edges.py` in about 15 minutes. |
| `research/clean/neuron/results/` | 0.95 GB | Neuron coarsening step 1 (2026-11-13): pooled final stream of ResNet-18 for the 35,000 reading photos, intact, with children removed, and with coarse children (`absorb_cache.npz`, `rand_cache.npz`, `absorb2_cache.npz`). Not in git. | When step 1 is closed. `absorb.py`, `rand_drop.py` and `absorb2.py` rebuild them in about 10 minutes |
| `research/clean/neuron/results/gap/cache/` | 7 GB | N6 (2026-11-16): stage-3 input stream (fp16) and Base logits for the train, flipped train, held, calibration and test photos, for the training reference. Not in git. | When N6 ends. `neuron/gap/cache3.py` rebuilds it in about 20 minutes |
| `research/clean/neuron/results/gap/*.pt` | 1.3 GB | N6 (2026-11-16): trained references T, T-greedy, T-random, T-local, T-all, the build T_build (176 MB each) and fisher.pt (82 MB). Copies from the UVA cluster (/bigtemp has no backups). Not in git. | When N6 ends. `gap/train_ref.py` on the cluster rebuilds each in about 10 minutes |
| `research/clean/onecoarse/results/oc_ladder/n*/*.pt` | 1.3 GB | Vector-math ladder (2026-11-24): every build (C, C_held, W, L, R4, R, WC, R4C) at cuts 11/64/192/384, 30 files of 47 MB. Copies from the UVA cluster (/bigtemp has no backups), md5 checked 2026-11-25. Not in git. | When the vector-math track ends. `onecoarse/code/oc_ladder.py` on the cluster rebuilds them in about 5 GPU hours |
| `research/clean/neuron/results/coarse/r18/distill_whole/` and `.../r18/distill/` | 3.4GB (2026-11-26) | Distill track (2026-11-25/26): 12 whole-ResNet-18 students at 100/95/50/5% of neurons, each with 10 checkpoints, plus T0 (block 0 at 128); wave 3 added 13 more (LOG section 10). Copies from the UVA cluster (/bigtemp has no backups), md5 checked. Not in git. | When the distill track ends. `distill/train_whole.py` on the cluster rebuilds each in 20-130 GPU minutes |
| `research/clean/onecoarse/results/oc_ladder/n*/w_*.npz` | 1.0 GB | Numpy exports of the 30 ladder builds (stage 4 + head), read by the `oc_vs_*.py` scripts without torch (2026-11-25). A derived working copy of the `.pt` files above, not a backup. Not in git. | When VECMATH section 8 work closes. `onecoarse/code/oc_vs_extract.py` rebuilds them in about 1 minute |
| `research/clean/neuron/results/read/` | 0.25 GB | N10 (2026-11-19): spare bird photos' stage-3 stream (`cache/spare_x3.npy`, 0.18 GB, 910 photos), B and T logits on held and test, pilot captures (a few neurons, pooled values). Not in git. | When N10 ends. `read/rcommon.py` rebuilds the spare cache in about a minute |
| `research/clean/vecread/data/broden/` | 1.3 GB | Vector reader (2026-11-23): Broden broden1_224 (Network Dissection data set: photos + region labels), the anchors of the concept dictionary. Internet Archive copy of the official file (the MIT server timed out). Not in git. | When the vector reader is retired. `vecread/get_broden.sh` and `data/pget.sh` re-download it in about 5 minutes |
| `research/clean/vecread/results/` | 4.8 GB | Vector reader (2026-11-23): concept sums for 4 spaces (`dict_build.npz`) and the held-out maps of two sets of 8,000 Broden photos (`held_*.npy`, `held_*_h2.npy` = the confirmation set, fp16). Not in git. | When the vector reader is retired. `vecread/build_dict.py` rebuilds it in about 4 minutes |
| `research/clean/neuron/results/lens/*.pt` | 0.36 GB | N7 (2026-11-19): two saved no-training builds, base-write + spread (`base_otw_saved.pt`) and lens + spread (`lens_ot_saved.pt`), 180 MB each. Used by `lens/gates.py` and `lens/split.py`. Not in git. | When N7 ends. `lens/diag2.py` rebuilds each in about 5 minutes |
| `research/clean/curves/map_agents/resnet18/sheets/` | 0.13 GB | Crop sheets the agents read. Not in git. | Any time. `ev.py sheet` rebuilds one in seconds. |
| `CrakBox/runs/` | 0.3 GB | CrakBox run records, judge packets and keys (moved from `research/mapping/runs/` on 2026-11-04). Not in git. | Packets: when their round is scored. Records: when the ResNet-18 map is finished or dropped. |
| `CrakBox/.venv` | 0.25 GB | CrakBox's Python for the spaCy tagger: spaCy 3.8, en_core_web_sm 3.8.0, numpy, pillow, and since 2026-11-07 scipy and pandas (testers and the orchestrator asked for them). No torch. Not in git. | When the tagger's map ends. `uv` rebuilds it in about a minute |
| `CrakBox/runs/cache/` | 13 MB now (8 MB per ResNet-18 neuron) | The bench's value store (2026-11-08): each neuron's values on the reading sample, so a bench call takes 1 s instead of 35 s. Not in git. A full `s4b1` layer would be about 4 GB, so fill only the neurons a run reads | Any time. The bench rebuilds a neuron's file on its next call. `CRAKBOX_VALUE_STORE=off` turns it off |
| `CrakBox/runs/cache/` (2026-11-08, later) | values 163 MB, tree 1.1 GB | The store key changed on 2026-11-08 (`mapper/store.py`, fuller fingerprint). Files with an old key are no longer read: in `values/`, the 26 files not keyed `dfe8cb8741749b04`; in `tree/`, every `sample.*.npz` except `sample.564f8e559fb9d17e.npz` (about 0.9 GB). The new row store (`NEURON.contrib_REGIME.KEY.npz`) grows by about 4 kB per neuron and stored row | The old-key files: any time |
| `CrakBox/runs/cache/resnet18/tree/` | 0.9 GB (3 files of 294 MB: layers `s4b1`, `s4b1i`, `s4b0`) | The tree's sample store (2026-11-08): every neuron of a layer on the tree sample, so a `tree try` or `tree predict` reads it in seconds instead of running the network for minutes. Not in git | Any time. The next try on that layer rebuilds its file (27 s to 13 min). `CRAKBOX_VALUE_STORE=off` turns it off |
| `CrakBox/runs/tagger.v1/` | 0.54 GB | The tagger's peak cache and the sealed text copy (2026-11-04). Not in git. | When the tagger's map is finished or dropped. `networks/tagger/build.py` rebuilds the cache in about a minute |
| `CrakBox/.venv-hf` | 0.9 GB | CrakBox's Python for language models (2026-11-09): torch 2.15, transformers 5.22, safetensors, accelerate. Not in git | When the mapper bake-off ends. `uv` rebuilds it |
| `CrakBox/runs/hf/` | 3.2 GB now; about 139 GB when all five are in | Bake-off weights in bf16 (2026-11-09, plan `okay-i-ve-been-thinking-harmonic-snowflake.md`): LFM3-1.7B 3.2 GB, Granite 4.2 8B about 16 GB, then Ministral 3 8B 20.8 GB, Qwen3.5-9B about 19 GB, Flint 79.6 GB last. Each model maps its own neurons, so testers' tries need them through the bake-off. Not in git | When the bake-off picks a model: delete all but the chosen one |
| `CrakBox/runs/cache/hf/` | about 1 GB per stored layer of a 2,048-wide model, 1 GB per 4,096-wide (bf16 bits) | The stream at one layer's MLP for the bench's 120,000-example sample (`networks/hf_mlp/base.py`, 2026-11-09). Any neuron of a stored layer then reads in seconds. Not in git | Any time. The next bench call on that layer rebuilds it (about 90 s on LFM3) |
| `~/BlinkFix-models/d1-spike/` (outside this repo, owned by BlinkFix) | 11 GB | CrakBox's decision model since 2026-11-05: d1-2B MLX weights in `hf/` and its Python in `venv/`, used by `CrakBox/mapper/decide.py`. The BlinkFix disk guard lists it as a stale candidate, so its `--apply` would delete it | Not while CrakBox scripts use it. Ask Shonu before any removal. Of the 9.2 GB in `hf/`, only `models--LiquidAI--d1-2B-MLX-8bit` is needed here |
| `research/clean/hf/vision-ck/`, `research/clean/hf/cifar100/` | ~0.2 GB | Vision lab (lineage round 3, 2026-10-21): CIFAR-100 arrays and 4 small trained nets with checkpoints. | When the vision lab ends |
| `research/clean/hf/convnext/atto`, `research/clean/hf/convnext/nano` | 77 MB | Real small ConvNeXts (timm atto.d2, nano.d1h) as the second reference for the map track (2026-10-21). | When the map track ends |
| `research/clean/hf/imagenet/n7cache/` | 2.6 GB | Closed-form nudge caches (block-input tensors for 12 families, teacher logits, trained neurons), 2026-10-22. | When the nudge track ends |
| `research/clean/curves/.venv`, `research/clean/curves/cache/`, `~/.cache/torch/hub/checkpoints/inception5h.pth` | 1.2 GB | Curves track (2026-10-22): PyTorch + Lucent venv and lucid InceptionV1 weights (27 MB) for the essay's 3b crush. | When the curves track ends |
| `research/clean/curves/cache/r18/` | 2.4 GB | Curves track zoom on ResNet-18 (2026-10-28, plan okay-i-really-enjoyed-humble-allen.md): last-block pre-activations z (float16) and logits of 35000 train + 10000 sealed photos (`zoom/rz_cache.py`), pooled h, head. | When the ResNet-18 zoom is written up |
| `research/clean/curves/cache/map/` | 0.75 GB | Curves track map agents pilot (2026-11-01): pooled last-layer values of ResNet-18 and ResNet-50 on 20000 validation crops, the small CIFAR-100 net's features; `research/clean/curves/map_agents/sheets/` 52 MB of contact sheets. Keep for session 3 |
| `~/.cache/torch/hub/checkpoints/resnet{18,34,50,101,152}-*.pth` | 0.66 GB | Curves track Part C (2026-10-28): torchvision ResNet ladder (IMAGENET1K_V1) for H-weak and H-scale (`pc_weak.py`, `pc_scale.py`). | When the curves track ends |
| `research/tree/data/` | 12 MB | wikitext-2 train/test text (calibration and held-out). The Hugging Face dataset cache was deleted after saving. | When the tree research ends |
| `results/g3n-*.json`, `results/bench/g3n-*.json` | <1 MB | Phase 3 measurements | Keep |
| `results/clean-a0.json`, `clean-a1.json`, `clean-a2.json`, `clean-b1*.json`, `clean-q1.json`-`clean-q8.json`, `clean-w1.json`-`clean-w5.json`, `clean-v1.json`-`clean-v3.json`, `clean-s1.json`, `clean-g-l16.json`, `clean-g-l22.json`, `clean-t1.json`, `clean-t1-150s.json`, `clean-g75.json`, `clean-es.json` | <1 MB | RCF scale + channels tests (2026-10-20), CHILDREN-LOG "RCF SCALE + CHANNELS" | Keep |
| `data/gemma3n-slicing` | 0.04 MB | Google's official size table (source of `nest/gemma3n_levels.json`) | Never needed again after gate A |

## Removed

| Path | Size | Date | Reason |
|---|---|---|---|
| `models/g12-nest` | 7.3 GB | 2026-10-09 | Old 12B one-file build. Replaced by Gemma 3n. Shonu agreed. |
| `models/qg4` | 7.0 GB | 2026-10-09 | Same weights as `g12-nest`. Shonu agreed. |
| `models/gemma-4-e4b-it-4bit`, `models/gemma-4-e2b-it-4bit` | 8.1 GB | 2026-10-09 | Gemma 4 baselines from phase 1. Shonu agreed. |
| `models/qat-assistant-4bit` | 0.27 GB | 2026-10-09 | Drafter for the old 12B. |
| `demo/inbox` | 29 MB | 2026-10-09 | Old demo app, needed the 12B Cale. Replaced by Open WebUI and Aider. Not in git. |
| `models/tmp-gemma-3n-e2b-4bit` | 4.2 GB | 2026-10-09 | Gate A done: our E2B slice matches the official E2B on real text (loss 2.200 vs 2.200) |
| `research/clean/hf/imagenet/l1cache` | 2.5 GB | 2026-11-15 | Superseded by n7cache. Shonu agreed. |
| `models/nest8-id` | 12 GB | 2026-10-01 | Original-order copy. Reorder checked exact, so it was no longer needed. |
| `models/gemma-4-12B-it-4bit` | 6.3 GB | 2026-10-01 | Same as our own 4-bit build at full width |
| `data/cache` + HF dataset entries | 0.14 GB | 2026-10-01 | Samples already saved in `data/` |
| `models/nest8` | 12 GB | 2026-10-01 | 8-bit tests done; 4-bit QAT is the main test model now |
| `models/nestq4e8` | 6.8 GB | 2026-10-01 | QAT standard-4-bit test done; exact-grid build is better |
| `models/qg4-id` | 7.0 GB | 2026-10-01 | Ceiling confirmed equal to the nested build at full width |
| `models/nest4e8` | 6.8 GB | 2026-10-01 | Superseded by QAT builds |
| `models/nest4` | 6.3 GB | 2026-10-01 | `nest4e8` scored almost the same (answer drift 0.683 vs 0.704) |
| `models/gemma-4-12B-it-qat-bf16` | 22 GB | 2026-10-01 | `qg4` holds the QAT weights; not needed by the ladder (plan decision) |
| `models/q4e8` | 6.8 GB | 2026-10-01 | Encoder comparison done; exact-grid `qg4` is the 4-bit build |
| `models/gemma-4-12B-it-bf16` | 22 GB | 2026-10-01 | Precision ladder dropped (ocean ≈ QAT 4-bit); Shonu agreed. Re-download if needed. |

## Phase 5 additions (2026-10-11)

| Path | Size | Reason |
|---|---|---|
| `llama/dist/` | 18 MB | Built Windows and Linux libraries. Needed by the installer. Keep. |
| `/private/tmp/claude-501/xplat/llama.cpp` | 0.2 GB | llama.cpp source clone used to make the patch. Delete when the patch is final. |
| HP laptop `C:\cale\`, `~/cale-test` (WSL), `~/cale-apps` (WSL) | about 40 GB | Models (two copies per OS), builds, Open WebUI and Aider environments. Test laptop only. |

## Phase 6 stage 2b additions (2026-10-13)

| Path | Size | Reason |
|---|---|---|
| HP `C:\cale\xplat\gfx_win_101.2145.exe` | 0.3 GB | Intel driver installer, already installed. Delete when no longer needed. |
| `results/gpu/` | small | Raw GPU benchmark logs and scripts. Keep. |

## Phase 6 stages 3 to 8 additions (2026-10-13)

| Path | Size | Reason |
|---|---|---|
| `models/gguf/gemma-3n-E4B-it-Q4_0.cale.gguf` | 4.4 GB | Mac copy of the llama engine file, for the llama tests. Keep while testing. |
| `/private/tmp/claude-501/llvmmingw` | 0.8 GB | Windows arm64 cross compiler. Delete when done. |
| `/private/tmp/claude-501/lbuild_mac`, `lbuild_winarm` | about 1 GB | Mac and Windows arm64 builds. Delete when done. |
| Docker volume `cale-arm64-build` (colima) | about 2 GB | Linux arm64 build. Remove with `docker volume rm cale-arm64-build`, then `colima stop`. |

## Cleaned at the end of the 2026-10-13 session

Removed: `/private/tmp/claude-501/llvmmingw`, `/private/tmp/claude-501/lbuild_winarm`, docker volume `cale-arm64-build`.
Left alone on purpose: colima (`~/.colima`, 52 GB on disk) holds containers of another project (Supabase `Mapta`), so Cale does not stop or prune it.

## Cleaned at the end of the later 2026-10-13 session

Removed: docker volume `cale-arm64-build`, scratchpad packages, the HP test homes (`C:\cale\insttest*`, `tunehome*`, WSL `~/insttest4`, `~/hf-gemma3n` 15 GB). Mac free space: about 5 GB (colima, 52 GB, belongs to another project).

## Cleaned at the end of the 2026-10-14 session (prompt reading)

Removed: the Mac llama build folders and the llama.cpp clone in the scratchpad, `~/.cache/cale-arm64-out`, docker volume `cale-arm64-build`; on the HP: `C:\cale\insttest15`, `insttest21`, `zoo15`, `zoo21`, the `flat*` folders, and in WSL `~/insttest15`, `~/zoo15`, `~/zoohome*`, `~/flat*`.
Left on the HP on purpose (about 15 GB, HP has 235 GB free): `C:\cale\models\raw.gguf` and `gemma-3n-E4B-it-Q4_0.cale.gguf`, the build folders `C:\cale\lbuild2` (b_noomp8, b_dl14), WSL `~/lbuild9`, `~/lbuild_dl`. They save a download and a build next time.

## Consumer demo session (2026-10-14/15)

| Path | Size | Reason |
|---|---|---|
| `dist/` (git-ignored) | about 0.6 GB | the one-download installers (`Cale-mac.dmg`, `.zip`, `Cale-linux-x86_64.run`, `win/`) and `dist/cache/` (uv and Node downloads). Rebuild with `scripts/build_installer.py`. Delete when not needed. |
| `/private/tmp/claude-501/cale-act-home` | tiny | dev server home for the demo tests (its model is a hard link). |
| HP `C:\cale\apptest`, `C:\cale\winpack2`, `winpack3`, WSL `~/cale-apptest` | about 10 GB | throwaway installs made from the one-download files (real model download, Windows and Linux). Delete when the demo scorecard is done. |

Removed: the Mac test install `cale-app-home` (venv only; its model was hard-linked), `/private/tmp/claude-501/standalone`.

## Quant quality test (2026-10-15)

Downloaded one at a time into the scratchpad, measured, deleted: Q8_0 (7.4 GB, twice), Q4_0 (4.4 GB, twice), Q4_K_M (about 5 GB, twice). Free space never went below 17 GB. Nothing of these is kept. Removed: the llama.cpp b11558 clone and build in the scratchpad (0.27 GB, scratchpad only).

## Chat window bundle (2026-10-17)

Kept in `dist/chat` (not in the repo, git-ignored; 0.9 GB, output of `scripts/build_chat.py`, moved here from the scratchpad on 2026-10-17 so the Mac installer can carry it; can be rebuilt in about 1 minute with a warm yarn cache). Kept in the scratchpad (not in the repo): `scratchpad/anythingllm-src` (source only, about 0.1 GB), `scratchpad/node22`, `scratchpad/yarnbox` (small). Removed: all node_modules of the AnythingLLM source tree (about 2 GB), the Chromium that puppeteer fetched into `~/.cache/puppeteer` (0.26 GB), the build work folder, the test data folder. The yarn download cache (`~/Library/Caches/Yarn`) was not touched.

## Windows chat bundle (2026-10-17)

Kept: `dist/chat-win` (git-ignored, 0.87 GB, made on the Mac with `scripts/build_chat.py --target win-x64 --out dist/chat-win`; rebuild takes about 1 minute). The Windows installer carries it as `chat/` (about 0.3 GB compressed). Scratchpad copies for the HP test (`scratchpad/serve`: chat-win.zip 0.31 GB, node.exe 0.08 GB) are deleted when the test ends. On the HP, `%TEMP%\cw2` (about 1.2 GB: zip, unpacked payload, test home) is deleted when the test ends. HP free space before: 204 GB.

## 2026-10-18 (sameness work)

`dist/chat` (0.8 GB) and `dist/chat-win` (0.85 GB) were rebuilt with the Quicker/Smarter slider and replace the old ones (no second copy kept). Scratchpad: `anythingllm-src` (0.1 GB source, kept for rebuilds), `node22` (small). Test homes and chat data folders from this session were deleted. HP: `C:\cale\xb7` (the chat bundle unpacked from the old installer, about 0.9 GB), `C:\cale\code_*`, `C:\cale\insth_*`, `C:\cale\th*`, `C:\cale\abh_*`, `C:\cale\ord_*` are test folders: delete when the HP work ends.

`dist/chat` (0.82 GB, 28,987 files) and `dist/chat-win` (0.87 GB, 26,473 files) were rebuilt again later on 2026-10-18 with the server and the document reader packed into one file each, and the files nothing can reach removed (`scripts/pack_chat.mjs`, `scripts/chat_prune.py`, the default of `scripts/build_chat.py`). Before: about 61,000 files each. They replace the earlier ones (no second copy). `dist/Cale-mac.dmg` was rebuilt (443 MB, was 559 MB) and `dist/Cale-mac.zip` (315 MB, was 338 MB). Scratchpad: `anythingllm-src` (0.1 GB) and `node22` and `esb` (small) stay for rebuilds.


## 2026-10-18 (cache clean)

Deleted about 18 GB of caches: `~/Library/Caches/ms-playwright-mcp` (6.3 GB), `~/Library/Caches/Yarn` (4.4 GB, the next chat build downloads it again), `~/.cache/huggingface/hub/blobs` (3.1 GB), the parakeet model folder in the same hub (0.7 GB), and three app-update caches (VSCode.ShipIt, todesktop ShipIt, com.openai.codex, about 4 GB). Nothing else was touched.


## 2026-10-19 (raw-neuron interpretability tools pilot)

All under `research/clean/tools/` (git-ignored `.venv`). The "system torch 2.7.1" did not exist on this Mac, so the one virtualenv holds the only torch copy (2.14.1, 0.58 GB); it must stay until the pilot ends.
- `research/clean/tools/.venv`: 0.81 GB total (torch 0.58 GB, wandb 0.06 GB pulled in by transformer-lens, nnsight, transformer-lens, neuronscope-cli, eai-delphi installed with `--no-deps`, small helper packages). Created with `--system-site-packages`.
- `research/clean/tools/src/delphi` 1.3 MB, `src/griffin` 50 MB (shallow clone, ships committed .pyc files: never import them), `src/moefication` 0.2 MB. Read-only references.
- `research/clean/tools/pilots`: small scripts, JSON results and a folder of symlinks to the cached gpt2 (no copy).
- pip cache `~/Library/Caches/pip` grew to 0.16 GB (not removable by name; `pip cache purge` clears it).
- No model weights were downloaded. SAE tools (sparsify, SAELens, circuit-tracer) were not installed.
Delete `research/clean/tools/.venv` and `src/griffin` when the pilot ends.

## Vision lab (2026-10-21)
- `research/clean/hf/imagenet/`: 9.0 GB. `val_224.npy` (all 50k ImageNet val images, 224 px, uint8, ~7.5 GB), the
  two source parquet shards (~0.9 GB, removable: `val_224.npy` is built), labels, cached activations
  `acts_tiny_ana20k.npz` and `feat_tiny_ana20k.npy` (small).
- `research/clean/hf/convnext/`: 0.1 GB (timm convnext_tiny.fb_in1k weights).
Delete the parquet shards first if space is short. Delete the whole folder when the vision track ends.
- Map track (2026-10-21), in `research/clean/hf/imagenet/`: `m3_matrices.npz` 2.4 GB (containment of every neuron
  in every earlier neuron; the map needs it), `e1_edges.npz` 1.2 GB (direct-path edges), `m3_null.npz` 0.3 GB (null
  readouts). Deleted: `m1_means_15-16-17.npz` (0.6 GB, failed grouping method). Also `research/clean/map/deep/`
  0.29 GB of neuron views and `research/clean/map/tests/` 0.12 GB. Keep while the map is built; delete the npz files
  once `map/network_map_v2.json` is final.
| `research/clean/curves/cache/e1_self_256.npy` | 39 MB | 256 self-made class-inversion inputs (DATA-FREE check E, 2026-10-23). Regenerate with `e1_selfmade.py 256 256` (15 min). Keep while the data-free track is open. |
| `research/clean/curves/lean/coarse/.lake` | 7.2 GB | Lean 4 + Mathlib v4.35 build cache for the NEURON-MATH proofs (2026-10-24). Rebuild: `lake exe cache get` (~10 min). Delete when the proofs are final or disk is short. |
| `research/clean/curves/cache/x3c_stats_35k.pt` | 710 MB | Coupled-formula statistic m_3s on all 35000 train images (NEURON-MATH 14.12, 2026-10-24). Regenerate with `x3c_population.py` (~25 min). Delete when the exactness track moves past k = 0. (The 2000-image version was deleted: 1.5 min to rebuild with `x3_coupled.py`.) |
| `research/clean/curves/cache/c45_readerq_368*.npy` | 54 MB each (256-step set; 512-step set for K4) | Reader questions: one network-made picture per family reader (NEURON-MATH Part VIII, 2026-10-24). Regenerate with `c45_readerq.py 256` (17 min). Keep while Part VIII is open. Small c40-c48 caches (< 10 MB each) sit next to them. |

2026-11-03: map round 2 adds research/clean/curves/cache/map18 2.0G (total), map_agents/resnet18/sheets2 231M, map_agents/resnet18/review2 214M. The images regenerate from ev2.py and build_crop_review.py and are git-ignored from now on. About 445 MB of them remain in local git history (49 commits ahead of origin, not pushed). 32 GB free.

## Cleaned on 2026-11-09 (Shonu asked)

Removed: both colima machines with their data (`default` 42 GB, `canvasfresh` 26 GB, this includes the Mapta Supabase containers), the Parallels Windows 11 VM (32 GB), and `~/DunnitOS/.runtime/model-contract-study` (30 GB of downloaded GGUF models). Mac free space went from 50 GB to 185 GB.

## Added on 2026-11-16 (CrakBox feature-extraction session)

| Path | Size | Why | Delete when |
|---|---|---|---|
| `CrakBox/runs/captures/fx_p28_w0`, `_w1`, `_w2`, `fx_p10_w0`, `_w1`, `_w2` | 5.3 GB (0.9 GB each) | Frozen captures of the 9B's own windows of problems 28 and 10: R5 of the feature ruler and M1/M3 | The ruler moves to new data; they can be recaptured in about 3 minutes from `checks/fx_capture.py windows` |
| `CrakBox/runs/captures/fx_cache` | 0.3 GB | Cached residual reads and log P(" Wait") for `checks/fx_ruler.py` | Any time; rebuilds in about 5 minutes |
| `CrakBox/runs/captures/fx_m_*` (118 folders) | about 7 GB after `checks/fx_shrink.py` | The 9B's own windows around 119 blind-labelled match lines: R2 of the feature ruler, M6 | The source-of-reference work ends; recapture with `checks/fx_capture.py matches` (about 35 min) |

Free after the session: 26 GB.

## Added on 2026-11-25 (CLIP-KD known answer)

| Path | Size | Why | Delete when |
|---|---|---|---|
| `research/clean/distill/clipkd/weights/` | 1.7 GB | CLIP-KD teacher ViT-B/16 + two ViT-T/16 students (known answer for the calculation, `distill/clipkd/README.md`) | The calculation work ends; re-download from github.com/winycg/CLIP-KD release CLIP-KDv0.1 (md5s in the README) |
| `research/clean/distill/clipkd/out/` | 0.1 GB | Saved embeddings and class logits, read by `distill/CALC.md` and `eval_fast.py` | Same; rebuild with `score_clip.py` (about 10 min) |
| `research/clean/distill/clipkd/pylib/` | 7 MB | Private open_clip 4.2.2 + ftfy install | Same |

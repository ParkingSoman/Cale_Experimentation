---
name: project-map-track
description: "CURRENT focus (2026-10-21): map ConvNeXt-Tiny coarse-to-fine; Shonu's depth-chain idea and vision-track history; state file is research/clean/MAP-STATE.md"
metadata:
  node_type: memory
  type: project
  originSessionId: 2ba42140-21a0-426d-8405-942d9986529c
  modified: 2026-10-21T17:59:43.081Z
---

**2026-10-22 D-series (derive track) is the current focus:** read `research/clean/DERIVE-REPORT-2026-10-22.md` first. Coarse neuron = consensus of a co-firing group (siblings + depth), fires as often as its parts; toy derived build works for coarse classes; in ConvNeXt dropping a whole block (.104) beats thinning two blocks + training (.224); derived linear repairs (stream MSE, conditional-expectation reads, mean bias) hurt a clean drop. D6: SELF-HEALING LAW (LayerNorm replaces the lost gate on each neuron's territory, slope ~1.07; block 16 adds ~one copy of what readers see, lambda 1.07, 93% proportional); best training reference .0836 (data-limited, 90k images); fixes fail via size restoration, variance collapse, broken joint pattern, bias confound.

**Read first (map):** `/Users/shonusengupta/Cale_Experimentation/research/clean/MAP-STATE.md` (current state, measured results, tools, next steps). This memory holds only the idea and the track history. Logs: `research/clean/VISION-LOG.md`, `CHILDREN-LOG.md` ("TREE FOLD"). Goal and "do not stop" rule: [[feedback-research-method]].

## Shonu's depth-chain idea (clarified 2026-10-21 with essay figures)
Figures he means by "neuron gating":
1. Gate in front: pose neuron groups cars + dogs; a gate neuron (+ cars, - dogs) routes cars to car detectors, which split further (taxi vs ambulance on each other's negative poles).
2. Large ConvNeXt, one sports car, blocks 14-18: each block's top-contributing neuron draws a finer split (cars vs buses -> rectangular vs snakes/dogs -> convertibles vs minibuses -> vs vans -> vs station wagons).
3. Smallest model's LAST layer: ground vehicles vs boats/planes: "still differentiating, but more coarsely, and it ends early."

Meaning: the tree runs along DEPTH inside one model. A neuron's parent is an earlier neuron on its attribution path. A small model = the large model's chain of distinctions stopped at a coarser node. Coarsening = cut each chain's fine tail and let the earlier gate carry the output. The lineage track (checkpoints over TIME) looked in the wrong place. This is where his "any neuron coarsens to its parent" goal comes from. RCF is a one-step, unstructured version ([[project-research-history]]).

## Results before the map track
- Test A (CHILDREN-LOG "TREE FOLD", results clean-tf1/tf-diag/tf-ctx.json): one fixed parent per child fails (2-12% of RCF's gain). Upstream support is spread: top-10 parents reach ~20-27% of the dense fit; a child's top contributor changes token to token (modal parent top on ~10% of its strongest tokens). The essay chain is a per-input path; its linear version is RCF's dense fold.
- Vision-first follow-up (VISION-LOG): ConvNeXt-Tiny (timm, MLX port in vision2/, ImageNet val 50k, per-class split) reproduces the chain visually (block 17: chrome car parts vs dogs). One-parent tree fold FAILS in vision too (KL, attribution parents); dense fold works (~1/3 at block 14). SmolLM2-1.7B matches 135M (tf_big.py): not a weak-model artifact. Last-stage partial cuts are worse than dropping the whole block; single blocks drop for 1-4 pts -> depth dial ("ends early") is a candidate, plus a vision depth-readout test.
- R8: two kinds of chain. "Evidence chain" (top contributor on one image, essay's figure method) is NOT nested (7/32 links; follows color/background). "Territory chain" (each neuron's classes contain the next's) exists far above shuffled null (62-69% of blocks 15-17 neurons have a nested parent one block back), coarser going back (cars 13 cls <- 15 <- road vehicles 25 <- all vehicles 49). Wiring between territory links is mixed (one link inhibits). Tree fold R7 used evidence parents; territory parents untested.
- R9-R11: label-free recipe = average activations in k-means groups (k=1000) of the model's final feature, then territory/containment; reproduces class result. LM analog: k-means on final hidden states of tokens. ALL neurons: late stage ~all nested child or broad root (orphans 1-8% vs null 16-47%); early blocks merge two parents' territories (29-41% vs null 6%). Nested parents are bimodal in wiring, half feed, half inhibit, almost none neutral. grad x act is fooled by LayerNorm saturation; use integrated gradients.
- M0-M3 (whole-network label-free map at image positions, natural on-region z>0, parents in any earlier block, either pole; dog specialists sit in the gate's NEGATIVE pole): 60% of blocks 1-17 neurons get a parent (null 18%), but parents weak (median lift ~2, 25% lift >= 3). Groups (k-means) fail at position level. Next ruler: families from connections (essay: edge motifs beat activation groupings).
- 2026-10-21 evening (Shonu away, "small = parent + child", dropdown/composable tree): real small ConvNeXts (timm atto, nano in research/clean/hf/convnext/) are the second reference. They keep the map's trunk neurons and lack its leaves. One-neuron vote merge closed (E1-E3). Tree is not a better removal order than energy within a block; brief retraining erases order differences (A5, A6b). Details: MAP-STATE.md section "Small = parent + child", VISION-LOG E1-E3, A1-A6.
- Later in MAP-STATE.md: S2/S3 hand study (17 neurons), edit fingerprints, link types, tools in `research/clean/vision2/`, next steps.

Related: [[feedback-research-essay-rules]], [[project-research-history]].

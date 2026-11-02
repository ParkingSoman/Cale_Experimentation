# ZOOM review (independent, adversarial) — config v11, 2026-10-19

Scope: `zoom1.py`, `zoomfinal.py`, helpers, `r1_weights/zoom_v11.npz`, `results/clean-zoom1.json`, `CHILDREN-LOG.md` (ZOOM).
No files were edited except this one. No model was loaded. Checks were numpy/MLX on the saved file and on random inputs.

## Checks that passed (no issue)

| Check | Result |
|---|---|
| Test data in fitting | None. Parents, families, order, resid table, dD, eD and the cut all use wikitext train blocks 0-31 only (`zoom1.py:171-182`). |
| Blocks 60-99 vs 0-19 | Disjoint. Both come from the same deterministic `token_blocks(tok, "test", 100)`; blocks are contiguous 513-token slices. |
| Code train/test split | Clean. 105 of 17,149 long lines (0.6%) are shared, all boilerplate (licence lines, `with open(...)`). Code text is not used by zoom1 at all. |
| zoom1 vs zoomfinal layer | Same function. In float64 the two formulas agree to 2e-8 relative at N = 48, 96, 500 and more. The 5e-4 seen in float32 is MLX kernel noise (MLX float32 vs float64 differs by 1.7e-3 on the same weights). |
| Neuron count | Same in both. `tot` (zoom1) equals `neurons` (zoomfinal) at all 16 settings tried. Parents with alpha > 0 are counted in both. |
| Resid table | Length = family size + 1 for all 48 families. Last entry 0. No zero before the end. Monotone. Within-family return order equals importance order, so indexing by `k_on` is right. |
| `tot` monotone in M | Yes (checked for all M). The binary search in `zoom1.py:264` is valid. |
| Full setting | Weights bit-identical to the original (gate, up, down). beta = 0.0 exactly. All alpha = 0. |

## Issues

| Issue | Severity | Evidence (file:line or check output) | Suggested fix |
|---|---|---|---|
| "No refit" is misleading. dD and eD are a learned per-size output refit stored in the file. | major | Measured on v11: the folded correction `beta*dD + alpha*eD` is 42% of the on-neurons' output-column norm at N=64, 45% at 96, 40% at 192, 32% at 384, 19% at 768. `zoom1.py:58-61`, `zoomfinal.py:63-66`. | Say "no per-size refit step; a learned two-scalar output schedule is stored". Do not say "no refit" without that. |
| The "neurons" count hides storage. dD and eD are two extra 576 x 1536 matrices. | major | Saved file: original 2.65M params + parents 83k + dD/eD 1.77M = 4.50M, about 1.7x the original layer. Five separate per-size fitted layers (48..768) cost 2.57M. | Report parameters stored next to neurons. Compare against "store one fitted layer per size". |
| Runtime cost may be above the neuron count. | minor | Each dial change needs `D_eff = D + beta*dD + alpha*eD` for all on columns. Without that rebuild, each token runs three down-projections. | State the rebuild cost (576 x M per change) or count the extra matmuls. |
| Criterion 1 strict already fails for v11 on the dev set. | major | `results/clean-zoom1.json` attempt 12: `1_steady: False`, `max_rise 6.6e-4`. Dev sweep: 48 -> 64 rises 0.1162 -> 0.1166. | Headline must say strict fails; only the practical 1e-3 passes. The 1e-3 rule was added after v7 (`CHILDREN-LOG.md:144`). |
| The final steady sweep is coarser than dev. A coarser grid can only hide rises. | major | `zoomfinal.py:90,129`: step 32. Dev used step 16 and found rises a 64-step sweep missed (`CHILDREN-LOG.md:158`). The claim is "steps of one neuron"; no run checks that. | Use step 16 or less in zoomfinal. Say the claim is checked on a grid only. |
| No error bars on criterion 2. Margins at large sizes are near the noise. | major | Dev v11 at 768: dial 0.0460 vs cut 0.0500 (0.004). Log notes build-to-build spread ~0.002 (`CHILDREN-LOG.md:159`). `zoomfinal.py:154` uses a strict `<` on one draw. `r1.py:60` `boot_diff` exists but is unused. | Report a paired per-block bootstrap SE for dial - cut at each size. Build v11 with 2-3 seeds and report the spread. |
| Blocks 60-99 were used before, in earlier rounds that chose methods the dial uses. | minor | `r2round3.py:98`, `r2round4.py:126`, `r2round5.py:115` evaluate on all 100 test blocks (and on `code_test.txt`). Those rounds picked fit_tail, the cut init and LS refit. | Say "unused by the zoom loop", not "never used". Fitting is still clean. |
| Code split origin is not recorded. | minor | No script in the repo builds `data/code_train.txt` / `code_test.txt` (grep found only readers). | Add the build script or a note listing the stdlib files in each. |
| Criterion 3 is true by construction. It is not evidence. | minor | Full-setting weights are bit-identical to the original; beta = 0 exactly; alpha = 0. | Report it as a design property, not as a passed test. |
| zoomfinal tests size 1536 in criteria 2 and 4. zoom1 does not. The claim says 48..768. | minor | `zoomfinal.py:128` has no `< n` filter (zoom1 has it, `zoom1.py:294`). At 1536 the dial is exactly the original, so criterion 2 is trivial. Criterion 4 can fail by noise if `fitted` lands slightly below 0 (0 <= 1.25*fitted + 1e-4 fails for fitted < -8e-5). Also spends 330 s on a useless fit. | Filter test sizes to `< n` for criteria 2 and 4. |
| zoomfinal sweep starts at N=32, below K=48. | minor | `zoomfinal.py:129`. `DialLayer(N=32)` never meets `tot <= N`, so M = 0 and `neurons` = 48. Silent budget overrun and a duplicate point. | Start the sweep at K, or assert `neurons <= N`. |
| Cut baseline is weak. It gets no loss-aware training; the dial gets 330 s. | minor | `zoom1.py:201-203`, `zoomfinal.py:120-122`: importance + least-squares output fit only. | Frame criterion 2 as a floor. Criterion 4 is the real test. |
| Per-size fitted reference: steps not recorded; the Mac is shared during the final run. | major | `zoomfinal.py:142` discards the step count (`Pf, _ = fit_tail(...)`). Budget is wall-clock (`r2round5.py:86`). The dial's steps were 216 + 1849. Other jobs on the Mac cut the reference's steps. Equal time already favours the reference (an N-neuron MLP is cheaper per step than the 1584-neuron ZoomMLP with two extra matmuls). | Record reference steps. Better: give a fixed step budget. Add one long reference (about 10x time) to show whether the 25% bar is against an under-trained layer. |
| Position 0 is left out of the cut's fit only. | minor (possible, not measured) | `zoom1.py:176`, `zoomfinal.py:115`: `Xc` drops position 0 for the LS refit, importance, families, resid. The dial's nested fit and the fitted reference run the tail with position 0 in the forward pass, so later tokens that attend to it are trained for. The cut never sees position-0 inputs. | After the final run: rerun the cut with position 0 in the LS fit and compare. |
| Builds are not reproducible. | minor | Time-based loops (`zoom1.py:113`, `r2round5.py:86`). Steps vary 195-248 (fit) and 1849-1912 (nested) across attempts. This is the source of the ~0.002 spread. | Use fixed step counts. Record `n_cal` (not saved in `res`, `zoom1.py:304`). |
| One shot on blocks 60-99. | process | 13 attempts on dev; practical tolerance added after v7; K changed after v10B. All disclosed in the log (good). | Pre-register: if the final run fails, do not iterate on 60-99. Use a new block range for any next final check. |

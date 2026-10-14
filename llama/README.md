# llama/: the engine for Windows and Linux

Cale runs one Gemma 3n model at many sizes. On a Mac that is done with MLX.
On Windows and Linux it is done with llama.cpp, plus a small patch that adds the sizes.
There is still one model file. There is no second file for the small size.

## What the patch does

| Part | Detail |
|---|---|
| `patches/0001-cale-matryoshka-widths.patch` | Applied to llama.cpp `b11558`. About 100 lines of C++. |
| `patches/0002-cale-readback-repack.patch` | A repacked Q4_0 weight can be read back in its plain layout, and the scheduler may give its operations to a GPU. Then a GPU too small for the model can read long prompts while the weights stay repacked in RAM for the CPU. Check: `tests/test_unrepack.cpp` and `scripts/check_unrepack.sh`. Rule: `cale/backends/devices.py`. Numbers: `FINDINGS.md`, "Option D". |
| `patches/0003-cale-prefill-kv-exit.patch` | Gemma 3n only. A step with no output token (the middle of a long prompt) stops after layer 19: layers 20 to 34 have no key/value memory of their own, so nothing later reads what they compute. The shim reads the last prompt token in a step of its own (`cale_set_option` key 4), so every other piece skips those layers. About 1.7 times faster prompts on CPUs. The early exit changes no number (`CALE_NO_KV_EXIT=1` turns it off, for the test `cale/tests/prefill_exit.py`). Numbers: `FINDINGS.md`, "Prompt reading". |
| New call | `llama_set_layer_widths(ctx, widths, n_layers)`: one number per layer. |
| Width 0 | The layer is skipped. Only layers 20 to 24 may be skipped: they own no KV cache. |
| Width N | The layer's feed-forward block uses its first N channels. This is a view of the stock weights. Nothing is copied. |
| Full width | The same code path as stock llama.cpp. |
| New call | `llama_model_get_tensor(model, name)`: lets Cale see where a layer's weights live in memory. |

Shim calls added in Phase 6: `cale_begin2(..., top_p, seed)` (top-p sampling). The library is loaded with the model's full 32768-token context (8 KB of KV cache per token: 0.27 GB).

The sizes are the official Gemma 3n sizes (`nest/gemma3n_levels.json`). A size is a list of widths.
Switching size takes about 0.03 ms.

## What the shim does

`shim/cale_llama.cpp` is a small C interface. Python calls it through `ctypes` (`cale/backends/llamalib.py`).
It exists because llama.cpp's own structures change between versions. The shim is built with the pinned llama.cpp,
so the two always match. It also holds the only code that talks to the OS about memory pages:

| OS | Drop pages | Read pages in | Count pages in RAM |
|---|---|---|---|
| Linux, macOS | `madvise(MADV_DONTNEED)` | `madvise(MADV_WILLNEED)` and a touch | `/proc/self/pagemap` (Linux) |
| Windows | `VirtualUnlock` | `PrefetchVirtualMemory` and a touch | `QueryWorkingSetEx` |

The model file is memory-mapped. A weight is in RAM only while its pages are. Dropping pages is how a smaller size
uses less memory. Reading them back is "reading from disk" for a call that needs a bigger size.

## Build

| System | Command | Needs |
|---|---|---|
| Linux, macOS | `llama/build.sh [OUT_DIR]` | git, cmake, a C++ compiler |
| Windows | `powershell -ExecutionPolicy Bypass -File llama\build.ps1 -Out C:\cale\lbuild -Python python.exe` | Python. It fetches a portable GCC (w64devkit). No admin rights, no Visual Studio. |

OpenMP is off on every system (`-DGGML_OPENMP=OFF`). With the mingw OpenMP runtime, Windows decoding was 43-50% slower than with llama.cpp's own thread pool. On Linux the two were equal, and an OpenMP-free library needs no `libgomp`, which a small container may lack. Linux also links the C++ runtime statically.

| Build option | Meaning |
|---|---|
| `CALE_BACKENDS_DL=1` | CPU variants for each instruction set (SSE4.2, AVX2, AVX-512 and more) and GPU backends are separate libraries. Cale loads the best one that this computer can run. A computer with an old CPU still works. |
| `CALE_CPU_VARIANTS=OFF` | One CPU build. Used for Windows arm64, which has no variants. |

`scripts/make_dist.py` packs a build into `llama/dist/<os>-<arch>.zip` (Windows) or `.tar.gz` (Linux). The installer unpacks the archive for this computer. Systems built so far: `windows-x86_64`, `linux-x86_64`, `linux-arm64`. `windows-arm64` builds with a cross compiler and is **not tested**.

Several answers at once: the library holds `CALE_SLOTS` conversations (default 4) in one shared memory pool. `cale/batching.py` runs one scheduler thread over them.

`STOCK=1 llama/build.sh` builds against unpatched llama.cpp, for the full-size comparison below.

## Checks that were run

| Check | Result |
|---|---|
| Full size equals stock llama.cpp (stock file, no repacking) | Identical: 3 of 3 greedy answers, loss difference 0.0 on every sample |
| Sliced E4B at the E2B size vs Google's E2B file (stock file) | Loss 2.1049 vs 2.1053. Greedy answers identical 3 of 3. |
| The same two checks on the prepared file | See the numbers below. Greedy answers still identical. |
| Server tests on Linux and Windows | `e2e.py` 37 of 37, `relative.py` all, `objects.py` all, `backend_llama.py` all |

Details: `results/xplat/llama-gates.json` and `results/xplat/bakeoff.json`.

## The prepared file: why it exists and what it does

llama.cpp can repack Q4 weights into a different byte order for faster math on a CPU. Repacking is worth about a third in speed
on the test laptop. But a view of a repacked weight only works if it covers whole row groups, and repacked weights cannot be mapped from the file.

| Matrix | First half of the channels | Works with repacking? |
|---|---|---|
| `gate`, `up` | a block of whole rows | Yes. A size is a view. |
| `down` | a slice of every row (spread over the file) | No |

So `llama/prepare_gguf.py` stores each `down` matrix as two halves (a 3-D tensor: 2 x rows x half the columns).
Each half is one block. Every other tensor and all metadata are copied unchanged, and the key `cale.ffn_down_split = 1` is added.
It only reorders bytes: it took 67 s on the test laptop and the check script confirms 35 matrices split and 812 tensors unchanged.
The installer runs it once after the download, then deletes the downloaded copy. If it fails, the stock file is used (slower, still correct).

With the prepared file:

| What | How |
|---|---|
| Full size | the two halves of `down` are two matrix multiplies that are added |
| Half size | only the first half |
| Memory for a small size | the second halves of `gate`, `up` and `down` are given back |
| Getting them back | repacked weights live in ordinary memory, so a dropped weight is read from the file again (`cale_restore`). Weights that stay mapped (the `down` matrices here are a 4-bit type that cannot be repacked) come back page by page when used. |
| File pages after load | the original file pages of repacked weights are given back at once (`cale_drop_file_mapping`), so the model costs its repacked size once |

## Numbers (Intel i7-8550U, 4 threads, Linux in WSL2, a 541-token prompt)

| File | Size | Prompt | Answer | Memory |
|---|---|---|---|---|
| stock, no repacking | full | 11.3 tok/s | 3.2 tok/s | 3.3 GB |
| stock, no repacking | smallest | 22.3 | 8.4 | 3.3 GB |
| prepared, repacking | full | 15.2 | 4.0 | 3.4 GB |
| prepared, repacking | smallest | 29.5 | 12.4 | 3.4 GB, and 1.14 GB freed for a held small size (0.85 GB with the stock file) |

Answer speeds here are measured as the difference between two runs, so they are noisy. The prompt speeds are steady.
The quality checks for this file: full size loss 1.9417 against 1.9384 for stock llama.cpp, with identical greedy answers (repacking uses different kernels, so it is no longer bit-exact).
The sliced E2B size: 2.1078 against 2.1053 for Google's E2B file, identical greedy answers.

## What was measured about MLX on a CPU

MLX has Windows and Linux builds that run on the CPU. On the test laptop it was unusable: one matrix multiply
for a 256-token prompt took 9.7 s (about 1.8 GFLOP/s, one core). The model warm-up did not finish in 7 minutes.
llama.cpp uses all cores and AVX2. That is why Windows and Linux use llama.cpp.

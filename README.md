# Cale

Cale is a local AI runtime for Mac, Windows and Linux.
It keeps one AI model on the computer, and every app on that computer shares it.
An app says what it needs, for example a speed. Cale picks the model size that fits this computer.
The model is Gemma 3n, where the smaller sizes live inside the large one, so there is only one download.

This repository also holds the research behind that idea: how to get smaller working models out of a bigger one
without training them again.

## Where to look

| Read | What it is |
|---|---|
| [`demo/README.md`](demo/README.md) | The demo: a chat and a desktop app on one local model, measured on 2026-10-17; a website using the same model was tested later |
| [`llama/README.md`](llama/README.md) | The Windows and Linux engine (a small llama.cpp patch), and the check that its small size matches Google's own E2B file |
| [`research/clean/NEURON-LOG.md`](research/clean/NEURON-LOG.md), section N5 | The tree: one weight set plus a 2.4 MB config gives ConvNeXt-Base (stages 3 and 4) smaller sizes, tested at 50%, 40% and 25%, with no training |
| [`research/clean/onecoarse/VECMATH.md`](research/clean/onecoarse/VECMATH.md), section 7 | A formula against training, for cuts of four sizes in ResNet-18 |

## Rerun the results

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ParkingSoman/Cale_Experimentation/blob/phase5-cross-platform/repro/Cale_rerun.ipynb)

Open the notebook, pick a GPU runtime, and use Runtime → Run all.
It downloads the public ImageNet validation photos, checks that they are the same bytes as in the original run,
and prints both result tables next to the committed numbers.

To run the same steps on your own computer, read [`repro/README.md`](repro/README.md).

## Install the runtime

| Type | What happens |
|---|---|
| `./cale/install.sh` (Mac, Linux) | Installs Cale, downloads the model once, sets up the `cale` command |
| `powershell -ExecutionPolicy Bypass -File cale\install.ps1` (Windows) | Same |
| `python cale/install.py` | Same, on any system that has Python 3.9 or newer |
| `cale status` | Shows if Cale runs and how much memory it uses |
| `./cale/uninstall.sh`, `cale\uninstall.ps1`, `python cale/uninstall.py` | Removes everything |

| | Mac | Windows and Linux |
|---|---|---|
| Needs | Apple silicon, macOS 14 or newer | 64-bit Intel or AMD, or ARM64 |
| Engine | MLX on the GPU | llama.cpp on the CPU |
| Model download | about 5.5 GB | about 4.4 GB |

Apps call Cale over plain HTTP with a local key. [`cale/README.md`](cale/README.md) has the commands and client code,
and `docs/API.md` has the whole interface.

## Dates

| Item | Date |
|---|---|
| Commits | 2026-10-01 onward |
| Research logs made public | 2026-11-26 |
| Dates inside the logs | The day each measurement was made |

"""Download the model copies used in phase 0 into models/."""
import sys
from huggingface_hub import snapshot_download

REPOS = {
    "gemma-4-e4b-it-4bit": "mlx-community/gemma-4-e4b-it-4bit",
    "gemma-4-e2b-it-4bit": "mlx-community/gemma-4-e2b-it-4bit",
    "gemma-4-12B-it-4bit": "mlx-community/gemma-4-12B-it-4bit",
    "gemma-4-12B-it-bf16": "mlx-community/gemma-4-12B-it-bf16",
    "gemma-4-12B-it-qat-bf16": "mlx-community/gemma-4-12B-it-qat-bf16",
}
for name in (sys.argv[1:] or REPOS):
    path = snapshot_download(REPOS[name], local_dir=f"models/{name}")
    print("done", name, path, flush=True)

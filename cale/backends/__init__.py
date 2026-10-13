"""The inference engines. Mac: MLX on the GPU. Windows and Linux: llama.cpp on the CPU.

Both give the engine the same model object (see nest/gemma3n.py and cale/backends/llama.py).
CALE_BACKEND=mlx|llama chooses by hand (for tests).
"""
import os

from .. import oslayer


def name():
    return os.environ.get("CALE_BACKEND") or ("mlx" if oslayer.MAC else "llama")


def load_model(path):
    if name() == "mlx":
        import mlx.core as mx
        from nest.gemma3n import NestedGemma3n
        mx.set_cache_limit(1024 ** 3)
        return NestedGemma3n.load(path, width=1.0, ple_on_disk=True)      # costs under 4% speed, saves 1.3 GB
    from .llama import LlamaNested
    return LlamaNested.load(path)


def supports():
    """Which request options this engine can honour: {"top_p", "seed", "stop"} -> bool. Does not load the model."""
    if name() == "mlx":
        return {"top_p": True, "seed": True, "stop": True, "guided": True, "images": True}
    from .llama import LlamaNested
    return dict(LlamaNested.supports)


def context_limit():
    """The longest prompt plus answer, in tokens, before the model is loaded."""
    if name() == "mlx":
        from nest.levels3n import MAX_CONTEXT
        return MAX_CONTEXT
    from .llama import N_CTX
    return N_CTX

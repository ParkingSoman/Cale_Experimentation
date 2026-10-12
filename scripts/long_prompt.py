"""Manual check: read a prompt near the context limit and report time and peak memory. Slow on weak computers.

usage: python scripts/long_prompt.py MODEL [TOKENS]       (default 30000, MLX only for the memory line)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cale import backends  # noqa: E402

m = backends.load_model(sys.argv[1])
target = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
filler = " ".join(f"Item {i} is a number that appears in the list." for i in range(target // 14))
msgs = [{"role": "user", "content": filler + "\n\nQuestion: what is 2+2? One word."}]
print("prompt tokens", m.prompt_len(msgs), "limit", m.max_context, flush=True)
t = time.time()
out = "".join(ev.get("delta", "") for ev in m.chat_stream(msgs, max_tokens=6))
print(repr(out), f"{time.time() - t:.1f}s", flush=True)
if backends.name() == "mlx":
    import mlx.core as mx
    print("peak GB", round(mx.get_peak_memory() / 1e9, 2))

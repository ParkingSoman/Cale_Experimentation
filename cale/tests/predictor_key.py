"""The saved speeds keep their key when only the free-text reason of a placement changes, and a one-time measurement is saved. Run: python cale/tests/predictor_key.py
No model, no engine.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cale import predict  # noqa: E402
from cale.predict import Predictor, placement_key  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:240]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


a = "device=Vulkan1;gpu_layers=0; chosen because the saved measurement"
b = "device=Vulkan1;gpu_layers=0; chosen because the card was cool"
c = "device=cpu; chosen because the CPU (the graphics card failed a trial load)"
check("same way of running, different reason: same key", placement_key(a) == placement_key(b) == "device=Vulkan1;gpu_layers=0", (placement_key(a), placement_key(b)))
check("a different way of running: different key", placement_key(a) != placement_key(c))
check("no placement: None", placement_key(None) is None and placement_key("") is None)
check("a plain placement is unchanged", placement_key("device=cpu") == "device=cpu")

# a one-time measurement is saved (n_dec + n_pre = 2 is skipped by the periodic save rule)
tmp = Path(tempfile.mkdtemp(prefix="cale-pred-test-"))
levels = [{"id": "e4b", "pct": 100, "eff_params_b": 3.98, "quality": 1.0}, {"id": "e2b", "pct": 48, "eff_params_b": 1.91, "quality": 0.8}]
p = Predictor("llama", 20e9, {"engine": "llama", "placement": "device=cpu"}, tmp / "predict.json", levels=levels)
p.observe({"tier": "e4b", "tokens": 16, "decode_s": 1.5, "first_token_s": 0.4, "prompt_tokens": 64, "seconds": 1.9})
check("one measurement counted once for each of answer and prompt", p.n_dec == 1 and p.n_pre == 1, (p.n_dec, p.n_pre))
check("the periodic rule alone does not save it (the bug)", not (tmp / "predict.json").exists())
p.save()
saved = json.loads((tmp / "predict.json").read_text()) if (tmp / "predict.json").exists() else {}
check("an explicit save writes the file, and a new start finds it", saved.get("n_dec") == 1 and saved["key"]["placement"] == "device=cpu", saved.get("n_dec"))
q = Predictor("llama", 20e9, {"engine": "llama", "placement": "device=cpu"}, tmp / "predict.json", levels=levels)
check("a new predictor with the same key loads it (no new measurement)", q.n_dec == 1, q.n_dec)

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)

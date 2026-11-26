"""Step 5 of the CLIP-KD plan (2026-11-25): zero-shot ImageNet scores of the teacher and the two ViT-T/16 students.

Models (weights/ folder, from github.com/winycg/CLIP-KD release CLIP-KDv0.1):
  teacher  = ViT_B_16-laion400m_e32.pt                         (OpenCLIP ViT-B/16, Laion-400M)
  kd       = ViT_B_16_laion400m_kd_ViT_T_16_cc3m_12m_ep32.pt   (ViT-T/16 distilled from that teacher)
  baseline = ViT_T_16_cc3m_12m_ep32.pt                         (ViT-T/16, same data, no teacher)
Photos: the reading photos (curves/map_agents/resnet18/net.py Photos("read"), 224 x 224 crops).
Test split as in neuron/coarse/r18.py: odd positions inside each class. We score the first N per class.
Each model scores classes with ITS OWN text tower: 1,000 class names x 80 OpenAI templates, mean of unit vectors.
Class logits = that model's logit_scale x cos(image, class).
Output: out/clip_scores.json (table) and out/<model>_test.npz (embeddings, logits, labels, photo ids) for reuse.
Usage: PYTHONPATH=distill/clipkd/pylib curves/.venv/bin/python -I distill/clipkd/score_clip.py [--per_class 5]
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "pylib"))
sys.path.insert(0, os.path.join(CLEAN, "curves", "map_agents", "resnet18"))
import open_clip  # noqa: E402
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES  # noqa: E402

import net  # noqa: E402

W = os.path.join(HERE, "weights")
OUT = os.path.join(HERE, "out")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MODELS = {
    "teacher": ("ViT-B-16", "ViT_B_16-laion400m_e32.pt"),
    "kd": ("ViT-T-16", "ViT_B_16_laion400m_kd_ViT_T_16_cc3m_12m_ep32.pt"),
    "baseline": ("ViT-T-16", "ViT_T_16_cc3m_12m_ep32.pt"),
}
# CLIP-KD's own config: src/open_clip/model_configs/ViT-T-16.json in github.com/winycg/CLIP-KD
VIT_T_16 = {"embed_dim": 512,
            "vision_cfg": {"image_size": 224, "layers": 12, "width": 192, "patch_size": 16},
            "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 384, "heads": 6, "layers": 12}}
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def safe_load(path):
    from numpy._core.multiarray import scalar
    ok = [(scalar, "numpy.core.multiarray.scalar"), np.dtype] + [type(np.dtype(t)) for t in ("float64", "float32", "int64")]
    with torch.serialization.safe_globals(ok):
        return torch.load(path, map_location="cpu", weights_only=True)


def build(arch, fname):
    if arch == "ViT-T-16" and "ViT-T-16" not in open_clip.list_models():
        open_clip.factory._MODEL_CONFIGS["ViT-T-16"] = VIT_T_16
    model = open_clip.create_model(arch, pretrained=None)
    ck = safe_load(os.path.join(W, fname))
    sd = ck.get("state_dict", ck)
    sd = {re.sub(r"^module\.", "", k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.eval().to(DEV)


@torch.no_grad()
def classifier(model, tok):
    rows = []
    for name in IMAGENET_CLASSNAMES:
        t = tok([tpl(name) for tpl in OPENAI_IMAGENET_TEMPLATES]).to(DEV)
        e = F.normalize(model.encode_text(t).float(), dim=-1).mean(0)
        rows.append(F.normalize(e, dim=0))
    return torch.stack(rows, 1)  # 512 x 1000


def test_photos(ph, per_class):
    ids = []
    for c in range(1000):
        w = np.where(ph.lab == c)[0]
        ids += w[1::2][:per_class].tolist()
    return np.array(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class", type=int, default=5)
    ap.add_argument("--bs", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ph = net.Photos("read")
    ids = test_photos(ph, a.per_class)
    lab = ph.lab[ids]
    tok = open_clip.get_tokenizer("ViT-B-16")
    res, logits = {}, {}
    for name, (arch, fname) in MODELS.items():
        t0 = time.time()
        model = build(arch, fname)
        cls = classifier(model, tok)
        scale = float(model.logit_scale.exp())
        emb, lg = [], []
        with torch.no_grad():
            for i in range(0, len(ids), a.bs):
                x = torch.from_numpy(ph.raw(ids[i:i + a.bs]).copy()).permute(0, 3, 1, 2).float() / 255.0
                x = ((x - CLIP_MEAN) / CLIP_STD).to(DEV)
                e = model.encode_image(x).float()
                emb.append(e.cpu())
                lg.append((scale * F.normalize(e, dim=-1) @ cls).cpu())
        emb, lg = torch.cat(emb), torch.cat(lg)
        logits[name] = lg
        np.savez(os.path.join(OUT, f"{name}_test.npz"), emb=emb.numpy(), logits=lg.numpy(), labels=lab,
                 photo_ids=ids, logit_scale=scale, text_classifier=cls.cpu().numpy())
        res[name] = {"arch": arch, "file": fname, "logit_scale": scale,
                     "top1": float((lg.argmax(1).numpy() == lab).mean()), "seconds": round(time.time() - t0, 1)}
        print(name, res[name], flush=True)
        del model
        if DEV == "mps":
            torch.mps.empty_cache()
    pt = F.log_softmax(logits["teacher"], 1)
    for name in ("kd", "baseline"):
        ps = F.log_softmax(logits[name], 1)
        res[name]["kl_teacher_to_student"] = float((pt.exp() * (pt - ps)).sum(1).mean())
        res[name]["agree_top1_with_teacher"] = float((logits[name].argmax(1) == logits["teacher"].argmax(1)).float().mean())
    res["photos"] = {"set": "reading photos, test split (odd positions per class)", "per_class": a.per_class, "n": int(len(ids))}
    json.dump(res, open(os.path.join(OUT, "clip_scores.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

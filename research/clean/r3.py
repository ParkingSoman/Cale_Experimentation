"""R3: what does a smaller model lose? Fine detail (right family, wrong member) or whole concepts?

Plan: ~/.claude/plans/okay-we-ve-been-experimenting-vectorized-salamander.md

  python3 research/clean/r3.py download          # SmolLM2-360M, 1.7B, GPT-2, DistilGPT2 into research/clean/hf
  python3 research/clean/r3.py run               # per-word loss and top-5 guesses, one model at a time
  python3 research/clean/r3.py report            # bins, near-miss share, our compressed 135M against the size gap

Ladders: SmolLM2 135M / 360M / 1.7B (trained apart, same tokenizer) and GPT-2 / DistilGPT2 (real distillation).
Our compressed 135M: cut to 50% of neurons in every layer (Z2's cut_act + least squares) and the per-word
dial at 25% (C1). Writes research/clean/r3_cache/*.npz, results/clean-r3.json, results/clean-r3-spots.md.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "research", "tree"))
sys.path.insert(0, HERE)
from tree_lib import (load_model, Weights, capture_mlp_inputs, hidden, cut_keep, ls_down,  # noqa: E402
                      canonical_sign, SEQ)
from tree_lib import HERE as TREE  # noqa: E402

HF = os.path.join(HERE, "hf")
CACHE = os.path.join(HERE, "r3_cache")
RES = os.path.join(ROOT, "results")
N_HELD = 100
REPOS = {"smol360": "HuggingFaceTB/SmolLM2-360M", "smol1700": "HuggingFaceTB/SmolLM2-1.7B",
         "gpt2": "openai-community/gpt2", "distilgpt2": "distilbert/distilgpt2"}
LADDERS = {"smol": ["smol135", "smol360", "smol1700", "smol135_cut50", "smol135_dial25"],
           "gpt2": ["distilgpt2", "gpt2"]}
PAIRS = [("smol", "smol135", "smol360"), ("smol", "smol135", "smol1700"), ("smol", "smol360", "smol1700"),
         ("gpt2", "distilgpt2", "gpt2"),
         ("smol", "smol135_cut50", "smol135"), ("smol", "smol135_dial25", "smol135")]


def model_dir(name):
    if name.startswith("smol135"):
        return None
    return glob.glob(f"{HF}/models--{REPOS[name].replace('/', '--')}/snapshots/*")[0]


def cmd_download(args):
    from huggingface_hub import snapshot_download
    for name, repo in REPOS.items():
        t0 = time.time()
        p = snapshot_download(repo, cache_dir=HF, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
        print(f"{name}: {p} [{time.time() - t0:.0f}s]", flush=True)


def blocks_for(tok):
    """The same held-out text for every model of a ladder: wikitext-2 test, N_HELD blocks of SEQ+1 tokens."""
    text = open(f"{TREE}/data/wikitext2_test.txt").read()
    ids = tok.encode(text[:2_000_000])
    need = N_HELD * (SEQ + 1)
    ids = np.array(ids[:need], dtype=np.int32)
    assert len(ids) == need, len(ids)
    return ids.reshape(N_HELD, SEQ + 1)


def score(model, blocks, batch=4):
    """Per position: cross-entropy of the true next token and the top-5 guesses."""
    ce, top = [], []
    for i in range(0, blocks.shape[0], batch):
        b = mx.array(blocks[i:i + batch])
        lg = model(b[:, :-1]).astype(mx.float32)
        lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        ce.append(np.array(-mx.take_along_axis(lg, b[:, 1:, None], axis=-1).squeeze(-1)))
        top.append(np.array(mx.argsort(-lg, axis=-1)[..., :5]))
        mx.eval()
    return np.concatenate(ce).reshape(-1), np.concatenate(top).reshape(-1, 5)


def compress_cut50(model, tok):
    """Z2's best cut: keep the half of each layer's neurons with the biggest mean |activation| x |output|,
    then refit the output weights by least squares on calibration text (intact-model inputs)."""
    from tree_lib import token_blocks
    W = Weights(model)
    X = capture_mlp_inputs(model, token_blocks(tok, "train", 32))
    for i in range(len(W.G)):
        G, U, D = W.G[i], W.U[i], W.D[i]
        H = hidden(X[i], G, U)
        keep = cut_keep(np.abs(H).mean(0) * np.linalg.norm(D, axis=0), G.shape[0] // 2)
        Y = H @ D.T
        D2 = ls_down(X[i], G[keep], U[keep], Y)
        m = model.model.layers[i].mlp
        import mlx.nn as nn
        m.gate_proj, m.up_proj = nn.Linear(576, len(keep), bias=False), nn.Linear(576, len(keep), bias=False)
        m.down_proj = nn.Linear(len(keep), 576, bias=False)
        m.gate_proj.weight, m.up_proj.weight = mx.array(G[keep]), mx.array(U[keep])
        m.down_proj.weight = mx.array(D2)
    mx.eval(model.parameters())


def cmd_run(args):
    from mlx_lm import load
    os.makedirs(CACHE, exist_ok=True)
    names = args.models.split(",") if args.models else [n for v in LADDERS.values() for n in v]
    for name in names:
        out = os.path.join(CACHE, f"{name}.npz")
        if os.path.exists(out) and not args.force:
            print(f"{name}: cached", flush=True)
            continue
        t0 = time.time()
        if name.startswith("smol135"):
            model, tok = load_model()
            if name == "smol135_cut50":
                compress_cut50(model, tok)
            elif name == "smol135_dial25":
                from c1 import set_mlps
                orig = [l.mlp for l in model.model.layers]
                set_mlps(model, orig, [(0.25, "hidden")] * len(orig))
        else:
            model, tok = load(model_dir(name))
        blocks = blocks_for(tok)
        ce, top = score(model, blocks)
        np.savez(out, ce=ce, top=top, blocks=blocks)
        print(f"{name}: mean CE {ce.mean():.4f} [{time.time() - t0:.0f}s]", flush=True)
        del model
        mx.clear_cache()


# ---------- report ----------

def token_info(tok, ladder):
    """Per vocabulary id: frequency rank in calibration text, and a word type."""
    text = open(f"{TREE}/data/wikitext2_train.txt").read()[:3_000_000]
    ids = np.array(tok.encode(text))
    V = max(int(ids.max()) + 1, len(tok.vocab) if hasattr(tok, "vocab") else 0, 50257 if ladder == "gpt2" else 49152)
    cnt = np.bincount(ids, minlength=V)
    rank = np.empty(V, int)
    rank[np.argsort(-cnt, kind="stable")] = np.arange(V)
    kind = []
    for v in range(V):
        s = tok.decode([v])
        st = s.strip()
        if not st:
            kind.append("space")
        elif st.replace(",", "").replace(".", "").isdigit():
            kind.append("number")
        elif not any(c.isalnum() for c in st):
            kind.append("punct")
        elif s[0] == " " and st[0].isupper():
            kind.append("Capital word")
        elif s[0] == " ":
            kind.append("lower word")
        else:
            kind.append("word piece")
    return rank, np.array(kind), cnt


def freq_bin(rank):
    return np.select([rank < 100, rank < 1000, rank < 10000], ["top100", "100-1k", "1k-10k"], "rarer")


def seen_before(blocks):
    """True where the target token already appeared earlier in the same block (copying can help)."""
    tgt = blocks[:, 1:]
    out = np.zeros_like(tgt, bool)
    for b in range(blocks.shape[0]):
        seen = set(blocks[b, :1].tolist())
        for t in range(tgt.shape[1]):
            out[b, t] = tgt[b, t] in seen
            seen.add(int(tgt[b, t]))
    return out.reshape(-1)


def neighbors(emb, ids, k=20):
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return {int(i): set(np.argsort(-(e @ e[i]))[1:k + 1].tolist()) for i in ids}


def cmd_report(args):
    from mlx_lm import load
    rng = np.random.default_rng(0)
    res = {"pairs": []}
    spots_md = ["# R3 spots: the smaller model is much worse and the bigger model is right\n",
                "Each spot: context, the true next token, top-5 guesses of the bigger and the smaller model.\n"]
    info = {}
    for ladder, small, big in PAIRS:
        a, b = np.load(os.path.join(CACHE, f"{small}.npz")), np.load(os.path.join(CACHE, f"{big}.npz"))
        assert (a["blocks"] == b["blocks"]).all()
        if ladder not in info:
            ref = "smol1700" if ladder == "smol" else "gpt2"
            model, tok = load(model_dir(ref))
            emb = np.array(model.model.embed_tokens.weight.astype(mx.float32)) if ladder == "smol" else \
                np.array(model.model.wte.weight.astype(mx.float32))
            del model
            rank, kind, cnt = token_info(tok, ladder)
            info[ladder] = (tok, emb, rank, kind, cnt, seen_before(a["blocks"]))
        tok, emb, rank, kind, cnt, seen = info[ladder]
        tgt = a["blocks"][:, 1:].reshape(-1)
        gap = a["ce"] - b["ce"]
        bins = {"freq": freq_bin(rank[tgt]), "type": kind[tgt],
                "seen_before": np.where(seen, "seen before in block", "new in block")}
        row = {"ladder": ladder, "small": small, "big": big, "ce_small": float(a["ce"].mean()),
               "ce_big": float(b["ce"].mean()), "gap": float(gap.mean()), "bins": {}}
        for bn, lab in bins.items():
            row["bins"][bn] = {}
            for v in np.unique(lab):
                m = lab == v
                row["bins"][bn][str(v)] = {"share_tokens": float(m.mean()), "ce_small": float(a["ce"][m].mean()),
                                           "ce_big": float(b["ce"][m].mean()), "gap": float(gap[m].mean()),
                                           "share_of_total_gap": float(gap[m].sum() / gap.sum())}
        # Near miss vs far miss: the bigger model's top guess is right, the smaller one's is wrong.
        miss = (b["top"][:, 0] == tgt) & (a["top"][:, 0] != tgt)
        mi = np.where(miss)[0]
        sub = rng.choice(mi, min(3000, len(mi)), replace=False)
        nb = neighbors(emb, np.unique(tgt[sub]))
        near = np.array([int(a["top"][i, 0]) in nb[int(tgt[i])] for i in sub])
        same_kind = np.array([kind[a["top"][i, 0]] == kind[tgt[i]] for i in sub])
        in_top5 = np.array([tgt[i] in a["top"][i] for i in sub])
        # Null: a random wrong token drawn by frequency, is it a neighbor of the target?
        p = cnt / cnt.sum()
        rnd = rng.choice(len(p), len(sub), p=p)
        near_null = np.mean([int(r) in nb[int(tgt[i])] for r, i in zip(rnd, sub)])
        row["misses"] = {"share_positions": float(miss.mean()), "n_sampled": int(len(sub)),
                         "small_top1_is_embedding_neighbor_of_target": float(near.mean()),
                         "null_random_token_is_neighbor": float(near_null),
                         "small_top1_same_word_type": float(same_kind.mean()),
                         "truth_in_small_top5": float(in_top5.mean())}
        # Correlation of where losses happen, against the natural 135M -> 360M gap.
        res["pairs"].append(row)
        print(f"{small:>15s} vs {big:<9s} CE {row['ce_small']:.3f} vs {row['ce_big']:.3f}  gap {row['gap']:.3f}  "
              f"misses {row['misses']['share_positions']:.3f}  near {row['misses']['small_top1_is_embedding_neighbor_of_target']:.2f}"
              f" (null {near_null:.2f})  same type {same_kind.mean():.2f}  truth in top5 {in_top5.mean():.2f}", flush=True)
        if (small, big) in (("smol135", "smol1700"), ("distilgpt2", "gpt2"), ("smol135_cut50", "smol135")):
            spots_md.append(f"\n## {small} (smaller) vs {big} (bigger)\n")
            blocks = a["blocks"]
            for i in rng.choice(mi, 25 if small != "smol135_cut50" else 15, replace=False):
                bi, t = divmod(int(i), SEQ)
                ctx = tok.decode(blocks[bi, max(0, t - 30):t + 1].tolist())
                dec = lambda ids: ", ".join(repr(tok.decode([int(x)])) for x in ids)  # noqa: E731
                spots_md.append(f"- spot {i}: ...{ctx[-160:]!r}\n  - truth: {tok.decode([int(tgt[i])])!r}\n"
                                f"  - bigger: {dec(b['top'][i])}\n  - smaller: {dec(a['top'][i])}")
    # Do our compressed models lose the same tokens as the natural size gap?
    base = np.load(os.path.join(CACHE, "smol135.npz"))["ce"]
    nat = base - np.load(os.path.join(CACHE, "smol360.npz"))["ce"]
    nat17 = base - np.load(os.path.join(CACHE, "smol1700.npz"))["ce"]
    res["same_tokens"] = {}
    for v in ("smol135_cut50", "smol135_dial25"):
        loss = np.load(os.path.join(CACHE, f"{v}.npz"))["ce"] - base
        res["same_tokens"][v] = {"corr_with_135_to_360_gap": float(np.corrcoef(loss, nat)[0, 1]),
                                 "corr_with_135_to_1700_gap": float(np.corrcoef(loss, nat17)[0, 1])}
    res["same_tokens"]["smol135_to_360_vs_135_to_1700"] = float(np.corrcoef(nat, nat17)[0, 1])
    print("same tokens:", res["same_tokens"])
    json.dump(res, open(os.path.join(RES, "clean-r3.json"), "w"), indent=1)
    open(os.path.join(RES, "clean-r3-spots.md"), "w").write("\n".join(spots_md) + "\n")
    print("wrote results/clean-r3.json and results/clean-r3-spots.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "run", "report"])
    ap.add_argument("--models", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"download": cmd_download, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()

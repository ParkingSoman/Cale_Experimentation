# serves: H-map
"""Channel cards for card 3-22 (zoom plan, step 1): what a ResNet-18 last-layer channel's strongest entries show,
against its everyday activity and what the logits read from it.

The 6 channels with the largest c_u = s_u |P_perp W_u| (DM10: the channels the logit differences depend on most), on
the train split. Per channel:
  strong   the 8 photos holding its largest entries (one per photo), cropped around the strongest position
  classes  top 5 classes of the strong part (entries above the channel's 99.9 % quantile) and of all activity, with
           shares and the effective number of classes
  typical  6 photos where the pooled activity sits between its 45th and 55th percentile (what it does most of the time)
  reads    the 5 classes whose logits gain most from the channel (largest entries of W[:, u])
Writes zoom/cards.html (thumbnails as data URIs).
"""
import base64, io, json, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "..", "cache", "r18")
VAL = os.path.join(HERE, "..", "..", "hf", "imagenet", "val_224.npy")
Z = np.load(os.path.join(C, "z_train.npy"), mmap_mode="r"); ix = np.load(os.path.join(C, "ix_train.npy"))
lab = np.load(os.path.join(C, "lab_train.npy")); H = np.load(os.path.join(C, "h_train.npy"))
hd = np.load(os.path.join(C, "head.npz")); W = hd["W"]; names = hd["names"]
X = np.load(VAL, mmap_mode="r")
n = len(Z)
ssq = np.zeros(512)
for i in range(0, n, 2000):
    ssq += (Z[i:i + 2000].astype(np.float32) ** 2).sum((0, 2))
s = np.sqrt(ssq / (n * 49)); Wp = W - W.mean(0, keepdims=True); c = s * np.linalg.norm(Wp, axis=0)
chans = list(np.argsort(c)[::-1][:6])
zc = np.zeros((len(chans), n, 49), np.float32)
for i in range(0, n, 2000):
    zc[:, i:i + 2000] = Z[i:i + 2000][:, chans].astype(np.float32).transpose(1, 0, 2)
rng = np.random.default_rng(3)


def thumb(img_i, pos=None, size=88):
    im = np.asarray(X[ix[img_i]])
    if pos is not None:
        r, q = divmod(int(pos), 7); cy, cx = 16 + 32 * r, 16 + 32 * q
        y0, x0 = int(np.clip(cy - 56, 0, 224 - 112)), int(np.clip(cx - 56, 0, 224 - 112))
        im = im[y0:y0 + 112, x0:x0 + 112]
    b = io.BytesIO(); Image.fromarray(im).resize((size, size), Image.LANCZOS).save(b, "JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def spread(act):
    A = np.bincount(lab, weights=act, minlength=1000); S = A / A.sum()
    ent = float(np.exp(-(S * np.log(S + 1e-12)).sum())); top = np.argsort(S)[::-1][:5]
    return ent, [(str(names[k]), float(S[k])) for k in top]


cards = []
for j, u in enumerate(chans):
    z = np.maximum(zc[j], 0)
    peak = z.max(1); order = np.argsort(peak)[::-1][:8]
    strong_imgs = [{"src": thumb(i, z[i].argmax()), "cls": str(names[lab[i]])} for i in order]
    thr = np.quantile(z, 0.999); strong_act = np.where(z > thr, z, 0).sum(1)
    e_s, top_s = spread(strong_act); e_a, top_a = spread(H[:, u])
    lo, hi = np.quantile(H[:, u], [0.45, 0.55]); mid = np.where((H[:, u] >= lo) & (H[:, u] <= hi))[0]
    typ = [{"src": thumb(i), "cls": str(names[lab[i]])} for i in rng.choice(mid, 6, replace=False)]
    reads = [(str(names[k]), float(W[k, u])) for k in np.argsort(W[:, u])[::-1][:5]]
    share_strong = float(strong_act.sum() / z.sum())
    cards.append({"u": int(u), "c": float(c[u]), "strong": strong_imgs, "e_s": e_s, "top_s": top_s, "e_a": e_a,
                  "top_a": top_a, "typ": typ, "reads": reads, "share_strong": share_strong})
    print(u, round(e_s), round(e_a), top_s[:2], reads[:2], flush=True)
json.dump([{k: v for k, v in cd.items() if k not in ("strong", "typ")} for cd in cards],
          open(os.path.join(HERE, "results_rz_cards.json"), "w"), indent=1)


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bars(items):
    m = max(v for _, v in items)
    return "".join(f'<li><span class="nm">{esc(nm)}</span><span class="bar"><i style="width:{100*v/m:.0f}%"></i></span>'
                   f'<span class="v">{100*v:.1f}%</span></li>' for nm, v in items)


body = []
for cd in cards:
    st = "".join(f'<figure><img src="{im["src"]}" alt="{esc(im["cls"])}" width="88" height="88"><figcaption>{esc(im["cls"])}</figcaption></figure>' for im in cd["strong"])
    ty = "".join(f'<figure><img src="{im["src"]}" alt="{esc(im["cls"])}" width="88" height="88"><figcaption>{esc(im["cls"])}</figcaption></figure>' for im in cd["typ"])
    rd = ", ".join(esc(nm) for nm, _ in cd["reads"])
    body.append(f'''<section class="card">
<header><h2>Channel {cd["u"]}</h2><p class="meta">The strongest 0.1% of its entries hold {100*cd["share_strong"]:.1f}% of its activity</p></header>
<div class="row"><div class="col"><h3>Strongest entries</h3><div class="grid">{st}</div></div></div>
<div class="row two">
<div class="col"><h3>Classes of the strong part <span class="eff">{cd["e_s"]:.0f} effective classes</span></h3><ul class="bars">{bars(cd["top_s"])}</ul></div>
<div class="col"><h3>Classes of all its activity <span class="eff">{cd["e_a"]:.0f} effective classes</span></h3><ul class="bars">{bars(cd["top_a"])}</ul></div>
</div>
<div class="row"><div class="col"><h3>Typical photos (activity near its median)</h3><div class="grid">{ty}</div></div></div>
<p class="reads"><b>The outputs read it as evidence for:</b> {rd}</p>
</section>''')

html = f'''<title>ResNet-18 Channel Cards</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,650&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono&display=swap">
<style>
/* Layout: a lab notebook page. Findings first, then one card per channel, photo strips in a wrapping grid. */
:root {{ --bg:#f6f7f4; --panel:#ffffff; --fg:#1d2420; --muted:#5d6a63; --rule:#d9dfda; --accent:#2f6f5e; --bar:#9cc3b6;
  --display:"Source Serif 4", Georgia, serif; --body:"IBM Plex Sans", system-ui, sans-serif; --mono:"IBM Plex Mono", ui-monospace, monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#151a18; --panel:#1d2421; --fg:#e4ebe6; --muted:#9aa8a0; --rule:#2e3833; --accent:#7cc2ad; --bar:#3f6d5f; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#151a18; --panel:#1d2421; --fg:#e4ebe6; --muted:#9aa8a0; --rule:#2e3833; --accent:#7cc2ad; --bar:#3f6d5f; color-scheme:dark; }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--body); font-size:15px; line-height:1.55; }}
main {{ max-width:1080px; margin:0 auto; padding-block:28px 56px; padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); display:flex; flex-direction:column; gap:28px; }}
h1 {{ font-family:var(--display); font-weight:650; font-size:2rem; line-height:1.15; margin:0; text-wrap:balance; }}
h2 {{ font-family:var(--display); font-weight:650; font-size:1.3rem; margin:0; }}
h3 {{ font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:0 0 8px; font-weight:600; }}
.lede {{ max-width:68ch; margin:8px 0 0; color:var(--muted); }}
.findings {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(230px, 1fr)); gap:12px; }}
.finding {{ background:var(--panel); border:1px solid var(--rule); border-radius:6px; padding:14px 16px; min-width:0; }}
.finding b {{ display:block; font-family:var(--mono); font-size:1.35rem; color:var(--accent); font-variant-numeric:tabular-nums; }}
.card {{ background:var(--panel); border:1px solid var(--rule); border-radius:6px; padding:18px; display:flex; flex-direction:column; gap:16px; }}
.card header {{ display:flex; flex-wrap:wrap; align-items:baseline; gap:6px 14px; }}
.meta {{ margin:0; color:var(--muted); font-size:.9rem; }}
.row.two {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:18px; }}
.col {{ min-width:0; }}
.grid {{ display:flex; flex-wrap:wrap; gap:8px; }}
figure {{ margin:0; width:88px; }}
figure img {{ display:block; border-radius:3px; max-width:100%; height:auto; }}
figcaption {{ font-size:.68rem; color:var(--muted); line-height:1.25; margin-top:3px; overflow-wrap:anywhere; }}
.eff {{ font-family:var(--mono); text-transform:none; letter-spacing:0; color:var(--accent); margin-left:6px; }}
.bars {{ list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:5px; }}
.bars li {{ display:grid; grid-template-columns:minmax(0, 1.4fr) 1fr 3.4em; gap:8px; align-items:center; font-size:.85rem; }}
.nm {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.bar {{ height:8px; background:var(--rule); border-radius:4px; overflow:hidden; }}
.bar i {{ display:block; height:100%; background:var(--bar); }}
.v {{ font-family:var(--mono); font-size:.78rem; text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); }}
.reads {{ margin:0; font-size:.92rem; }}
</style>
<main>
<header><h1>ResNet-18 Channel Cards</h1>
<p class="lede">The six channels of ResNet-18's last layer that the outputs depend on most. Each card shows the photos where the channel fires hardest, where its activity goes across the 1000 classes, and what it does on a typical photo. Photos are from the ImageNet validation set (35 per class).</p></header>
<div class="findings">
<div class="finding"><b>~860 of 1000</b>classes share a typical channel's activity. No channel stays quiet outside a small set of classes.</div>
<div class="finding"><b>0.24</b>median correlation between a channel and its closest neighbour. No channel is a copy of another.</div>
<div class="finding"><b>KL 4.9</b>if only each channel's strongest 0.1% is kept. The outputs rely on the broad activity, not on the peaks.</div>
</div>
{"".join(body)}
</main>
'''
open(os.path.join(HERE, "cards.html"), "w").write(html)
print("wrote cards.html", len(html) // 1024, "KB")

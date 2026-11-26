# serves: H-map
"""Build the page Shonu reads to judge round 2: a tree of channels, with each child nested under its main parent.
Reading photos only. Crops are small JPEGs inside the page. Usage: build_page2.py OUTPUT.html"""
import base64
import glob
import html
import io
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ev2 import CROP, grid, parse, peaks  # noqa: E402
from net import HERE, STORE, STRIDE, Photos, class_names  # noqa: E402

P = Photos("read")
cats = class_names()
y = np.load(os.path.join(STORE, "read", "y.npy"), mmap_mode="r")
RES = os.path.join(HERE, "results")
LAYER = {"s4b1": "last stream layer", "s4b1i": "inner channels of the last block", "s4b0": "stream of stage 4, block 0", "s4b0i": "inner channels of stage 4, block 0", "s3b1": "last stream layer of stage 3"}


def clean(t):
    return (t or "").replace(" (skip)", "").strip()


cards = {}
for f in sorted(glob.glob(os.path.join(HERE, "cards2", "*.jsonl"))):
    for line in open(f):
        if line.strip():
            d = json.loads(line)
            d["agent"] = os.path.basename(f)[:-6]
            cards.setdefault(d["node"], d)
scores = {r["node"]: r for r in json.load(open(os.path.join(RES, "cards2_scores.json")))["nodes"]}
crop = {r["node"]: r for r in json.load(open(os.path.join(HERE, "review2", "new", "scores.json"))) if not r["wrong"]}


def tile(raw, pos, s, value, cls):
    g, half = grid(s), CROP[s] // 2
    r, c = int(pos) // g, int(pos) % g
    cy, cx = int((r + .5) * STRIDE[s]), int((c + .5) * STRIDE[s])
    r0, c0 = min(max(cy - half, 0), 224 - 2 * half), min(max(cx - half, 0), 224 - 2 * half)
    im = Image.fromarray(raw[r0:r0 + 2 * half, c0:c0 + 2 * half]).resize((132, 132))
    if s == 4:  # the whole photo: mark where the channel fires hardest
        d = ImageDraw.Draw(im)
        px, py = (cx - c0) * 132 / (2 * half), (cy - r0) * 132 / (2 * half)
        d.ellipse((px - 11, py - 11, px + 11, py + 11), outline=(255, 214, 0), width=2)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=68)
    src = "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    return f'<figure class="{cls}"><img loading="lazy" alt="" src="{src}"><figcaption>{value:.1f} {html.escape(cats[int(np.argmax(y[i_now]))][:15])}</figcaption></figure>'


def strip(n):
    global i_now
    v, p = peaks(n)
    big = v.max()
    picks = [("top", i) for i in np.argsort(-v)[:6]] + [("half", i) for i in np.argsort(np.abs(v - .5 * big))[:3]]
    vn, pn = peaks(n, "neg")
    picks += [("neg", i) for i in np.argsort(vn)[:3]]
    raw = P.raw([q[1] for q in picks])
    out = {"top": [], "half": [], "neg": []}
    for j, (kind, i) in enumerate(picks):
        i_now = int(i)
        pos, val = (pn[i], vn[i]) if kind == "neg" else (p[i], v[i])
        out[kind].append(tile(raw[j], pos, n["s"], float(val), kind))
    return f'<div class="row"><span class="tag">strongest (largest {big:.1f})</span><div class="crops">{"".join(out["top"])}</div></div>' \
           f'<div class="row"><span class="tag">half strength</span><div class="crops">{"".join(out["half"])}</div></div>' \
           f'<div class="row"><span class="tag">negative side</span><div class="crops">{"".join(out["neg"])}</div></div>'


def e(x):
    return html.escape(str(x if x is not None else ""))


def pills(k):
    s, c = scores.get(k), crop.get(k)
    out = []
    if s:
        out.append('<span class="pill good">parent acts</span>' if s["tree"] else '<span class="pill bad">no tested parent</span>')
    if c:
        cls, word = ("good", "words pass") if c["gap"] >= .2 else ("bad", "words miss")
        out.append(f'<span class="pill {cls}">{word} {c["gap"]:+.2f}</span>')
    else:
        out.append('<span class="pill mid">words not scored</span>')
    return "".join(out)


children = {}
roots = []
for k, c in cards.items():
    p = clean((c.get("main_parent") or {}).get("node"))
    if p and p != k:
        children.setdefault(p, []).append(k)
    if not p or p not in cards or p == k:
        roots.append(k)
for v in children.values():
    v.sort()


def node_html(k, depth, seen):
    c = cards[k]
    seen.add(k)
    n = parse(k)
    layer = k.split(":")[0]
    mp = c.get("main_parent") or {}
    inputs = "".join(f'<tr><td class="mono">{e(clean(i.get("node")))}</td><td>{e(i.get("type"))}</td><td>{e(i.get("gives"))}</td></tr>' for i in c.get("inputs", []))
    inp = f'<table><tr><th>Other input</th><th>Kind</th><th>What it gives</th></tr>{inputs}</table>' if inputs else ""
    s = scores.get(k) or {}
    rank = f'Rank by effect among its inputs: {s["main_rank"]}.' if s.get("main_rank") else ""
    pred = {True: "held", False: "failed", None: "not tested"}.get(c.get("prediction_held"), "not tested")
    kids = "".join(node_html(q, depth + 1, seen) for q in children.get(k, []) if q in cards and q not in seen)
    return f"""<details class="node">
<summary><span class="cid">{e(k)}</span><span class="dist">{e(c.get("distinction"))}</span><span class="pills">{pills(k)}</span></summary>
<div class="body">
<p class="muted">{e(LAYER.get(layer, layer))}. Family: {e(c.get("family"))}. Read by {e(c.get("agent"))}, its own confidence {e(c.get("sure"))}. Prediction {pred}.</p>
<p><b>Fires on:</b> {e(c.get("positive"))}</p>
<p><b>Negative side:</b> {e(c.get("negative"))}</p>
{strip(n)}
<p><b>Main parent:</b> <span class="mono">{e(clean(mp.get("node")) or "none named")}</span> {e(mp.get("why"))} {rank}</p>
{inp}
<p><b>What reads it:</b> {e(c.get("readers"))}</p>
<p class="muted"><b>Not explained:</b> {e(c.get("unread"))}</p>
{f'<div class="kids"><p class="kidlabel">{len(children.get(k, []))} child channel(s) grew from this</p>{kids}</div>' if kids else ''}
</div></details>"""


# group the roots by what they hang from: a parent in the next layer down that no agent read, or nothing
seen = set()
by_outside = {}
loners = []
for k in sorted(roots):
    p = clean((cards[k].get("main_parent") or {}).get("node"))
    if p and p not in cards:
        by_outside.setdefault(p, []).append(k)
    else:
        loners.append(k)
blocks = []
order = sorted(by_outside.items(), key=lambda x: -len(x[1]))
for p, ks in order:
    inner = "".join(node_html(k, 1, seen) for k in ks)
    blocks.append(f'<details class="node"><summary><span class="cid">{e(p)}</span><span class="dist muted">named as a parent, not read yet</span></summary><div class="body"><div class="kids">{inner}</div></div></details>')
trees = "".join(blocks)
with_kids = [k for k in loners if children.get(k)]
no_kids = [k for k in loners if not children.get(k)]
trees2 = "".join(node_html(k, 0, seen) for k in with_kids)
trees3 = "".join(node_html(k, 0, seen) for k in no_kids)
left = [k for k in cards if k not in seen]
trees4 = "".join(node_html(k, 0, seen) for k in left)


def depth_of(k, memo={}):
    if k in memo:
        return memo[k]
    memo[k] = 1
    memo[k] = 1 + max([depth_of(q) for q in children.get(k, []) if q in cards], default=0)
    return memo[k]


longest = max(depth_of(k) for k in cards)
page = f"""<title>ResNet-18 Tree Map</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column. A verdict first, then the tree as nested dropdowns, because the tree is the object being judged. */
:root {{
  --bg: #f2f4f6; --panel: #ffffff; --fg: #17222c; --muted: #566573; --line: #d5dce3; --accent: #0e6e8c;
  --good: #1b7f4b; --bad: #b3372c; --mid: #9a6a00;
  --sans: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #11171d; --panel: #18212a; --fg: #e4eaf0; --muted: #9aa8b5; --line: #2c3945; --accent: #54b8d8;
  --good: #58c48a; --bad: #ee7f73; --mid: #dcae4b; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #11171d; --panel: #18212a; --fg: #e4eaf0; --muted: #9aa8b5; --line: #2c3945; --accent: #54b8d8;
  --good: #58c48a; --bad: #ee7f73; --mid: #dcae4b; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.5 var(--sans); padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 24px 48px; }}
main {{ max-width: 980px; margin: 0 auto; display: grid; gap: 22px; min-width: 0; }}
section {{ display: grid; gap: 10px; min-width: 0; }}
h1 {{ font-size: 26px; font-weight: 600; margin: 0; text-wrap: balance; }}
h2 {{ font-size: 18px; font-weight: 600; margin: 0; text-wrap: balance; }}
p {{ margin: 0; max-width: 72ch; }}
.muted {{ color: var(--muted); }}
.mono, code {{ font-family: var(--mono); font-size: 13px; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; padding: 6px 10px 6px 0; border-bottom: 1px solid var(--line); vertical-align: top; font-size: 14px; }}
th {{ color: var(--muted); font-weight: 500; }}
.pill {{ font: 500 12px var(--mono); padding: 1px 7px; border: 1px solid currentColor; white-space: nowrap; }}
.good {{ color: var(--good); }} .bad {{ color: var(--bad); }} .mid {{ color: var(--mid); }}
details.node {{ background: var(--panel); border: 1px solid var(--line); min-width: 0; }}
summary {{ cursor: pointer; padding: 9px 12px; display: flex; flex-wrap: wrap; gap: 3px 10px; align-items: baseline; }}
summary:focus-visible {{ outline: 2px solid var(--accent); outline-offset: -2px; }}
.cid {{ font: 500 13px var(--mono); color: var(--accent); }}
.dist {{ flex: 1 1 280px; min-width: 0; }}
.pills {{ display: flex; flex-wrap: wrap; gap: 4px; }}
.body {{ padding: 2px 12px 12px; display: grid; gap: 8px; min-width: 0; }}
.kids {{ display: grid; gap: 6px; border-left: 2px solid var(--accent); padding-left: 10px; min-width: 0; }}
.kidlabel {{ font-size: 13px; color: var(--muted); }}
.row {{ display: grid; gap: 3px; min-width: 0; }}
.tag {{ font-size: 12px; color: var(--muted); letter-spacing: .02em; }}
.crops {{ display: flex; flex-wrap: wrap; gap: 4px; }}
figure {{ margin: 0; width: 132px; max-width: 100%; }}
figure img {{ display: block; width: 100%; height: auto; }}
figcaption {{ font: 400 11px var(--mono); color: var(--muted); overflow-wrap: anywhere; }}
.tree {{ display: grid; gap: 6px; min-width: 0; }}
</style>
<main>
<section>
<h1>ResNet-18 Tree Map</h1>
<p><b>The map is not clean yet, and your read decides. This page shows stage 4: 126 channels, each child nested under the parent it grew from, plus 9 parents from stage 3.</b> Stages 1 to 3 are not read yet. Every number is on sealed photos. These are ImageNet validation crops, not the photos the network trained on.</p>
</section>
<section>
<h2>The lines I fixed before reading</h2>
<div class="scroll"><table>
<tr><th>Line</th><th>Pass</th><th>Result</th><th>Verdict</th></tr>
<tr><td>Real parents: a blind reader picks the true parent of 5, from the words alone</td><td>70%</td><td>79%. 85% when all candidates come from one agent. 85% of 34 when a fresh agent wrote the parent's words without seeing any card. Last round: 57%, on links four convolutions apart. These links are one convolution apart, so they are easier, and the two numbers do not compare like with like</td><td class="good">Passes</td></tr>
<tr><td>Wrong cards lose: a card shown on another channel's photos</td><td>under .05</td><td>-.11</td><td class="good">Passes</td></tr>
<tr><td>Parents act: filling a named input moves the child 2 times more than controls</td><td>90%</td><td>100% of 369. This line is too easy, so it shows little</td><td class="mid">Passes, weak</td></tr>
<tr><td>Tree cover: the main parent passes that test</td><td>90%</td><td>88%. Last layer 100%, inner channels of the last block 77%</td><td class="bad">Narrow miss</td></tr>
<tr><td>Parents carry: the named inputs explain enough of the child</td><td>80%</td><td>67%. Last layer 93%, inner channels of the last block 47%</td><td class="bad">Fails</td></tr>
<tr><td>Words: a blind reader picks 12 firing crops of 24, from the words alone</td><td>85%</td><td>62%. Chance is about 11%. Strong channels 71 to 90%, weak channels 26 to 62%</td><td class="bad">Fails</td></tr>
</table></div>
</section>
<section>
<h2>Three things that go against this map</h2>
<p>1. On the same 30 top-layer channels, the old cards pass the words test at 77% and these at 70%. The new words are no better there. The gain is in the links.</p>
<p>2. In 80% of cases the main parent is simply the input my tool ranks first. So the numbers cannot show that the agents' judgment added anything.</p>
<p>3. The test with a fresh agent covers 34 strong links only, so it may flatter the tree.</p>
</section>
<section>
<h2>How to read the tree</h2>
<div class="scroll"><table>
<tr><th>Name</th><th>What it is</th></tr>
<tr><td class="mono">s4b1:179</td><td>Stream channel 179 of stage 4, block 1. This is the last layer before the logits</td></tr>
<tr><td class="mono">s4b1i:191</td><td>Inner channel 191 of the same block, between its two convolutions</td></tr>
<tr><td class="mono">s4b0, s4b0i</td><td>The same two kinds of channel, one block lower</td></tr>
<tr><td class="mono">s3b1</td><td>The last layer of stage 3. Named as parents here. Only 9 have a thin card</td></tr>
</table></div>
<p>Open a channel to see its photos, its card and its children. The photos are the network's real firings. A yellow ring marks where the channel fires hardest. The number under a photo is the firing, and the class is the network's top guess for that photo. The words on the cards are the agents' readings. The pills say what the sealed tests found.</p>
<p>The longest chain in this tree has {longest} channels. Parents sit at the outside and their children nest inside them, so the tree grows from lower layers up to the last layer.</p>
</section>
<section>
<h2>What I would like you to judge</h2>
<p>Is this your tree? Do the chains read as a parent that builds its children? When you open a weak channel, does it look wrong to you, or just hard to name? Your answer overrules every line above.</p>
</section>
<section>
<h2>Trees whose parent no agent has read yet ({len(order)})</h2>
<div class="tree">{trees}</div>
</section>
<section>
<h2>Trees that start at a channel an agent read ({len(with_kids)})</h2>
<div class="tree">{trees2}</div>
</section>
<section>
<h2>Channels with no parent and no children yet ({len(no_kids)})</h2>
<p class="muted">These are the open leads. An agent named no parent that it could defend, or read nothing that grew from it.</p>
<div class="tree">{trees3}{trees4}</div>
</section>
</main>
"""
open(sys.argv[1], "w").write(page)
print(len(cards), "nodes", len(seen), "placed; size MB", round(len(page) / 1e6, 2), "longest chain", longest, "outside parents", len(order), "inside trees", len(with_kids), "loners", len(no_kids), "left", len(left))

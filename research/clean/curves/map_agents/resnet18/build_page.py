# serves: H-map
"""Build the dropdown page Shonu reads to judge the map: families -> stages -> channels -> crops, card, tested links.
Reading photos only. Crops are small JPEGs inside the page. Usage: build_page.py OUTPUT.html"""
import base64
import glob
import html
import io
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net import CROP, HERE, STORE, STRIDE, Photos, class_names, parse  # noqa: E402

READ = os.path.join(STORE, "read")
RES = os.path.join(HERE, "results")
P = Photos("read")
cats = class_names()
y = np.load(os.path.join(READ, "y.npy"), mmap_mode="r")


def strip(s, c, n_top=8, n_neg=4):
    """A row of crops: strongest firings, then the negative pole."""
    b = 0 if s == 0 else 1
    tiles = []
    hx = np.asarray(np.load(os.path.join(READ, f"s{s}b{b}_hx.npy"), mmap_mode="r")[:, c], dtype=np.float32)
    hp = np.load(os.path.join(READ, f"s{s}b{b}_hp.npy"), mmap_mode="r")[:, c]
    picks = [(i, int(hp[i]), "t") for i in np.argsort(-hx)[:n_top]]
    if s > 0:
        zn = np.asarray(np.load(os.path.join(READ, f"s{s}b{b}_zn.npy"), mmap_mode="r")[:, c], dtype=np.float32)
        zp = np.load(os.path.join(READ, f"s{s}b{b}_zp.npy"), mmap_mode="r")[:, c]
        picks += [(i, int(zp[i]), "n") for i in np.argsort(zn)[:n_neg]]
    g, half, out = 224 // STRIDE[s], CROP[s] // 2, []
    raw = P.raw([p[0] for p in picks])
    for j, (i, pos, kind) in enumerate(picks):
        r, q = pos // g, pos % g
        cy, cx = int((r + .5) * STRIDE[s]), int((q + .5) * STRIDE[s])
        r0, c0 = min(max(cy - half, 0), 224 - 2 * half), min(max(cx - half, 0), 224 - 2 * half)
        im = Image.fromarray(raw[j][r0:r0 + 2 * half, c0:c0 + 2 * half]).resize((76, 76), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=62)
        out.append(dict(k=kind, c=cats[int(np.argmax(y[i]))], d="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()))
    return out


cards = {}
for f in sorted(glob.glob(os.path.join(HERE, "cards", "*.jsonl"))):
    for line in open(f):
        if line.strip():
            d = json.loads(line)
            d["agent"] = os.path.basename(f)[:-6]
            cards.setdefault(d["channel"], d)
fam = json.load(open(os.path.join(RES, "families.json")))
acts = json.load(open(os.path.join(RES, "acts_sealed.json")))
blind = {r["channel"]: r for r in json.load(open(os.path.join(RES, "blind_a.json")))}
famres = {r["name"]: r for r in json.load(open(os.path.join(RES, "family_sealed.json")))}
need = {r["child"]: r for r in acts["need"]}
links = {}
for r in acts["links"]:
    links.setdefault(r["child"], []).append(r)
data = {}
for ch, d in cards.items():
    s, c = parse(ch)
    data[ch] = dict(channel=ch, distinction=d["distinction"], positive=d["positive"], negative=d["negative"], gate=d.get("gate"), offshoots=d.get("offshoots"),
                    unread=d.get("unread"), sure=d.get("sure"), agent=d["agent"], gap=(blind.get(ch) or {}).get("gap"), parents=d.get("parents", []),
                    tests=links.get(ch, []), together=(need.get(ch) or {}).get("together"), z=(need.get(ch) or {}).get("z"), crops=strip(s, c))
page = dict(families=[dict(name=x["name"], pattern=x["pattern"], channels=x.get("main", x["channels"]), evidence=x.get("evidence"), sure=x.get("sure"), test=famres.get(x["name"])) for x in fam["families"]],
            no_family=fam["no_family"], shape=fam["shape"], channels=data)
tpl = open(os.path.join(HERE, "page_template.html")).read()
SCORES = [
    ["Specific enough to lose: a reviewer predicts firing from the card alone", "85% of channels", "44%. Stage 4 80%, stage 3 29%, stage 2 20%, stage 1 13%", "fails"],
    ["Real parents: a reviewer picks the true parent of 5 from cards alone", "70%", "57%. Chance is 20%", "fails"],
    ["Parents act: removing a parent moves the child 2 times more than matched controls", "most links", "92% of 119 links", "passes"],
    ["Families act: predict the top losing classes, from the words alone", "most of 15 families", "5 of 15. Mean 4.9 of 10 named, chance is 1", "fails"],
    ["One parent's effect on a stage 4 channel", "none set", "Median 3.8% of its firing. All listed parents together: 6%", "finding"],
    ["Keep only the top 64 of 256 feeders of a stage 4 channel", "none set", "77% of its squared firing changes", "finding"],
    ["Whole network, remove the vehicle cards, dogs and cats photos", "none set", "KL .090, agreement .885. Same-size random channels: .090 to .124", "finding"],
    ["Your read of this page", "you decide", "waiting for you", "open"],
]
out = tpl.replace("__SCORES__", json.dumps(SCORES)).replace("__DATA__", json.dumps(page).replace("</", "<\\/"))
open(sys.argv[1], "w").write(out)
print(sys.argv[1], round(len(out) / 1e6, 2), "MB", len(data), "channels")

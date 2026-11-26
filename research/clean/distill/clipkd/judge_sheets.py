"""CALC 3.6 Step 3 judge sheets (plan id-like-you-to-fizzy-cherny). SCORE photos only (positions 4-7).

Per group g and version v (A = ranks 1-12, B = ranks 13-24 of each target unit):
  targets: 12 top SCORE crops of member m2 + 12 of member m3 (24)
  distractors: the top SCORE crop of 24 pool neurons of the same blocks (A and B use different pool neurons)
  48 tiles, shuffled, numbered 1-48. Tile = crop above the whole photo with the patch boxed (as the reader sheets).
Words per condition: S = the student's words, T = the best single teacher neuron's words, X = a shuffled control
(the student words of group (g + 5) % 10). Judge j (0-5) sees version j // 3 and condition (j + g) % 3 for group g,
so each judge sees each group once.
Output: judge/sheet_g{g}{v}.png, judge/ANSWER.json (judges never open it), judge/ASSIGN.json
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crops import T, tile  # noqa: E402
from score_clip import OUT, net  # noqa: E402

JD = os.path.join(HERE, "judge")


def main():
    P = dict(np.load(os.path.join(OUT, "poles.npz")))
    ph = net.Photos("read")
    role, grp, blk = P["unit_role"], P["unit_group"], P["unit_block"]
    sc = np.where(P["pos"] >= 4)[0]
    G, tok, ids = P["gmax"], P["gtok"], P["photo_ids"]
    os.makedirs(JD, exist_ok=True)
    rng = np.random.default_rng(17)
    ans, assign = {}, {f"judge{j}": [] for j in range(6)}
    for g in range(10):
        m2 = np.where((grp == g) & (role == "m2"))[0][0]
        m3 = np.where((grp == g) & (role == "m3"))[0][0]
        pool = np.where((role == "random") & np.isin(blk, [blk[m2], blk[m3]]))[0]
        pool = rng.permutation(pool)
        for vi, v in enumerate("AB"):
            items, used = [], set()
            for m in (m2, m3):
                o = sc[np.argsort(-G[sc, m])]
                o = [r for r in o if r not in used][12 * vi:12 * vi + 12]
                used |= set(o)
                items += [(r, tok[r, m], 1) for r in o]
            for p in pool[24 * vi:24 * vi + 24]:
                r = next(r for r in sc[np.argsort(-G[sc, p])] if r not in used)
                used.add(r)
                items.append((r, tok[r, p], 0))
            items = [items[i] for i in rng.permutation(len(items))]
            th = T + T // 2 + 4
            im = Image.new("RGB", (8 * (T + 4), 6 * (th + 18)), "white")
            d = ImageDraw.Draw(im)
            raws = ph.raw(ids[[it[0] for it in items]])
            for n, (img, it) in enumerate(zip(raws, items)):
                x, y = (n % 8) * (T + 4), (n // 8) * (th + 18)
                d.text((x + 2, y + 2), str(n + 1), fill="black")
                im.paste(tile(img, it[1]), (x, y + 16))
            im.save(os.path.join(JD, f"sheet_g{g}{v}.png"))
            ans[f"g{g}{v}"] = [n + 1 for n, it in enumerate(items) if it[2] == 1]
        for j in range(6):
            assign[f"judge{j}"].append(dict(sheet=f"sheet_g{g}{'AB'[j // 3]}.png", group=g, cond="STX"[(j + g) % 3]))
    json.dump(ans, open(os.path.join(JD, "ANSWER.json"), "w"))
    json.dump(assign, open(os.path.join(JD, "ASSIGN.json"), "w"), indent=1)
    print("ok", len(ans))


if __name__ == "__main__":
    main()

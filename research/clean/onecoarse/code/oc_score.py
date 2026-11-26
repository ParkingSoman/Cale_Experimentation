"""Floor B scorer: precision at 12 for own words and control words, per neuron (judge/answers vs judge/truth).
After scoring it also writes, for the reader, one marked sheet per neuron (results/miss_<n>.png: green = a target
the judge picked, yellow = a target it missed, red = a non-target it picked) and results/judge_score.json with the
hit, miss and wrong-pick lists. Only this scorer reads the key; the reader reads its outputs."""
import json
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
J = os.path.join(HERE, "..", "judge")
OUT = os.path.join(HERE, "..", "results")
S = 120
print(f"{'neuron':7s} {'own':>5s} {'ctrl':>5s} {'gap':>5s}  B1(>=.75) B2(gap>=.3)")
CODE = {v: k for k, v in json.load(open(os.path.join(J, "truth", "codes.json"))).items()}
report = {}
for f in sorted(os.listdir(os.path.join(J, "truth"))):
    if f in ("codes.json", "judges.json"):
        continue
    t = json.load(open(os.path.join(J, "truth", f))); n = t["neuron"]; T = set(t["targets"])
    sc, picks = {}, {}
    for kind in ("own", "ctrl"):
        p = os.path.join(J, "answers", f"{CODE[n + '_' + kind]}.json")
        picks[kind] = set(json.load(open(p))["ranked"][:12]) if os.path.exists(p) else set()
        sc[kind] = len(picks[kind] & T) / 12 if os.path.exists(p) else float("nan")
    g = sc["own"] - sc["ctrl"]
    print(f"{n:7s} {sc['own']:5.2f} {sc['ctrl']:5.2f} {g:+5.2f}  {'pass' if sc['own'] >= .75 else 'FAIL'}      {'pass' if g >= .3 else 'FAIL'}")
    own = picks["own"]
    report[n] = dict(own=sc["own"], ctrl=sc["ctrl"], ctrl_from=t.get("control_words_from"), hits=sorted(own & T), missed=sorted(T - own), wrong=sorted(own - T))
    img = Image.open(os.path.join(J, "packets", f"{n}.png")).convert("RGB")
    dr = ImageDraw.Draw(img)
    for num in range(1, 37):
        x0, y0 = ((num - 1) % 6) * S, ((num - 1) // 6) * (S + 16)
        col = (0, 200, 0) if num in T and num in own else (255, 210, 0) if num in T else (230, 0, 0) if num in own else None
        if col:
            dr.rectangle((x0 + 1, y0 + 1, x0 + S - 2, y0 + S + 14), outline=col, width=4)
    img.save(os.path.join(OUT, f"miss_{n}.png"))
json.dump(report, open(os.path.join(OUT, "judge_score.json"), "w"), indent=1)

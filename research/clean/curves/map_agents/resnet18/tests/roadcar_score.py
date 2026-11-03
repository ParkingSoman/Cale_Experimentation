# serves: H-map
"""roadcar: predictions written BEFORE the run, then ev.py score on class names from each channel's one-line description.
Each entry is (channel, classes the description says should fire, classes it says should not).
High classes are mostly NOT on the sheets I looked at, so the test is not circular."""
import subprocess
import torchvision

CAT = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
ID = {n: i for i, n in enumerate(CAT)}
P = [
    ("s4:32", ["submarine", "bullet train", "killer whale", "limousine"], ["barn", "plunger", "cello", "isopod"]),
    ("s4:400", ["cauliflower", "artichoke", "head cabbage"], ["rotisserie", "impala", "ski"]),
    ("s4:461", ["gondola", "canoe", "violin", "speedboat"], ["soap dispenser", "mask", "trombone"]),
    ("s4:269", ["school bus", "hamster", "beaver", "mink"], ["cannon", "fountain pen", "Weimaraner", "projectile"]),
    ("s4:344", ["flamingo", "parking meter", "impala", "white stork"], ["trombone", "oboe", "Bedlington terrier"]),
    ("s4:486", ["warplane", "black stork", "kite", "albatross"], ["padlock", "waffle iron", "gasmask"]),
    ("s4:329", ["suspension bridge", "tripod", "harp", "catamaran"], ["coyote", "tabby", "Siamese cat", "wood rabbit"]),
    ("s4:277", ["fireboat", "armadillo", "planetarium", "radio telescope"], ["analog clock", "wall clock", "water ouzel"]),
    ("s4:403", ["revolver", "oboe", "assault rifle", "harmonica"], ["analog clock", "steel drum", "ladle", "vault"]),
    ("s4:367", ["abaya", "gasmask", "mortarboard", "backpack"], ["killer whale", "dragonfly", "rhinoceros beetle", "airliner"]),
]
for ch, hi, lo in P:
    h = ",".join(str(ID[n]) for n in hi)
    l = ",".join(str(ID[n]) for n in lo)
    r = subprocess.run([".venv/bin/python", "map_agents/resnet18/ev.py", "score", ch, "--high", h, "--near", l],
                       capture_output=True, text=True)
    print(ch, "|", " ".join((r.stdout + r.stderr).strip().splitlines()[-3:]), flush=True)

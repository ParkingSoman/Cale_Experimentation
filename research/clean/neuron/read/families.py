"""N10 step 1: a partition of the 1,000 ImageNet classes into WordNet families. Fixed before any gap number is seen.
Each class goes to the FIRST family in FAMS whose synset is an ancestor of the class (any hypernym path). Rest = 'other artifact' / 'other'.
Also writes, per class pair, nothing; the 'near' test uses lca_up(): steps from the true class up to the lowest common ancestor.
WordNet data: nltk 'wordnet' (downloaded to the session scratchpad; set NLTK_DATA)."""
import json, os, sys
from nltk.corpus import wordnet as wn
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "results", "read")
IDX = json.load(open(os.path.join(OUT, "imagenet_class_index.json")))
WNID = [IDX[str(i)][0] for i in range(1000)]
# order matters: specific before general
FAMS = [
    ("dogs", "dog.n.01"), ("other canines", "canine.n.02"), ("cats", "feline.n.01"), ("primates", "primate.n.02"),
    ("other mammals", "mammal.n.01"), ("birds", "bird.n.01"), ("reptiles+amphibians", "reptile.n.01"),
    ("reptiles+amphibians", "amphibian.n.03"), ("fish", "fish.n.01"), ("insects+spiders", "arthropod.n.01"),
    ("other animals", "animal.n.01"), ("food+plants", "food.n.02"), ("food+plants", "food.n.01"),
    ("food+plants", "plant_part.n.01"), ("food+plants", "plant.n.02"), ("food+plants", "fungus.n.01"),
    ("nature+places", "geological_formation.n.01"), ("people", "person.n.01"),
    ("vehicles+craft", "conveyance.n.03"), ("clothing+covering", "covering.n.02"), ("containers", "container.n.01"),
    ("structures", "structure.n.01"), ("furniture+furnishing", "furnishing.n.02"),
    ("devices+instruments", "device.n.01"), ("devices+instruments", "instrument.n.01"),
    ("tools+implements", "implement.n.01"), ("other artifact", "artifact.n.01"),
]
def syn(i):
    w = WNID[i]; return wn.synset_from_pos_and_offset(w[0], int(w[1:]))
def ancestors(s):
    return {a for p in s.hypernym_paths() for a in p}
def lca_up(i, j):
    """Fewest steps up from class i to a common ancestor with class j (over all paths)."""
    a, b = syn(i), syn(j)
    best = 99
    for l in a.lowest_common_hypernyms(b):
        best = min(best, min(len(p) - 1 - p.index(l) for p in a.hypernym_paths() if l in p))
    return best
if __name__ == "__main__":
    fam = []
    for i in range(1000):
        anc = ancestors(syn(i)); f = "other"
        for name, s in FAMS:
            if wn.synset(s) in anc: f = name; break
        fam.append(f)
    sys.path.insert(0, os.path.join(HERE, "..", "tree")); import common as TC
    g = TC.groups(); names = TC.net.class_names()
    dogs = [i for i in range(1000) if fam[i] == "dogs"]
    print("dogs == 151..268:", dogs == list(range(151, 269)), len(dogs))
    birds = sorted(i for i in range(1000) if fam[i] == "birds")
    print("birds == N5:", birds == sorted(g["birds"]), len(birds), len(g["birds"]), set(birds) ^ set(g["birds"]))
    print("N5 vehicles' families:", {fam[i] for i in g["vehicles"]})
    from collections import Counter
    for k, v in Counter(fam).most_common(): print(f"{v:4d} {k}")
    for k in ("other", "other artifact", "other animals"):
        print(k, ":", ", ".join(names[i] for i in range(1000) if fam[i] == k)[:600])
    json.dump(dict(families=fam, rule=FAMS), open(os.path.join(OUT, "families.json"), "w"))

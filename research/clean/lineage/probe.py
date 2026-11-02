"""Small reasoning / recitation probe, run next to dCE. Score = share of target tokens that are the top-1 guess
(teacher forced). Recitation: famous public-domain passages, scored after a short prefix. Arithmetic: few-shot
single-digit sums, scored on the answer token only."""
import numpy as np
import mlx.core as mx

RECITE = [
    "We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of America.",
    "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.",
    "The Lord is my shepherd; I shall not want. He maketh me to lie down in green pastures: he leadeth me beside the still waters. He restoreth my soul: he leadeth me in the paths of righteousness for his name's sake.",
    "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles",
    "In the beginning God created the heaven and the earth. And the earth was without form, and void; and darkness was upon the face of the deep.",
    "I pledge allegiance to the Flag of the United States of America, and to the Republic for which it stands, one Nation under God, indivisible, with liberty and justice for all.",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity",
    "One, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, nineteen, twenty",
]
PREFIX = 6


def _arith(rng, n=40):
    out = []
    for _ in range(n):
        shots = []
        for _ in range(4):
            a, b = rng.integers(0, 10, 2); shots.append(f"{a} + {b} = {a + b}")
        a, b = rng.integers(0, 10, 2)
        out.append(("\n".join(shots) + f"\n{a} + {b} =", f" {a + b}"))
    return out


def probe(model, tok):
    rec = []
    for s in RECITE:
        ids = tok.encode(s)
        x = mx.array(np.array(ids, dtype=np.int32)[None])
        p = np.array(mx.argmax(model(x[:, :-1]), -1))[0]
        rec.append(float((p[PREFIX - 1:] == np.array(ids[PREFIX:])).mean()))
    ok = []
    for q, a in _arith(np.random.default_rng(0)):
        qi, ai = tok.encode(q), tok.encode(a)
        x = mx.array(np.array(qi, dtype=np.int32)[None])
        ok.append(float(int(mx.argmax(model(x)[0, -1])) == ai[0]))
    return {"recite": float(np.mean(rec)), "recite_each": rec, "arith": float(np.mean(ok))}


if __name__ == "__main__":
    from common import load_pythia
    m, tok = load_pythia()
    print(probe(m, tok))

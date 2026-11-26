"""LABELLED READING CHECK (photos, off the board; run with --quick only). Row d with the read scale (mu, sd, S_M) and
the gate quantile taken from 64 HELD photos instead of pink. Tree, kept set and P4 fold stay on pink X.
Photo use: 64 held photos inside the calculation, for this check only."""
import os
import sys

from rules import r2_p1p3_merge

_H = {}


def _held():
    if "X" not in _H:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "clipkd"))
        import eval_fast
        ref = eval_fast.teacher_ref("held")
        _H["X"] = eval_fast.photos(ref["ids"][:64]).float()
    return _H["X"]


def vit(T, f, X=None):
    return r2_p1p3_merge.build(T, f, X, tree="r1rms", gate="or", tag="photo-d", statX=_held())

"""L3 drift law (kept nodes one layer down carry what deleted nodes no longer write): mode="full", hfold=True, morder="root", drift=("mlp",). See rules/l1_stream.py. Photo use: none."""
from rules import l1_stream as L

vit = L.make(mode="full", hfold=True, morder="root", drift=("mlp",))

#!/usr/bin/env python3
"""Make a Cale-layout copy of a Gemma 3n GGUF: the same model, with each FFN `down` matrix stored as two halves.

usage: python llama/prepare_gguf.py IN.gguf OUT.gguf

Why: a Cale size uses the first half of the feed-forward channels, or all of them. For the `gate` and `up` matrices the
first half is a block of whole rows, so a size is a plain view. The `down` matrix is cut the other way (by columns), so
its first half is spread over every row. Stored as two halves (a 3-D tensor, 2 x rows x half-columns) each half is one
block. Then llama.cpp can keep its fast weight repacking on, and a smaller size can free the second half.

Every other tensor and all metadata are copied unchanged. The key `cale.ffn_down_split` = 1 is added.
The numbers do not change: this only reorders bytes. Needs the `gguf` and `numpy` packages (pip install gguf).
"""
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:                                           # use the copy inside a llama.cpp checkout if there is one
    for cand in Path(__file__).resolve().parents:
        if (cand / "gguf-py" / "gguf").exists():
            sys.path.insert(0, str(cand / "gguf-py"))
            break
    import gguf

KEY = "cale.ffn_down_split"


def halves(data, n_cols_per_half_bytes):
    """data: (rows, row_bytes) bytes of a quantized matrix. Returns (2, rows, row_bytes // 2): the left and right half of every row."""
    rows, row_bytes = data.shape
    assert row_bytes % 2 == 0 and row_bytes // 2 == n_cols_per_half_bytes
    return np.ascontiguousarray(data.reshape(rows, 2, row_bytes // 2).transpose(1, 0, 2))


def convert(src, dst):
    reader = gguf.GGUFReader(str(src))
    if reader.get_field(KEY) is not None:
        sys.exit(f"{src} is already in the Cale layout.")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    if arch != "gemma3n":
        sys.exit(f"This is for Gemma 3n files. {src} is '{arch}'.")
    writer = gguf.GGUFWriter(str(dst), arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    writer.add_uint32(KEY, 1)

    plan = []
    for t in reader.tensors:
        if t.name.endswith(".ffn_down.weight"):
            n_ff, n_embd = int(t.shape[0]), int(t.shape[1])              # GGUF order: columns first
            assert n_ff % 2 == 0 and n_ff // 2 % 256 == 0, (t.name, n_ff)  # a half must hold whole quantization blocks
            data = t.data
            assert data.ndim == 2 and data.shape[0] == n_embd, (t.name, data.shape)
            new_shape = (2, data.shape[0], data.shape[1] // 2)           # numpy order: (halves, rows, bytes)
            writer.add_tensor_info(t.name, new_shape, data.dtype, data.nbytes, t.tensor_type)
            plan.append((t, True))
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            plan.append((t, False))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    done = 0
    for t, split in plan:
        writer.write_tensor_data(halves(t.data, t.data.shape[1] // 2) if split else t.data, tensor_endianess=reader.endianess)
        done += t.n_bytes
        print(f"\r{done / 1e9:.2f} GB written", end="", flush=True)
    writer.close()
    print()


def check(src, dst):
    """Read the new file back: shapes are as expected and the bytes of one matrix match."""
    a, b = gguf.GGUFReader(str(src)), gguf.GGUFReader(str(dst))
    assert b.get_field(KEY).contents() == 1
    ta = {t.name: t for t in a.tensors}
    tb = {t.name: t for t in b.tensors}
    assert ta.keys() == tb.keys()
    n_split = 0
    for name, x in ta.items():
        y = tb[name]
        if name.endswith(".ffn_down.weight"):
            n_split += 1
            assert tuple(y.shape) == (x.shape[0] // 2, x.shape[1], 2), (name, y.shape, x.shape)
            rows = x.data.shape[0]
            left, right = y.data.reshape(2, rows, -1)
            assert np.array_equal(np.concatenate([left, right], axis=1), x.data), name
        else:
            assert tuple(y.shape) == tuple(x.shape) and y.tensor_type == x.tensor_type, name
    print(f"checked: {n_split} down matrices split, {len(ta) - n_split} other tensors unchanged")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    convert(sys.argv[1], sys.argv[2])
    check(sys.argv[1], sys.argv[2])

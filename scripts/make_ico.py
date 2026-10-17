#!/usr/bin/env python3
"""Make a Windows .ico file with the standard library only.

  python scripts/make_ico.py dist/cale.ico                      -> Cale's kale-leaf icon
  python scripts/make_ico.py out.ico --bg 61,107,214            -> the same drawing in another colour (an app icon)
  python scripts/make_ico.py out.ico --png icon.png             -> turn an existing 1024 px PNG into an .ico

An .ico file can hold PNG pictures (Windows Vista and newer). So this draws the icon with build_installer.make_icon,
shrinks it to several sizes, and packs the PNGs into one file.
"""
import argparse
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SIZES = (256, 128, 64, 32, 16)                    # every size is a power of two below 1024, so each step is an exact halving


def read_png(data):
    """Read an 8-bit RGBA PNG without interlacing (any row filter). Returns (width, height, rows of RGBA bytes)."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos, idat, w = 8, b"", 0
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if (depth, ctype, interlace) != (8, 6, 0):
                raise ValueError("only 8-bit RGBA PNG files without interlacing are supported")
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    stride, bpp = w * 4, 4
    rows, prev = [], bytearray(stride)
    pos = 0
    for y in range(h):
        ftype = raw[pos]
        line = bytearray(raw[pos + 1:pos + 1 + stride])
        pos += 1 + stride
        if ftype == 1:                                  # Sub
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 255
        elif ftype == 2:                                # Up
            line = bytearray((x + u) & 255 for x, u in zip(line, prev))
        elif ftype == 3:                                # Average
            for i in range(stride):
                a_ = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a_ + prev[i]) >> 1)) & 255
        elif ftype == 4:                                # Paeth
            for i in range(stride):
                a_ = line[i - bpp] if i >= bpp else 0
                b_ = prev[i]
                c_ = prev[i - bpp] if i >= bpp else 0
                pa, pb, pc = abs(b_ - c_), abs(a_ - c_), abs(a_ + b_ - 2 * c_)
                pred = a_ if pa <= pb and pa <= pc else (b_ if pb <= pc else c_)
                line[i] = (line[i] + pred) & 255
        elif ftype != 0:
            raise ValueError("unknown PNG filter")
        rows.append(bytes(line))
        prev = line
    return w, h, rows


def write_png(w, h, rows):
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(r) for r in rows)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def halve(w, rows):
    """Shrink to half size: each new pixel is the average of a 2x2 block, weighted by opacity so clear corners stay clean."""
    out = []
    for y in range(0, len(rows), 2):
        a, b = rows[y], rows[y + 1]
        r = bytearray(w // 2 * 4)
        alpha = [p + q + s + t for p, q, s, t in zip(a[3::8], a[7::8], b[3::8], b[7::8])]
        for c in range(3):
            summed = [p * pa + q * qa + s * sa + t * ta
                      for p, pa, q, qa, s, sa, t, ta in zip(a[c::8], a[3::8], a[c + 4::8], a[7::8], b[c::8], b[3::8], b[c + 4::8], b[7::8])]
            r[c::4] = bytes((v // al if al else 0) for v, al in zip(summed, alpha))
        r[3::4] = bytes((al + 2) // 4 for al in alpha)
        out.append(bytes(r))
    return w // 2, out


def pack_ico(images):
    """images: list of (size, png bytes). Returns the bytes of an .ico file."""
    head = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, blobs = b"", b""
    for size, png in images:
        entries += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(png), offset + len(blobs))
        blobs += png
    return head + entries + blobs


def ico_from_png(png_bytes):
    w, h, rows = read_png(png_bytes)
    if w != h or w & (w - 1):
        raise ValueError("the picture must be square, with a power-of-two size")
    made = {}
    while w >= 16:
        if w <= 256:
            made[w] = write_png(w, w, rows)
        if w == 16:
            break
        w, rows = halve(w, rows)
    return pack_ico([(s, made[s]) for s in SIZES if s in made])


def make_ico(path, **icon_args):
    """Draw Cale's icon (build_installer.make_icon, with the same arguments) and save it as an .ico at `path`."""
    import tempfile
    import build_installer as bi
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        png = Path(d) / "icon.png"
        bi.make_icon(png, **icon_args)
        path.write_bytes(ico_from_png(png.read_bytes()))
    return path


def color(text):
    return tuple(int(v) for v in text.split(","))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--png", help="use this 1024 px PNG instead of drawing")
    ap.add_argument("--glyph", default="kale")
    ap.add_argument("--bg", type=color, default=(47, 93, 80))
    ap.add_argument("--fg", type=color, default=(247, 245, 238))
    a = ap.parse_args()
    if a.png:
        Path(a.out).write_bytes(ico_from_png(Path(a.png).read_bytes()))
    else:
        make_ico(a.out, bg=a.bg, fg=a.fg, glyph=a.glyph)
    print("built:", a.out)


if __name__ == "__main__":
    main()

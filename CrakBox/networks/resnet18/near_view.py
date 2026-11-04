"""near NODE: what looks like this node and leaves it quiet. Reading examples only. The top row is where this node fires. The rows below are strong places of its most similar nodes, where this node stays under a fifth of its largest firing.
Usage: near_view.py NODE FOLDER. Prints a PNG path, then the similar nodes."""
import importlib.util
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("mapper_command", os.path.join(ROOT, "mapper.py"))
cmd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmd)
from mapper import judge  # noqa: E402

SPLIT = "read"
FIRES, NEAR, COLS, TILE = 6, 18, 6, 110


def main(node, folder):
    net = cmd.get_net("resnet18")
    v, p = net.peaks(node, "pos", SPLIT)
    big = float(v.max())
    top = np.argsort(-v)[:FIRES]
    similar = net.similar(node, SPLIT, 12)
    near = judge._others(net, node, "pos", similar, SPLIT, big, NEAR, {int(i) for i in top})
    if near is None:
        print(f"{node}: its similar nodes have too few places where it is quiet")
        return
    shown = [(int(i), int(p[i]), f"fires {v[i] / big:.2f}") for i in top] + [(i, pos, f"quiet {share:.2f}") for i, pos, share in near]
    tiles = net.render(SPLIT, node, np.array([s[0] for s in shown]), np.array([s[1] for s in shown]), TILE)
    rows = int(np.ceil(len(tiles) / COLS))
    sheet = Image.new("RGB", (COLS * (TILE + 4), rows * (TILE + 16)), "white")
    d = ImageDraw.Draw(sheet)
    for k, (tile, s) in enumerate(zip(tiles, shown)):
        x, y = (k % COLS) * (TILE + 4), (k // COLS) * (TILE + 16)
        sheet.paste(tile, (x, y + 14))
        d.text((x + 1, y + 1), s[2], fill="black" if k >= FIRES else "red")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, node.replace(":", "_") + ".png")
    sheet.save(path)
    print(f"{path}: top row, where this node fires. Other rows: strong places of similar nodes, where this node stays quiet. The number is its firing there, as a share of its largest.")
    print("similar nodes, the closest first:", ", ".join(similar))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

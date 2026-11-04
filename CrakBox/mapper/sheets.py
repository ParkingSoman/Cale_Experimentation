"""One code pass per node, with no tokens. The features select a workflow. The tools' outputs are stored once, so a reader does not pay to fetch them again."""
import json
import os
import re

import numpy as np

STRONG = 64


def route_split(D, seed=0):
    """How far apart this node's strong firings are, by what feeds them. 0: one route. Near 1: routes share almost nothing.
    Local clustering inside one node, as evidence only. It makes no group of nodes."""
    x = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), 3, replace=False)]
    lab = np.zeros(len(x), dtype=int)
    for _ in range(25):
        lab = np.argmax(x @ centers.T, 1)
        new = [x[lab == k].mean(0) if (lab == k).any() else centers[k] for k in range(3)]
        centers = np.stack(new) / (np.linalg.norm(new, axis=1, keepdims=True) + 1e-9)
    big = [k for k in range(3) if (lab == k).sum() >= 8]
    if len(big) < 2:
        return 0.0
    return float(1 - max(centers[a] @ centers[b] for a in big for b in big if a < b))


def features(net, node, split="read"):
    v, p = net.peaks(node, "pos", split)
    largest = float(v.max())
    on = v > .01 * largest
    top = np.argsort(-v)[:STRONG]
    values, counts = np.unique(p[top], return_counts=True)
    labels = net.example_labels(split, top)
    readers = net.readers(node, split, 10)
    out = dict(
        node=node,
        largest=largest,
        on_share=float(on.mean()),
        # low: a detector, where few firings reach the strong range. High: a dial, where many do. Survey: about .02 late, .2 to .3 early
        cliff=float((v >= .5 * largest).sum() / max(int((v >= .1 * largest).sum()), 1)),
        output_effect=float(net.output_effect(node, split, top[:24])),
        strength=largest / max(float(np.median(net.largest_in_layer(net.layer_of(node), split))), 1e-9),
        position_share=float(counts.max() / len(top)),
        label_spread=float(len(set(labels)) / len(top)) if labels is not None else 0.0,
        reader_mix=float(sum(s < 0 for _, s in readers) / max(len(readers), 1)),
        route_split=0.0,
        has_inputs=net.has_inputs(node),
        inputs_for_half=0,
        n_inputs=0,
        top_input_share=0.0,
        top_input="",
    )
    if net.has_inputs(node):
        D, names, _, _ = net.inputs(node, split, top, p[top])
        weight = np.abs(D).mean(0)
        order = np.argsort(-weight)
        cum = np.cumsum(weight[order]) / max(float(weight.sum()), 1e-12)
        out.update(inputs_for_half=int(np.searchsorted(cum, .5)) + 1, n_inputs=len(names), top_input_share=float(weight[order[0]] / max(float(weight.sum()), 1e-12)), top_input=names[order[0]], route_split=route_split(D))
    if "neg" in net.poles(node):
        vn, _ = net.peaks(node, "neg", split)
        out["neg_ratio"] = float(vn.max() / max(largest, 1e-9))
    else:
        out["neg_ratio"] = 0.0
    return out


def link_candidates(net, node, split="read", count=10):
    """Code proposes links. A reader may overrule one with evidence."""
    if not net.has_inputs(node):
        return []
    v, p = net.peaks(node, "pos", split)
    top = np.argsort(-v)[:STRONG]
    D, names, _, _ = net.inputs(node, split, top, p[top])
    weight = np.abs(D).mean(0)
    mean = D.mean(0)
    order = np.argsort(-weight)[:count]
    return [dict(node=names[j], effect=float(weight[j]), signed=float(mean[j])) for j in order]


def file_name(node):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", node) + ".json"


def write(net, node, folder, split="read"):
    os.makedirs(folder, exist_ok=True)
    sheet = dict(features=features(net, node, split), inputs=link_candidates(net, node, split), readers=[dict(node=n, signed=float(s)) for n, s in net.readers(node, split, 10)])
    path = os.path.join(folder, file_name(node))
    with open(path, "w") as f:
        json.dump(sheet, f, indent=1)
    return path, sheet

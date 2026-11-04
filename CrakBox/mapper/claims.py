"""Each understood node writes its claims. A parent claim says: this node's value does not change when the layer below carries only its means outside a set Q.
Its error is e / V, as in the mean-fill parent claims of the curves paper: e is the mean squared change, V the variance of the value. Code computes it. No agent's word enters."""
import numpy as np


def parent_claim(net, node, q, split, examples, positions):
    """Over given examples and positions. A caller that samples positions gets an estimate, and the record says so.
    Inputs the architecture keeps stay at the network's values, whatever Q holds."""
    D, names, _, true = net.inputs(node, split, examples, positions)
    q = set(q)
    missing = [n for n in q if n not in names]
    if missing:
        raise ValueError(f"{node}: not direct inputs: {missing}")
    out = [j for j, n in enumerate(names) if n not in q and not net.input_kept(n)]
    e = float((D[:, out].sum(1) ** 2).mean())
    v = float(true.var())
    return dict(type="parent", node=node, q=sorted(q), eps=e / v if v > 0 else e, e=e, v=v, examples=len(true))


def sampled_parent_claim(net, node, q, split, count, seed=0):
    """One random position per example. An estimate of the mean over all positions, and it is marked as one."""
    rng = np.random.default_rng(seed)
    n = len(net.peaks(node, "pos", split)[0])
    examples = np.sort(rng.choice(n, min(count, n), replace=False))
    positions = rng.integers(0, net.n_positions(node), len(examples))
    return dict(parent_claim(net, node, q, split, examples, positions), estimate=True, split=split)

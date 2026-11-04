"""Code chooses a scope from a goal. An agent checks the result. It groups nothing: it follows exact inputs down from the outputs, by effect on the goal.

Start: the output layer's nodes, by what they write to the goal's own outputs. Then one layer at a time: each chosen node gives credit to its inputs, in proportion to their exact effect on it, at its strongest places on the goal's examples.
Keep the top nodes per layer. Then close the set: each kept node's strongest input is kept too, so that a parent test can be built."""
import numpy as np

EXAMPLES = 32


def strongest_input(net, node, goal_examples, split="read"):
    v, p = net.peaks(node, "pos", split)
    ex = goal_examples[np.argsort(-v[goal_examples])[:EXAMPLES]]
    D, names, _, _ = net.inputs(node, split, ex, p[ex])
    return names, np.abs(D).mean(0)


def trace(net, goal_examples, per_layer, split="read", log=print):
    """Returns (scope nodes in order from the output, scores). Exact inputs only. No training and no agents."""
    layers = net.layers()
    scores = {layers[0]: net.output_scores(goal_examples, split)}
    chosen = {}
    for layer in layers:
        here = scores.get(layer, {})
        top = sorted(here, key=lambda n: -here[n])[:per_layer]
        chosen[layer] = top
        total = sum(here[n] for n in top) or 1.0
        for node in top:
            if not net.has_inputs(node):
                continue
            names, weight = strongest_input(net, node, goal_examples, split)
            share = weight / max(float(weight.sum()), 1e-12)
            for name, s in zip(names, share):
                real = net.canonical(name)
                target = net.layer_of(real)
                scores.setdefault(target, {})
                scores[target][real] = scores[target].get(real, 0.0) + here[node] / total * float(s)
        log(f"{layer}: kept {len(top)}")
    # closure: a kept node's strongest input is kept, down to the raw input
    kept = {n for nodes in chosen.values() for n in nodes}
    added = []
    for layer in layers:
        for node in list(chosen.get(layer, [])):
            if not net.has_inputs(node):
                continue
            names, weight = strongest_input(net, node, goal_examples, split)
            parent = net.canonical(names[int(np.argmax(weight))])
            if parent not in kept:
                kept.add(parent)
                added.append(parent)
                chosen[net.layer_of(parent)].append(parent)
        log(f"{layer}: {len(chosen[layer])} after closure")
    nodes = [n for layer in layers for n in chosen[layer]]
    return nodes, dict(per_layer=per_layer, added_by_closure=len(added), counts={layer: len(chosen[layer]) for layer in layers})

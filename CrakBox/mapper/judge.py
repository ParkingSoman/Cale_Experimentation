"""The blind judge: packets, wrong cards, the key, scoring and pass marks. Rules: CONSTITUTION.md sections 5 and 6.

A pole sheet shows 36 items from sealed examples, numbered 1 to 36.
   12 fire    a random sample of ALL firings at or above half of the largest. Not the top ones only, so an unrelated part of the job is in the pool and a narrow card gets caught.
   12 near    the peaks of the most similar nodes, where this node is quiet.
   12 random  the peaks of random nodes, where this node is quiet.
From words alone, a judge ranks the 12 items most likely to fire, strongest first. Code scores three things from that one answer:
   center  the firing reached by the first 6, as a share of the largest, against random picks and the best 6 on this sheet.
   edge    the share of fire items in the 12, minus the share of near items. Near misses must stay out.
   far     the share of fire items in the 12, minus the share of random items. It also falls when a card misses part of the job, since fire items are a sample of all strong firings.
So a judge's answer is scored by the network and not by a person. An earlier form asked judges to name near misses and far items. On 80 items it did not separate real words from wrong words (LOG, section 3), so it was replaced.

Verdicts go by judges' votes and never by a median across judges. Two judges must agree. A split waits for a third. One answer decides nothing.

Each item in a packet has a random id. The key stays apart, under sealed_keys, and only this module reads it."""
import hashlib
import json
import os
import random
import re
import secrets

import numpy as np
from PIL import Image, ImageDraw

N_FIRE, N_NEAR, N_RANDOM = 12, 12, 12
N_TOP = 6
N_RANK = 12
N_ORDER = 8
BATTERY_FLOOR = .25
QUANTILES = (.5, .6, .7, .75, .8, .85, .9, .95)
QUIET = .2
STRONG_SHARE = .02
SHEET_FORM = 3
FIRE_FLOOR = .3
TILE = 112
LINK_CHOICES = 8
STRONG_LINKS = 8  # a node's strongest links in one direction. These are not offered as wrong choices, since they are real
WRONG_SHARE_OF_ITEMS = .25
MAX_FALSE_PASS = .10
REQUIRED = ("pole:pos", "pole:neg", "parent", "reader", "critic")


def strong_links(net, kind, node, count=STRONG_LINKS):
    """A node's strongest links by code, the strongest first. kind is "parent" or "reader"."""
    if kind == "reader":
        return [net.canonical(n) for n, _ in net.readers(node, "read", count)]
    if not net.has_inputs(node):
        return []
    v, p = net.peaks(node, "pos", "read")
    examples = np.argsort(-v)[:32]
    D, names, _, _ = net.inputs(node, "read", examples, p[examples])
    return [net.canonical(names[j]) for j in np.argsort(-np.abs(D).mean(0))[:count]]


class ShortPool(Exception):
    pass


class Leak(Exception):
    pass


class BadAnswer(ValueError):
    pass


def leaks(packet, literal, patterns=()):
    """Names found in a packet's words. A judge must not see a node's name, its layer or any other position in the network. Case does not hide a name."""
    text = json.dumps(packet.get("words", {})) + json.dumps(packet.get("candidates", []))
    found = [b for b in literal if re.search(r"(?<![A-Za-z0-9_])" + re.escape(b) + r"(?![A-Za-z0-9_])", text, re.I)]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            found.append(m.group(0))
    return found


def _check_blind(packet, literal, patterns):
    found = leaks(packet, literal, patterns)
    if found:
        raise Leak(f"a card's words name {found}, so a judge could see its place in the network. Rewrite the words")


def _quiet_items(net, node, pole, other, split, big, count, avoid):
    """Up to count strong places of another node where this node is quiet. Judged per place, since a dial fires somewhere in nearly every example."""
    v, p = net.peaks(other, "pos", split)
    order = [int(i) for i in np.argsort(-v)[:200] if int(i) not in avoid]
    got = []
    for start in range(0, len(order), 32):
        batch = np.array(order[start:start + 32])
        here = net.value_at(node, pole, split, batch, p[batch])
        for i, val in zip(batch, here):
            if val < QUIET * big:
                got.append((int(i), int(p[i]), float(val / big)))
        if len(got) >= count:
            break
    return got[:count]


def _others(net, node, pole, candidates, split, big, count, avoid):
    """Items from as many candidate nodes as it needs. None when they run out, since a short sheet is a weak test."""
    got = []
    for other in candidates:
        if other != node:
            got += _quiet_items(net, node, pole, other, split, big, count - len(got), avoid | {g[0] for g in got})
        if len(got) >= count:
            return got[:count]
    return None


def route_values(net, node, pole, route, split, examples, positions):
    """The firing on this pole with only this route's inputs real, by the exact sum. Other inputs carry their means, and an input the architecture keeps stays real."""
    D, names, base, _ = net.inputs(node, split, examples, positions)
    keep = [j for j, n in enumerate(names) if n in route or net.input_kept(n)]
    value = base + D[:, keep].sum(1)
    return np.maximum(value if pole == "pos" else -value, 0)


def strong_pool(values, big):
    """Firings at or above half of the largest, and never more than the strongest share of examples. Half of the largest can cover a quarter of all examples on a broad pole, and those are not strong firings (LOG, smoke run)."""
    pool = np.where(values >= .5 * big)[0]
    cap = max(N_FIRE, int(STRONG_SHARE * len(values)))
    if len(pool) > cap:
        pool = pool[np.argsort(-values[pool])[:cap]]
    return pool


def build_pole_sheet(net, node, pole, split, rng, route=None):
    """Returns tiles and truth, in the order shown. Stops with ShortPool rather than build a weak sheet."""
    v, p = net.peaks(node, pole, split)
    full_big = float(v.max())
    if full_big <= 0:
        raise ShortPool(f"{node} {pole}: this pole never fires")
    share_of = None
    if route is None:
        big = full_big
        values = v
        pool = strong_pool(v, big)
    else:
        strong = np.argsort(-v)[:300]
        rv = route_values(net, node, pole, route, split, strong, p[strong])
        big = float(rv.max())
        if big <= 0:
            raise ShortPool(f"{node} {pole}: this route never fires alone")
        values = np.zeros(len(v))
        values[strong] = rv
        pool = strong_pool(values, big)
    if len(pool) >= N_FIRE:
        fire = rng.permutation(pool)[:N_FIRE]
    else:
        # few strong firings: take the next strongest, unless they are so far down that they are not this node's job
        fire = np.argsort(-values)[:N_FIRE]
        if values[fire].min() < FIRE_FLOOR * big:
            raise ShortPool(f"{node} {pole}: fewer than {N_FIRE} firings near its largest. This sheet cannot test it, so the node needs another workflow")
    avoid = {int(i) for i in fire}
    similar = net.similar(node, split, 12)
    near = _others(net, node, pole, similar, split, full_big, N_NEAR, avoid)
    if near is None:
        raise ShortPool(f"{node} {pole}: not enough similar items where it is quiet")
    avoid |= {g[0] for g in near}
    layer_nodes = [n for n in net.nodes(net.layer_of(node)) if n != node and n not in set(similar)]
    shuffled = [layer_nodes[i] for i in rng.permutation(len(layer_nodes))]
    rand = _others(net, node, pole, shuffled[:24], split, full_big, N_RANDOM, avoid)
    if rand is None:
        raise ShortPool(f"{node} {pole}: not enough random items where it is quiet")
    rows = [dict(example=int(i), position=int(p[i]), share=float(values[i] / big), kind="fire") for i in fire]
    rows += [dict(example=i, position=pos, share=s, kind="near") for i, pos, s in near]
    rows += [dict(example=i, position=pos, share=s, kind="random") for i, pos, s in rand]
    if route is not None:
        quiet = [r for r in rows if r["kind"] != "fire"]
        here = route_values(net, node, pole, route, split, np.array([r["example"] for r in quiet]), np.array([r["position"] for r in quiet]))
        for r, val in zip(quiet, here):
            r["share"] = float(val / big)
    rows = [rows[j] for j in rng.permutation(len(rows))]
    tiles = net.render(split, node, np.array([r["example"] for r in rows]), np.array([r["position"] for r in rows]), TILE)
    return tiles, rows


def draw_sheet(tiles, path, stamp=""):
    """The stamp is the item's random id. A judge who reads several sheets at once can get them back in another order, so a sheet must say which item it is (LOG, control run)."""
    cols = 6
    rows = int(np.ceil(len(tiles) / cols))
    top = 14 if stamp else 0
    sheet = Image.new("RGB", (cols * (TILE + 4), rows * (TILE + 18) + top), "white")
    d = ImageDraw.Draw(sheet)
    if stamp:
        d.text((2, 1), stamp, fill="black")
    for k, t in enumerate(tiles):
        x, y = (k % cols) * (TILE + 4), (k // cols) * (TILE + 18) + top
        d.text((x + 2, y + 2), str(k + 1), fill="black")
        sheet.paste(t, (x + 2, y + 16))
    sheet.save(path)


def words_for(card, pole, route_id=None):
    """A judge ranks what fits these words. The negative side's words stand alone: with the main job beside them, judges rank the positive side first (LOG, smoke run). A route has its own words."""
    if route_id is not None:
        for r in card.get("routes", []):
            if r.get("id") == route_id and r.get("words"):
                return dict(this_side=r["words"])
    if pole == "neg":
        return dict(this_side=card.get("negative", ""))
    return dict(job=card.get("distinction", ""), this_side=card.get("positive", ""))


def _digest(words):
    """The same words on the same form of sheet get the same item. A new form of the test gets new items."""
    return hashlib.sha256(json.dumps([SHEET_FORM, words], sort_keys=True).encode()).hexdigest()[:16]


class Bench:
    """Judge work for one run. Packets go to a folder with a neutral name. The key goes to sealed_keys, which a guard blocks.
    Each item draws from the seed and the count of items so far, so one command per item still gives different sheets and a true share of wrong cards."""

    def __init__(self, root, seed=0, packets=None):
        self.packets = packets or os.path.join(root, "packets")
        self.keys = os.path.join(root, "sealed_keys")
        os.makedirs(self.packets, exist_ok=True)
        os.makedirs(self.keys, exist_ok=True)
        self.key_path = os.path.join(self.keys, "key.json")
        self.key = {}
        if os.path.exists(self.key_path):
            with open(self.key_path) as f:
                self.key = json.load(f)
        self.seed = seed

    def _rng(self):
        return np.random.default_rng([self.seed, len(self.key)])

    def _pick(self):
        return random.Random(f"{self.seed}/{len(self.key)}/pick")

    def _save(self):
        tmp = self.key_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.key, f)
        os.replace(tmp, self.key_path)

    def _write_packet(self, packet):
        with open(os.path.join(self.packets, packet["item"] + ".json"), "w") as f:
            json.dump(packet, f, indent=0)

    def _existing(self, node, test, digest):
        """The same words on the same test get the same item. Rebuilding cannot shop for an easier sheet."""
        for item, k in self.key.items():
            if k["node"] == node and k["test"] == test and k.get("digest") == digest and not k["wrong_card"]:
                return item
        return None

    def want_wrong_item(self):
        """Whether to add one wrong item next to a real one, so that about a quarter of all items are wrong."""
        return random.Random(f"{self.seed}/{len(self.key)}/wrong").random() < WRONG_SHARE_OF_ITEMS / (1 - WRONG_SHARE_OF_ITEMS)

    def build_pole(self, net, node, pole, cards, split="sealed", route=None, route_id=None, wrong=False):
        """A real item shows the node's own words. A wrong item shows the words of another card from the layer, on this node's sheet."""
        if split != "sealed":
            raise ValueError("a judge's test uses sealed examples only")
        test = f"route:{route_id}" if route is not None else f"pole:{pole}"
        rng, pick = self._rng(), self._pick()
        if wrong:
            same = [n for n in cards if n != node and net.layer_of(n) == net.layer_of(node)]
            if not same:
                raise ShortPool(f"{node}: no other card in its layer to use as a wrong card")
            source = pick.choice(sorted(same))
        else:
            source = node
        words = words_for(cards[source], pole, route_id if route is not None and not wrong else None)
        digest = _digest(words)
        if not wrong:
            old = self._existing(node, test, digest)
            if old:
                return old
        packet = dict(item="item_" + secrets.token_hex(6), test="pole", words=words, count=N_FIRE + N_NEAR + N_RANDOM, rank=N_RANK)
        _check_blind(packet, list(net.layers()) + [node, source], net.leak_patterns())
        tiles, rows = build_pole_sheet(net, node, pole, split, rng, route)
        packet["sheet"] = os.path.join(self.packets, packet["item"] + ".png")
        draw_sheet(tiles, packet["sheet"], packet["item"])
        self._write_packet(packet)
        self.key[packet["item"]] = dict(test=test, node=node, pole=pole, wrong_card=bool(wrong), words_from=source, digest=digest, form=SHEET_FORM, shares=[r["share"] for r in rows], kinds=[r["kind"] for r in rows])
        self._save()
        return packet["item"]

    def build_link(self, net, kind, node, truth, cards, choices=LINK_CHOICES, wrong=False):
        """kind is "parent" or "reader". A judge sees a node's words and the words of several candidates, and picks the true one.
        A wrong item shows another node's words, so the true link should not be found. An output has plain words of its own, and its candidates are other outputs."""
        pick = self._pick()
        truth_words = net.words_of(truth)
        real = set(strong_links(net, kind, node))
        if truth_words is not None:
            pool = [n for n in net.output_names() if n != truth and n not in real]
            describe = lambda n: dict(job=net.words_of(n))  # noqa: E731
        else:
            if truth not in cards:
                raise ShortPool(f"{node}: its {kind} has no card yet")
            author = cards[node].get("_author")
            if author is not None and cards[truth].get("_author") == author:
                raise ShortPool(f"{node}: one reader wrote both this card and its {kind}'s, so a shared style could give the link away. An independent reader must describe {truth} first")
            # A wrong choice comes from the truth's layer. A small layer is widened to its nearest layers, so that the chance stays the same.
            # A shared author could draw a judge by style, and a real link is not a wrong answer. So neither is offered.
            author = cards[node].get("_author")
            layers = list(net.layers())
            home = layers.index(cards[truth].get("_layer"))
            pool = []
            for layer in sorted(layers, key=lambda x: (abs(layers.index(x) - home), layers.index(x))):
                pool += [n for n in cards if cards[n].get("_layer") == layer and n not in (truth, node) and n not in real and not (author is not None and cards[n].get("_author") == author)]
                if len(pool) >= choices - 1:
                    break
            describe = lambda n: dict(job=cards[n].get("distinction", ""), this_side=cards[n].get("positive", ""))  # noqa: E731
        if len(pool) < choices - 1:
            raise ShortPool(f"{node}: {len(pool)} candidates for its {kind}, and the test needs {choices - 1}")
        names = pick.sample(sorted(pool), choices - 1) + [truth]
        pick.shuffle(names)
        if wrong:
            # words whose own real link is the truth would make a right answer look like a false pass
            same = [n for n in cards if n not in names and n != node and cards[n].get("_layer") == cards[node].get("_layer") and truth not in strong_links(net, kind, n)]
            if not same:
                raise ShortPool(f"{node}: no card to use as a wrong card")
            source = pick.choice(sorted(same))
        else:
            source = node
        words = dict(job=cards[source].get("distinction", ""))
        digest = _digest([words, truth])
        if not wrong:
            old = self._existing(node, kind, digest)
            if old:
                return old
        packet = dict(item="item_" + secrets.token_hex(6), test="link", kind=kind, words=words, candidates=[dict(number=k + 1, **describe(n)) for k, n in enumerate(names)])
        _check_blind(packet, list(net.layers()) + [node, source, truth], net.leak_patterns())
        self._write_packet(packet)
        self.key[packet["item"]] = dict(test=kind, node=node, truth=truth, wrong_card=bool(wrong), words_from=source, digest=digest, true_number=names.index(truth) + 1, chance=1 / choices)
        self._save()
        return packet["item"]

    def build_ordering(self, net, node, cards, wrong=False):
        """A dial's test. A sheet of fixed drawn items. From words alone, a judge ranks the 8 most likely to fire. Code scores the rank correlation with the firing.
        The firing must vary enough on these items, or this test says nothing and the node needs another workflow."""
        items = net.battery()
        if items is None:
            raise ShortPool("this network has no battery of fixed items")
        values = net.battery_values(node)
        if values.max() < BATTERY_FLOOR:
            raise ShortPool(f"{node}: fires under {BATTERY_FLOOR} of its largest on every drawn item, so this test cannot read it")
        rng, pick = self._rng(), self._pick()
        if wrong:
            same = [n for n in cards if n != node and net.layer_of(n) == net.layer_of(node)]
            if not same:
                raise ShortPool(f"{node}: no other card in its layer to use as a wrong card")
            source = pick.choice(sorted(same))
        else:
            source = node
        words = words_for(cards[source], "pos")
        digest = _digest(words)
        if not wrong:
            old = self._existing(node, "ordering", digest)
            if old:
                return old
        packet = dict(item="item_" + secrets.token_hex(6), test="ordering", words=words, count=len(items), rank=N_ORDER)
        _check_blind(packet, list(net.layers()) + [node, source], net.leak_patterns())
        order = rng.permutation(len(items))
        packet["sheet"] = os.path.join(self.packets, packet["item"] + ".png")
        draw_sheet([items[j][1].resize((TILE, TILE)) for j in order], packet["sheet"], packet["item"])
        self._write_packet(packet)
        self.key[packet["item"]] = dict(test="ordering", node=node, pole="pos", wrong_card=bool(wrong), words_from=source, digest=digest, values=[float(values[j]) for j in order])
        self._save()
        return packet["item"]

    def barred_judges(self, item, author_of):
        k = self.key[item]
        return {a for a in (author_of(k["node"]), author_of(k["words_from"])) if a}

    def judge_scores(self, item, answers):
        k = self.key[item]
        if k["test"] in ("parent", "reader"):
            return [dict(correct=a.get("pick") == k["true_number"]) for a in answers.values()]
        if k["test"] == "ordering":
            return [score_battery(k, a) for a in answers.values()]
        return [score_pole(k, a) for a in answers.values()]


def score_pole(key, answer):
    shares = key["shares"]
    kinds = key["kinds"]
    ranked = answer.get("ranked") or []
    if len(set(ranked)) != N_RANK or len(ranked) != N_RANK or not all(isinstance(j, int) and 1 <= j <= len(shares) for j in ranked):
        raise BadAnswer(f"a judge must rank {N_RANK} different items")
    # Nodes differ in how strong their sheet's items are, so a raw share does not compare across nodes.
    # 0 is what random picks reach on this sheet. 1 is the best 6 items on this sheet.
    best = float(np.mean(sorted(shares, reverse=True)[:N_TOP]))
    chance = float(np.mean(shares))
    center = float(np.mean([shares[j - 1] for j in ranked[:N_TOP]]))

    def rate(kind):
        total = kinds.count(kind)
        return sum(kinds[j - 1] == kind for j in ranked) / max(total, 1)

    fire, near, rand = rate("fire"), rate("near"), rate("random")
    return dict(center=center, center_norm=(center - chance) / max(best - chance, 1e-9), edge=fire - near, far=fire - rand, fire_rate=fire, near_rate=near, random_rate=rand)


def score_battery(key, answer):
    values = key["values"]
    ranked = answer.get("ranked") or []
    if len(set(ranked)) != N_ORDER or len(ranked) != N_ORDER or not all(isinstance(j, int) and 1 <= j <= len(values) for j in ranked):
        raise BadAnswer(f"a judge must rank {N_ORDER} different items")
    predicted = np.zeros(len(values))
    for place, j in enumerate(ranked):
        predicted[j - 1] = N_ORDER - place
    return dict(rho=score_ordering(predicted, values))


def medians(scores):
    """For reports only. A median across judges can pass an item that every judge failed, so no verdict uses it."""
    return {k: float(np.median([s[k] for s in scores])) for k in scores[0]}


def passes(score, marks):
    return score["center_norm"] >= marks["center"] and score["edge"] >= marks["edge"] and score["far"] >= marks["far"]


def judge_passed(score, marks):
    if "correct" in score:
        return score["correct"]
    if "rho" in score:
        return score["rho"] >= marks["ordering"]
    return passes(score, marks)


def vote(passed):
    """True, False, or None when it is not settled yet. One answer settles nothing. Two who split wait for a third."""
    if len(passed) < 2:
        return None
    yes = sum(passed)
    if len(passed) == 2:
        return passed[0] if yes != 1 else None
    if yes * 2 == len(passed):
        return None
    return yes * 2 > len(passed)


def need_third(scores, marks):
    return len(scores) == 2 and vote([judge_passed(s, marks) for s in scores]) is None


def agreement(pairs, marks):
    """Share of items where two judges agree on the verdict."""
    same = [judge_passed(a, marks) == judge_passed(b, marks) for a, b in pairs]
    return float(np.mean(same)) if same else float("nan")


def clopper_upper(hits, n, alpha=.05):
    """Upper end of a one-sided 95% range for a false-pass rate. Standard library only: a simple search over the binomial tail."""
    if n == 0:
        return 1.0
    from math import comb
    lo, hi = hits / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        tail = sum(comb(n, k) * mid ** k * (1 - mid) ** (n - k) for k in range(hits + 1))
        if tail > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def marks_at(pooled, q):
    return {name: float(np.quantile([s[field] for s in pooled], q)) for name, field in (("center", "center_norm"), ("edge", "edge"), ("far", "far"))}


def rate_of(items, marks):
    """Wrong items that pass by vote. A split counts as a pass, which is the cautious reading."""
    return sum(vote([passes(s, marks) for s in item]) is not False for item in items)


def calibrate(wrong_items, target=.05):
    """Pass marks from wrong cards ONLY, so real cards cannot move them. wrong_items is one list of judges' scores per wrong item.
    All three marks sit at one quantile of the wrong judgments. It is the lowest quantile that keeps wrong items at the target on the half used to fit.
    With 20 or more wrong items, the stated rate comes from the other half, so it is not flattered."""
    usable = [i for i in wrong_items if i]
    if len(usable) < 2:
        raise ValueError("calibration needs wrong cards")
    held_out = len(usable) >= 20
    fit = usable[::2] if held_out else usable
    state = usable[1::2] if held_out else usable
    pooled = [s for item in fit for s in item]
    chosen = QUANTILES[-1]
    for q in QUANTILES:
        if rate_of(fit, marks_at(pooled, q)) / len(fit) <= target:
            chosen = q
            break
    marks = marks_at(pooled, chosen)
    hits = rate_of(state, marks)
    return dict(marks=marks, quantile=chosen, wrong_items=len(usable), held_out=held_out, stated_on=len(state), wrong_pass=hits, false_pass=hits / len(state), false_pass_upper=clopper_upper(hits, len(state)))


def calibrate_ordering(wrong_items, target=.05):
    """The mark for a dial's test, from wrong items only."""
    pooled = [s["rho"] for item in wrong_items for s in item]
    return float(np.quantile(pooled, 1 - target)) if len(pooled) >= 2 else None


def chance_center(shares, trials=2000, seed=0):
    """What 6 random picks score on one sheet. A second check on the marks, with no judge in it."""
    rng = np.random.default_rng(seed)
    shares = np.asarray(shares)
    return np.array([shares[rng.choice(len(shares), N_TOP, replace=False)].mean() for _ in range(trials)])


def node_results(bench, answers_of, node, card, marks):
    """Each test's vote for this node, on its current words only. A wrong item never counts, and a test with no settled vote is left out."""
    results = {}
    for item, k in bench.key.items():
        if k["node"] != node or k["wrong_card"]:
            continue
        if k["test"] in ("parent", "reader"):
            words = dict(job=card.get("distinction", ""))
            if k["digest"] != _digest([words, k["truth"]]):
                continue
        elif k["digest"] != _digest(words_for(card, k.get("pole", "pos"), k["test"].split(":")[1] if k["test"].startswith("route:") else None)):
            continue
        answers = answers_of(item)
        if answers:
            settled = vote([judge_passed(s, marks) for s in bench.judge_scores(item, answers)])
            if settled is not None:
                results[k["test"]] = settled
    return results


def decide(results, poles, routes=(), workflow=None, has_parent=True):
    """A node is understood when every required test passes. Returns dict(understood, failed, missing, need).
    A test not yet settled is missing, and it is not a pass. A dial workflow may replace pole sheets with an ordering. Only a node that reads the raw input has no parent test."""
    need = [t for t in REQUIRED if not (t.startswith("pole:") and t.split(":")[1] not in poles)]
    if workflow is not None and workflow.replaces_poles:
        need = [t for t in need if not t.startswith("pole:")]
    if not has_parent:
        need = [t for t in need if t != "parent"]
    need += [f"route:{r}" for r in routes]
    if workflow is not None:
        need += list(workflow.tests)
    failed = [t for t in need if results.get(t) is False]
    missing = [t for t in need if t not in results or results[t] is None]
    return dict(understood=not failed and not missing, failed=failed, missing=missing, need=need)


def _ranks(a):
    """Ranks with ties shared, so a prediction that says nothing cannot score by index order."""
    a = np.asarray(a, float)
    order = np.argsort(a)
    r = np.empty(len(a))
    r[order] = np.arange(len(a), dtype=float)
    for v in np.unique(a):
        r[a == v] = r[a == v].mean()
    return r


def score_ordering(predicted, values):
    """A dial's test, scored by code with no model: rank correlation between the predicted order and the measured firing."""
    if len(predicted) < 5:
        raise ValueError("an ordering needs at least 5 items")
    a, b = _ranks(predicted), _ranks(values)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

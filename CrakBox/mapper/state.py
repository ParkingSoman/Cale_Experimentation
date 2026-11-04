"""Node states and full history in a store on disk. Every change is one transaction, so a crash loses nothing that was finished, and leaves nothing half done.

The legal moves are the loop of CONSTITUTION.md section 4. Limits of section 11 are enforced here, not in a brief."""
import json
import re
import os
import sqlite3
import time

MAX_WORKFLOWS = 3
SMOKE_BATCH = 10
ROLES = {"reader", "critic"}

ATTACK_POWER = .5  # share of the largest firing that at least one input in an attack must reach

MOVES = {
    "queued": {"routed", "open"},
    "routed": {"reading", "open"},
    "reading": {"claimed", "routed", "open"},
    "claimed": {"attacked", "reading", "open"},
    "attacked": {"judging", "reading", "claimed"},
    "judging": {"understood", "open", "reading"},
    "open": {"routed"},
    "understood": {"open"},
}
SUSPECTS = {"tool", "explanation", "data", "motif", "offshoot", "regime", "learning"}


class IllegalMove(Exception):
    pass


class LimitHit(Exception):
    pass


class Store:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS nodes(id TEXT PRIMARY KEY, layer TEXT, state TEXT, workflow TEXT, tried TEXT, suspect TEXT);
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, node TEXT, kind TEXT, data TEXT, at REAL);
            CREATE TABLE IF NOT EXISTS cards(id INTEGER PRIMARY KEY AUTOINCREMENT, node TEXT, author TEXT, workflow TEXT, card TEXT, at REAL);
            CREATE TABLE IF NOT EXISTS steps(id INTEGER PRIMARY KEY AUTOINCREMENT, node TEXT, role TEXT, workflow TEXT, command TEXT, output TEXT, at REAL);
            CREATE TABLE IF NOT EXISTS plans(id INTEGER PRIMARY KEY AUTOINCREMENT, node TEXT, critic TEXT, prediction TEXT, attack TEXT, after_step INTEGER, at REAL);
            CREATE TABLE IF NOT EXISTS answers(item TEXT, judge TEXT, answer TEXT, at REAL, PRIMARY KEY(item, judge));
            CREATE TABLE IF NOT EXISTS tokens(id INTEGER PRIMARY KEY AUTOINCREMENT, node TEXT, role TEXT, workflow TEXT, count INTEGER, at REAL);
            CREATE TABLE IF NOT EXISTS claims(id INTEGER PRIMARY KEY AUTOINCREMENT, node TEXT, claim TEXT, at REAL);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
        """)
        self.db.commit()

    # scope and states
    def set_scope(self, scope, pairs):
        missing = [k for k in ("nodes", "goal", "control_goal", "threshold", "firings") if not scope.get(k)]
        if missing:
            raise IllegalMove(f"a scope must name {missing} before any reading starts")
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES('scope',?)", (json.dumps(scope),))
            self.db.executemany("INSERT OR IGNORE INTO nodes VALUES(?,?,'queued',NULL,'[]',NULL)", list(pairs))

    def scope(self):
        row = self.db.execute("SELECT value FROM meta WHERE key='scope'").fetchone()
        return json.loads(row[0]) if row else None

    def add_nodes(self, pairs):
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO nodes VALUES(?,?,'queued',NULL,'[]',NULL)", list(pairs))

    def state(self, node):
        row = self.db.execute("SELECT state FROM nodes WHERE id=?", (node,)).fetchone()
        if row is None:
            raise KeyError(node)
        return row[0]

    def tried(self, node):
        return json.loads(self.db.execute("SELECT tried FROM nodes WHERE id=?", (node,)).fetchone()[0])

    def workflow(self, node):
        return self.db.execute("SELECT workflow FROM nodes WHERE id=?", (node,)).fetchone()[0]

    def suspect(self, node):
        return self.db.execute("SELECT suspect FROM nodes WHERE id=?", (node,)).fetchone()[0]

    def _check(self, node, to, data):
        now = self.state(node)
        if to not in MOVES[now]:
            raise IllegalMove(f"{node}: {now} -> {to} is not a legal move")
        if to == "open":
            if data.get("suspect") not in SUSPECTS:
                raise IllegalMove(f"{node}: an open node must name a suspect, one of {sorted(SUSPECTS)}")
            if not data.get("record"):
                raise IllegalMove(f"{node}: an open node needs a stuck record: what was tried, and what a new tool would show")
        if to == "understood":
            verdict = data.get("verdict") or {}
            if not verdict.get("understood") or verdict.get("failed") or verdict.get("missing") or not verdict.get("results"):
                raise IllegalMove(f"{node}: only a judged verdict with every test passed reaches understood")
            if not all(v is True for v in verdict["results"].values()):
                raise IllegalMove(f"{node}: a verdict holds a test that did not pass")
        return now

    def _write_move(self, node, now, to, data):
        self.db.execute("UPDATE nodes SET state=? WHERE id=?", (to, node))
        if to == "open":
            self.db.execute("UPDATE nodes SET suspect=? WHERE id=?", (data["suspect"], node))
        self.db.execute("INSERT INTO events(node,kind,data,at) VALUES(?,?,?,?)", (node, f"{now}->{to}", json.dumps(data), time.time()))

    def move(self, node, to, **data):
        now = self._check(node, to, data)
        with self.db:
            self._write_move(node, now, to, data)

    def route(self, node, workflow):
        tried = self.tried(node)
        if workflow in tried:
            raise IllegalMove(f"{node}: already tried {workflow}")
        if len(tried) >= MAX_WORKFLOWS:
            raise LimitHit(f"{node}: {MAX_WORKFLOWS} workflows tried, so the node stays open with its stuck record")
        now = self._check(node, "routed", {})
        with self.db:
            self._write_move(node, now, "routed", dict(workflow=workflow))
            self.db.execute("UPDATE nodes SET workflow=?, tried=?, suspect=NULL WHERE id=?", (workflow, json.dumps(tried + [workflow]), node))

    def history(self, node):
        return [(k, json.loads(d)) for k, d in self.db.execute("SELECT kind,data FROM events WHERE node=? ORDER BY seq", (node,))]

    # steps of a workflow
    def steps_used(self, node, role, workflow=None):
        workflow = workflow or self.workflow(node)
        after = max([d["after_step"] for k, d in self.history(node) if d.get("retool")], default=0)
        return self.db.execute("SELECT COUNT(*) FROM steps WHERE node=? AND role=? AND workflow=? AND id>?", (node, role, workflow, after)).fetchone()[0]

    def retool(self, node, reason):
        """An open node whose suspect is a tool gets one more read in the same workflow when that tool is fixed. It spends no workflow slot, since the workflow was not what failed. Once per node, so it cannot become a way around the limit."""
        if not reason:
            raise IllegalMove(f"{node}: a retool needs a reason")
        if self.state(node) != "open":
            raise IllegalMove(f"{node}: only an open node can be retooled")
        if any(d.get("retool") for _, d in self.history(node)):
            raise LimitHit(f"{node}: it already had its one read after a tool fix. The next read spends a workflow slot")
        self.move(node, "routed", retool=reason, after_step=self.last_step())
        with self.db:
            self.db.execute("UPDATE nodes SET suspect=NULL WHERE id=?", (node,))

    def record_step(self, node, role, command, output, limit):
        if role not in ROLES:  # a role name is fixed
            raise IllegalMove(f"role must be one of {sorted(ROLES)}, so that nobody gets a second allowance of steps")
        workflow = self.workflow(node)
        if self.steps_used(node, role, workflow) >= limit:
            raise LimitHit(f"{node}: {limit} steps used by the {role} in {workflow}. Take an exit")
        with self.db:
            cur = self.db.execute("INSERT INTO steps(node,role,workflow,command,output,at) VALUES(?,?,?,?,?,?)", (node, role, workflow, command, output, time.time()))
        return cur.lastrowid

    def step(self, step_id):
        row = self.db.execute("SELECT node,role,workflow,command,output FROM steps WHERE id=?", (step_id,)).fetchone()
        return None if row is None else dict(zip(("node", "role", "workflow", "command", "output"), row))

    def last_step(self):
        return self.db.execute("SELECT COALESCE(MAX(id),0) FROM steps").fetchone()[0]

    def add_reader_prediction(self, node, author, prediction):
        if self.state(node) != "reading":
            raise IllegalMove(f"{node}: a reader's prediction goes on file while reading")
        if not prediction:
            raise IllegalMove(f"{node}: an empty prediction")
        with self.db:
            self.db.execute("INSERT INTO events(node,kind,data,at) VALUES(?,?,?,?)", (node, "reader-prediction", json.dumps(dict(author=author, prediction=prediction, workflow=self.workflow(node))), time.time()))

    def reader_prediction(self, node):
        for kind, data in reversed(self.history(node)):
            if data.get("retool"):
                return None  # a new read needs its own prediction
            if kind == "reader-prediction" and data.get("workflow") == self.workflow(node):
                return data
        return None

    def revise(self, node, author, card):
        """An author corrects its own card before any critic has it. The first words count for nothing."""
        if self.state(node) != "claimed":
            raise IllegalMove(f"{node}: only a claimed card can be corrected")
        if self.plan_for_current_card(node) is not None:
            raise IllegalMove(f"{node}: a critic already has this card")
        if self.card(node)["_author"] != author:
            raise IllegalMove(f"{node}: only its author can correct it")
        with self.db:
            self.db.execute("INSERT INTO cards(node,author,workflow,card,at) VALUES(?,?,?,?,?)", (node, author, self.workflow(node), json.dumps(card), time.time()))
            self.db.execute("INSERT INTO events(node,kind,data,at) VALUES(?,?,?,?)", (node, "card-revised", json.dumps(dict(author=author)), time.time()))

    def plan_for_current_card(self, node):
        last_card = self.db.execute("SELECT MAX(at) FROM cards WHERE node=?", (node,)).fetchone()[0] or 0
        row = self.db.execute("SELECT critic FROM plans WHERE node=? AND at>=?", (node, last_card)).fetchone()
        return row[0] if row else None

    # a critic's prediction is on file before its test runs
    def add_plan(self, node, critic, prediction, attack):
        if self.state(node) != "claimed":
            raise IllegalMove(f"{node}: a critic attacks a claimed card")
        if not prediction or not attack:
            raise IllegalMove(f"{node}: an attack needs a prediction and a plan, written before the test")
        with self.db:
            self.db.execute("INSERT INTO plans(node,critic,prediction,attack,after_step,at) VALUES(?,?,?,?,?,?)", (node, critic, prediction, attack, self.last_step(), time.time()))

    def plan(self, node):
        row = self.db.execute("SELECT critic,prediction,attack,after_step FROM plans WHERE node=? ORDER BY id DESC LIMIT 1", (node,)).fetchone()
        return None if row is None else dict(zip(("critic", "prediction", "attack", "after_step"), row))

    def attack_has_power(self, steps):
        """An attack must be able to lose. A test on the node's real examples can. A test only on inputs that barely fire the node cannot: a weak firing says little about what the node does."""
        shares, other = [], False
        for step_id in steps:
            s = self.step(step_id)
            if s is None:
                continue
            found = [float(x) for x in re.findall(r"(-?\d+\.\d+) of the largest (?:negative )?firing", s["output"])]
            if s["command"].startswith("probe") and found:
                shares += found
            else:
                other = True
        return other or not shares or max(shares) >= ATTACK_POWER

    def void_attack(self, node, reason):
        """A past attack that could not have failed counts for nothing, either way. The same card goes back to a critic."""
        moves = self.history(node)
        last = next((d for k, d in reversed(moves) if k in ("claimed->attacked", "claimed->reading")), None)
        if last is None or self.state(node) not in ("attacked", "reading"):
            raise IllegalMove(f"{node}: no attack to void")
        self.move(node, "claimed", void=reason, voided=last["critic"]["critic"])

    def conclude_attack(self, node, critic, steps, observed, broke):
        plan = self.plan(node)
        if plan is None or plan["critic"] != critic:
            raise IllegalMove(f"{node}: no prediction on file from this critic. Run critic-plan before the test")
        card = self.card(node)
        if card["_author"] == critic:
            raise IllegalMove(f"{node}: a card's author cannot attack it")
        if not steps:
            raise IllegalMove(f"{node}: an attack must cite the logged evidence calls that ran it")
        for step_id in steps:
            s = self.step(step_id)
            if s is None or s["node"] != node or s["role"] != "critic":
                raise IllegalMove(f"step {step_id} is not a critic's logged call for {node}")
            if step_id <= plan["after_step"]:
                raise IllegalMove(f"step {step_id} ran before the prediction was on file")
        if not self.attack_has_power(steps):
            raise IllegalMove(f"{node}: no input in this attack fires the node at {ATTACK_POWER} of its largest firing, so the test could not break the card. Start from one of the node's real examples and edit it")
        self.move(node, "reading" if broke else "attacked", critic=dict(plan, steps=steps, observed=observed, broke=broke))

    def critic_held(self, node):
        """True when the latest attack on the current card held. A new card needs a new attack."""
        moves = self.history(node)
        for kind, data in reversed(moves):
            if kind in ("reading->claimed", "attacked->claimed"):
                return None
            if kind == "claimed->attacked":
                return True
            if kind == "claimed->reading":
                return False
        return None

    # cards: the card and its move are one transaction, so a node cannot hold words that nothing judged
    def claim(self, node, author, card):
        now = self._check(node, "claimed", {})
        with self.db:
            self.db.execute("INSERT INTO cards(node,author,workflow,card,at) VALUES(?,?,?,?,?)", (node, author, self.workflow(node), json.dumps(card), time.time()))
            self._write_move(node, now, "claimed", dict(author=author))

    def card(self, node):
        row = self.db.execute("SELECT author,card FROM cards WHERE node=? ORDER BY id DESC LIMIT 1", (node,)).fetchone()
        return None if row is None else dict(json.loads(row[1]), _author=row[0])

    def all_cards(self):
        latest = {}
        for node, author, card, layer in self.db.execute("SELECT c.node,c.author,c.card,n.layer FROM cards c JOIN nodes n ON n.id=c.node ORDER BY c.id"):
            latest[node] = dict(json.loads(card), _author=author, _layer=layer)
        return latest

    # judges
    def add_answer(self, item, judge, answer, barred=()):
        if judge in barred:
            raise IllegalMove(f"{judge} wrote these words, so {judge} cannot judge them")
        try:
            with self.db:
                self.db.execute("INSERT INTO answers VALUES(?,?,?,?)", (item, judge, json.dumps(answer), time.time()))
        except sqlite3.IntegrityError:
            raise IllegalMove(f"{judge} already answered {item}. An answer cannot be changed") from None

    def answers(self, item):
        return {j: json.loads(a) for j, a in self.db.execute("SELECT judge,answer FROM answers WHERE item=? ORDER BY judge", (item,))}

    # claims and cost
    def add_claim(self, node, claim):
        with self.db:
            self.db.execute("INSERT INTO claims(node,claim,at) VALUES(?,?,?)", (node, json.dumps(claim), time.time()))

    def claims(self):
        return [json.loads(c) for (c,) in self.db.execute("SELECT claim FROM claims ORDER BY id")]

    def add_tokens(self, node, role, count):
        with self.db:
            self.db.execute("INSERT INTO tokens(node,role,workflow,count,at) VALUES(?,?,?,?,?)", (node, role, self.workflow(node), count, time.time()))

    # batches
    def smoke_passed(self):
        return bool(self.db.execute("SELECT value FROM meta WHERE key='smoke'").fetchone())

    def set_smoke_passed(self, note):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES('smoke',?)", (note,))

    def take(self, state, n):
        """Counts every node already started, so repeated small batches cannot walk past the smoke gate."""
        if not self.smoke_passed():
            started = self.db.execute("SELECT COUNT(*) FROM nodes WHERE state!='queued'").fetchone()[0]
            room = SMOKE_BATCH - started if state == "queued" else SMOKE_BATCH
            if n > room:
                raise LimitHit(f"{started} nodes started. Above {SMOKE_BATCH} runs only after a smoke batch passes")
        return [r[0] for r in self.db.execute("SELECT id FROM nodes WHERE state=? ORDER BY id LIMIT ?", (state, n))]

    # the bar
    def status(self):
        rows = self.db.execute("SELECT layer,state,COUNT(*) FROM nodes GROUP BY layer,state").fetchall()
        layers, total, done = {}, 0, 0
        for layer, state, count in rows:
            entry = layers.setdefault(layer, dict(scope=0, understood=0))
            entry["scope"] += count
            total += count
            if state == "understood":
                entry["understood"] += count
                done += count
        states = dict(self.db.execute("SELECT state,COUNT(*) FROM nodes GROUP BY state").fetchall())
        open_nodes = [dict(node=n, suspect=s, tried=json.loads(t)) for n, s, t in self.db.execute("SELECT id,suspect,tried FROM nodes WHERE state='open' ORDER BY id")]
        cost = {w or "none": dict(tokens=t, nodes=c) for w, t, c in self.db.execute("SELECT workflow,SUM(count),COUNT(DISTINCT node) FROM tokens GROUP BY workflow")}
        return dict(understood=done, scope=total, layers=layers, states=states, open=open_nodes, cost=cost)

    def close(self):
        self.db.close()

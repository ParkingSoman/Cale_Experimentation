"""What `cale tune` measures, one probe at a time. Each probe is timed, and what it found is written to a report, so the cost of every
measurement can be set against what it is worth. Run when Cale is installed: no app ever waits for any of it.

Profiles say how much to measure:
  quick   the few measurements everything else depends on (about a minute on a weak laptop, seconds on a fast Mac)
  full    quick, plus the ones that choose settings: threads, long context, several answers at once, graphics card pieces
  max     everything, including the long test of sustained speed (minutes)

The rules for every probe: use the real model and the real engine (a made-up test misses the code path that is used), run a warm-up first
and discard it, take the median of several runs for the numbers that matter, and never change a setting unless the new one is clearly
better (more than the noise).
"""
import json
import statistics
import threading
import time

from . import gemma3n_facts as facts, oslayer

LEVEL = {"quick": 1, "full": 2, "max": 3}
NOISE = 1.08                       # a setting must be this much better than the default to be chosen: smaller differences are noise


class Report:
    def __init__(self):
        self.entries, self.t0 = [], time.time()

    def add(self, name, seconds, ok, result):
        self.entries.append({"probe": name, "seconds": round(seconds, 1), "ok": ok, "result": result})

    def total(self):
        return round(sum(e["seconds"] for e in self.entries), 1)

    def to_json(self):
        return {"seconds": self.total(), "probes": self.entries, "at": time.strftime("%Y-%m-%d %H:%M:%S")}


class Tuner:
    def __init__(self, eng, profile="max", say=lambda t: None, log=print):
        self.eng, self.level, self.say, self.log = eng, LEVEL.get(profile, 3), say, log
        self.report = Report()
        self.warmed = False

    # --- the probes, in the order they run ----------------------------------------------------------------------------------
    def probes(self):
        m = self.eng.m
        place = str(getattr(m, "placement", "device=cpu"))
        cpu = place.startswith("device=cpu") or "gpu_layers=0" in place          # the CPU answers: weights in RAM, even if a graphics card reads prompts
        gpu_reads = bool(getattr(m, "gpu_reads_prompts", False))
        return [
            ("environment", 1, self.environment, True),
            ("disk", 1, self.disk, True),
            ("sizes", 1, self.sizes, True),
            ("prefill_curve", 1, self.prefill_curve, True),
            ("working_memory", 1, self.working_memory, True),
            ("threads", 2, self.threads, cpu and hasattr(m, "set_threads")),
            ("context", 2, self.context, True),
            ("batch", 3, self.batch, self.eng.can_batch()),
            ("gpu_pieces", 2, self.gpu_pieces, self.eng.can_batch() and gpu_reads),
            ("sustained", 2, self.sustained, True),
            ("drift", 1, self.drift, True),
        ]

    def run(self):
        for name, level, fn, applies in self.probes():
            if level > self.level or not applies:
                continue
            t = time.time()
            try:
                result, ok = fn() or {}, True
            except Exception as e:                     # a probe that fails must not stop the rest: Cale still works with what it has
                result, ok = {"error": f"{type(e).__name__}: {str(e)[:160]}"}, False
            self.report.add(name, time.time() - t, ok, result)
            self.log(f"tune: {name} {'ok' if ok else 'FAILED'} in {time.time() - t:.1f}s {json.dumps(result)[:200]}")
        return self.report

    # --- helpers --------------------------------------------------------------------------------------------------------------
    @property
    def pred(self):
        return self.eng.pred

    def full_level(self):
        return max(self.eng.levels, key=lambda l: l["pct"])

    def one(self, level, tokens, answer):
        d = self.eng._run_prompt(tokens, answer, width=level)
        if not d or d.get("tokens", 0) < 9 or d.get("decode_s", 0) <= 0:
            return None
        dec = (d["tokens"] - 1) / d["decode_s"]
        return d["prompt_tokens"] / max(d["first_token_s"] - 1.0 / dec, 1e-6), dec

    def warm_up(self):
        """Run the full size until two runs in a row agree (within 8%): clocks rise, caches fill and memory settles before the first
        reading. The runs are thrown away. At most 6 runs."""
        last = None
        for _ in range(6):
            r = self.one(self.full_level(), 64, 20)
            if r and last and abs(r[1] / last[1] - 1) < 0.08 and abs(r[0] / last[0] - 1) < 0.15:
                break
            last = r or last
        self.warmed = True

    def point(self, level, tokens, answer=24, repeats=3):
        """(prompt tokens/s, answer tokens/s) at this size and prompt length: the median of `repeats` runs, after the warm-up."""
        if not self.warmed:
            self.warm_up()
        runs = [r for r in (self.one(level, tokens, answer) for _ in range(repeats)) if r]
        if not runs:
            return None
        return statistics.median(r[0] for r in runs), statistics.median(r[1] for r in runs)

    # --- probes ---------------------------------------------------------------------------------------------------------------
    def environment(self):
        power = oslayer.power_state()
        info = {"power": "battery" if power.get("on_battery") else "plugged in", "hot": bool(power.get("hot")),
                "cores_physical": getattr(self.eng.m, "threads", None), "cores_logical": __import__("os").cpu_count(),
                "core_types": oslayer.core_types(), "ram_gb": round(oslayer.total_ram() / 1e9, 1), "placement": getattr(self.eng.m, "placement", None)}
        self.pred.extra["tuned_on"] = {"power": info["power"], "core_types": info["core_types"]}
        return info

    def disk(self):
        files = list(self.eng.m.model_files())
        speed = oslayer.read_uncached(files[0], 1 << 30) if files else 0.0
        if speed > 0:
            self.pred.disk_bps = speed
            self.eng.mach["disk_bps"] = speed
        self.pred.extra["disk_bps"] = speed
        return {"gb_per_s": round(speed / 1e9, 2), "load_s": self.eng.load_s}

    def sizes(self):
        """Prompt and answer speed at spread-out sizes: the table the predictor uses (sizes between are read off by cost)."""
        levels = sorted(self.eng.levels, key=lambda l: l["pct"])
        picks = {levels[round(i * (len(levels) - 1) / 4)]["id"]: levels[round(i * (len(levels) - 1) / 4)] for i in range(5)}
        picks[self.full_level()["id"]] = self.full_level()
        table, out = {}, {}
        for l in sorted(picks.values(), key=lambda l: -l["pct"]):
            r = self.point(l, self.pred.table_prompt, answer=20, repeats=2 if l is self.full_level() else 1)
            if r:
                table[l["id"]] = {"pre": r[0], "dec": r[1]}
                out[l["id"]] = [round(r[0], 1), round(r[1], 2)]
        if table:
            self.pred.table, self.pred.corr_dec, self.pred.corr_pre = table, 1.0, 1.0
            self.pred.n_dec = self.pred.n_pre = max(1, len(table))
        return out

    def prefill_curve(self):
        """Prompt speed by prompt length at the full size (long prompts are a little slower per token)."""
        full, t = self.full_level(), self.pred.table.get(self.full_level()["id"])
        pts = [(self.pred.table_prompt, t["pre"])] if t else []
        for n in (64, 1024):
            r = self.point(full, n, repeats=1)
            if r:
                pts.append((n, r[0]))
        if len(pts) >= 2:
            self.pred.pre_curve = sorted(pts)
        return {str(n): round(v, 1) for n, v in pts}

    def working_memory(self):
        """The most memory the process holds while the longest prompt piece is read, beyond the weights, the base and the context."""
        m, piece = self.eng.m, getattr(self.eng.m, "batch_tokens", 512)
        oslayer.give_back_heap()
        before = int(m.active_bytes())
        peak, stop = [before], threading.Event()

        def watch():
            while not stop.is_set():
                try:
                    peak[0] = max(peak[0], int(m.active_bytes()))
                except Exception:
                    return
                time.sleep(0.02)
        th = threading.Thread(target=watch, daemon=True)
        th.start()
        try:
            d = self.eng._run_prompt(piece, 4)
        finally:
            stop.set()
            th.join(timeout=2)
        if not d:
            return {}
        ctx = int(d.get("prompt_tokens", 0)) + int(d.get("tokens", 0))
        scratch = max(0, peak[0] - before - facts.kv_bytes(ctx))
        self.pred.extra["scratch_bytes"] = scratch
        self.eng.mach["scratch_bytes"] = scratch
        return {"piece_tokens": d.get("prompt_tokens"), "mb": round(scratch / 1e6)}

    def threads(self):
        """Threads for answers and for prompts, separately: a count is chosen only if it is clearly better than the number of cores."""
        m, full = self.eng.m, self.full_level()
        phys = int(getattr(m, "threads_default", None) or m.threads)
        logical = __import__("os").cpu_count() or phys
        types = oslayer.core_types()
        cands = {max(1, phys // 2), phys, min(logical, phys * 2), logical}
        if types:
            cands.add(types["fast"])
        cands = sorted(c for c in cands if 1 <= c <= logical)
        table = {}
        if len(cands) > 1:
            for n in cands:
                m.set_threads(n, n)
                r = self.point(full, 128, answer=16, repeats=1)
                if r:
                    table[n] = r
            base = table.get(phys)
            best_dec = best_pre = phys
            if base:
                best_dec = max(table, key=lambda n: table[n][1]) if table[max(table, key=lambda n: table[n][1])][1] >= NOISE * base[1] else phys
                best_pre = max(table, key=lambda n: table[n][0]) if table[max(table, key=lambda n: table[n][0])][0] >= NOISE * base[0] else phys
            m.set_threads(best_dec, best_pre)
            # the pair must really be better than the plain default, measured together
            if (best_dec, best_pre) != (phys, phys):
                chosen = self.point(full, 128, answer=16, repeats=2)
                m.set_threads(phys, phys)
                default = self.point(full, 128, answer=16, repeats=2)
                if chosen and default and chosen[0] >= default[0] * 0.97 and chosen[1] >= default[1] * 0.97 and \
                        (chosen[0] >= NOISE * default[0] or chosen[1] >= NOISE * default[1]):
                    m.set_threads(best_dec, best_pre)
                else:
                    best_dec = best_pre = phys
            self.pred.extra["thread_table"] = {str(k): [round(v[0], 1), round(v[1], 2)] for k, v in table.items()}
        else:
            best_dec = best_pre = phys
        m.set_threads(best_dec, best_pre)
        if (best_dec, best_pre) != (phys, phys):
            self.pred.extra["threads"], self.pred.extra["threads_batch"] = best_dec, best_pre
        else:
            self.pred.extra.pop("threads", None)
            self.pred.extra.pop("threads_batch", None)
        return {"cores": phys, "answers": best_dec, "prompts": best_pre, "table": self.pred.extra.get("thread_table")}

    def context(self):
        """How much answer speed is lost per 1000 tokens of context (more to read from the key and value memory). The context is as long as
        about 25 seconds of prompt reading allow, between 1024 and 4096 tokens: a longer test gives a clearer slope."""
        full, t = self.full_level(), self.pred.table.get(self.full_level()["id"])
        if not t:
            return {}
        n = int(max(1024, min(4096, t["pre"] * 25)))
        r = self.point(full, n, answer=24, repeats=1)
        if not r:
            return {}
        self.pred.dec_ctx = max(0.0, (t["dec"] / r[1] - 1.0) / ((n - self.pred.table_prompt) / 1000.0))
        return {"context_tokens": n, "answer_tps_there": round(r[1], 2), "lost_per_1000": round(self.pred.dec_ctx, 4)}

    def batch(self):
        """What each extra answer running at the same time costs. Answers share the pass over the weights, so on a computer limited by
        memory speed it is little, and on one limited by arithmetic it is a lot."""
        eng, t = self.eng, self.pred.table.get(self.full_level()["id"])
        if not t:
            return {}
        out, eps = {}, []
        for n in (2, 4):
            speeds, lock = [], threading.Lock()

            def one(i):
                for ev in eng.chat_stream([{"role": "user", "content": "Write the whole numbers from 1 to 80, separated by spaces."}], 40, 0.0, app=f"cale tune {i}"):
                    if "done" in ev:
                        d = ev["done"]
                        if d.get("tokens", 0) >= 9 and d.get("decode_s", 0) > 0:
                            with lock:
                                speeds.append((d["tokens"] - 1) / d["decode_s"])
            ths = [threading.Thread(target=one, args=(i,)) for i in range(n)]
            for th in ths:
                th.start()
            for th in ths:
                th.join()
            if speeds:
                each = statistics.median(speeds)
                eps.append(max(0.0, min(2.0, (t["dec"] / each - 1.0) / (n - 1))))
                out[str(n)] = round(each, 2)
        if eps:
            self.pred.batch_eps = statistics.median(eps)
        return {"each_answer_tps": out, "extra_answer_cost": round(self.pred.batch_eps, 2)}

    def gpu_pieces(self):
        """A graphics card that reads prompts: the speed of each prompt piece length, tried now so that no app pays for trying."""
        eng, m = self.eng, self.eng.m
        piece = getattr(m, "batch_tokens", 512)
        unit = "The sea is wide and deep. "
        n_unit = max(1, int(m.prompt_len([{"role": "user", "content": unit}])))
        text = unit * max(1, int(3 * piece / max(n_unit - 8, 1)))
        for _ in range(4):
            for ev in eng.chat_stream([{"role": "user", "content": text}], 1, 0.0, app="cale tune"):
                pass
        tuner = getattr(eng, "_tuner", None)
        if tuner is not None:
            self.pred.extra["piece_speed"] = {str(k): round(v, 2) for k, v in tuner.speed.items()}
        return {"tokens_per_s_by_piece": self.pred.extra.get("piece_speed")}

    def sustained(self, seconds=120):
        """Speed after minutes of work against speed in the first seconds. A laptop that heats up slows down: answers that run long
        are predicted at the sustained speed. The whole `seconds` are used on purpose: a thin laptop can run at full speed for 90 seconds
        and then slow down (measured on a fanless Mac), and a test that stopped early would call it steady."""
        eng, full = self.eng, self.full_level()
        t = self.pred.table.get(full["id"])
        if not t:
            return {}
        n = int(min(4000, max(200, t["dec"] * seconds * 1.4)))
        stamps, t0 = [], time.perf_counter()

        def rate(a, b):
            return sum(1 for s_ in stamps if a <= s_ < b) / (b - a)
        for ev in eng.m.chat_stream([{"role": "user", "content": "Count upward from one, one number on each line, and do not stop."}],
                                    max_tokens=n, width=full, temperature=0.0):
            if "delta" not in ev:
                continue
            now = time.perf_counter() - t0
            stamps.append(now)
            if now > seconds:
                break
        eng.m.clear_cache()
        if len(stamps) < 60 or stamps[-1] < 20:
            return {"note": "the answer ended too early to tell", "tokens": len(stamps)}
        end = stamps[-1]
        first, last = rate(3, 13), rate(end - 15, end)
        sustained = min(1.0, last / first) if first > 0 else 1.0
        burst = end
        for a in range(3, int(end) - 5, 5):                             # when the speed first falls below 95% of the start
            if rate(a, a + 5) < 0.95 * first:
                burst = a
                break
        self.pred.sustained = sustained if sustained < 0.95 else 1.0
        self.pred.extra["burst_s"] = float(burst)
        return {"seconds": round(end), "first_tps": round(first, 2), "last_tps": round(last, 2), "sustained": round(sustained, 3), "burst_s": burst}

    def drift(self):
        """The first measurement again, at the end: a number that moved by more than the noise says the computer was busy or heating up."""
        full, t = self.full_level(), self.pred.table.get(self.full_level()["id"])
        if not t:
            return {}
        r = self.point(full, self.pred.table_prompt, repeats=1)
        if not r:
            return {}
        d = {"answers_changed_by": round(r[1] / t["dec"] - 1, 3), "prompts_changed_by": round(r[0] / t["pre"] - 1, 3)}
        self.pred.extra["noise"] = max(abs(d["answers_changed_by"]), abs(d["prompts_changed_by"]))
        return d


# --- which way of running is fastest here: the CPU, or each graphics card (before the model is loaded for the rest of the tuning) ----------
def probe_one(env_gpu, log=print, extra_env=None):
    """Run one way of running (CALE_GPU=env_gpu) in its own process, and return {"pre", "dec", "placement"} or None.
    A separate process: a graphics driver that fails or hangs must not take the installer with it."""
    import os
    import subprocess
    import sys
    import tempfile
    home = tempfile.mkdtemp(prefix="cale-probe-")
    try:
        r = subprocess.run([sys.executable, "-m", "cale.tune", "--probe"], capture_output=True, text=True, timeout=900,
                           env=dict(os.environ, CALE_GPU=env_gpu, CALE_HOME=home, PYTHONUNBUFFERED="1", **(extra_env or {})))
        for line in r.stdout.splitlines():
            if line.startswith("PLACEMENT_PROBE "):
                return json.loads(line[len("PLACEMENT_PROBE "):])
        log(f"tune: way of running {env_gpu} gave no result ({r.returncode}) {r.stderr[-200:]}")
    except Exception as e:
        log(f"tune: way of running {env_gpu} failed ({type(e).__name__})")
    finally:
        import shutil
        shutil.rmtree(home, ignore_errors=True)
    return None


def choose_placement(model_path, lib_path, home, say=lambda t: None, log=print):
    """Time the CPU and every graphics card on a real prompt and a few answer tokens, each in its own process, and keep the fastest way in
    placement.json. A card is chosen only if prompts are clearly faster (more than the noise) and answers are not slower than on the CPU.
    So a card can never make things slower, whatever it is. Returns the decision, or None if there is no card."""
    import os
    from .backends import devices, llamalib
    devs = llamalib.devices(lib_path)
    cards = [d for d in devs if d["type"] in ("GPU", "IGPU")]
    if not cards:
        return None
    say("pieces")
    results = {"cpu": probe_one("off", log)}
    cpu_first = (results["cpu"] or {}).get("first_pre")
    for d in cards:
        fits = llamalib.fit_all(model_path, d["handle"], 32768, os.cpu_count() or 4, lib_path)
        how = "full" if fits else "prompts"
        r = probe_one(f"force:{how}:{d['name']}", log, {"CALE_PROBE_SKIP_BELOW": str(0.5 * cpu_first)} if cpu_first else None)
        if r:
            results[f"{d['name']}|{how}"] = r
    base = results.get("cpu")
    best, best_score = ("cpu", None), None
    if base:
        for key, r in results.items():
            if key == "cpu":
                continue
            if r["pre"] >= NOISE * base["pre"] and r["dec"] >= 0.92 * base["dec"]:
                score = (r["pre"] * r["dec"]) ** 0.5
                if best_score is None or score > best_score:
                    best, best_score = tuple(key.split("|")), score
    decision = {"signature": json.loads(json.dumps(devices.signature(devs))), "use": best[0], "mode": best[1] if len(best) > 1 else None,
                "measured": results, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (home / "placement.json").write_text(json.dumps(decision, indent=1))
    log(f"tune: fastest way of running: {best[0]} {best[1] if len(best) > 1 else ''}")
    return decision


def decision_env(decision):
    """The CALE_GPU value that runs the way the placement decision chose."""
    if not decision or decision.get("use") in (None, "cpu"):
        return "off"
    return f"force:{decision['mode']}:{decision['use']}"


def choose_engine_options(decision, home, say=lambda t: None, log=print):
    """Flash attention on or off, and bigger or smaller passes (n_batch = n_ubatch), each tried in its own process on the way of running
    that was chosen. An option is kept only if prompts or answers get more than the noise faster and nothing else gets slower.
    Written to engine-options.json: the engine reads it when it loads the model. Passes bigger than 512 are tried only when a graphics card
    reads prompts (the weights are copied once per pass, so a longer pass copies less): on a CPU they cost memory and gain nothing."""
    gpu = decision_env(decision)
    key = (decision.get("use") or "cpu") + ("|" + decision["mode"] if decision.get("mode") else "")
    base = (decision.get("measured") or {}).get("cpu" if gpu == "off" else key)
    if not base:
        return {}
    say("settings")
    chosen, tried = {}, {"default": base}

    def better(r, a_key):
        pre_ok, dec_ok = r["pre"] >= base["pre"] * 0.97, r["dec"] >= base["dec"] * 0.97
        return pre_ok and dec_ok and (r["pre"] >= NOISE * base["pre"] or r["dec"] >= NOISE * base["dec"])
    flash = {}
    for v in (0, 1):
        r = probe_one(gpu, log, {"CALE_FLASH": str(v)})
        if r:
            flash[v] = r
            tried[f"flash={v}"] = r
    cand = [v for v, r in flash.items() if better(r, "flash")]
    if cand:
        chosen["flash"] = max(cand, key=lambda v: flash[v]["pre"] * flash[v]["dec"])
    if gpu != "off":
        sizes = {}
        for n in (1024, 2048):
            r = probe_one(gpu, log, {"CALE_BATCH": str(n), "CALE_UBATCH": str(n), **({"CALE_FLASH": str(chosen["flash"])} if "flash" in chosen else {})})
            if r:
                sizes[n] = r
                tried[f"batch={n}"] = r
        ok = [n for n, r in sizes.items() if better(r, "batch")]
        if ok:
            chosen["batch"] = chosen["ubatch"] = max(ok, key=lambda n: sizes[n]["pre"])
    (home / "engine-options.json").write_text(json.dumps(chosen))
    log(f"tune: engine options {chosen or '(the defaults)'}")
    return {"chosen": chosen, "tried": {k: {"pre": round(v["pre"], 1), "dec": round(v["dec"], 2)} for k, v in tried.items()}}


def probe_cpu(log=print):
    return probe_one("off", log) or {}

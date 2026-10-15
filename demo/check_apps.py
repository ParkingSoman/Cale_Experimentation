#!/usr/bin/env python3
"""Scorecard check for the five Cale demo apps.

Run:  python demo/check_apps.py --hub http://127.0.0.1:47801 --label mac [--out results/demo/mac.json]
                                [--skip chat] [--no-together]

It uses plain HTTP only (standard library). It never opens a browser and never presses a key on the desktop.
It drives each app through the same HTTP calls the app's own web page makes, then reads the numbers from the Hub.
Note: if an app is NOT running yet, POST /api/open on the Hub starts it, and some apps then open a browser tab
by themselves. Start the apps first if the computer must stay quiet.
Exit code is 0 unless this script itself crashes: it reports verdicts, it does not judge the machine.
"""
import argparse
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
APPS = Path(os.environ["CALE_DEMO_APPS"]) if os.environ.get("CALE_DEMO_APPS") else HERE / "apps"      # where the apps' sample files are

QN_STARTS = [
    "Dear Maria, I wanted to follow up on",
    "The quarterly numbers look good, but we should",
    "Ingredients for the cake: two cups of",
]
QN_SLOW_S = 3.0
DOC_FACTS = [  # (any-of groups: every group needs one hit)
    [["5%", "five percent"], ["5 days", "five days", "5th day", "fifth day", "5 calendar", "five (5)", "5-day"]],
    [["60", "sixty"]],
    [["750"], ["30 days", "thirty days", "thirty (30)", "30) days", "30th", "30 calendar"]],
]
SORTER_TIMEOUT_S = int(os.environ.get("CALE_CHECK_SORTER_TIMEOUT_S", "900"))      # a slow computer may need longer: set the variable


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------------------------------- HTTP helpers
def http(method, url, body=None, timeout=30):
    """(status, parsed-json-or-text, content-type). Never raises: returns (0, error text, '') on a network error."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            ct = r.headers.get("Content-Type", "")
            return r.status, _parse(raw, ct), ct
    except urllib.error.HTTPError as e:
        try:
            raw = e.read()
        except Exception:
            raw = b""
        return e.code, _parse(raw, e.headers.get("Content-Type", "")), ""
    except Exception as e:  # URLError, timeout, connection reset, ...
        return 0, "%s: %s" % (type(e).__name__, e), ""


def _parse(raw, ct):
    text = raw.decode("utf-8", "replace")
    if "json" in ct or text[:1] in "{[":
        try:
            return json.loads(text)
        except ValueError:
            pass
    return text


def sse_stream(url, body, timeout=120, deadline_s=600):
    """POST and yield (event, data) from a Server-Sent-Events reply.
    A JSON reply (not a stream) is yielded once as ("json", obj). A network problem yields ("neterror", text)."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    t_end = time.time() + deadline_s
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        yield "neterror", "HTTP %s" % e.code
        return
    except Exception as e:
        yield "neterror", "%s: %s" % (type(e).__name__, e)
        return
    try:
        if "json" in r.headers.get("Content-Type", ""):
            try:
                yield "json", json.loads(r.read().decode("utf-8", "replace"))
            except ValueError:
                yield "neterror", "bad JSON reply"
            return
        event, data_lines = "message", []
        while True:
            if time.time() > t_end:
                yield "neterror", "stream took longer than %d s" % deadline_s
                return
            try:
                raw = r.readline()
            except Exception as e:
                yield "neterror", "%s: %s" % (type(e).__name__, e)
                return
            if not raw:
                return
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line == "":
                if data_lines:
                    try:
                        yield event, json.loads("\n".join(data_lines))
                    except ValueError:
                        pass
                event, data_lines = "message", []
            elif line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
    finally:
        try:
            r.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------------------------------- Hub
class Hub:
    """Reads /api/state and keeps every finished call it ever saw (the Hub's list is short), keyed by call id."""

    def __init__(self, base):
        self.base = base.rstrip("/")
        self.calls = {}
        self.lock = threading.Lock()
        self.last_state = {}

    def state(self):
        st, body, _ = http("GET", self.base + "/api/state", timeout=15)
        if st == 200 and isinstance(body, dict):
            self.last_state = body
            recent = ((body.get("activity") or {}).get("recent")) or []
            with self.lock:
                for c in recent:
                    if isinstance(c, dict) and "id" in c:
                        self.calls[c["id"]] = c
            return body
        return {}

    def mark(self):
        """Highest call id now known: calls with a higher id started after this moment."""
        self.state()
        with self.lock:
            return max(self.calls) if self.calls else 0

    def calls_for(self, cale_app, after_id):
        self.state()
        with self.lock:
            return sorted((c for i, c in self.calls.items() if c.get("app") == cale_app and i > after_id),
                          key=lambda c: c["id"])

    def app_info(self, app_id):
        for a in self.state().get("apps", []) or []:
            if a.get("id") == app_id:
                return a
        return {}

    def open(self, app_id):
        st, body, _ = http("POST", self.base + "/api/open", {"id": app_id}, timeout=60)
        if st == 200 and isinstance(body, dict) and body.get("url"):
            return body["url"].rstrip("/"), None
        msg = body.get("error") if isinstance(body, dict) else str(body)
        return None, "Hub could not open it (HTTP %s): %s" % (st, msg)


def summarize_calls(calls):
    """Numbers from Hub entries: lists and a small summary."""
    def col(k):
        return [c[k] for c in calls if isinstance(c.get(k), (int, float)) and not isinstance(c.get(k), bool)]
    out = {"calls": len(calls)}
    for k in ("size_pct", "tokens_per_s", "first_token_s", "waited_s", "tokens"):
        v = col(k)
        if v:
            out[k] = {"min": round(min(v), 2), "median": round(statistics.median(v), 2), "max": round(max(v), 2)}
            if k == "tokens":
                out[k]["total"] = int(sum(v))
    gm = [c.get("goal_met") for c in calls]
    out["goal_met"] = {"true": sum(1 for g in gm if g is True), "false": sum(1 for g in gm if g is False),
                       "null": sum(1 for g in gm if g is None)}
    out["outcomes"] = sorted({str(c.get("outcome")) for c in calls})
    return out


def missed_goals(calls):
    return [c for c in calls if c.get("goal_met") is False]


def result(verdict, notes, seconds, **numbers):
    return dict({"verdict": verdict, "notes": notes, "seconds": round(seconds, 1)}, **numbers)


# ----------------------------------------------------------------------------------------------------- quick-notes
def qn_ghost(url, text):
    t0 = time.perf_counter()
    st, body, _ = http("POST", url + "/api/ghost", {"text": text}, timeout=90)
    return st, body, time.perf_counter() - t0


def check_quick_notes(hub, url):
    t_begin = time.time()
    mark = hub.mark()
    notes, fails, ms_list = [], [], []
    st0, b0, secs0 = qn_ghost(url, QN_STARTS[0])             # the first call may have to load the model: it is reported apart, as the cold start
    cold_ms = round((b0.get("ms") if isinstance(b0, dict) and isinstance(b0.get("ms"), (int, float)) else secs0 * 1000))
    mark = hub.mark()
    for s in QN_STARTS:
        st, body, secs = qn_ghost(url, s)
        if st != 200 or not isinstance(body, dict):
            fails.append("%r: HTTP %s %s" % (s, st, str(body)[:100]))
            continue
        if body.get("error"):
            fails.append("%r: %s" % (s, body["error"]))
            continue
        text = (body.get("text") or "").strip()
        if not text:
            fails.append("%r: empty suggestion" % s)
            continue
        ms_list.append(body.get("ms") if isinstance(body.get("ms"), (int, float)) else secs * 1000)
        notes.append("%r -> %r" % (s, text[:60]))
    calls = hub.calls_for("Quick Notes", mark)
    num = summarize_calls(calls)
    firsts = [c["first_token_s"] for c in calls if isinstance(c.get("first_token_s"), (int, float))]
    # first word: the Hub's number when it has one; otherwise the app's own whole-call time (an upper bound)
    first_vals = firsts if firsts else [m / 1000.0 for m in ms_list]
    first_max = max(first_vals) if first_vals else None
    verdict, why = "WELL", []
    if fails or len(ms_list) < len(QN_STARTS):
        verdict = "FAIL"
        why = fails or ["not every suggestion came back"]
    else:
        if first_max is not None and first_max > QN_SLOW_S:
            verdict = "POOR"
            why.append("first word %.2f s is slower than %.0f s" % (first_max, QN_SLOW_S))
        if missed_goals(calls):
            verdict = "POOR"
            why.append("%d call(s) missed Cale's goal" % len(missed_goals(calls)))
    if not why:
        why = ["3 of 3 suggestions non-empty"]
    return result(verdict, "; ".join(why) + " | " + " | ".join(notes), time.time() - t_begin,
                  first_word_s=[round(v, 3) for v in first_vals], whole_call_ms=[round(m) for m in ms_list], cold_start_ms=cold_ms,
                  first_word_source="hub" if firsts else "app (whole call)", hub=num)


# ----------------------------------------------------------------------------------------------------- doc-reader
def has_facts(answer, groups):
    low = answer.lower()
    return all(any(alt in low for alt in grp) for grp in groups)


def check_doc_reader(hub, url):
    t_begin = time.time()
    mark = hub.mark()
    st, samp, _ = http("GET", url + "/api/sample", timeout=30)
    if st != 200 or not isinstance(samp, dict) or not samp.get("text"):
        return result("FAIL", "could not get the sample (HTTP %s)" % st, time.time() - t_begin)
    qs = samp.get("questions") or []
    fresh = samp["text"] + "\n\n(Check run %s.)\n" % datetime.now().strftime("%Y%m%d%H%M%S")      # new text, so the reading is not reused from an earlier run
    st, loaded, _ = http("POST", url + "/api/load", {"name": samp.get("name", "lease.md"), "text": fresh}, timeout=120)
    if st != 200 or not isinstance(loaded, dict) or not loaded.get("ok"):
        msg = loaded.get("error") if isinstance(loaded, dict) else loaded
        return result("FAIL", "load failed: %s" % str(msg)[:200], time.time() - t_begin)
    predicted = loaded.get("predicted_s")
    done, err = None, None
    for ev, d in sse_stream(url + "/api/read", {}, timeout=300, deadline_s=900):
        if ev == "neterror":
            err = d
            break
        if ev == "json":
            err = (d.get("error") if isinstance(d, dict) else str(d)) or "unexpected reply"
            break
        if isinstance(d, dict):
            if d.get("error"):
                err = d["error"]
                break
            if d.get("done"):
                done = d
                break
    if not done:
        return result("FAIL", "read failed: %s" % err, time.time() - t_begin)
    read_s = done.get("seconds")
    pred = done.get("predicted_s", predicted)
    answers, fails, firsts, reused = [], [], [], []
    for i, q in enumerate(qs[:3]):
        text, fin, e2 = [], None, None
        for ev, d in sse_stream(url + "/api/ask", {"question": q}, timeout=300, deadline_s=600):
            if ev == "neterror":
                e2 = d
                break
            if ev == "json":
                e2 = (d.get("error") if isinstance(d, dict) else str(d)) or "unexpected reply"
                break
            if isinstance(d, dict):
                if d.get("d"):
                    text.append(d["d"])
                if d.get("error"):
                    e2 = d["error"]
                    break
                if d.get("done"):
                    fin = d
                    break
        ans = "".join(text).strip()
        if e2 or not fin:
            fails.append("Q%d error: %s" % (i + 1, e2 or "no end of answer"))
            continue
        firsts.append(fin.get("first_s"))
        reused.append(fin.get("reused"))
        ok = i < len(DOC_FACTS) and has_facts(ans, DOC_FACTS[i])
        answers.append({"q": q, "ok": ok, "first_s": fin.get("first_s"), "reused_tokens": fin.get("reused"),
                        "tps": fin.get("tps"), "answer": ans[:300]})
        if not ok:
            fails.append("Q%d answer missing key facts: %r" % (i + 1, ans[:120]))
    calls = hub.calls_for("Private Doc Reader", mark)
    num = summarize_calls(calls)
    if len(qs) < 3:
        fails.append("the app returned only %d sample questions" % len(qs))
    verdict, why = "WELL", []
    if fails:
        verdict, why = "FAIL", fails
    else:
        if missed_goals(calls):
            verdict = "POOR"
            why.append("%d call(s) missed Cale's goal" % len(missed_goals(calls)))
        sizes = (num.get("size_pct") or {}) if isinstance(num, dict) else {}
        if isinstance(sizes.get("min"), (int, float)) and sizes["min"] < 100:      # this app asks for the full size, and holds it
            verdict = "POOR"
            why.append("asked for the full size but Cale ran %s%% for some calls" % sizes["min"])
        if isinstance(read_s, (int, float)) and isinstance(pred, (int, float)) and read_s > 2 * pred + 10:
            verdict = "POOR"
            why.append("reading took %.1f s, far above the predicted %.1f s" % (read_s, pred))
        if not why:
            why = ["3 of 3 answers correct"]
    return result(verdict, "; ".join(why), time.time() - t_begin, read_s=read_s, predicted_s=pred,
                  prompt_tokens=done.get("prompt_tokens"), first_word_s=firsts, reused_tokens=reused,
                  answers=answers, hub=num)


# ----------------------------------------------------------------------------------------------------- background-sorter
def expected_amounts(folder):
    """{file name: amount} for files whose text has a total/amount-due line (parsed by regex)."""
    out = {}
    pat = re.compile(r"(?:total paid|order total|amount due|total)\D{0,12}?\$?\s*([0-9]+(?:\.[0-9]{1,2})?)", re.I)
    for p in sorted(Path(folder).glob("*.txt")):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        hits = [m for m in pat.finditer(text) if not re.match(r"sub", text[max(0, m.start() - 3):m.start()], re.I)]
        # prefer a line that starts with a total word, not "Subtotal"
        good = []
        for line in text.splitlines():
            if re.match(r"\s*(subtotal)", line, re.I):
                continue
            m = pat.search(line)
            if m:
                good.append(float(m.group(1)))
        if good and hits:
            out[p.name] = good[-1]
    return out


def sorter_run(base, folder, timeout_s=SORTER_TIMEOUT_S, progress=True, hub=None):
    """Point the sorter at `folder`, wait until it is done. Returns (state or None, seconds, error)."""
    st, _, _ = http("POST", base + "/api/resume", {}, timeout=15)
    st, body, _ = http("POST", base + "/api/folder", {"path": str(folder)}, timeout=30)
    if st != 200:
        return None, 0.0, "could not set the folder (HTTP %s: %s)" % (st, str(body)[:100])
    t0 = time.time()
    next_print = t0 + 30
    n_files = len(list(Path(folder).glob("*.txt")))
    last = None
    while time.time() - t0 < timeout_s:
        if hub is not None:
            hub.state()                       # the Hub keeps only the last 50 calls: collect them as we go
        st, s, _ = http("GET", base + "/api/state", timeout=20)
        if st == 200 and isinstance(s, dict):
            last = s
            same = Path(str(s.get("folder", ""))).resolve() == Path(folder).resolve()
            if same and s.get("total", 0) >= n_files and (s.get("done", 0) + s.get("failed", 0)) >= s.get("total", 0) \
                    and s.get("status") in ("done", "idle"):
                return s, time.time() - t0, None
        if progress and time.time() >= next_print:
            next_print += 30
            if last:
                log("    sorter: %s/%s files, status %s" % (last.get("done"), last.get("total"), last.get("status")))
        time.sleep(2)
    return last, time.time() - t0, "not finished after %d s" % timeout_s


def judge_sorter(state, folder):
    rows = state.get("rows") or []
    fields = ("kind", "date", "title", "amount", "summary")
    kinds = {"receipt", "meeting_note", "idea", "todo", "email", "other"}
    problems = []
    if len(rows) != 20:
        problems.append("%d rows, expected 20" % len(rows))
    for r in rows:
        if r.get("error"):
            problems.append("%s: error %s" % (r.get("file"), r["error"]))
        elif r.get("kind") not in kinds or any(f not in r for f in fields):
            problems.append("%s: missing kind or fields" % r.get("file"))
    by_file = {r.get("file"): r for r in rows}
    exp = expected_amounts(folder)
    checked = 0
    for name, amount in exp.items():
        r = by_file.get(name)
        if not r:
            continue
        checked += 1
        got = r.get("amount")
        if r.get("kind") != "receipt":
            problems.append("%s: kind %s, expected receipt" % (name, r.get("kind")))
        if not isinstance(got, (int, float)) or abs(got - amount) > 0.005:
            problems.append("%s: amount %s, text says %.2f" % (name, got, amount))
    return problems, checked


def pause_sorter(url):
    http("POST", url + "/api/pause", {}, timeout=15)


def check_sorter(hub, url, orig_folder_holder):
    t_begin = time.time()
    mark = hub.mark()
    tmp = Path(tempfile.mkdtemp(prefix="cale_check_sorter_"))
    folder = tmp / "sample"
    try:
        shutil.copytree(APPS / "background-sorter" / "sample", folder)
        st, s0, _ = http("GET", url + "/api/state", timeout=20)
        if isinstance(s0, dict):
            orig_folder_holder["folder"] = s0.get("folder")
        log("    sorting a fresh copy of the sample folder (timeout 15 min)")
        state, secs, err = sorter_run(url, folder, hub=hub)
        if err:
            pause_sorter(url)                       # stop its work: it must not slow the next checks
        calls = hub.calls_for("Background Sorter", mark)
        if err and state and (state.get("rows") or []):          # slow, not wrong: report the part that was done
            n = len(state.get("rows") or [])
            return result("POOR", "%d of 20 files in %d s: slow for a background job on this computer" % (n, secs), time.time() - t_begin,
                          files=n, sort_seconds=round(secs, 1), files_per_min=round(n / secs * 60, 2), size=state.get("size"),
                          hub=summarize_calls(calls))
        if err or not state:
            return result("FAIL", err or "no state", time.time() - t_begin, hub=summarize_calls(calls))
        problems, checked = judge_sorter(state, folder)
        num = summarize_calls(calls)
        n = len(state.get("rows") or [])
        verdict, why = "WELL", []
        if problems:
            verdict, why = "FAIL", problems[:6]
        else:
            if missed_goals(calls):
                verdict = "POOR"
                why.append("%d call(s) missed Cale's goal" % len(missed_goals(calls)))
            else:
                why.append("20 of 20 files sorted, %d receipt amounts match" % checked)
        fpm = round(n / secs * 60, 1) if secs > 0 else None
        return result(verdict, "; ".join(why), time.time() - t_begin, files=n, sort_seconds=round(secs, 1),
                      files_per_min=fpm, app_files_per_min=state.get("files_per_min"), size=state.get("size"),
                      memory_pct_actual=state.get("memory_pct_actual"), receipts_checked=checked, hub=num)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------------------------------- meeting-recap
def check_meeting_recap(hub, url):
    t_begin = time.time()
    mark = hub.mark()
    try:
        text = (APPS / "meeting-recap" / "sample" / "standup.txt").read_text(encoding="utf-8").strip()
    except OSError as e:
        return result("FAIL", "cannot read the sample: %s" % e, 0.0)
    recap, items, done, err = [], None, None, None
    first_local = None
    t0 = time.time()
    for ev, d in sse_stream(url + "/api/run", {"text": text}, timeout=300, deadline_s=900):
        if ev == "neterror":
            err = d
            break
        if not isinstance(d, dict):
            continue
        if ev == "recap":
            if first_local is None and d.get("text"):
                first_local = time.time() - t0
            recap.append(d.get("text", ""))
        elif ev == "actions":
            items = d.get("items")
        elif ev == "actions_error":
            err = d.get("message")
        elif ev == "fail":
            err = d.get("message")
            break
        elif ev == "done":
            done = d
            break
    total = time.time() - t0
    recap_text = "".join(recap).strip()
    calls = hub.calls_for("Meeting Recap", mark)
    num = summarize_calls(calls)
    good_items = [i for i in (items or []) if isinstance(i, dict) and str(i.get("owner") or "").strip() and str(i.get("task") or "").strip()]
    problems = []
    if err:
        problems.append(str(err)[:150])
    if not recap_text:
        problems.append("recap is empty")
    if len(good_items) < 4:
        problems.append("only %d action items with owner and task (need 4)" % len(good_items))
    if not done and not err:
        problems.append("no 'done' event")
    verdict, why = "WELL", []
    if problems:
        verdict, why = "FAIL", problems
    else:
        if missed_goals(calls):
            verdict = "POOR"
            why.append("%d call(s) missed Cale's goal" % len(missed_goals(calls)))
        else:
            why.append("recap written, %d action items" % len(good_items))
    first = (done or {}).get("first_s", first_local)
    return result(verdict, "; ".join(why), time.time() - t_begin, first_word_s=first, whole_s=round(total, 1),
                  action_items=len(good_items), recap_chars=len(recap_text), size=(done or {}).get("size"), hub=num)


# ----------------------------------------------------------------------------------------------------- together
def together(hub, sorter_url, qn_url, orig_folder_holder):
    t_begin = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="cale_check_together_"))
    folder = tmp / "sample"
    out = {}
    try:
        shutil.copytree(APPS / "background-sorter" / "sample", folder)
        st, s0, _ = http("GET", sorter_url + "/api/state", timeout=20)
        if isinstance(s0, dict) and not orig_folder_holder.get("folder"):
            orig_folder_holder["folder"] = s0.get("folder")
        mark = hub.mark()
        cpu = []
        st, body, _ = http("POST", sorter_url + "/api/resume", {}, timeout=15)
        st, body, _ = http("POST", sorter_url + "/api/folder", {"path": str(folder)}, timeout=30)
        if st != 200:
            return {"error": "could not start the sorter (HTTP %s)" % st}
        time.sleep(3)  # let the first file start
        firsts, whole = [], []
        for i in range(5):
            st, b, secs = qn_ghost(qn_url, QN_STARTS[i % 3])
            ok = st == 200 and isinstance(b, dict) and (b.get("text") or "").strip()
            whole.append(round((b.get("ms") if isinstance(b, dict) and isinstance(b.get("ms"), (int, float)) else secs * 1000)))
            h = hub.state().get("host") or {}
            if isinstance(h.get("cpu_busy"), (int, float)):
                cpu.append(h["cpu_busy"])
            log("    together %d/5: %s, %d ms" % (i + 1, "ok" if ok else "FAILED", whole[-1]))
            time.sleep(2)
        sorter_done = http("GET", sorter_url + "/api/state", timeout=20)[1]
        sorter_busy = isinstance(sorter_done, dict) and sorter_done.get("status") == "running"
        qn_calls = hub.calls_for("Quick Notes", mark)
        firsts = [c["first_token_s"] for c in qn_calls if isinstance(c.get("first_token_s"), (int, float))]
        sorter_calls = hub.calls_for("Background Sorter", mark)
        out = {"quick_notes_first_word_s": [round(v, 3) for v in firsts],
               "quick_notes_whole_call_ms": whole,
               "quick_notes_hub": summarize_calls(qn_calls),
               "sorter_calls_during": len(sorter_calls),
               "sorter_still_running_after_5": sorter_busy,
               "host_cpu_busy": {"samples": [round(c, 3) for c in cpu], "max": round(max(cpu), 3) if cpu else None}}
        # wait for the sorter to finish so the next run starts quiet (but do not wait forever)
        log("    waiting for the sorter to finish the copy")
        _, _, err2 = sorter_run(sorter_url, folder, timeout_s=SORTER_TIMEOUT_S, progress=False, hub=hub)
        if err2:
            pause_sorter(sorter_url)
        out["seconds"] = round(time.time() - t_begin, 1)
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Scorecard check for the Cale demo apps")
    ap.add_argument("--hub", default="http://127.0.0.1:47801")
    ap.add_argument("--label", default="check")
    ap.add_argument("--out", default=None, help="JSON file to write (default results/demo/<label>.json)")
    ap.add_argument("--skip", action="append", default=[], help="app id to skip (repeat or use commas)")
    ap.add_argument("--no-together", action="store_true")
    ap.add_argument("--apps", default=None, help="folder that holds the demo apps (default: apps next to this script, or CALE_DEMO_APPS)")
    args = ap.parse_args(argv)
    if args.apps:
        global APPS
        APPS = Path(args.apps)
    skip = {x.strip() for s in args.skip for x in s.split(",") if x.strip()}

    hub = Hub(args.hub)
    st0 = hub.state()
    if not st0:
        log("Cannot reach the Hub at %s. Is it running?" % args.hub)
    apps = {a["id"]: a for a in (st0.get("apps") or []) if isinstance(a, dict)}
    report = {"label": args.label, "when": datetime.now().astimezone().isoformat(timespec="seconds"),
              "hub": args.hub, "host": st0.get("host"), "apps": {}}
    checks = [("chat", None), ("quick-notes", check_quick_notes), ("doc-reader", check_doc_reader),
              ("meeting-recap", check_meeting_recap), ("background-sorter", None)]      # the sorter runs last: a slow one must not disturb the others
    urls = {}
    orig_folder = {}
    sorter_info = apps.get("background-sorter")
    if sorter_info and sorter_info.get("up"):       # it sorts by itself whenever it runs: stop it, or it slows every other check
        pause_sorter("http://127.0.0.1:%s" % sorter_info["port"])
    for app_id, fn in checks:
        t0 = time.time()
        log("== %s" % app_id)
        info = apps.get(app_id)
        try:
            if app_id in skip:
                res = result("SKIPPED", "skipped by --skip", 0.0)
            elif info is None:
                res = result("FAIL", "the Hub does not list this app", 0.0)
            elif not info.get("present"):
                res = result("SKIPPED", "not installed on this computer (present: false)", 0.0)
            elif app_id == "chat":
                res = result("SKIPPED", "the Chat app needs its own pip install; this check does not drive it", 0.0)
            else:
                url, err = hub.open(app_id)
                if err:
                    res = result("FAIL", err, time.time() - t0)
                else:
                    urls[app_id] = url
                    if app_id == "background-sorter":
                        res = check_sorter(hub, url, orig_folder)
                    else:
                        res = fn(hub, url)
        except Exception as e:  # the script must not show a stack trace
            res = result("FAIL", "check crashed: %s: %s" % (type(e).__name__, e), time.time() - t0)
        report["apps"][app_id] = res
        log("   %s  %s" % (res["verdict"], res["notes"][:160]))

    if not args.no_together:
        log("== together (sorter running + quick-notes typing)")
        if "quick-notes" in urls and "background-sorter" in urls and "quick-notes" not in skip and "background-sorter" not in skip:
            try:
                tg = together(hub, urls["background-sorter"], urls["quick-notes"], orig_folder)
                quiet = report["apps"].get("quick-notes", {}).get("first_word_s") or []
                tg["quiet_first_word_s"] = quiet
                if quiet and tg.get("quick_notes_first_word_s"):
                    q, t = statistics.median(quiet), statistics.median(tg["quick_notes_first_word_s"])
                    tg["median_quiet_s"], tg["median_together_s"] = round(q, 3), round(t, 3)
                    tg["slowdown_x"] = round(t / q, 2) if q > 0 else None
                report["together"] = tg
            except Exception as e:
                report["together"] = {"error": "%s: %s" % (type(e).__name__, e)}
        else:
            report["together"] = {"error": "needs both quick-notes and background-sorter"}
        log("   %s" % json.dumps(report["together"])[:300])

    # leave the sorter on the folder it had before we started
    if orig_folder.get("folder") and "background-sorter" in urls:
        http("POST", urls["background-sorter"] + "/api/folder", {"path": orig_folder["folder"]}, timeout=15)

    # table
    log("")
    log("%-18s %-8s %s" % ("app", "verdict", "numbers"))
    for app_id, _ in checks:
        r = report["apps"].get(app_id)
        if not r:
            continue
        bits = []
        if r.get("first_word_s") is not None:
            fw = r["first_word_s"]
            bits.append("first word %s s" % (fw if not isinstance(fw, list) else ",".join("%.2f" % v for v in fw if v is not None)))
        if r.get("read_s") is not None:
            bits.append("read %s s (predicted %s)" % (r.get("read_s"), r.get("predicted_s")))
        if r.get("files_per_min") is not None:
            bits.append("%s files/min" % r["files_per_min"])
        if r.get("whole_s") is not None:
            bits.append("whole %s s" % r["whole_s"])
        size = (r.get("hub") or {}).get("size_pct")
        if size:
            bits.append("size %s%%" % size["median"])
        tps = (r.get("hub") or {}).get("tokens_per_s")
        if tps:
            bits.append("%s tok/s" % tps["median"])
        log("%-18s %-8s %s" % (app_id, r["verdict"], ", ".join(bits)))
        if r["verdict"] in ("POOR", "FAIL", "SKIPPED"):
            log("%-18s          %s" % ("", r["notes"][:140]))
    tg = report.get("together")
    if tg and not tg.get("error"):
        log("together: quick-notes median first word %s s quiet vs %s s with the sorter running; host cpu_busy max %s"
            % (tg.get("median_quiet_s"), tg.get("median_together_s"), (tg.get("host_cpu_busy") or {}).get("max")))

    out = Path(args.out) if args.out else ROOT / "results" / "demo" / ("%s.json" % args.label)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        log("saved " + str(out))
    except OSError as e:
        log("could not save %s: %s" % (out, e))
    return 0


if __name__ == "__main__":
    sys.exit(main())

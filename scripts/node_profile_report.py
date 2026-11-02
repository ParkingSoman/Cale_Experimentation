"""Summarize .cpuprofile files from scripts/node_profile_preload.js: where the start-up CPU time goes. usage: python scripts/node_profile_report.py PREFIX [--top 25]"""
import collections
import glob
import json
import sys

prefix = sys.argv[1]
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
for f in sorted(glob.glob(prefix + ".*.cpuprofile")):
    d = json.load(open(f))
    nodes = {n["id"]: n for n in d["nodes"]}
    parent = {c: n["id"] for n in d["nodes"] for c in n.get("children", [])}
    self_t, by_file = collections.Counter(), collections.Counter()
    busy = 0.0
    for s, t in zip(d["samples"], d["timeDeltas"]):
        cf = nodes[s]["callFrame"]
        if cf["functionName"] in ("(idle)", "(program)", "(garbage collector)"):
            if cf["functionName"] == "(garbage collector)":
                self_t[("(garbage collector)", "", 0)] += t
                busy += t
            continue
        busy += t
        self_t[(cf["functionName"] or "(anonymous)", cf["url"].split("/")[-1][:48], cf["lineNumber"])] += t
        by_file[cf["url"].split("node_modules/")[-1][:60] if "node_modules" in cf["url"] else cf["url"].split("/")[-1][:60]] += t
    print(f"\n{f.split('/')[-1]}: busy {busy / 1000:.0f} ms")
    for (fn, url, line), t in self_t.most_common(top):
        print(f"  {t / 1000:7.0f} ms  {fn[:36]:36s} {url}:{line}")
    print("  by file:", [(k, round(v / 1000)) for k, v in by_file.most_common(8)])

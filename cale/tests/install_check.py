"""The installer copies a fixed list of files from nest/. Every nest module that Cale imports must be on it. Run: python cale/tests/install_check.py"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cale.install import NEST_FILES  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def nest_imports(path):
    """The nest modules a file imports (import nest.x, from nest.x import y, from nest import x, and relative imports inside nest/)."""
    found = set()
    tree = ast.parse(path.read_text())
    inside_nest = path.parent.name == "nest"
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            if n.module and n.module.startswith("nest") and n.level == 0:
                parts = n.module.split(".")
                if len(parts) > 1:
                    found.add(parts[1])
                else:
                    found.update(a.name for a in n.names)
            elif inside_nest and n.level == 1:
                if n.module:
                    found.add(n.module.split(".")[0])
                else:
                    found.update(a.name for a in n.names)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("nest."):
                    found.add(a.name.split(".")[1])
    return found


needed, todo, seen = set(), [*ROOT.glob("cale/*.py"), *ROOT.glob("cale/backends/*.py")], set()
while todo:
    f = todo.pop()
    if f in seen:
        continue
    seen.add(f)
    for mod in nest_imports(f):
        needed.add(mod)
        target = ROOT / "nest" / f"{mod}.py"
        if target.exists():
            todo.append(target)
missing = sorted(m for m in needed if f"{m}.py" not in NEST_FILES and m != "__init__")
check("every nest module that cale imports is in the installer's list", not missing, missing)
check("every file on the list exists", all((ROOT / "nest" / f).exists() for f in NEST_FILES), [f for f in NEST_FILES if not (ROOT / "nest" / f).exists()])
print(f"(cale needs: {', '.join(sorted(needed))})")
print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)

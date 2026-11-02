"""The installer moves its own unpacked chat folder (no second copy of 60,000 files) and copies any other chat folder. Run: python cale/tests/install_chat_move.py
No model, no network.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cale import install  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:240]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def bundle(root):
    (root / "server" / "storage").mkdir(parents=True)
    (root / "collector" / "hotdir").mkdir(parents=True)
    (root / "server" / "index.js").write_text("// server")
    for i in range(300):
        (root / "server" / f"f{i}.js").write_text("x" * 50)
    (root / "bundle.json").write_text("{}")


# 1. the installer's own copy: <home>/app/chat, and the code runs from <home>/app
home = Path(tempfile.mkdtemp(prefix="cale-move-test-"))
app = home / "app"
bundle(app / "chat")
real_src = install.SRC
install.SRC = app
try:
    t = time.time()
    ok = install.install_chat(home)
    took = time.time() - t
    check("the install works", ok is True)
    check("the chat is in <home>/chat", (home / "chat" / "server" / "index.js").exists())
    check("the unpacked copy is gone (moved, not copied)", not (app / "chat").exists())
    check("it took no time", took < 1.0, took)
    check("the folders made at run time are not shipped", not (home / "chat" / "server" / "storage").exists() and not (home / "chat" / "collector" / "hotdir").exists())
    check("a leftover chat.new is gone", not (home / "chat.new").exists())
    # again, with an older chat in place: it is replaced
    bundle(app / "chat")
    (home / "chat" / "old.txt").write_text("old")
    install.install_chat(home)
    check("an older chat is replaced", not (home / "chat" / "old.txt").exists() and (home / "chat" / "server" / "index.js").exists())
finally:
    install.SRC = real_src

# 2. any other chat folder is copied and stays where it is (a developer's dist/chat)
home2 = Path(tempfile.mkdtemp(prefix="cale-copy-test-"))
dev = Path(tempfile.mkdtemp(prefix="cale-dev-chat-")) / "chat"
bundle(dev)
os.environ["CALE_CHAT_SRC"] = str(dev)
try:
    install.install_chat(home2)
    check("another chat folder is copied", (home2 / "chat" / "server" / "index.js").exists())
    check("...and the original stays", (dev / "server" / "index.js").exists())
finally:
    del os.environ["CALE_CHAT_SRC"]

# 3. a chat folder inside <home>/app but code running from elsewhere: copied
home3 = Path(tempfile.mkdtemp(prefix="cale-copy2-test-"))
bundle(home3 / "app" / "chat")
os.environ["CALE_CHAT_SRC"] = str(home3 / "app" / "chat")
try:
    install.install_chat(home3)
    check("code not running from <home>/app: copied, not moved", (home3 / "app" / "chat" / "server" / "index.js").exists() and (home3 / "chat" / "server" / "index.js").exists())
finally:
    del os.environ["CALE_CHAT_SRC"]

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)

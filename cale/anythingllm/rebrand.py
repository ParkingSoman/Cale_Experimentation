"""Make AnythingLLM look like Cale. A patch for the SOURCE of AnythingLLM's frontend, applied before `yarn build`.

    python -m cale.anythingllm.rebrand <frontend folder> [--icon icon.png] [--wordmark wordmark.png]

Not a fork. Every step looks for a known text and stops with a clear message if it is not there, so a new AnythingLLM
version cannot be patched half way without anybody noticing. Run it again after a version bump and fix what it reports.

What the person sees afterwards: the name "Cale", the kale icon, garden colours, no settings pages about providers,
embedders, vector databases or models, no links to the AnythingLLM web site, and an "About Cale" page
(public/about.html) with the MIT notice of AnythingLLM. The only place the word "AnythingLLM" stays is that page.
"""
import re
import shutil
import sys
from pathlib import Path

KALE, CREAM, LEAF = "#2f5d50", "#f7f5ee", "rgb(90, 148, 102)"
TESTED_VERSION = "1.18.0"


class PatchError(Exception):
    pass


def _read(p):
    return Path(p).read_text(encoding="utf-8")


def _swap(path, old, new, count=1):
    """Replace `old` with `new` in a file. It must be there."""
    text = _read(path)
    if old not in text:
        raise PatchError(f"{path}: the text to patch was not found: {old[:70]!r}")
    Path(path).write_text(text.replace(old, new, count), encoding="utf-8")


def _sub(path, pattern, new, flags=re.S):
    text = _read(path)
    out, n = re.subn(pattern, lambda m: new, text, count=1, flags=flags)
    if n != 1:
        raise PatchError(f"{path}: the pattern to patch was not found: {pattern[:70]!r}")
    Path(path).write_text(out, encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------------
THEME_CSS = f"""
/* ---- Cale garden theme (appended by cale/anythingllm/rebrand.py) ---- */
[data-theme="light"] {{
  --theme-bg-primary: {CREAM};
  --theme-bg-secondary: #ffffff;
  --theme-bg-sidebar: #e9efe6;
  --theme-bg-container: {CREAM};
  --theme-bg-chat: {CREAM};
  --theme-bg-chat-input: #ffffff;
  --theme-popup-menu-bg: #dfe8dc;
  --theme-text-primary: #1f2d28;
  --theme-text-secondary: #5d6f67;
  --theme-sidebar-item-default: #f7f5ee;
  --theme-sidebar-item-selected: #ffffff;
  --theme-sidebar-item-hover: #d9e6d8;
  --theme-sidebar-item-text-active: {KALE};
  --theme-sidebar-subitem-selected: #dfe8dc;
  --theme-sidebar-subitem-hover: #dfe8dc;
  --theme-sidebar-border: #cfdccf;
  --theme-chat-input-border: #c4d3c4;
  --theme-button-primary: {KALE};
  --theme-button-cta: {LEAF};
  --theme-home-button-primary: {KALE};
  --theme-home-button-primary-hover: rgba(47, 93, 80, 0.9);
  --theme-file-row-selected-even: {LEAF};
  --theme-file-row-selected-odd: {LEAF};
  --theme-file-picker-hover: #dfe8dc;
  --theme-settings-input-bg: #ffffff;
}}
:root:not([data-theme="light"]) {{
  --theme-bg-primary: #14231e;
  --theme-bg-secondary: #1b2e27;
  --theme-bg-sidebar: #14231e;
  --theme-bg-container: #14231e;
  --theme-bg-chat: #1b2e27;
  --theme-bg-chat-input: #243b32;
  --theme-popup-menu-bg: #0f1b17;
  --theme-text-primary: {CREAM};
  --theme-sidebar-item-selected: rgba(247, 245, 238, 0.25);
  --theme-sidebar-item-hover: #2d4a3f;
  --theme-sidebar-border: rgba(247, 245, 238, 0.12);
  --theme-chat-input-border: #4a6a5c;
  --theme-button-primary: {LEAF};
  --theme-button-cta: {LEAF};
  --theme-home-button-primary: {LEAF};
  --theme-settings-input-bg: #14231e;
}}
/* Links and buttons that lead to the AnythingLLM web site, to its community, or to pages about providers and models */
a[href^="mailto:"], a[href*="/settings/privacy"], a[href*="github.com"], a[href*="docs.anythingllm.com"],
a[href*="discord"], a[href*="useanything.com"], a[href*="anythingllm.com"] {{ display: none !important; }}
"""


def footer_component():
    return '''import React from "react";
import { isMobile } from "react-device-detect";
import { Info } from "@phosphor-icons/react";
import SettingsButton from "../SettingsButton";

// Cale: the original footer links to the AnythingLLM web site and its chat community. These icons stay (other files import them).
export const MAX_ICONS = 3;
export const ICON_COMPONENTS = { Info };

export default function Footer() {
  return (
    <div className="flex justify-center items-center mb-2 gap-x-4">
      <a
        href="/about.html"
        className="text-xs text-theme-text-secondary hover:text-theme-text-primary transition-all duration-300 flex items-center gap-x-1"
        aria-label="About Cale"
      >
        <Info weight="fill" className="h-4 w-4" />
        About Cale
      </a>
      {!isMobile && <SettingsButton />}
    </div>
  );
}
'''


ABOUT_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>About Cale</title>
<style>
 body{margin:0;background:#f7f5ee;color:#1f2d28;font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
 main{max-width:640px;margin:0 auto;padding:40px 24px}
 h1{color:#2f5d50;margin:0 0 4px} a{color:#2f5d50} pre{white-space:pre-wrap;font:12px/1.5 ui-monospace,Menlo,monospace;background:#e9efe6;padding:16px;border-radius:10px}
 .back{display:inline-block;margin-bottom:24px}
</style></head><body><main>
<a class="back" href="/">&larr; Back to chat</a>
<h1>Cale</h1>
<p>Your private assistant. It runs on this computer. Your chats and files stay here.</p>
<h2>Credit</h2>
<p>This chat window is built on AnythingLLM by Mintplex Labs, which is free software under the MIT licence. Thank you to its authors.</p>
<pre>The MIT License

Copyright (c) Mintplex Labs Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.</pre>
</main></body></html>
"""

# Words the person reads in the chat screens. {old text in locales/en/common.js: new text}
WORDING = {
    'thought_in_progress: "Model is Thinking..."': 'thought_in_progress: "Cale is thinking..."',
    'this will stop the model from generating the response': 'this will stop Cale from writing the answer',
}


def apply(frontend, icon=None, wordmark=None):
    fe = Path(frontend)
    src, pub = fe / "src", fe / "public"
    version = re.search(r'"version":\s*"([^"]+)"', _read(fe / "package.json")).group(1)
    if version != TESTED_VERSION:
        print(f"rebrand: tested with AnythingLLM {TESTED_VERSION}, this is {version}. Look at every message below.", file=sys.stderr)

    # 1. words: the name, then the few model words on chat screens
    loc = src / "locales" / "en" / "common.js"
    text = _read(loc).replace("AnythingLLM", "Cale")
    for old, new in WORDING.items():
        if old not in text:
            raise PatchError(f"{loc}: wording not found: {old!r}")
        text = text.replace(old, new)
    loc.write_text(text, encoding="utf-8")
    for f in ("index.html", "public/manifest.json"):
        p = fe / f
        if p.exists():
            p.write_text(_read(p).replace("AnythingLLM | Your personal LLM trained on anything", "Cale").replace("AnythingLLM", "Cale"), encoding="utf-8")

    # 2. pictures: the wide logo (header), the icon, the favicon
    if wordmark:
        for p in (src / "media/logo/anything-llm.png", src / "media/logo/anything-llm-dark.png",
                  pub / "anything-llm-light.png", pub / "anything-llm-dark.png"):
            shutil.copyfile(wordmark, p)
    if icon:
        shutil.copyfile(icon, src / "media/logo/anything-llm-icon.png")
        shutil.copyfile(icon, pub / "favicon.png")
        if (pub / "favicon.ico").exists():
            (pub / "favicon.ico").unlink()
    (pub / "about.html").write_text(ABOUT_HTML, encoding="utf-8")

    # 3. colours
    with open(src / "index.css", "a", encoding="utf-8") as f:
        f.write(THEME_CSS)

    # 4. footer: no links to the AnythingLLM site; one "About Cale" link
    (src / "components/Footer/index.jsx").write_text(footer_component(), encoding="utf-8")

    # 5. settings: only Appearance stays. Nothing about providers, embedders, vector databases or models.
    side = src / "components/SettingsSidebar/index.jsx"
    _sub(side, r"const SidebarOptions = \(.*?\n\);\n\nfunction HoldToReveal",
         '''const SidebarOptions = ({ user = null, t }) => (
  <Option
    btnText={t("settings.interface")}
    icon={<PencilSimpleLine className="h-5 w-5 flex-shrink-0" />}
    href={paths.settings.interface()}
    user={user}
    flex={true}
    roles={["admin", "manager"]}
  />
);

function HoldToReveal''')
    _sub(side, r"function SupportEmail\(\) \{.*?\n\}\n", "function SupportEmail() {\n  return null;\n}\n")
    _sub(side, r"function AppVersion\(\) \{.*?\n\}\n", "function AppVersion() {\n  return null;\n}\n")
    print("rebrand: done")


def main(argv):
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    args = dict(zip(argv[1::2], argv[2::2]))
    try:
        apply(argv[0], icon=args.get("--icon"), wordmark=args.get("--wordmark"))
    except PatchError as e:
        print(f"rebrand: STOP. {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

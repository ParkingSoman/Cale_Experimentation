"""Make AnythingLLM look like Cale. A patch for the SOURCE of AnythingLLM's frontend, applied before `yarn build`.

    python -m cale.anythingllm.rebrand <frontend folder> [--carrot-svg f] [--carrot-png f] [--icon f] [--wordmark f] [--wordmark-dark f]

Not a fork. Every step looks for a known text and stops with a clear message if it is not there, so a new AnythingLLM
version cannot be patched half way without anybody noticing. Run it again after a version bump and fix what it reports.

What the person sees afterwards: the name "Cale", the carrot mascot (cale/brand/carrot.svg), the carrot colours
(cale/brand/README.md), no settings pages about providers, embedders, vector databases or models, no links to the
AnythingLLM web site, and an "About Cale" page (public/about.html) with the MIT notice of AnythingLLM.
The only place the word "AnythingLLM" stays is that page.
"""
import re
import shutil
import sys
from pathlib import Path

CARROT, BROWN, CREAM, SKY = "#f58a1f", "#3b2314", "#fff1d6", "#8fd3f4"      # cale/brand/README.md
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
GREETING = "Hi, I am Cale. Ask me anything."

THEME_CSS = f"""
/* ---- Cale look: carrot, cream, brown (appended by cale/anythingllm/rebrand.py) ---- */
html, body, #root {{ background-color: var(--theme-bg-primary); }}
body, button, input, textarea, select {{
  font-family: ui-rounded, "SF Pro Rounded", "Arial Rounded MT Bold", "Nunito", system-ui, sans-serif !important;
}}
[data-theme="light"] {{
  --theme-loader: {CARROT};
  --theme-bg-primary: {CREAM};
  --theme-bg-secondary: #fff8e8;
  --theme-bg-sidebar: #ffe7b3;
  --theme-bg-container: {CREAM};
  --theme-bg-chat: #fff8e8;
  --theme-bg-chat-input: #ffffff;
  --theme-popup-menu-bg: #ffe7b3;
  --theme-text-primary: {BROWN};
  --theme-text-secondary: #7a5238;
  --theme-placeholder: #a07a5e;
  --theme-sidebar-item-default: {CREAM};
  --theme-sidebar-item-selected: {SKY};
  --theme-sidebar-item-hover: #ffd98f;
  --theme-sidebar-item-text-active: {BROWN};
  --theme-sidebar-subitem-selected: {SKY};
  --theme-sidebar-subitem-hover: #ffd98f;
  --theme-sidebar-thread-selected: {SKY};
  --theme-sidebar-border: {BROWN};
  --theme-chat-input-border: {BROWN};
  --theme-action-menu-bg: #ffe7b3;
  --theme-settings-input-bg: #fff8e8;
  --theme-settings-input-text: {BROWN};
  --theme-modal-border: {BROWN};
  --theme-button-primary: {CARROT};
  --theme-button-primary-hover: #ffa447;
  --theme-button-cta: {CARROT};
  --theme-button-text: {BROWN};
  --theme-file-row-even: #fff8e8;
  --theme-file-row-odd: {CREAM};
  --theme-file-row-selected-even: {SKY};
  --theme-file-row-selected-odd: {SKY};
  --theme-file-picker-hover: #ffd98f;
  --theme-home-button-primary: {CARROT};
  --theme-home-button-primary-hover: #ffa447;
  --theme-home-button-secondary: #ffe7b3;
  --theme-home-bg-card: #fff8e8;
}}
:root:not([data-theme="light"]) {{
  --theme-loader: {CARROT};
  --theme-bg-primary: #1e140e;
  --theme-bg-secondary: #2a1c13;
  --theme-bg-sidebar: #1e140e;
  --theme-bg-container: #1e140e;
  --theme-bg-chat: #2a1c13;
  --theme-bg-chat-input: #3a2819;
  --theme-popup-menu-bg: #140c08;
  --theme-text-primary: {CREAM};
  --theme-text-secondary: rgba(255, 241, 214, 0.65);
  --theme-sidebar-item-default: rgba(255, 241, 214, 0.08);
  --theme-sidebar-item-selected: rgba(245, 138, 31, 0.35);
  --theme-sidebar-item-hover: #4a321f;
  --theme-sidebar-subitem-default: rgba(255, 241, 214, 0.05);
  --theme-sidebar-border: rgba(255, 241, 214, 0.18);
  --theme-chat-input-border: #7a5238;
  --theme-action-menu-bg: #3a2819;
  --theme-settings-input-bg: #1e140e;
  --theme-modal-border: #5a3c26;
  --theme-button-primary: {CARROT};
  --theme-button-primary-hover: #ffa447;
  --theme-button-cta: {CARROT};
  --theme-home-button-primary: {CARROT};
  --theme-home-button-primary-hover: #ffa447;
  --theme-file-row-even: #1e140e;
  --theme-file-row-odd: #2a1c13;
  --theme-file-row-selected-even: rgba(245, 138, 31, 0.3);
  --theme-file-row-selected-odd: rgba(245, 138, 31, 0.2);
}}
/* A few boxes have their colours written into the page code, not in variables */
:root:not([data-theme="light"]) .bg-zinc-900 {{ background-color: #2a1c13 !important; }}
:root:not([data-theme="light"]) .bg-zinc-800 {{ background-color: #3a2819 !important; }}
:root:not([data-theme="light"]) .bg-zinc-700, :root:not([data-theme="light"]) .hover\\:bg-zinc-700:hover {{ background-color: #4a321f !important; }}
[data-theme="light"] .light\\:bg-white {{ background-color: #fff8e8 !important; }}
[data-theme="light"] .light\\:border-slate-300, [data-theme="light"] .light\\:border-theme-modal-border {{ border-color: {BROWN} !important; border-width: 2px !important; }}
/* Thick brown outline on the message box and on buttons: the cartoon look */
[data-theme="light"] .rounded-\\[20px\\] {{ border: 3px solid {BROWN} !important; background-color: #ffffff !important; }}
:root:not([data-theme="light"]) .rounded-\\[20px\\] {{ border: 3px solid #7a5238 !important; }}
button.rounded-full {{ border: 2px solid {BROWN}; color: {BROWN}; background-color: {CARROT} !important; }}
:root:not([data-theme="light"]) button.rounded-full {{ border-color: {BROWN}; }}
button.rounded-full svg, button.rounded-full svg * {{ color: {BROWN} !important; }}
.bg-theme-button-primary, .bg-primary-button {{ color: {BROWN} !important; }}
/* The wrench and the "back" button at the bottom of the sidebar */
a.bg-theme-sidebar-footer-icon {{ background-color: {CARROT} !important; border: 2px solid {BROWN}; }}
a.bg-theme-sidebar-footer-icon svg {{ color: {BROWN} !important; }}
a.bg-theme-sidebar-footer-icon:hover {{ background-color: #ffa447 !important; }}
/* Boxes: thick brown outline in the light theme */
[data-theme="light"] [class*="border-theme-modal-border"], [data-theme="light"] .bg-theme-bg-secondary {{ border: 2px solid {BROWN}; }}
/* Text on a carrot-coloured box is brown, never white */
[class*="bg-sky-"], [class*="bg-blue-"], [class*="bg-cyan-"], .bg-theme-button-primary, .bg-theme-home-button-primary {{ color: {BROWN} !important; }}
/* Every orange button and pill: on hover and keyboard focus a lighter Carrot, pressed a deeper Carrot; brown text and outline stay.
   (The page's own hover colours are grey or brown and would hide the label. The doubled :root makes this win over them.) */
:root:root :is(button.rounded-full, a.bg-theme-sidebar-footer-icon, [class*="bg-[#f58a1f]"], .bg-theme-button-primary, .bg-primary-button):is(:hover, :focus-visible):not(:disabled) {{
  background-color: #ffa447 !important; color: {BROWN} !important; border-color: {BROWN} !important; opacity: 1 !important;
}}
:root:root :is(button.rounded-full, a.bg-theme-sidebar-footer-icon, [class*="bg-[#f58a1f]"], .bg-theme-button-primary, .bg-primary-button):active:not(:disabled) {{
  background-color: #e07a10 !important; color: {BROWN} !important; border-color: {BROWN} !important;
}}
:root:root :is(button.rounded-full, [class*="bg-[#f58a1f]"]):is(:hover, :focus-visible, :active) svg, :root:root :is(button.rounded-full, [class*="bg-[#f58a1f]"]):is(:hover, :focus-visible, :active) svg * {{ color: {BROWN} !important; }}
/* Small pop-up labels: brown with cream text, not black */
.tooltip, .react-tooltip, [role="tooltip"] {{ background-color: {BROWN} !important; color: {CREAM} !important; border-radius: 12px !important; }}
/* Links and buttons that lead to the AnythingLLM web site, or to chat tools and the model picker */
a[href^="mailto:"], a[href*="/settings/privacy"], a[href*="github.com"], a[href*="docs.anythingllm.com"],
a[href*="discord"], a[href*="useanything.com"], a[href*="anythingllm.com"] {{ display: none !important; }}
"""


def footer_component():
    return '''import React from "react";
import { isMobile } from "react-device-detect";
import SettingsButton from "../SettingsButton";

// Cale: the original footer links to the AnythingLLM web site and its chat community. These exports stay (other files import them).
export const MAX_ICONS = 3;
export const ICON_COMPONENTS = {};

export default function Footer() {
  return (
    <div className="flex justify-center items-center mb-2 gap-x-4">
      <a
        href="/about.html"
        className="text-xs text-theme-text-secondary hover:text-theme-text-primary transition-all duration-300 flex items-center gap-x-1"
        aria-label="About Cale"
      >
        <img src="/carrot.png" alt="" className="h-5 w-5" />
        About Cale
      </a>
      {!isMobile && <SettingsButton />}
    </div>
  );
}
'''


def about_html(carrot_svg):
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>About Cale</title>
<style>
 body{margin:0;background:#fff1d6;color:#3b2314;font:17px/1.55 ui-rounded,"SF Pro Rounded","Arial Rounded MT Bold",system-ui,sans-serif}
 main{max-width:640px;margin:0 auto;padding:40px 24px}
 h1{margin:8px 0 4px;font-size:40px} a{color:#3b2314;font-weight:700}
 .art{width:140px;height:140px;background:#8fd3f4;border:4px solid #3b2314;border-radius:32px;padding:14px;box-sizing:border-box}
 .art svg{width:100%;height:100%}
 pre{white-space:pre-wrap;font:12px/1.5 ui-monospace,Menlo,monospace;background:#ffe7b3;border:3px solid #3b2314;padding:16px;border-radius:16px}
 .back{display:inline-block;margin-bottom:24px;background:#f58a1f;border:3px solid #3b2314;border-radius:999px;padding:6px 16px;text-decoration:none;color:#3b2314}
 .back:hover,.back:focus-visible{background:#ffa447}.back:active{background:#e07a10}
</style></head><body><main>
<a class="back" href="/">&larr; Back to chat</a>
<div class="art">""" + carrot_svg + """</div>
<h1>Cale</h1>
<p>Hi, I am Cale. I live on this computer. Your chats and your files stay here.</p>
<h2>Thank you</h2>
<p>This chat window is built on AnythingLLM by Mintplex Labs. It is free software under the MIT licence.</p>
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


# Words the person reads. {old text in locales/en/common.js: new text}
WORDING = {
    'thought_in_progress: "Model is Thinking..."': 'thought_in_progress: "Cale is thinking..."',
    'this will stop the model from generating the response': 'this will stop Cale from writing the answer',
    '    search: "Search",': '    search: "Search chats",',
    'uploadDocument: "Upload a Document"': 'uploadDocument: "Add a document"',
    'send_message: "Send a message"': 'send_message: "Ask Cale anything"',
    'send: "Send prompt message to workspace"': 'send: "Send your message"',
    'title: "Instance Settings"': 'title: "Settings"',
    'interface: "UI Preferences"': 'interface: "Look and feel"',
    'title: "UI Preferences"': 'title: "Look and feel"',
    "These files will be uploaded to the document processor running on this Cale instance. These files are not sent or shared with a third party.":
        "Your files stay on this computer. Nothing is sent anywhere.",
}


# ---------------------------------------------------------------------------------------------------------------------
# After `yarn build`: every stock blue and every grey-blue in the built page becomes a carrot-palette colour.
import colorsys

KEEP = {"8fd3f4", "f58a1f", "3b2314", "fff1d6", "fff8e8", "1b1722", "4aa34f", "8a5a3b"}      # the brand colours themselves


def _recolor(r, g, b, accent_only=False):
    """(r, g, b) -> new (r, g, b), or None to keep it. Blues become carrot or brown, greys become warm greys, the rest stays."""
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    if l > 0.985 or l < 0.03:
        return None
    hue = h * 360
    if 15 <= hue <= 50 and s > 0.12:
        return None                                              # already warm: the brand
    if s < 0.12 or (165 <= hue <= 275 and s < 0.45):
        if accent_only:
            return None
        nh, nl, ns = (36, l, 0.8) if l > 0.8 else (30, l, 0.4)  # grey and grey-blue -> warm grey/cream (same lightness)
    elif 165 <= hue <= 275:
        nh, nl, ns = (36, l, 1.0) if l > 0.8 else (31, l, 0.91) if l > 0.35 else (24, l, 0.55)    # blue -> pale carrot, carrot, brown
    else:
        return None
    nr, ng, nb = colorsys.hls_to_rgb(nh / 360, nl, ns)
    return round(nr * 255), round(ng * 255), round(nb * 255)


def recolor_built(dist):
    """Patch the built files of the frontend (a folder with index.css and assets/*.js). Returns how many colours changed."""
    import glob
    n = 0
    hexre = re.compile(r"#([0-9a-fA-F]{6})\b|#([0-9a-fA-F]{3})\b")
    rgbre = re.compile(r"rgba?\(\s*(\d{1,3})([\s,]+)(\d{1,3})([\s,]+)(\d{1,3})")

    def hexsub(accent_only):
        def f(m):
            nonlocal n
            h = m.group(1) or "".join(c * 2 for c in m.group(2))
            if h.lower() in KEEP:
                return m.group(0)
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            new = _recolor(r, g, b, accent_only)
            if new is None:
                return m.group(0)
            n += 1
            return "#%02x%02x%02x" % new
        return f

    def rgbsub(m):
        nonlocal n
        new = _recolor(int(m.group(1)), int(m.group(3)), int(m.group(5)))
        if new is None:
            return m.group(0)
        n += 1
        return f"rgb({new[0]}{m.group(2)}{new[1]}{m.group(4)}{new[2]}" if m.group(0).startswith("rgb(") else \
            f"rgba({new[0]}{m.group(2)}{new[1]}{m.group(4)}{new[2]}"
    for css in glob.glob(str(Path(dist) / "*.css")) + glob.glob(str(Path(dist) / "assets" / "*.css")):
        if Path(css).name in ("github.css", "github-dark.css"):
            continue                                              # the code colours: left alone
        text = _read(css)
        text = hexre.sub(hexsub(False), text)
        text = rgbre.sub(rgbsub, text)
        Path(css).write_text(text, encoding="utf-8")
    for js in glob.glob(str(Path(dist) / "assets" / "*.js")) + glob.glob(str(Path(dist) / "*.js")):
        text = _read(js)
        new_text = hexre.sub(hexsub(True), text)                  # in scripts only the blues (inline styles), never greys
        if new_text != text:
            Path(js).write_text(new_text, encoding="utf-8")
    return n


def apply(frontend, carrot_svg=None, carrot_png=None, icon=None, wordmark=None, wordmark_on_dark=None):
    fe = Path(frontend)
    src, pub = fe / "src", fe / "public"
    version = None
    for pj in (fe / "package.json", fe.parent / "server" / "package.json"):
        m = pj.exists() and re.search(r'"version":\s*"([^"]+)"', _read(pj))
        if m:
            version = m.group(1)
            break
    if version != TESTED_VERSION:
        print(f"rebrand: tested with AnythingLLM {TESTED_VERSION}, this is {version}. Look at every message below.", file=sys.stderr)

    # 1. words: the name, then the few words the person reads
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

    # 2. pictures: the wide logo (header, one for each theme), the icon, the favicon, the carrot itself
    if wordmark:
        # anything-llm.png is shown on the dark theme, anything-llm-dark.png on the light theme
        for p in (src / "media/logo/anything-llm-dark.png", pub / "anything-llm-dark.png"):
            shutil.copyfile(wordmark, p)
        for p in (src / "media/logo/anything-llm.png", pub / "anything-llm-light.png"):
            shutil.copyfile(wordmark_on_dark or wordmark, p)
    if icon:
        shutil.copyfile(icon, src / "media/logo/anything-llm-icon.png")
        shutil.copyfile(icon, pub / "favicon.png")
        if (pub / "favicon.ico").exists():
            (pub / "favicon.ico").unlink()
    if carrot_png:
        shutil.copyfile(carrot_png, pub / "carrot.png")
    if carrot_svg:
        shutil.copyfile(carrot_svg, pub / "carrot.svg")
    svg_text = _read(carrot_svg) if carrot_svg else ""
    (pub / "about.html").write_text(about_html(svg_text), encoding="utf-8")
    _swap(pub / "manifest.json", '"sizes": "any"', '"sizes": "any",\n      "type": "image/png"')

    # 3. colours and the cartoon look
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

    # 6. home screen: no "Create an Agent" or "Edit Workspace" buttons
    qa = src / "components/lib/QuickActions/index.jsx"
    _swap(qa, 'show={!user || ["admin"].includes(user?.role)}', "show={false}")
    _sub(qa, r"show=\{\s*hasAvailableWorkspace &&.*?\}\n", "show={false}\n")

    # 7. the empty chat: the carrot and one friendly line, no model picker
    for page in ("components/WorkspaceChat/ChatContainer/index.jsx", "pages/Main/Home/index.jsx"):      # a new chat, and the home screen
        cc = src / page
        text = _read(cc)
        text, n2 = re.subn(r"<ChatSettingsMenu(\s[^>]*?)?/>", "{null}", text, flags=re.S)       # the text-size and export button covers the first message
        if n2 < 1:
            raise PatchError(f"{cc}: chat settings button not found")
        text, n1 = re.subn(r"<WorkspaceModelPicker workspaceSlug=\{[^}]*\} />", "{null}", text)
        if "{greeting}" not in text or n1 < 1:
            raise PatchError(f"{cc}: greeting or model picker not found")
        text = text.replace("{greeting}", '<img src="/carrot.png" alt="" className="h-28 w-28 mx-auto mb-4" />' + GREETING)
        cc.write_text(text, encoding="utf-8")

    # 7b. no "Memories" panel (it keeps notes about the person; it is closed and has no button now, and it is removed so it cannot be reached)
    for page in ("components/WorkspaceChat/ChatContainer/index.jsx", "pages/Main/Home/index.jsx"):
        mp = src / page
        text, n3 = re.subn(r"<MemoriesSidebar workspace=\{workspace\} />", "{null}", _read(mp))
        if n3 < 1:
            raise PatchError(f"{mp}: memories panel not found")
        mp.write_text(text, encoding="utf-8")

    # 8. the message box: no "@" agent button, no "Tools", no reasoning button
    pi = src / "components/WorkspaceChat/ChatContainer/PromptInput/index.jsx"
    _sub(pi, r"<AgentSessionButton\s.*?/>", "{null}")
    _sub(pi, r"<ReasoningEffortButton\s.*?/>", "{null}")
    _sub(pi, r"<ToolsButton\s.*?/>", "{null}")
    _sub(pi, r"<ToolsMenu\s.*?/>", "{null}")

    # 9. the workspace list: no gear (it opens chat, vector database and agent settings); plain words for threads
    aw = src / "components/Sidebar/ActiveWorkspaces/index.jsx"
    _sub(aw, r"<button\s+onClick=\{\(e\) => \{\s*e\.preventDefault\(\);\s*e\.stopPropagation\(\);\s*navigate\(.*?<GearSix.*?/>\s*</button>", "")
    _swap(aw, "Upload documents to this workspace for RAG indexing", "Add a document")
    tc = src / "components/Sidebar/ActiveWorkspaces/ThreadContainer/index.jsx"
    _swap(tc, 'name: "default"', 'name: "Main chat"')
    _swap(tc, 'name: "*New Thread"', 'name: "New chat"')
    _sub(tc, r"New Thread\n", "Start another chat\n")
    sb = src / "components/Sidebar/SearchBox/index.jsx"        # the "+" next to Search: carrot with a brown outline, not white
    _swap(sb, 'className="border-none flex items-center justify-center bg-white  rounded-lg p-[8px] hover:bg-white/80 light:hover:bg-slate-300 transition-all duration-300"',
          'className="flex items-center justify-center bg-[#f58a1f] border-2 border-solid border-[#3b2314] rounded-lg p-[8px] hover:opacity-80 transition-all duration-300"')
    _swap(tc, '<div className="bg-zinc-800 light:bg-slate-50 p-2 rounded-lg h-[24px] w-[24px] flex items-center justify-center">',
          '<div className="bg-[#f58a1f] border-2 border-solid border-[#3b2314] p-2 rounded-lg h-[24px] w-[24px] flex items-center justify-center">')
    _swap(tc, 'className="shrink-0 text-white light:text-theme-text-primary"\n            />\n          )}', 'className="shrink-0 text-[#3b2314]"\n            />\n          )}')
    _swap(sb, 'className="text-black light:text-slate-500"', 'className="text-[#3b2314]"')

    # 9b. the add-a-document window: documents only (no web links, no "data connectors"), plain words
    mw = src / "components/Modals/ManageWorkspace/index.jsx"
    _sub(mw, r"<ModalTabSwitcher\s.*?/>", "null")                       # one tab only: no tab box
    uf = src / "components/Modals/ManageWorkspace/Documents/UploadFile/index.jsx"
    _sub(uf, r"<div className=\"text-center text-white text-opacity-50 text-xs font-medium w-\[560px\] py-2\">\s*\{t\(\"connectors\.upload\.or-submit-link\"\)\}\s*</div>\s*<form onSubmit=\{handleSendLink\}.*?</form>", "{null}")
    _swap(src / "components/Sidebar/ActiveWorkspaces/ThreadContainer/ThreadItem/index.jsx", "Delete Thread</p>", "Delete chat</p>")

    # 10. the loading screen: the carrot
    lc = src / "components/WorkspaceChat/LoadingChat/index.jsx"
    _swap(lc, '      <Skeleton.default\n        height="100px"\n        width="100%"',
          '      <img src="/carrot.png" alt="" className="h-24 w-24 mx-auto animate-bounce" />\n      <Skeleton.default\n        height="100px"\n        width="100%"')
    print("rebrand: done")


def main(argv):
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    args = dict(zip(argv[1::2], argv[2::2]))
    try:
        apply(argv[0], carrot_svg=args.get("--carrot-svg"), carrot_png=args.get("--carrot-png"), icon=args.get("--icon"),
              wordmark=args.get("--wordmark"), wordmark_on_dark=args.get("--wordmark-dark"))
    except PatchError as e:
        print(f"rebrand: STOP. {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

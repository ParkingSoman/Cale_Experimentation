#!/usr/bin/env python3
"""Build the small download for one Cale app. The app does NOT carry Cale: Cale is a separate, one-time download.

  python scripts/build_app.py mac quick-notes      -> dist/apps/Quick-Notes-mac.dmg   (Quick Notes.app inside)
  python scripts/build_app.py win quick-notes      -> dist/apps/win-quick-notes/      (then run build_app_exe.ps1 on Windows)
  python scripts/build_app.py mac all              -> every consumer-demo app

What the launcher does when the person double-clicks:
  1. Cale is installed on this computer: start Cale if it is down, start the app, open it in the browser.
  2. Cale is not installed: say so in plain words and offer to get it (the Cale download next to this file, in Downloads,
     or CALE_DOWNLOAD_URL if the build was given one).
The app runs with the Python that Cale installed. That is why the download is only a few hundred kilobytes.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_installer as bi  # noqa: E402

DIST = ROOT / "dist" / "apps"
APPS = ROOT / "demo" / "apps"
CONSUMER_APPS = ("quick-notes", "rewrite")
COLORS = {"quick-notes": ((61, 107, 214), (247, 245, 238)), "rewrite": ((196, 98, 63), (247, 245, 238))}
COPY = ("app.py", "index.html", "cale_client.py")
CALE_URL = os.environ.get("CALE_DOWNLOAD_URL", "")


def spec(app_id):
    reg = json.loads((APPS / "registry.json").read_text())["apps"]
    s = next(a for a in reg if a["id"] == app_id)
    return s["name"], s["port"]


MAC_LAUNCHER = r'''#!/bin/sh
here="$(cd "$(dirname "$0")/../Resources/app" && pwd)"
home="${CALE_HOME:-$HOME/Library/Application Support/Cale}"
name="__NAME__"
port="__PORT__"
url="http://127.0.0.1:$port/"
if curl -s -o /dev/null "$url"; then open "$url"; exit 0; fi                  # already running: show it again
if [ ! -x "$home/venv/bin/python" ] || [ ! -f "$home/installed.ok" ]; then
  r=$(osascript -e 'display dialog "'"$name"' needs Cale to run. Cale is a one-time download that lets apps like this work on your computer." buttons {"Not now", "Get Cale"} default button "Get Cale" with title "'"$name"'" with icon note' 2>/dev/null)
  case "$r" in
    *"Get Cale"*)
      for d in "$(cd "$here/../../../.." && pwd)" "$HOME/Downloads" "$HOME/Desktop"; do
        if [ -e "$d/Cale-mac.dmg" ]; then open "$d/Cale-mac.dmg"; exit 0; fi
      done
      if [ -n "__CALE_URL__" ]; then open "__CALE_URL__"; else
        osascript -e 'display dialog "Cale was not found on this computer. Ask the person who gave you this app for the Cale download." buttons {"OK"} default button "OK" with title "'"$name"'"' >/dev/null 2>&1
      fi ;;
  esac
  exit 0
fi
data="$HOME/Library/Application Support/$name"
mkdir -p "$data"
export CALE_HOME="$home" CALE_APP_DATA="$data" PATH="$home/bin:$PATH"
cd "$here" && exec "$home/venv/bin/python" app.py
'''

INFO_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>__NAME__</string>
  <key>CFBundleDisplayName</key><string>__NAME__</string>
  <key>CFBundleIdentifier</key><string>ai.cale.app.__ID__</string>
  <key>CFBundleExecutable</key><string>__NAME__</string>
  <key>CFBundleIconFile</key><string>App</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
</dict></plist>
"""


def copy_app(app_id, dest):
    dest.mkdir(parents=True)
    for f in COPY:
        shutil.copy2(APPS / app_id / f, dest / f)


def build_mac(app_id):
    name, port = spec(app_id)
    work = DIST / f"mac-{app_id}"
    shutil.rmtree(work, ignore_errors=True)
    app = work / f"{name}.app"
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "Resources").mkdir(parents=True)
    copy_app(app_id, app / "Contents" / "Resources" / "app")
    (app / "Contents" / "Info.plist").write_text(INFO_PLIST.replace("__NAME__", name).replace("__ID__", app_id))
    launcher = app / "Contents" / "MacOS" / name
    launcher.write_text(MAC_LAUNCHER.replace("__NAME__", name).replace("__PORT__", str(port)).replace("__CALE_URL__", CALE_URL))
    launcher.chmod(0o755)
    png = work / "icon.png"
    bi.make_icon(png, *COLORS[app_id])
    iconset = work / "App.iconset"
    iconset.mkdir()
    for size in (16, 32, 64, 128, 256, 512, 1024):
        out = iconset / f"icon_{size}x{size}.png"
        subprocess.run(["sips", "-z", str(size), str(size), str(png), "--out", str(out)], check=True, capture_output=True)
        if size in (32, 64, 256, 512, 1024):
            half = size // 2
            shutil.copy2(out, iconset / f"icon_{half}x{half}@2x.png")
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(app / "Contents" / "Resources" / "App.icns")], check=True)
    shutil.rmtree(iconset)
    identity = os.environ.get("CALE_SIGN_IDENTITY", "-")                  # "-" is ad hoc; a Developer ID name signs for real
    subprocess.run(["codesign", "--force", "--deep", "--sign", identity] + (["--options", "runtime"] if identity != "-" else []) + [str(app)],
                   check=True, capture_output=True)
    dmg = DIST / f"{name.replace(' ', '-')}-mac.dmg"
    dmg.unlink(missing_ok=True)
    subprocess.run(["hdiutil", "create", "-volname", name, "-srcfolder", str(app), "-ov", "-format", "UDZO", str(dmg)],
                   check=True, capture_output=True)
    return dmg


WIN_LAUNCH = r'''@echo off
setlocal
set "NAME=__NAME__"
set "HERE=%~dp0"
set "HOMEDIR=%LOCALAPPDATA%\Cale"
if defined CALE_HOME set "HOMEDIR=%CALE_HOME%"
curl.exe -s -o nul http://127.0.0.1:__PORT__/ && ( start "" http://127.0.0.1:__PORT__/ & exit /b )
if not exist "%HOMEDIR%\venv\Scripts\python.exe" goto nocale
if not exist "%HOMEDIR%\installed.ok" goto nocale
set "CALE_HOME=%HOMEDIR%"
set "CALE_APP_DATA=%LOCALAPPDATA%\%NAME%"
if not exist "%CALE_APP_DATA%" mkdir "%CALE_APP_DATA%"
set "PATH=%HOMEDIR%\bin;%PATH%"
cd /d "%HERE%app"
"%HOMEDIR%\venv\Scripts\pythonw.exe" app.py
exit /b
:nocale
wscript.exe "%HERE%needcale.vbs" "%NAME%" "__CALE_URL__"
exit /b
'''

WIN_NEEDCALE = r'''name = WScript.Arguments(0)
url = WScript.Arguments(1)
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
r = MsgBox(name & " needs Cale to run. Cale is a one-time download that lets apps like this work on your computer." & vbCrLf & vbCrLf & "Get Cale now?", vbYesNo + vbInformation, name)
If r = vbYes Then
  found = ""
  For Each d In Array(sh.ExpandEnvironmentStrings("%USERPROFILE%\Downloads"), sh.ExpandEnvironmentStrings("%USERPROFILE%\Desktop"))
    If fso.FileExists(d & "\Cale-Setup.exe") Then found = d & "\Cale-Setup.exe"
  Next
  If found <> "" Then
    sh.Run """" & found & """"
  ElseIf url <> "" Then
    sh.Run url
  Else
    MsgBox "Cale was not found on this computer. Ask the person who gave you this app for the Cale download.", vbOKOnly, name
  End If
End If
'''

WIN_SETUP_CMD = r'''@echo off
set "D=%LOCALAPPDATA%\__NAME__\files"
if not exist "%D%" mkdir "%D%"
tar -xf "%~dp0payload.zip" -C "%D%"
call "%D%\launch.cmd"
'''

WIN_SETUP_VBS = r'''Set sh = CreateObject("WScript.Shell")
here = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
sh.Run "cmd /c """"" & here & "\setup.cmd""""", 0, True
'''


def build_win(app_id):
    name, port = spec(app_id)
    work = DIST / f"win-{app_id}"
    shutil.rmtree(work, ignore_errors=True)
    stage = work / "stage"
    copy_app(app_id, stage / "app")
    crlf = lambda t: t.replace("\n", "\r\n")  # noqa: E731
    (stage / "launch.cmd").write_text(crlf(WIN_LAUNCH.replace("__NAME__", name).replace("__PORT__", str(port)).replace("__CALE_URL__", CALE_URL)))
    (stage / "needcale.vbs").write_text(crlf(WIN_NEEDCALE))
    with zipfile.ZipFile(work / "payload.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in stage.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(stage))
    shutil.rmtree(stage)
    (work / "setup.cmd").write_text(crlf(WIN_SETUP_CMD.replace("__NAME__", name)))
    (work / "setup.vbs").write_text(crlf(WIN_SETUP_VBS))
    shutil.copy2(Path(__file__).with_name("build_app_exe.ps1"), work / "build_app_exe.ps1")
    (work / "name.txt").write_text(name.replace(" ", "-") + "-Setup")
    return work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("os", choices=["mac", "win"])
    ap.add_argument("app", help="an app id from demo/apps/registry.json, or 'all'")
    a = ap.parse_args()
    DIST.mkdir(parents=True, exist_ok=True)
    for app_id in (CONSUMER_APPS if a.app == "all" else (a.app,)):
        print("built:", (build_mac if a.os == "mac" else build_win)(app_id))


if __name__ == "__main__":
    main()

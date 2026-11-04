#!/usr/bin/env python3
"""PreToolUse guard. Exit 2 blocks the call.

For everyone: no tool call may name sealed data or a judge's key, and no search may sweep the runs folder.
For a judge: only its own packets and answers.
For a reader, critic or scientist: the shell runs only the mapper's own commands and drawing scripts, and packets and the state store stay closed.

Its limit, said plainly. It reads the text of a call. It stops mistakes and most shortcuts. It is not a wall against code written to get around it, so a role's shell is narrowed to named commands, and sealed scoring stays in the judge module."""
import json
import os
import re
import sys

SEALED = re.compile(r"seal|sea\*|s\?al", re.I)
RUNS = re.compile(r"CrakBox/runs|(^|[\s/\"'])runs/")
PACKETS = re.compile(r"packets_|/packets|(^|[\s\"'])packets")
STATE = re.compile(r"state\.db")
SHELL_MARKS = re.compile(r"[;&|`$<>\n]")
MAPPER_COMMAND = re.compile(r"^\S*python\S*\s+(-W\s+ignore\s+)?\S*mapper\.py\s+(status|next|sheet|route|read|note|predict|revise|switch|ev|card|critic-plan|critic|open)\s")
DRAW_COMMAND = re.compile(r"^\S*python\S*\s+(\S*CrakBox/runs/drawn/[\w.-]+\.py)(\s|$)")
DRAW_BARRED = re.compile(r"seal|state\.db|packets|glob|walk|listdir|scandir|subprocess|socket|system\(|exec\(|eval\(|__import__|chr\(", re.I)
PATH_KEYS = ("file_path", "path", "pattern", "glob", "command")
READERS = {"map-reader", "map-critic", "map-scientist"}


def draw_script_is_clean(path, cwd):
    full = path if os.path.isabs(path) else os.path.join(cwd or ".", path)
    try:
        with open(full, encoding="utf-8") as f:
            return not DRAW_BARRED.search(f.read())
    except OSError:
        return False


def decide(data):
    tool = data.get("tool_name", "")
    tool_input = data.get("tool_input") or {}
    role = data.get("agent_type")
    # what a call reaches: not the text a file edit writes, so a test or a note may name these paths
    text = " ".join(str(tool_input[k]) for k in PATH_KEYS if k in tool_input)
    command = tool_input.get("command", "")
    if SEALED.search(text):
        return "sealed data and judge keys score the work, so no agent may reach them. Use the mapper's judge commands"
    if tool in ("Grep", "Glob") and RUNS.search(text):
        return "a search may not sweep the runs folder. Read a named file"
    if role == "map-judge":
        if tool not in ("Read", "Write"):
            return "a judge reads its packets and writes its answers, and nothing else"
        path = tool_input.get("file_path", "")
        if not re.search(r"CrakBox/runs/(packets|answers)_[0-9a-f]+/", path):
            return "a judge may touch only its packets and answers"
    if role in READERS:
        if PACKETS.search(text) or STATE.search(text):
            return "a judge's packets and the state store are closed to this role. Use the mapper's commands"
        if tool == "Bash":
            if SHELL_MARKS.search(command):
                return "one plain command at a time for this role"
            if MAPPER_COMMAND.match(command):
                return None
            draw = DRAW_COMMAND.match(command)
            if draw:
                return None if draw_script_is_clean(draw.group(1), data.get("cwd")) else "this drawing script reaches for more than drawing"
            return "this role's shell runs only the mapper's commands and scripts under runs/drawn"
    return None


def main():
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return 0
    reason = decide(data)
    if reason:
        sys.stderr.write("sealed guard: " + reason + "\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

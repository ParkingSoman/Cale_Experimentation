#!/usr/bin/env bash
# Run a bash command in Ubuntu (WSL2) on the Windows test laptop (ssh host "hp").
# Usage: scripts/remote_linux.sh 'uname -a'
# The command text is copied to the laptop as a file first. This avoids
# PowerShell quote problems.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 '<bash command>'" >&2; exit 2; }
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
printf '%s\n' "$*" > "$tmp"
scp -q "$tmp" hp:C:/cale/last-linux-cmd.sh
ssh hp 'wsl -d Ubuntu-24.04 -u cale -- bash /mnt/c/cale/last-linux-cmd.sh 2>&1' | tr -d '\0\r'

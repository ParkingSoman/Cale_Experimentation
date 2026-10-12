#!/usr/bin/env bash
# Run a PowerShell command on the Windows test laptop (ssh host "hp").
# Usage: scripts/remote_win.sh 'Get-Date'
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 '<powershell command>'" >&2; exit 2; }
ssh hp "$*" | tr -d '\0\r'

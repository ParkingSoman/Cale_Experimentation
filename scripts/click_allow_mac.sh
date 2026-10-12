#!/usr/bin/env bash
# Click "Allow" on Cale's own Allow dialog (macOS). For automated tests, only with the owner's OK.
#
#   scripts/click_allow_mac.sh [SECONDS]      wait up to SECONDS (default 30) for the dialog, then click Allow
#
# Safety: it clicks only a window titled "Cale" in the osascript process that has an "Allow" button.
# Needs Accessibility permission for the terminal app. Exit 0 = clicked, 1 = no dialog appeared.
set -uo pipefail
limit="${1:-30}"
for ((i = 0; i < limit * 2; i++)); do
  if osascript -e 'tell application "System Events" to click button "Allow" of (first window of process "osascript" whose name is "Cale")' >/dev/null 2>&1; then
    echo "clicked Allow"
    exit 0
  fi
  sleep 0.5
done
echo "no Cale Allow dialog appeared within ${limit}s"
exit 1

#!/bin/sh
# ONE-TIME prep of a Mac DEMO computer, so a downloaded Cale.app opens with no "unknown developer" warning.
# This lowers this Mac's protection. Use it only on a Mac you use for demos. Undo it with:  sh scripts/prep_mac_demo.sh undo
#   Run:   sh scripts/prep_mac_demo.sh        (it asks for your password: it uses sudo)
# What it changes: it allows apps from anywhere (spctl --master-disable). Then open
#   System Settings > Privacy & Security > "Allow applications from" and choose "Anywhere" once (macOS asks you to confirm).
# A public release needs an Apple Developer ID and notarization. This is for the demo only.
set -e
if [ "$1" = "undo" ]; then
  sudo spctl --master-enable
  echo "Undone: Gatekeeper is on again."
  exit 0
fi
sudo spctl --master-disable
echo "Done. Gatekeeper status now:"
spctl --status || true
echo "If macOS still warns, open System Settings > Privacy & Security and choose 'Anywhere'."

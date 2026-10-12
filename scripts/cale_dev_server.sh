#!/bin/sh
# Start a test Cale on port 47899 with its own home. Does not touch the installed Cale (port 47821).
# usage: scripts/cale_dev_server.sh HOME_DIR
cd "$(dirname "$0")/.." || exit 1
CALE_HOME="$1" CALE_PORT=47899 CALE_DEBUG=1 CALE_AUTO_APPROVE=1 CALE_IDLE_S=${CALE_IDLE_S:-6} CALE_MODEL="$PWD/models/gemma-3n-e4b-4bit" \
  exec .venv/bin/python -m cale.server

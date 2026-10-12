#!/usr/bin/env bash
# Forward a port of the HP laptop to this Mac, and reconnect by itself when the link drops.
# usage: scripts/hp_forward.sh LOCAL_PORT REMOTE_PORT        (stop it with: pkill -f hp_forward.sh)
# Open WebUI on the HP (Windows or Linux) is then at http://127.0.0.1:LOCAL_PORT on this Mac.
# WSL2 passes its listening ports to Windows localhost, so one forward works for both systems.
L="${1:?local port}"; R="${2:?remote port}"
while true; do
  ssh -N -L "$L:127.0.0.1:$R" -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ConnectTimeout=15 hp 2>/dev/null
  sleep 2
done

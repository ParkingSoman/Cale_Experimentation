#!/usr/bin/env bash
# Download Broden (broden1_224, the Network Dissection data set) into data/broden/.
# The official address (NetDissect-Lite script/dlbroden.sh) is http://netdissect.csail.mit.edu/data/broden1_224.zip.
# That server timed out on 2026-11-23, so this uses the Internet Archive copy of the same file (2024-07-07, 937,582,103 bytes).
# Untrusted data: it stays in its own folder and is only read as data (python -I), never run.
set -euo pipefail
cd "$(dirname "$0")/data/broden"
URL="https://web.archive.org/web/20240707052113id_/https://netdissect.csail.mit.edu/data/broden1_224.zip"
curl -L --fail --retry 5 -C - -o broden1_224.zip "$URL"
test "$(stat -f%z broden1_224.zip)" = 937582103
unzip -tq broden1_224.zip
unzip -q broden1_224.zip
rm broden1_224.zip
ls broden1_224 | head

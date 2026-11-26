#!/usr/bin/env bash
# Parallel range download of the rest of broden1_224.zip (12 connections), then append in order.
set -euo pipefail
cd "$(dirname "$0")/data/broden"
URL="https://web.archive.org/web/20240707052113id_/https://netdissect.csail.mit.edu/data/broden1_224.zip"
TOTAL=937582103
START=$(stat -f%z broden1_224.zip)
N=12
STEP=$(( (TOTAL - START + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  a=$(( START + i*STEP )); b=$(( a + STEP - 1 )); [ $b -ge $TOTAL ] && b=$((TOTAL-1))
  [ $a -ge $TOTAL ] && continue
  ( until curl -sL --fail -r $a-$b -o part.$i "$URL" && [ "$(stat -f%z part.$i)" = $((b-a+1)) ]; do sleep 2; done ) &
done
wait
for i in $(seq 0 $((N-1))); do [ -f part.$i ] && cat part.$i >> broden1_224.zip && rm part.$i; done
test "$(stat -f%z broden1_224.zip)" = $TOTAL
unzip -tq broden1_224.zip
unzip -q broden1_224.zip
rm broden1_224.zip
echo DONE; ls broden1_224 | head

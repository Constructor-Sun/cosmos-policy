#!/bin/bash
# One demo file -> one json.  All config arrives via exported env vars.
set -euo pipefail
f="$1"
b=$(basename "$f" .hdf5)
"$PY" "$R/scripts/libero_pro/replay_measure.py" \
    --mode demo --suite libero_10 --demo-file "$f" \
    --n-demos "$N_DEMOS" --open-frames "$OPEN_FRAMES" \
    --out "$OUT/json/$b.json" > "$OUT/logs/$b.log" 2>&1
echo "done: $b"

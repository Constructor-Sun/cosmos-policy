#!/usr/bin/env bash
set -euo pipefail

REPO=/data1/liu/exp/counterfactual/external/cosmos-policy
OUTPUT=/data1/liu/exp/counterfactual/external/cosmos-policy/scripts/experiments/libero10_robotinit_completion_smoke5

cd "$REPO/scripts"

NUM_CASES=5 \
COSMOS_INIT_STATE_OFFSET=0 \
OUTPUT_ROOT="$OUTPUT" \
COSMOS_ROLLOUT_SUBDIR=08-29-completion-smoke5 \
COSMOS_INITIAL_ALIGNMENT=1 \
COSMOS_SKILL_COMPLETION_ACTIVE=1 \
COSMOS_SKILL_COMPLETION_SHADOW=0 \
COSMOS_DEBUG_INIT_ALIGN=0 \
./run_libero10_robotinit_20.sh

echo "=== completion transitions ==="
grep -R -E \
  "active sequence loaded|VLA start phase|advance phase|SKILL_COMPLETION.*summary" \
  "$OUTPUT" || true

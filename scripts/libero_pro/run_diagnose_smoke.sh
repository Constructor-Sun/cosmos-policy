#!/bin/bash
# Identity-chain smoke for LIBERO-PRO (plan doc section 1): run diagnose on the
# PRO smoke census and check the chain lands on the LIBERO-PRO package / BDDL.
#
# Success looks like:
#   [libero] benchmark module: .../LIBERO-PRO/libero/...  LIBERO_CONFIG_PATH: .../configs/libero_pro
#   [<variant>] flavor=pro suite=libero_10_swap ... bddl=.../LIBERO-PRO/LIBERO-Pro/bddl_files/libero_10_swap/...
# and phase_record.json under --out carrying "flavor": "pro", "suite", "bddl_file".
#
# The library check itself fail-fasts inside diagnose_failed.py when the
# census says flavor=pro but LIBERO-PRO is not what got imported.
set -euo pipefail

R=$(cd "$(dirname "$0")/../.." && pwd)
cd "$R"

CENSUS=${CENSUS:-$R/experiments/tta_census/libero_pro_swap_smoke.json}
OUT=${OUT:-$R/experiments/tta_phase_check_pro}
TASKS=${TASKS:-}   # e.g. TASKS=pro_swap (default: all census entries)

export COSMOS_LIBERO_ROOT=${COSMOS_LIBERO_ROOT:-/data1/liu/exp/counterfactual/external/LIBERO-PRO}
export LIBERO_CONFIG_PATH="$COSMOS_LIBERO_ROOT/configs/libero_pro"
export PYTHONPATH="$COSMOS_LIBERO_ROOT:$R${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONNOUSERSITE=1

PY=${PY:-/data1/liu/miniconda3/envs/cosmospolicy/bin/python}

"$PY" memory_system/tta/diagnose_failed.py \
    --census "$CENSUS" \
    --out "$OUT" \
    ${TASKS:+--tasks "$TASKS"}

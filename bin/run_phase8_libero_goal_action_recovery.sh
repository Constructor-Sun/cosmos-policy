#!/bin/sh

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$REPO_ROOT"

SUITE=${SUITE:-libero_goal}
TASK=${TASK:-open_the_middle_drawer_of_the_cabinet}
TASK_LANGUAGE=${TASK_LANGUAGE:-}
CONDITION=${CONDITION:-camera_viewpoints}
POLICY_SEED=${POLICY_SEED:-1009}
ENV_SEED=${ENV_SEED:-0}
GPU_ID=${GPU_ID:-6}
NUM_INIT_STATES=${NUM_INIT_STATES:-50}
START_INIT_STATE=${START_INIT_STATE:-0}
INIT_STATE_INDICES=${INIT_STATE_INDICES:-}
MAX_PAIRS=${MAX_PAIRS:-0}
ALPHA=${ALPHA:-1.0}
CHECKPOINT=${CHECKPOINT:-experiments/phase8_latent_shift/main_suites_camera_grid_50init_5pseed_env0_train7500_clean_no_old_skips/last_action_20e_cuda6/best.pt}
POLICY_DIR=${POLICY_DIR:-"$REPO_ROOT/../../checkpoints/Cosmos-Policy-LIBERO-Predict2-2B"}
PERTURB_CONFIG=${PERTURB_CONFIG:-experiments/paired_smoke_vla_jepa_kitchen_scene4_seed7_all_20case/seed_and_init_state_alignment_audit.json}
LIBERO_PLUS_PATH=${LIBERO_PLUS_PATH:-"$REPO_ROOT/../LIBERO-plus"}
SUMMARY_DIR=${SUMMARY_DIR:-"experiments/phase8_eval_latent_correction/_tmp_${SUITE}_${TASK}_pseed${POLICY_SEED}_eseed${ENV_SEED}"}
OUTPUT_DIR=${OUTPUT_DIR:-"experiments/phase8_eval_latent_correction/last_action_20e_cuda6_${SUITE}_${TASK}_pseed${POLICY_SEED}_eseed${ENV_SEED}_cuda${GPU_ID}"}
PREPARE_ONLY=${PREPARE_ONLY:-0}
DRY_RUN=${DRY_RUN:-0}

if [ ! -f "$CHECKPOINT" ]; then
    echo "Missing checkpoint: $CHECKPOINT" >&2
    exit 1
fi
if [ ! -f "$PERTURB_CONFIG" ]; then
    echo "Missing perturb config: $PERTURB_CONFIG" >&2
    exit 1
fi
if [ ! -d "$LIBERO_PLUS_PATH" ]; then
    echo "Missing LIBERO-plus directory: $LIBERO_PLUS_PATH" >&2
    exit 1
fi
if [ ! -f "$POLICY_DIR/libero_dataset_statistics.json" ]; then
    echo "Missing dataset statistics: $POLICY_DIR/libero_dataset_statistics.json" >&2
    exit 1
fi
if [ ! -f "$POLICY_DIR/Cosmos-Policy-LIBERO-Predict2-2B.pt" ]; then
    echo "Missing policy checkpoint: $POLICY_DIR/Cosmos-Policy-LIBERO-Predict2-2B.pt" >&2
    exit 1
fi

export REPO_ROOT
export SUITE
export TASK
export TASK_LANGUAGE
export CONDITION
export POLICY_SEED
export ENV_SEED
export NUM_INIT_STATES
export START_INIT_STATE
export INIT_STATE_INDICES
export PERTURB_CONFIG
export LIBERO_PLUS_PATH
export SUMMARY_DIR

SUMMARY_PATH=$(python3 - <<'PY'
import json
import os
import re
import sys
from pathlib import Path


def repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(os.environ["REPO_ROOT"]) / path


def one_condition(config: dict, condition: str) -> dict:
    for item in config["conditions"]:
        if item.get("condition") == condition:
            return item
    raise SystemExit(f"missing condition in perturb config: {condition}")


def one_config_value(info: dict, key: str):
    values = info.get(key)
    if not isinstance(values, list) or len(values) != 1:
        raise SystemExit(f"expected one {key} value for {info.get('condition')}, got {values!r}")
    return values[0]


def init_indices() -> list[int]:
    explicit = os.environ.get("INIT_STATE_INDICES", "").strip()
    if explicit:
        return [int(item) for item in re.split(r"[\s,]+", explicit) if item]
    start = int(os.environ["START_INIT_STATE"])
    count = int(os.environ["NUM_INIT_STATES"])
    if count < 1:
        raise SystemExit("NUM_INIT_STATES must be >= 1")
    return list(range(start, start + count))


suite = os.environ["SUITE"]
task = os.environ["TASK"]
condition = os.environ["CONDITION"]
policy_seed = int(os.environ["POLICY_SEED"])
env_seed = int(os.environ["ENV_SEED"])
summary_dir = repo_path(os.environ["SUMMARY_DIR"])
perturb_config = repo_path(os.environ["PERTURB_CONFIG"])
libero_plus = Path(os.environ["LIBERO_PLUS_PATH"]).expanduser()
if not libero_plus.is_absolute():
    libero_plus = Path(os.environ["REPO_ROOT"]) / libero_plus

config = json.loads(perturb_config.read_text(encoding="utf-8"))
clean_info = one_condition(config, "clean")
pert_info = one_condition(config, condition)
if one_config_value(clean_info, "instruction_modes") != "task":
    raise SystemExit("expected clean instruction mode 'task'")
pert_instruction_mode = one_config_value(pert_info, "instruction_modes")
perturb = pert_info.get("perturb_parameters") or {}
if perturb.get("type") != condition:
    raise SystemExit(f"perturb config does not define {condition}")
view_part = perturb.get("view_part")
parsed_init_state = perturb.get("parsed_init_state")
if not isinstance(view_part, str) or not isinstance(parsed_init_state, int):
    raise SystemExit(f"invalid camera perturb parameters: {perturb!r}")

bddl_root = libero_plus / "libero" / "libero" / "bddl_files" / suite
clean_bddl = bddl_root / f"{task}.bddl"
if not clean_bddl.is_file():
    raise SystemExit(f"missing clean BDDL: {clean_bddl}")
# Match LIBERO benchmark.grab_language_from_filename() and the precomputed
# Cosmos policy T5 cache. Some BDDL (:language ...) strings differ from these
# cache keys and would force offline T5 recomputation.
language = os.environ.get("TASK_LANGUAGE", "").strip() or task.replace("_", " ")
pert_task = f"{task}_view_{view_part}_initstate_{parsed_init_state}"

clean_dir = summary_dir / "clean"
pert_dir = summary_dir / condition
clean_dir.mkdir(parents=True, exist_ok=True)
pert_dir.mkdir(parents=True, exist_ok=True)

clean_eps = []
pert_eps = []
for index in init_indices():
    common = {
        "suite": suite,
        "seed": policy_seed,
        "deterministic_reset": True,
        "deterministic_reset_seed": env_seed,
        "base_task": task,
        "language": language,
        "episode": index,
        "init_state_index": index,
        "success": None,
    }
    clean_eps.append(
        {
            **common,
            "condition": "clean",
            "category": "clean",
            "task_name": task,
            "instruction_mode": "task",
        }
    )
    pert_eps.append(
        {
            **common,
            "condition": condition,
            "category": "Camera Viewpoints",
            "task_name": pert_task,
            "instruction_mode": pert_instruction_mode,
        }
    )

(clean_dir / "episodes.json").write_text(json.dumps(clean_eps, indent=2), encoding="utf-8")
(pert_dir / "episodes.json").write_text(json.dumps(pert_eps, indent=2), encoding="utf-8")
summary = {
    "mode": "phase8_libero_goal_action_recovery_subset",
    "suite": suite,
    "seed": policy_seed,
    "deterministic_reset": True,
    "deterministic_reset_seed": env_seed,
    "num_pairs": len(clean_eps),
    "base_task": task,
    "conditions": [
        {
            "condition": "clean",
            "task_name": task,
            "episodes_path": str(clean_dir / "episodes.json"),
        },
        {
            "condition": condition,
            "task_name": pert_task,
            "episodes_path": str(pert_dir / "episodes.json"),
            "instruction_mode": pert_instruction_mode,
        },
    ],
}
summary_path = summary_dir / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(summary_path)
print(f"Prepared {len(clean_eps)} pairs for {suite}/{task}", file=sys.stderr)
print(f"Instruction: {language}", file=sys.stderr)
print(f"Perturbed task: {pert_task}", file=sys.stderr)
PY
)

echo "Summary: $SUMMARY_PATH"

if [ "$PREPARE_ONLY" = "1" ]; then
    exit 0
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run; command not executed."
    echo "CUDA_VISIBLE_DEVICES=$GPU_ID python3 bin/phase8_eval_latent_correction.py --checkpoint $CHECKPOINT --summary $SUMMARY_PATH --output-dir $OUTPUT_DIR"
    exit 0
fi

export LIBERO_PLUS_PATH

CUDA_VISIBLE_DEVICES="$GPU_ID" \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
HF_HUB_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
NUMBA_CACHE_DIR="/tmp/cosmospolicy-libero-goal-action-recovery-cuda${GPU_ID}" \
MPLCONFIGDIR="/tmp/cosmospolicy-matplotlib-libero-goal-action-recovery-cuda${GPU_ID}" \
python3 bin/phase8_eval_latent_correction.py \
    --checkpoint "$CHECKPOINT" \
    --summary "$SUMMARY_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --policy-dir "$POLICY_DIR" \
    --t5-extra-embeddings "" \
    --conditions "$CONDITION" \
    --groups unknown preserved flipped recovery both_fail \
    --max-pairs "$MAX_PAIRS" \
    --alpha "$ALPHA" \
    --seed "$POLICY_SEED" \
    --reset-seed "$ENV_SEED" \
    --device cuda

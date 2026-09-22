"""Re-run individual PRO repair cases with the current working tree (single-case debug).

原为 /tmp/run_k3_turnon1_liu.py，硬编码了 swap 分支的 suite / meta / requests 三处路径，
无法用于 libero_10_task。现在这三处全部参数化（suite 优先取 meta 里的 suite 字段）。

用法
    python scripts/libero_pro/run_single_case.py --tags KSCENE3-init006 --gpu 3
    python scripts/libero_pro/run_single_case.py --tags A,B --meta <...> --requests-dir <...> \
        --out-dir <绝对路径> --gpu 0

每个 tag 串行跑一次 run_libero_smoke_test.sh，结果写到 <out-dir>/results/<tag>，
repair 结果 JSON 写到 <out-dir>/outputs/<tag>.json。

调试探针（可选）
    PYTHONPATH 里会加入仓库内的 scripts/libero_pro/hooks；那里的 sitecustomize.py
    在设了 PROBE_TAG 时把 probe_<TAG>.txt 写到 /tmp，探测对象用
    moka_pot 指定。不需要探针时不要把 PROBE_TAG 导出即可
    （默认 tag 是 'x'，与搬迁前的行为一致）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOOKS = REPO / "scripts" / "libero_pro" / "hooks"
PRO = Path("/data1/liu/exp/counterfactual/external/LIBERO-PRO")

DEFAULT_META = REPO / "scripts/experiments/tta_repair_work_pro_full10/inputs_v3/diagnosis_meta.json"
DEFAULT_REQUESTS = REPO / "experiments/tta_repair_work_pro_full10_v4/requests"
DEFAULT_OUT = REPO / "experiments/tta_repair_work_k3_turnon1"
DEFAULT_SUITE = "libero_10_swap"  # 仅当 meta 里没有 suite 字段时的兜底


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="+", help="case tags, e.g. KSCENE3-init006")
    ap.add_argument("--meta", default=str(DEFAULT_META), help="diagnosis_meta.json")
    ap.add_argument("--requests-dir", default=str(DEFAULT_REQUESTS),
                    help="directory holding repair_<tag>.json")
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR", str(DEFAULT_OUT)))
    ap.add_argument("--gpu", default=os.environ.get("LAUNCH_GPU", "0"))
    ap.add_argument("--suite", default=DEFAULT_SUITE,
                    help="fallback suite when the meta entry has no 'suite' field")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    out = Path(args.out_dir)
    requests = Path(args.requests_dir)
    (out / "outputs").mkdir(parents=True, exist_ok=True)

    for tag in args.tags:
        entry = meta[tag]
        suite = entry.get("suite") or args.suite
        env = dict(os.environ)
        env.update({
            "PATH": "/data1/liu/miniconda3/envs/cosmospolicy/bin:" + os.environ["PATH"],
            "PYTHONPATH": str(HOOKS) + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
            "GPU_ID": args.gpu, "MUJOCO_EGL_DEVICE_ID": args.gpu,
            "SMOKE_ONLY_CONDITION": "perturb", "SMOKE_PAIR_SUITE": suite,
            "SMOKE_PAIR_BASE_TASK": entry["base_task"],
            "SMOKE_PAIR_CLEAN_LANGUAGE": entry["clean_language"],
            "SMOKE_PAIR_PERT_NAME": "pro_variant", "SMOKE_PAIR_PERT_CATEGORY": "",
            "SMOKE_PAIR_PERT_TASK": entry["base_task"], "SMOKE_NUM_PAIRS": "1", "SMOKE_SEED": "7",
            "SMOKE_RESULTS_DIR": str(out / "results" / tag), "SMOKE_RUN_ID": "tta_repair_" + tag,
            "COSMOS_INIT_STATE_OFFSET": str(entry["census_abs_init"]),
            "COSMOS_DATA_COLLECTION": "1", "SMOKE_DATA_COLLECTION": "1",
            "COSMOS_INITIAL_ALIGNMENT": "1", "COSMOS_TTA_RGBD": "1",
            "COSMOS_SKILL_COMPLETION_SHADOW": "1", "COSMOS_DEBUG_INIT_ALIGN": "1",
            "COSMOS_TTA_REPAIR": str(requests / ("repair_" + tag + ".json")),
            "COSMOS_TTA_REPAIR_OUT": str(out / "outputs" / (tag + ".json")),
            "COSMOS_TTA_REPAIR_TAG": tag, "COSMOS_LIBERO_ROOT": str(PRO),
            "LIBERO_CONFIG_PATH": str(PRO / "configs" / "libero_pro"),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "COSMOS_SKILL_READY_MEMORY": "timegrip", "COSMOS_TTA_PHASE_LOOP": "1",
        })
        print("[gpu%s] %s start (suite=%s)" % (args.gpu, tag, suite), flush=True)
        subprocess.run(["sh", "run_libero_smoke_test.sh"], env=env, cwd=str(REPO / "scripts"))
        print("[gpu%s] %s done" % (args.gpu, tag), flush=True)


if __name__ == "__main__":
    main()

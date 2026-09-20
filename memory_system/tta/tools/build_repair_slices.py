"""把 repair_all7dims 的成功修复轨迹切成 SFT 训练用的 HDF5。

每个案例输出一个文件 = 源 rollout 的 [t*, t*+L)，其中
    t* = requests/repair_<tag>.json 的 t_star          memory 介入点
    D  = outputs/<tag>.json 的 duration_steps          memory 介入到交还 VLA 的步数
    L  = ceil(D/16)*16
t*+D 之后是交还 VLA 之后的真实动作，切片顺带保留，只为了把长度补成 16 的倍数。

案例判定：task_success=True 且 HDF5 attr success=True，按（去掉 _gpuN 的干扰条件、
完整任务名、tag）去重，排除旧的 robotinit_p1 —— 共 121 条。

task_description 要写成评测时实际查表的 key（跑 LIBERO-plus 变体名经
strip_libero_plus_metadata 归一化后的 base 指令），否则训练和评测查到的是
两个不同的文本 embedding。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import h5py

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]

from cosmos_policy.experiments.robot.cosmos_utils import strip_libero_plus_metadata  # noqa: E402

GPU_RE = re.compile(r"_gpu\d+$")
ARRAYS = ("primary_images_jpeg", "wrist_images_jpeg", "actions", "proprio")


def pick_rollout(cond_dir: Path, tag: str) -> Path | None:
    """该案例 rollout_data 里最大的 success=True 文件（一个 tag 可能有多份）。"""
    best = None
    for path in sorted((cond_dir / "results" / tag / "logs" / "rollout_data").glob("*.hdf5")):
        with h5py.File(path, "r") as f:
            if not f.attrs.get("success"):
                continue
            n = f["actions"].shape[0]
        if best is None or n > best[0]:
            best = (n, path)
    return best[1] if best else None


def select_cases(repair_root: Path) -> list[dict]:
    cases: dict[tuple, dict] = {}
    for cond_dir in sorted(repair_root.iterdir()):
        if not (cond_dir / "outputs").is_dir() or cond_dir.name == "robotinit_p1":
            continue
        for out_json in sorted((cond_dir / "outputs").glob("*.json")):
            tag = out_json.stem
            req_path = cond_dir / "requests" / f"repair_{tag}.json"
            if not req_path.exists():
                continue
            result = json.loads(out_json.read_text())
            if not result.get("task_success"):
                continue
            src = pick_rollout(cond_dir, tag)
            if src is None:
                continue
            req = json.loads(req_path.read_text())
            duration = int(result["duration_steps"])
            key = (GPU_RE.sub("", cond_dir.name), req["task"], tag)
            cases.setdefault(key, {
                "tag": tag,
                "condition": GPU_RE.sub("", cond_dir.name),
                "t_star": int(req["t_star"]),
                "duration": duration,
                "L": math.ceil(duration / 16) * 16,
                "src": src,
            })
    return [cases[k] for k in sorted(cases, key=lambda k: k[2])]


def write_slice(case: dict, out_dir: Path) -> Path:
    t, L = case["t_star"], case["L"]
    with h5py.File(case["src"], "r") as src:
        n = src["actions"].shape[0]
        assert t + L <= n, f"{case['tag']}: t*+L={t + L} > n={n}"
        dst = out_dir / case["src"].name
        with h5py.File(dst, "w") as out:
            for name in ARRAYS:
                out.create_dataset(name, data=src[name][t:t + L])
            out.attrs["success"] = True
            out.attrs["task_description"] = strip_libero_plus_metadata(src.attrs["task_description"])
            out.attrs["slice_t_star"] = t
            out.attrs["slice_duration_steps"] = case["duration"]
            out.attrs["slice_len"] = L
            out.attrs["slice_source_hdf5"] = str(case["src"])
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair-root", default=str(REPO / "experiments/tta/repair_all7dims"))
    ap.add_argument("--out-dir", default=str(REPO / "training/tta_sft_repair_slices_v1"))
    args = ap.parse_args()

    cases = select_cases(Path(args.repair_root))
    fallback = sum(1 for c in cases if c["t_star"] == 10)
    print(f"[slices] {len(cases)} cases: {fallback} fallback (t*=10), {len(cases) - fallback} numeric")
    if len(cases) != 121:
        raise SystemExit(f"[slices] expected 121 cases, got {len(cases)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        write_slice(case, out_dir)
    print(f"[slices] wrote {len(cases)} files, {sum(c['L'] for c in cases)} steps -> {out_dir}")


if __name__ == "__main__":
    main()

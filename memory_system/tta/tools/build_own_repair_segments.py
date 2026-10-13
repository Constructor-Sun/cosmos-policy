"""从 tta_sft_success_v2 的完整轨迹内抽取控制器修复段，构造 SFT 训练用 HDF5。

与 build_repair_slices.py 的产物约定对齐（键名/attrs），但来源是 v2 自己：
    每个文件 = 源轨迹最长死区游程（夹爪 |g|<0.3 连续段，控制器跟踪段指纹）
    所在的 64 步窗口；窗口起点 = max(0, run_start - 16)（保留介入前上下文），
    超出轨迹尾部则回移。文件名 = 源文件名 + "--ownseg"（避免与拼接的完整轨迹重名）。

task_description 原样拷贝（宿主轨迹本身以此键训练且成功），不做
strip_libero_plus_metadata 归一化。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[3]
ARRAYS = ("primary_images_jpeg", "wrist_images_jpeg", "actions", "proprio")
DEADZONE = 0.3
PRE = 16          # 游程起点前保留的上下文步数
L = 64            # 窗口长度（16 的倍数）
MIN_RUN = 10      # 短于此的游程不视为修复段


def longest_run(g: np.ndarray) -> tuple[int, int]:
    dz = np.abs(g) < DEADZONE
    best_len = best_start = cur_len = cur_start = 0
    for i, d in enumerate(dz):
        if d:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    return best_start, best_len


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default=str(REPO / "training/tta_sft_success_v2"))
    ap.add_argument("--out-dir", default=str(REPO / "training/tta_sft_own_repair_segments_v1"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wrote, skipped, runs = 0, [], []
    for f in sorted(Path(args.src_dir).glob("*.hdf5")):
        with h5py.File(f, "r") as src:
            run_start, run_len = longest_run(src["actions"][:, -1])
            if run_len < MIN_RUN:
                skipped.append(f.name)
                continue
            n = src["actions"].shape[0]
            start = max(0, min(run_start - PRE, n - L))
            dst = out_dir / (f.stem + "--ownseg.hdf5")
            with h5py.File(dst, "w") as out:
                for name in ARRAYS:
                    out.create_dataset(name, data=src[name][start:start + L])
                out.attrs["success"] = True
                out.attrs["task_description"] = src.attrs["task_description"]
                out.attrs["slice_t_star"] = int(run_start - start)
                out.attrs["slice_duration_steps"] = int(min(run_len, L - (run_start - start)))
                out.attrs["slice_len"] = L
                out.attrs["slice_source_hdf5"] = str(f)
        wrote += 1
        runs.append(run_len)
    print(f"[ownseg] wrote {wrote} files (游程长 P50={int(np.median(runs))} "
          f"min={min(runs)} max={max(runs)}), skipped {len(skipped)} {skipped} -> {out_dir}")


if __name__ == "__main__":
    main()

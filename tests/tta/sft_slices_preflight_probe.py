"""切片数据的训前检验，任何一条不过就非零退出。

全部文件走 h5py 层检查：长度是 16 的倍数、success attr、文件名能被扫描解析、
切片起点确实对齐到源文件的 t*、task_description 在 t5 embedding 里查得到。

再抽前几个文件构造真实的 LIBERODataset（和训练完全相同的加载路径），
确认样本取得出来，并实测窗口步长和 padding 比例。
"""

from __future__ import annotations

import argparse
import pickle
import re
import shutil
import sys
import tempfile
from pathlib import Path

import h5py

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]

TAG_RE = re.compile(r"--([A-Za-z0-9]+-init\d+)--")
META = REPO / "training/tta_sft_metadata"
CHUNK = 16
SAMPLE_N = 3


def fail(msg: str) -> None:
    print(f"[preflight] FAIL: {msg}")
    sys.exit(1)


def check_files(files: list[Path], t5_keys) -> int:
    """全部文件的廉价检查，返回总步数。"""
    total = 0
    for path in files:
        if not TAG_RE.search(path.name) or "success=True" not in path.name:
            fail(f"filename not scannable: {path.name}")
        with h5py.File(path, "r") as f:
            length = f["actions"].shape[0]
            if length % CHUNK or length != f.attrs["slice_len"]:
                fail(f"{path.name}: len={length} slice_len={f.attrs['slice_len']}")
            if not f.attrs.get("success"):
                fail(f"{path.name}: success attr is not True")
            if f.attrs["task_description"] not in t5_keys:
                fail(f"{path.name}: task_description not in t5_embeddings.pkl")
            t_star = f.attrs["slice_t_star"]
            with h5py.File(f.attrs["slice_source_hdf5"], "r") as src:
                if not (f["actions"][0] == src["actions"][t_star]).all():
                    fail(f"{path.name}: actions[0] != source actions[t*={t_star}]")
        total += length
    return total


def check_dataset(files: list[Path]) -> None:
    """用真实 dataset 跑一遍加载路径：父类和丢弃尾部起点的子类各一次。"""
    from memory_system.tta.model import setup_offline_hf_cache

    setup_offline_hf_cache()
    from cosmos_policy.datasets.libero_dataset import LIBERODataset, RepairSliceLIBERODataset

    sample = files[:SAMPLE_N]
    lengths = [h5py.File(p, "r")["actions"].shape[0] for p in sample]
    expected = {
        LIBERODataset: sum(lengths),
        RepairSliceLIBERODataset: sum(n - CHUNK + 1 for n in lengths),
    }
    tmp = Path(tempfile.mkdtemp(prefix="tta_slice_probe."))
    try:
        for path in sample:
            (tmp / path.name).symlink_to(path)
        for cls, want in expected.items():
            ds = cls(
                data_dir=str(META),
                t5_text_embeddings_path=str(META / "t5_embeddings.pkl"),
                chunk_size=CHUNK,
                rollout_data_dir=str(tmp),
                demonstration_sampling_prob=1.0,
                treat_success_rollouts_as_demos=True,
            )
            if len(ds) != want or ds.num_steps != want:
                fail(f"{cls.__name__}: len={len(ds)} num_steps={ds.num_steps} != {want}")
            for idx in (0, len(ds) // 2, len(ds) - 1):
                shape = tuple(ds[idx]["actions"].shape)
                if shape != (CHUNK, 7):
                    fail(f"{cls.__name__} ds[{idx}]['actions'].shape={shape}, expected ({CHUNK}, 7)")

            ep_len = {i: ep["num_steps"] for i, ep in ds.data.items()}
            starts = list(ds._step_to_episode_map.values())
            padded = sum(1 for e, s in starts if s + CHUNK > ep_len[e])
            slots = sum(max(0, s + CHUNK - ep_len[e]) for e, s in starts)
            print(f"[preflight] {cls.__name__:24s} windows={len(ds):5d} "
                  f"padded_windows={padded:4d} padded_slots={slots}/{len(ds) * CHUNK} "
                  f"({100.0 * slots / (len(ds) * CHUNK):.1f}%)")
    finally:
        shutil.rmtree(tmp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices-dir", default=str(REPO / "training/tta_sft_repair_slices_v1"))
    args = ap.parse_args()

    files = sorted(Path(args.slices_dir).glob("*.hdf5"))
    if not files:
        fail(f"no hdf5 in {args.slices_dir}")

    with open(META / "t5_embeddings.pkl", "rb") as f:
        t5_keys = pickle.load(f).keys()

    total = check_files(files, t5_keys)
    print(f"[preflight] {len(files)} files OK, {total} steps total")

    check_dataset(files)
    print("[preflight] PASS")


if __name__ == "__main__":
    main()

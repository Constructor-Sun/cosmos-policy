"""Add place records for (item, destination) pairs the LIBERO-90 build never recorded.

背景
    LIBERO-PRO 的 libero_10_task 变体改的是 goal 里的目标物（或它的目的地）。
    例如 LIVING_ROOM_SCENE5 的变体把两个杯子的盘子对调，于是需要
    white_yellow_mug_1 -> plate_1 和 porcelain_mug_1 -> plate_2 这两组记录，
    而构库时（LIBERO-90 只有白杯在左盘 / 黄白杯在右盘那两条 demo）并不存在。

做法
    同杯子换盘子：复制 (porcelain_mug_1, plate_1) -> (porcelain_mug_1, plate_2)，
    (white_yellow_mug_1, plate_2) -> (white_yellow_mug_1, plate_1)。只改
    arguments['target'] 与 memory_id，其余字段原样保留。

    为什么换盘子是保帧的：记录里的 T_object_ee_ready 是相对【目的地】的位姿
    （anchor_role == destination，见 oracle_ready_eval.select），而 plate_1 /
    plate_2 在 BDDL 里同为 plate 类型、同一资产（稳定扫描物体，旋转对称），
    两盘只差位置。item 不动，所以不引入任何物体几何差异。

    运行时实际读取的字段只有 arguments（全等匹配）、T_object_ee_ready 与回放段
    （ee_pose_object_sequence / gripper_sequence / ready_frame / segment_end）；
    T_world_object_anchor、target_points_*、complete_points_* 等由运行时的
    object_frame 重算或在离线诊断里用，保留原值不影响。

注意
    本文件是 memory_system/pointcloud_action/offline/build_memory.py 的构建产物；重建 place 库会
    覆盖这里追加的记录，届时需要重跑本脚本。

用法
    python scripts/libero_pro/add_place_records.py --dry-run
    python scripts/libero_pro/add_place_records.py
"""
from __future__ import annotations

import argparse
import copy
import shutil
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY = REPO / "memory_system/pointcloud_action/pointcloud_action_memory_place.pt"

# (source item, source target, new target)
DERIVATIONS = [
    ("porcelain_mug_1", "plate_1", "plate_2"),
    ("white_yellow_mug_1", "plate_2", "plate_1"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--memory", default=str(DEFAULT_MEMORY))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = Path(args.memory)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload["records"]
    print(f"loaded {path.name}: {len(records)} records")

    added = 0
    for src_item, src_target, new_target in DERIVATIONS:
        sources = [
            r for r in records
            if r.get("skill") == "PlaceOn"
            and (r.get("arguments") or {}).get("item") == src_item
            and (r.get("arguments") or {}).get("target") == src_target
        ]
        existing = {
            r["memory_id"] for r in records
            if (r.get("arguments") or {}).get("item") == src_item
            and (r.get("arguments") or {}).get("target") == new_target
        }
        print(f"  {src_item}: {src_target} -> {new_target}: {len(sources)} source records, "
              f"{len(existing)} already present")
        for record in sources:
            clone = copy.deepcopy(record)
            clone["arguments"] = dict(record["arguments"], target=new_target)
            clone["memory_id"] = f"{record['memory_id']}@to_{new_target}"
            clone["derived_from"] = record["memory_id"]
            if clone["memory_id"] in existing:
                continue
            records.append(clone)
            added += 1

    if not added:
        print("nothing to add; library already up to date")
        return
    if args.dry_run:
        print(f"--dry-run: would append {added} records, {len(records)} total; not written")
        return

    backup = Path(str(path) + ".backup")
    if backup.exists():
        backup = Path(f"{path}.backup_{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(path, backup)
    print(f"backed up original -> {backup.name}")

    torch.save(payload, path)
    print(f"wrote {path.name}: {len(records)} records (+{added})")


if __name__ == "__main__":
    main()

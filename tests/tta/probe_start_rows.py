#!/usr/bin/env python
"""起步几步：模型预测的动作落在 A→B 相位轴的哪里（对照视频里观察到的"先 B"）。

对每个 episode：
  axis = a*(B 相供体) − a*(晚期 A 供体)，单位化、长度 = ||·||（记为 D）
  位置(p) = <p − a*(晚期 A), axis_unit> / D      # 0 ≈ 还在 A 相，1 ≈ 已到 B 相
对 row ∈ {0,1,2,3}（扰动初始态之后的头几步）和 row = 0.5·place_start（对照）：
  报告 位置(记录的 a*)、位置(模型预测 p)、||p − a*||、以及 row0 proprio 与晚期 A proprio 的距离。

用法：
  python tests/tta/probe_start_rows.py --ckpt training/tta_sft_v2fix_merged.pt --tag v2fix --out /tmp/start_v2fix.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from memory_system.tta.model import setup_offline_hf_cache  # noqa: E402

setup_offline_hf_cache()

import h5py  # noqa: E402

from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint  # noqa: E402
from cosmos_policy.datasets.libero_dataset import LIBERODataset  # noqa: E402

sys.path.insert(0, str(REPO / "tests/tta"))
from probe_action_input_reliance import build_sample, call_loss, collate, fixed_sigma_eps, read_action_flat  # noqa: E402


def predict(model, batch, sigma_value=80.0, seed=0):
    with torch.no_grad():
        _, x0, cond = model.get_data_and_condition(batch)
        sigma, eps = fixed_sigma_eps(model, x0, cond, sigma_value, seed)
        out, _ = call_loss(model, x0, cond, eps, sigma, batch)
    return torch.tensor(read_action_flat(out, 0, int(batch["action_latent_idx"][0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--episodes", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--rows", default="0,1,2,3")
    ap.add_argument("--control-frac", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = LIBERODataset(data_dir=str(REPO / "training/tta_sft_metadata"),
                       t5_text_embeddings_path=str(REPO / "training/tta_sft_metadata/t5_embeddings.pkl"),
                       chunk_size=16, rollout_data_dir=str(REPO / "training/tta_sft_success_v2"),
                       demonstration_sampling_prob=1.0, treat_success_rollouts_as_demos=True,
                       use_image_aug=False, use_stronger_image_aug=False)
    eps_list = sorted([(i, m) for i, m in ds.rollout_episode_metadata.items() if "LRSCENE5" in m["file_path"]],
                      key=lambda t: t[1]["file_path"])
    picks = [eps_list[int(x)] for x in args.episodes.split(",")]

    model, _ = load_model_from_checkpoint(
        experiment_name="tta_repair_sft", s3_checkpoint_dir=args.ckpt,
        config_file="cosmos_policy/config/config.py", load_ema_to_reg=False,
        instantiate_ema=False, to_device="cuda")
    model.eval()
    print(f"[{args.tag}] loaded", flush=True)

    rows = [int(x) for x in args.rows.split(",")]
    recs = []
    for ep_i, ep_meta in picks:
        name = Path(ep_meta["file_path"]).name.split("--")[1]
        with h5py.File(ep_meta["file_path"], "r") as f:
            place_start = int(f.attrs.get("place_start", 0))
        ret = ep_meta.get("returns")
        ret = None if ret is None else np.asarray(ret)
        late = max(1, int(0.9 * place_start))
        brow = min(place_start + 20, ep_meta["num_steps"] - 2)
        bl = collate(build_sample(ds, ep_i, late, ret), "cuda")
        bb = collate(build_sample(ds, ep_i, brow, ret), "cuda")
        a_late = bl["actions"].float().cpu().reshape(-1)
        a_B = bb["actions"].float().cpu().reshape(-1)
        axis = a_B - a_late
        D = float(axis.norm())
        u = axis / (D + 1e-9)
        q_late = bl["proprio"].detach().float().cpu().reshape(-1)

        for row in rows + [max(1, int(args.control_frac * place_start))]:
            b = collate(build_sample(ds, ep_i, row, ret), "cuda")
            a_star = b["actions"].float().cpu().reshape(-1)
            p = predict(model, b)
            q = b["proprio"].detach().float().cpu().reshape(-1)
            rec = {
                "episode": name, "row": row, "place_start": place_start, "axis_len_D": D,
                "pos_recorded": float((a_star - a_late).dot(u) / D),
                "pos_pred": float((p - a_late).dot(u) / D),
                "err_pred_vs_recorded": float((p - a_star).norm()),
                "proprio_dist_from_late_A": float((q - q_late).norm()),
                "is_control": row == max(1, int(args.control_frac * place_start)),
            }
            recs.append(rec)
            print(f"  {name} row={row:2d}/{place_start} 记录a*在轴={rec['pos_recorded']:+.2f} "
                  f"模型预测在轴={rec['pos_pred']:+.2f} ||p-a*||={rec['err_pred_vs_recorded']:5.2f} "
                  f"proprio距晚期A={rec['proprio_dist_from_late_A']:.2f}", flush=True)
            torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(recs, indent=2))
    start = [r for r in recs if not r["is_control"]]
    ctrl = [r for r in recs if r["is_control"]]
    print(f"\n[{args.tag}] 起步 {len(start)} 条：模型预测在 A→B 轴上的位置中位 = {np.median([r['pos_pred'] for r in start]):+.2f}"
          f"（记录动作 = {np.median([r['pos_recorded'] for r in start]):+.2f}）")
    print(f"[{args.tag}] 对照（0.5·place_start, n={len(ctrl)}）：{np.median([r['pos_pred'] for r in ctrl]):+.2f}")
    print(f"[{args.tag}] 预测落在轴 >+0.3 的比例（起步）：{np.mean([r['pos_pred'] > 0.3 for r in start]):.0%}")
    print(f"[{args.tag}] 写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()

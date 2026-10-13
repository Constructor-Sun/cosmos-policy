#!/usr/bin/env python
"""尺度对齐的"位姿 vs 图像"影响力配比。

问题：proprio 是 9 个数，图像是 2×16×28×28，任何"总梯度范数"都不可比。
做法：让两个模态都沿同一条"当前相位 → B 相供体"的方向走同一个比例 t：
    proprio_t = (1-t)·q_cur + t·q_donor
    latent_t  = (1-t)·z_cur  + t·z_donor      （只改 f2 腕部、f3 第三人称，编码后混合）
t ∈ {0.25, 0.5, 0.75, 1.0}。同一个 σ/噪声、同一个基线，量预测动作动了多少、
以及朝供体相位方向投影多少。这样两边的"剂量"都定义为"走了多少比例的路"。

用法：
  python tests/tta/probe_dose_matched.py --ckpt training/tta_sft_v2fix_merged.pt --tag v2fix --out /tmp/dose_v2fix.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from memory_system.tta.model import setup_offline_hf_cache  # noqa: E402

setup_offline_hf_cache()

import h5py  # noqa: E402

from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint  # noqa: E402
from cosmos_policy.datasets.libero_dataset import LIBERODataset  # noqa: E402

sys.path.insert(0, str(REPO / "tests/tta"))
from probe_action_input_reliance import build_sample, call_loss, collate, fixed_sigma_eps, read_action_flat  # noqa: E402


def forward_pred(model, batch, sigma_value, seed, mix=None):
    """mix: {frame_idx: tensor} 覆盖 gt_frames 与 x0 的对应帧（图像 latent 混合用）。"""
    orig = model.get_data_and_condition

    def patched(db):
        raw, x0, cond = orig(db)
        if mix:
            x0 = x0.clone()
            cond.gt_frames = cond.gt_frames.clone()
            for fi, val in mix.items():
                x0[:, :, fi] = val.to(x0.dtype)
                cond.gt_frames[:, :, fi] = val.to(cond.gt_frames.dtype)
        return raw, x0, cond

    model.get_data_and_condition = patched
    try:
        with torch.no_grad():
            _, x0, cond = model.get_data_and_condition(batch)
            sigma, eps = fixed_sigma_eps(model, x0, cond, sigma_value, seed)
            out, _ = call_loss(model, x0, cond, eps, sigma, batch)
    finally:
        model.get_data_and_condition = orig
    return read_action_flat(out, 0, int(batch["action_latent_idx"][0]))


def encode_frames(model, batch, frames):
    """编码一个 batch 的 video（no_grad），返回指定帧的 latent 副本。"""
    orig = model.get_data_and_condition
    try:
        with torch.no_grad():
            _, x0, _ = orig(batch)
    finally:
        model.get_data_and_condition = orig
    return {f: x0[0, :, f].clone() for f in frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--episodes", default="0,1,2,3")
    ap.add_argument("--fracs", default="0.3,0.6,0.85")
    ap.add_argument("--sigma", type=float, default=80.0)
    ap.add_argument("--levels", default="0.25,0.5,0.75,1.0")
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
    print(f"[{args.tag}] loaded {args.ckpt}", flush=True)

    levels = [float(x) for x in args.levels.split(",")]
    records = []
    for ep_i, ep_meta in picks:
        name = Path(ep_meta["file_path"]).name.split("--")[1]
        with h5py.File(ep_meta["file_path"], "r") as f:
            place_start = int(f.attrs.get("place_start", 0))
        ret = ep_meta.get("returns")
        ret = None if ret is None else np.asarray(ret)
        drow = min(place_start + 20, ep_meta["num_steps"] - 2)
        db = collate(build_sample(ds, ep_i, drow, ret), "cuda")
        d_star = db["actions"].float().cpu().reshape(-1)
        f_w = int(db["current_wrist_image_latent_idx"][0])
        f_t = int(db["current_image_latent_idx"][0])
        z_d = encode_frames(model, db, [f_w, f_t])

        for frac in [float(x) for x in args.fracs.split(",")]:
            row = max(1, min(int(round(frac * place_start)), ep_meta["num_steps"] - 2))
            b = collate(build_sample(ds, ep_i, row, ret), "cuda")
            q_cur = b["proprio"].detach().clone()
            z_c = encode_frames(model, b, [f_w, f_t])
            a_star = b["actions"].float().cpu().reshape(-1)
            dir_v = d_star - a_star
            dn = float(dir_v.norm()) + 1e-9

            p0 = torch.tensor(forward_pred(model, b, args.sigma, 0))
            rec = {"episode": name, "row": row, "place_start": place_start, "donor_row": drow,
                   "sigma": args.sigma, "donor_astar_dist": dn,
                   "baseline_err": float((p0 - a_star).norm()), "proprio": {}, "image": {}}
            for t in levels:
                # 位姿：在归一化 proprio 空间线性插值
                bq = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
                bq["proprio"] = ((1 - t) * q_cur + t * db["proprio"].detach().clone()).to(torch.bfloat16)
                pq = torch.tensor(forward_pred(model, bq, args.sigma, 0))
                # 图像：编码后按帧混合
                mix = {f_w: (1 - t) * z_c[f_w] + t * z_d[f_w], f_t: (1 - t) * z_c[f_t] + t * z_d[f_t]}
                pi = torch.tensor(forward_pred(model, b, args.sigma, 0, mix=mix))
                for key, p in (("proprio", pq), ("image", pi)):
                    rec[key][f"{t:g}"] = {
                        "dA": float((p - p0).norm()),
                        "to_donor": float((p - p0).dot(dir_v) / dn),
                        "frac_of_path": float((p - p0).dot(dir_v) / dn / dn),
                        "err_vs_astar": float((p - a_star).norm()),
                        "err_vs_donor": float((p - d_star).norm()),
                    }
            records.append(rec)
            q = rec["proprio"]
            i = rec["image"]
            print(f"  {name} row={row} (供体相隔 {dn:.2f} 动作单位) | " + " ".join(
                f"t={t}: 位姿{q[t]['frac_of_path']:5.1%}/图像{i[t]['frac_of_path']:5.1%}" for t in [f"{x:g}" for x in levels]),
                flush=True)
            torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(records, indent=2))
    print(f"\n[{args.tag}] 中位（n={len(records)}）")
    for key in ("proprio", "image"):
        for t in [f"{x:g}" for x in levels]:
            vals = [r[key][t]["frac_of_path"] for r in records]
            print(f"    {key:8s} t={t:4s} 走到供体相位的 {np.median(vals):6.1%}   "
                  f"Δa={np.median([r[key][t]['dA'] for r in records]):.2f}")
    print(f"[{args.tag}] 写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()

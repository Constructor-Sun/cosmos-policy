#!/usr/bin/env python
"""动作损失的动作帧梯度，在参数/模块层面分别落在谁身上。

同 LRSCENE5 真实修复样本 + 固定 σ/噪声，跑一次 fwd/bwd（不更新权重），然后按模块分组统计 |∂L/∂θ|：
    cross_attn.*            语言专用通路
    self_attn.* / mlp.*     隐 token 共享通路（proprio + 视觉 + 动作 + future 帧都走这里）
    x_embedder              输入 patch 嵌入
    final_layer / t_embedder / pos_embedder / 其他
另单独统计 LoRA（本配方唯一被优化的参数）内部的质量分布。
用法：python tests/tta/probe_param_grad_share.py --ckpt <路径> --tag v2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
from probe_action_input_reliance import build_sample, collate, fixed_sigma_eps, call_loss  # noqa: E402


def group_of(name: str) -> str:
    if "cross_attn" in name:
        return "cross_attn(语言通路)"
    if "self_attn" in name:
        return "self_attn(隐token通路)"
    if ".mlp." in name:
        return "mlp(隐token通路)"
    if "x_embedder" in name:
        return "x_embedder(输入嵌入)"
    if "final_layer" in name:
        return "final_layer(输出头)"
    if "t_embedder" in name or "t_embedding" in name:
        return "t_embedder(时间步)"
    if "pos_embed" in name:
        return "pos_embed"
    return "其他"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--episodes", default="0,1")
    ap.add_argument("--fracs", default="0.3,0.6,0.85")
    ap.add_argument("--sigma", type=float, default=80.0)
    ap.add_argument("--reinit-lora", action="store_true",
                    help="按训练启动器 train_tta_sft_fixed.py 的 reinit_lora 把 A 置 kaiming、B 置 0，"
                         "复现 SFT 第 0 步的 LoRA 状态（load_model_from_checkpoint 路径下 A=B=0，LoRA 梯度恒为 0）")
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
    net = model.net

    n_all = sum(p.numel() for p in net.parameters())
    trainable = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    print(f"[{args.tag}] net 参数 {n_all/1e6:.1f}M；requires_grad=True 的张量 {len(trainable)} 个"
          f"（{sum(p.numel() for _, p in trainable)/1e6:.1f}M）", flush=True)
    lora_share = defaultdict(float)
    for n, p in trainable:
        g = group_of(n)
        lora_share[g] += p.numel()
    if trainable:
        print("    可训练参数按模块: " + ", ".join(f"{k}={v/1e6:.2f}M" for k, v in sorted(lora_share.items())))
    if args.reinit_lora:
        import math

        n = 0
        for nm, mod in net.named_modules():
            if nm.endswith("lora_A.default"):
                torch.nn.init.kaiming_uniform_(mod.weight, a=math.sqrt(5))
                n += 1
            elif nm.endswith("lora_B.default"):
                torch.nn.init.zeros_(mod.weight)
        amod = next(m for nm, m in net.named_modules() if nm.endswith("lora_A.default"))
        bmod = next(m for nm, m in net.named_modules() if nm.endswith("lora_B.default"))
        print(f"[{args.tag}] reinit LoRA {n} 层；A|w|={float(amod.weight.abs().sum()):.3g} "
              f"B|w|={float(bmod.weight.abs().sum()):.3g}（B=0 → 前向仍等于合并权重）", flush=True)
    # 打开全参数梯度（不改变前向数值，只为看梯度流向）
    for p in net.parameters():
        p.requires_grad_(True)

    records = []
    for ep_i, ep_meta in picks:
        name = Path(ep_meta["file_path"]).name.split("--")[1]
        with h5py.File(ep_meta["file_path"], "r") as f:
            place_start = int(f.attrs.get("place_start", 0))
        ret = ep_meta.get("returns")
        ret = None if ret is None else np.asarray(ret)
        for frac in [float(x) for x in args.fracs.split(",")]:
            row = max(1, min(int(round(frac * place_start)), ep_meta["num_steps"] - 2))
            batch = collate(build_sample(ds, ep_i, row, ret), "cuda")
            for p in net.parameters():
                p.grad = None
            _, x0, cond = model.get_data_and_condition(batch)
            sigma, eps = fixed_sigma_eps(model, x0, cond, args.sigma, 0)
            out, _ = call_loss(model, x0, cond, eps, sigma, batch)
            L = out["demo_sample_action_mse_loss"]
            L.backward()

            mass = defaultdict(float)
            lora_mass = defaultdict(float)
            for n, p in net.named_parameters():
                if p.grad is None:
                    continue
                v = float(p.grad.abs().sum())
                mass[group_of(n)] += v
                if "lora" in n.lower():
                    lora_mass[group_of(n)] += v
            tot = sum(mass.values()) + 1e-30
            rec = {"episode": name, "row": row, "sigma": args.sigma, "L": float(L),
                   "share": {k: v / tot for k, v in mass.items()},
                   "lora_share": ({k: v / (sum(lora_mass.values()) + 1e-30) for k, v in lora_mass.items()}
                                  if lora_mass else {})}
            records.append(rec)
            print(f"  {name} row={row} L={float(L):.4f} | " +
                  " ".join(f"{k.split('(')[0]}={v/tot:.3f}" for k, v in sorted(mass.items(), key=lambda kv: -kv[1])),
                  flush=True)
            torch.cuda.empty_cache()

    agg = defaultdict(list)
    for r in records:
        for k, v in r["share"].items():
            agg[k].append(v)
    print(f"\n[{args.tag}] 中位占比: " + ", ".join(
        f"{k}={np.median(v):.3f}" for k, v in sorted(agg.items(), key=lambda kv: -np.median(kv[1]))))
    if records and records[0]["lora_share"]:
        aggl = defaultdict(list)
        for r in records:
            for k, v in r["lora_share"].items():
                aggl[k].append(v)
        print(f"[{args.tag}] LoRA 内部中位占比: " + ", ".join(
            f"{k}={np.median(v):.3f}" for k, v in sorted(aggl.items(), key=lambda kv: -np.median(kv[1]))))
    Path(args.out).write_text(json.dumps(records, indent=2))
    print(f"[{args.tag}] 写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()

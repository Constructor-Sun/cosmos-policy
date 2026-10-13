#!/usr/bin/env python
"""loss 对"最开始那两个输入"的更新归因（不碰隐藏层语义，只看更新量）。

两项测量，都基于 LRSCENE5 真实修复样本 + 固定 σ/噪声：

A. 第一层（x_embedder/PatchEmbed）权重梯度按 latent 帧精确分解
   PatchEmbed = Rearrange(b c (t r) (h m) (w n) -> b t h w (c r m n)) + Linear，
   r=1 ⇒ 每个输出 token 只吃一帧。于是 ∂L/∂W 可写成 Σ_t (帧 t 的贡献)，逐帧取 |·| 求和即得各帧占比。
   帧布局：0 全零 | 1 proprio | 2 腕部图 | 3 第三人称图 | 4 动作 | 5..8 future/value

B. 换掉单一输入后，**优化器真正消费的那份梯度**（LoRA，训练启动器 reinit 后的状态）变了多少
   Δg = ||g_替换 − g_基线|| / ||g_基线||，以及与原梯度的余弦。
   这不受"输入维数 9 vs 25088"的尺度问题影响：同一个基线更新做参照。

用法：
  python tests/tta/probe_input_update_attrib.py --ckpt training/tta_sft_merged_v2_2030.pt --tag v2 \
      --episodes 0,1 --fracs 0.3,0.6,0.85 --sigmas 20,80 --out /tmp/iua_v2.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from memory_system.tta.model import setup_offline_hf_cache  # noqa: E402

setup_offline_hf_cache()

import h5py  # noqa: E402

from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint  # noqa: E402
from cosmos_policy.datasets.libero_dataset import LIBERODataset  # noqa: E402

sys.path.insert(0, str(REPO / "tests/tta"))
from probe_action_input_reliance import (  # noqa: E402
    OTHER_LANGS, THIRD_FRAMES, WRIST_FRAMES, build_sample, call_loss, collate, fixed_sigma_eps,
)

FRAME_NAMES = {0: "f0 全零", 1: "f1 proprio", 2: "f2 腕部图", 3: "f3 第三人称图", 4: "f4 动作",
               5: "f5 future proprio", 6: "f6 future wrist", 7: "f7 future third", 8: "f8 value"}


def fam_of(name: str) -> str:
    if "cross_attn" in name:
        return "cross_attn(语言通路)"
    if "self_attn" in name:
        return "self_attn(隐token通路)"
    if ".mlp." in name:
        return "mlp(隐token通路)"
    return "其他"


def reinit_lora(net) -> int:
    n = 0
    for nm, mod in net.named_modules():
        if nm.endswith("lora_A.default"):
            torch.nn.init.kaiming_uniform_(mod.weight, a=math.sqrt(5))
            n += 1
        elif nm.endswith("lora_B.default"):
            torch.nn.init.zeros_(mod.weight)
    return n


def lora_vec(net):
    """把 LoRA 梯度按模块族拼成向量 + 各族的 |g| 质量。"""
    vecs, mass = defaultdict(list), defaultdict(float)
    for nm, p in net.named_parameters():
        if "lora" in nm.lower() and p.grad is not None:
            f = fam_of(nm)
            vecs[f].append(p.grad.detach().reshape(-1).float())
            mass[f] += float(p.grad.abs().sum())
    cat = {f: torch.cat(v) for f, v in vecs.items()}
    return cat, mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--episodes", default="0,1")
    ap.add_argument("--fracs", default="0.3,0.6,0.85")
    ap.add_argument("--sigmas", default="20,80")
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
    print(f"[{args.tag}] reinit LoRA {reinit_lora(net)} 层", flush=True)
    for p in net.parameters():
        p.requires_grad_(False)
    for nm, p in net.named_parameters():
        if "lora" in nm.lower() or nm.startswith("x_embedder"):
            p.requires_grad_(True)

    # 第一层钩子
    cap = {}
    pe = net.x_embedder

    def hook(_m, inp, out):
        cap["x_in"] = inp[0].detach()          # (B,C,T,H,W)
        out.retain_grad()
        cap["x_out"] = out

    pe.register_forward_hook(hook)
    lin = pe.proj[1]
    r, m, n = pe.temporal_patch_size, pe.spatial_patch_size, pe.spatial_patch_size

    lang_key = "KSCENE6"
    lang_swap = ds.t5_text_embeddings[OTHER_LANGS[lang_key]].squeeze().float()

    records = []
    for ep_i, ep_meta in picks:
        name = Path(ep_meta["file_path"]).name.split("--")[1]
        with h5py.File(ep_meta["file_path"], "r") as f:
            place_start = int(f.attrs.get("place_start", 0))
        ret = ep_meta.get("returns")
        ret = None if ret is None else np.asarray(ret)
        b_row = min(place_start + 20, ep_meta["num_steps"] - 2)
        d_b = collate(build_sample(ds, ep_i, b_row, ret), "cuda")
        donor = {"proprio": d_b["proprio"].detach().clone(), "video_raw": d_b["video"], "row": b_row,
                 "astar": d_b["actions"].float().cpu().reshape(-1).tolist()}

        for frac in [float(x) for x in args.fracs.split(",")]:
            row = max(1, min(int(round(frac * place_start)), ep_meta["num_steps"] - 2))
            batch0 = collate(build_sample(ds, ep_i, row, ret), "cuda")
            raw_video = batch0["video"].clone()
            sv = raw_video.clone()
            sv[:, :, WRIST_FRAMES] = donor["video_raw"][:, :, WRIST_FRAMES]
            sv[:, :, THIRD_FRAMES] = donor["video_raw"][:, :, THIRD_FRAMES]
            variants = {
                "基线": {},
                "换位姿(B相)": {"proprio": donor["proprio"].to(torch.bfloat16)},
                "换两路图像(B相)": {"video_raw": sv},
                "换语言(另一任务)": {"t5": lang_swap.to("cuda", dtype=torch.bfloat16)[None]},
            }
            for sig in [float(s) for s in args.sigmas.split(",")]:
                out_rec = {"episode": name, "row": row, "place_start": place_start, "sigma": sig}
                base_vec, base_mass = None, None
                for vname, over in variants.items():
                    b = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch0.items()}
                    b.pop("is_preprocessed", None)
                    b["video"] = over.get("video_raw", raw_video)
                    if over.get("proprio") is not None:
                        b["proprio"] = over["proprio"]
                    if over.get("t5") is not None:
                        b["t5_text_embeddings"] = over["t5"]
                    for p in net.parameters():
                        p.grad = None
                    _, x0, cond = model.get_data_and_condition(b)
                    sigma, eps = fixed_sigma_eps(model, x0, cond, sig, 0)
                    loss, _ = call_loss(model, x0, cond, eps, sigma, b)
                    loss["demo_sample_action_mse_loss"].backward()
                    vec, mass = lora_vec(net)

                    if vname == "基线":
                        base_vec, base_mass = vec, mass
                        # ---- A: 第一层逐帧分解 ----
                        gW = lin.weight.grad
                        pat = rearrange(cap["x_in"], "b c (t r) (h m) (w n) -> b t h w (c r m n)", r=r, m=m, n=n)
                        go = cap["x_out"].grad  # (B,T,H,W,D)
                        per = {}
                        tot = 0.0
                        for t in range(pat.shape[1]):
                            M = torch.einsum("bhwd,bhwj->dj", go[:, t].float(), pat[:, t].float())
                            v = float(M.abs().sum())
                            per[t] = v
                            tot += v
                        out_rec["first_layer_frame_share"] = {FRAME_NAMES.get(int(t), str(t)): v / tot for t, v in per.items()}
                        out_rec["first_layer_grad_mass"] = float(gW.abs().sum())
                        out_rec["loss"] = float(loss["demo_sample_action_mse_loss"])
                    else:
                        d = {}
                        for f in base_vec:
                            dv = (vec[f] - base_vec[f]).norm().item()
                            nv = base_vec[f].norm().item() + 1e-30
                            cos = torch.nn.functional.cosine_similarity(vec[f], base_vec[f], dim=0).item()
                            d[f] = {"dG_ratio": dv / nv, "cos": cos,
                                    "mass_ratio": mass[f] / (base_mass[f] + 1e-30)}
                        out_rec[vname] = d
                records.append(out_rec)
                A = out_rec["first_layer_frame_share"]
                print(f"  {name} row={row} σ={sig:g} L={out_rec['loss']:.4f} | 第一层帧占比: "
                      + " ".join(f"{k.split()[0]}={v:.3f}" for k, v in A.items() if v > 0.01)
                      + " || Δg: " + " ".join(
                          f"{vn.split('(')[0]}:{np.mean([x['dG_ratio'] for x in out_rec[vn].values()]):.3f}"
                          for vn in ("换位姿(B相)", "换两路图像(B相)", "换语言(另一任务)")), flush=True)
                torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(records, indent=2))

    def med(vals):
        return float(np.median(vals))

    print(f"\n[{args.tag}] 第一层（x_embedder）权重梯度按帧的中位占比：")
    keys = list(records[0]["first_layer_frame_share"].keys())
    for k in keys:
        print(f"    {k:22s} {med([r['first_layer_frame_share'][k] for r in records]):.3f}")
    print(f"[{args.tag}] 单一输入替换后 LoRA 梯度的变化（中位）：")
    for vn in ("换位姿(B相)", "换两路图像(B相)", "换语言(另一任务)"):
        fams = list(records[0][vn].keys())
        for f in fams:
            print(f"    {vn:16s} {f:22s} Δg/|g|={med([r[vn][f]['dG_ratio'] for r in records]):.3f} "
                  f"cos={med([r[vn][f]['cos'] for r in records]):+.3f}")
    print(f"[{args.tag}] 写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()

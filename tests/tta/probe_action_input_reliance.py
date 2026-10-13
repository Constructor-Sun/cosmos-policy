#!/usr/bin/env python
"""LRSCENE5 输入依赖探针：动作损失对 (视觉 / 语言 / 位姿) 的梯度 + 反事实替换。

不训练、不更新任何权重，复用 Cosmos Policy 原训练前向：
    L_action = demo_sample_action_mse_loss   (动作 latent 帧上的 x0 MSE)

该 checkpoint 配置 min=max_num_conditional_frames=4，latent 序列
    f0 全零 | f1 当前 proprio | f2 当前腕部图像 | f3 当前第三人称图像 | f4 动作(loss 目标)
    | f5 future proprio | f6 future wrist | f7 future third | f8 value
中 f0..f3 是干净条件帧（模型在 get_data_and_condition 里把 proprio/动作/value 注入
condition.gt_frames），故三种输入都是"干净条件输入"：
    g_V = ∂L/∂ z(f2), z(f3)   视觉        g_Q = ∂L/∂ proprio       g_L = ∂L/∂ t5

叶子：patch model.encode → latent 叶子（f32→.to(bf16) 进 gt_frames；in-place 注入发生
在非叶子 clone 上，与训练一致）；batch 的 proprio/t5 换 f32 叶子再 .to(bf16)。
注意 conditioner 的 embedder.is_trainable 为 None（falsy）会在 _forward 里开 no_grad，
切断 t5→crossattn_emb 的图（只影响记录图，不影响前向数值），故先置 True。

反事实替换（同 sigma/噪声，只换一个输入，供体为同 episode 的早期/晚期 A 相与 B 相）：
    proprio ← 供体 proprio        图像 ← 供体腕部+第三人称原始帧        text ← 另一任务指令
记录每次替换后的预测动作（112 维）与供体自身的 a*，用于判断"动作是否被位姿拉向供体的相位"。

用法：
  python tests/tta/probe_action_input_reliance.py --ckpt <路径> --tag v2 --out /tmp/probe_v2.json
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
from cosmos_policy.datasets.libero_dataset import LIBERODataset, build_action_chunk_sample  # noqa: E402

META = REPO / "training/tta_sft_metadata"
DATA = REPO / "training/tta_sft_success_v2"

OTHER_LANGS = {
    "LRSCENE2": "put both the alphabet soup and the tomato sauce in the basket",
    "LRSCENE6": "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "KSCENE6": "put the yellow and white mug in the microwave and close it",
}

# 原始视频序列中腕部/第三人称图像帧位置（各 4 帧，见 build_action_chunk_sample）
WRIST_FRAMES = slice(5, 9)
THIRD_FRAMES = slice(9, 13)


class OneSample(Dataset):
    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, i):
        return self.sample


def collate(sample, device):
    batch = next(iter(DataLoader(OneSample(sample), batch_size=1, shuffle=False, num_workers=0)))
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device) if (not v.dtype.is_floating_point or k == "video") else v.to(device, dtype=torch.bfloat16)
        else:
            out[k] = v
    return out


def build_sample(ds, ep_idx, rel_idx, returns, t5_override=None):
    ep_meta = ds.rollout_episode_metadata[ep_idx]
    ed = ds._load_rollout_episode_data(ep_meta)
    return build_action_chunk_sample(
        ed,
        rel_idx,
        ds.chunk_size,
        t5_override if t5_override is not None else ds.t5_text_embeddings,
        use_proprio=ds.use_proprio,
        use_wrist_images=ds.use_wrist_images,
        use_third_person_images=ds.use_third_person_images,
        return_value_function_returns=returns is not None,
        num_duplicates_per_image=ds.num_duplicates_per_image,
        final_image_size=ds.final_image_size,
        normalize_images=ds.normalize_images,
        use_image_aug=False,
        stronger_image_aug=False,
        decompress_jpeg=bool(ed["is_jpeg"]),
        returns=returns,
        rollout_data_mask=0,  # demo 样本（v2 SFT 的实际训练分布）
        rollout_data_success_mask=0,
        world_model_sample_mask=0,
        value_function_sample_mask=0,
        global_rollout_idx=-1,
        sample_key=rel_idx,
    )


def fixed_sigma_eps(model, x0, cond, sigma_value, seed):
    """固定噪声与时间步：同 (sample, sigma, seed) 完全可复现。"""
    sigma, _eps = model.draw_training_sigma_and_epsilon(x0.size(), cond)
    g = torch.Generator(device=sigma.device).manual_seed(seed)
    eps = torch.randn(_eps.shape, generator=g, device=_eps.device, dtype=_eps.dtype)
    return torch.full_like(sigma, float(sigma_value)), eps


def call_loss(model, x0, cond, eps, sigma, batch):
    out, kendall, _, _ = model.compute_loss_with_epsilon_and_sigma(
        x0,
        cond,
        eps,
        sigma,
        action_chunk=batch["actions"],
        action_indices=batch["action_latent_idx"],
        proprio=batch["proprio"],
        current_proprio_indices=batch["current_proprio_latent_idx"],
        future_proprio=batch["future_proprio"],
        future_proprio_indices=batch["future_proprio_latent_idx"],
        future_wrist_image_indices=batch["future_wrist_image_latent_idx"],
        future_wrist_image2_indices=None,
        future_image_indices=batch["future_image_latent_idx"],
        future_image2_indices=None,
        rollout_data_mask=batch["rollout_data_mask"],
        world_model_sample_mask=batch["world_model_sample_mask"],
        value_function_sample_mask=batch["value_function_sample_mask"],
        value_function_return=batch["value_function_return"],
        value_indices=batch["value_latent_idx"],
    )
    return out, kendall


def read_action_flat(out, b, idx):
    return out["model_pred"].x0[b, :, idx, :, :].reshape(-1)[:112].float().cpu().tolist()


class EncodeLeaf:
    """把 model.encode 换成可求导叶子，并保留 latent 供取梯度。"""

    def __init__(self, model):
        self.model = model
        self.orig = model.encode
        self.leaf = None

    def __enter__(self):
        def patched(*a, **kw):
            with torch.no_grad():
                z = self.orig(*a, **kw)
            leaf = z.detach().clone().float().requires_grad_(True)
            self.leaf = leaf
            return leaf.clone().to(z.dtype)

        self.model.encode = patched
        return self

    def __exit__(self, *exc):
        self.model.encode = self.orig
        return False


def probe_grad(model, batch, leaves, sigma_value, seed):
    with EncodeLeaf(model) as el:
        raw, x0, cond = model.get_data_and_condition(batch)
        sigma, eps = fixed_sigma_eps(model, x0, cond, sigma_value, seed)
        out, _ = call_loss(model, x0, cond, eps, sigma, batch)
        L = out["demo_sample_action_mse_loss"]
        L.backward()
        leaf = el.leaf

    a_idx = int(batch["action_latent_idx"][0])
    f_wrist = int(batch["current_wrist_image_latent_idx"][0])
    f_third = int(batch["current_image_latent_idx"][0])
    g_lat = leaf.grad[0].detach().float()

    def sp(g):
        return g.abs().sum(dim=0).cpu().numpy().ravel().tolist()

    return {
        "L_action": float(L),
        "action_idx": a_idx,
        "latent_idx": {"wrist": f_wrist, "third": f_third, "proprio": int(batch["current_proprio_latent_idx"][0])},
        "modality": {
            "vision_wrist_f%d" % f_wrist: {
                "grad_norm": float(g_lat[:, f_wrist].norm()),
                "input_norm": float(leaf[0, :, f_wrist].norm()),
                "input_std": float(leaf[0, :, f_wrist].std()),
                "spatial_flat": sp(g_lat[:, f_wrist]),
            },
            "vision_third_f%d" % f_third: {
                "grad_norm": float(g_lat[:, f_third].norm()),
                "input_norm": float(leaf[0, :, f_third].norm()),
                "input_std": float(leaf[0, :, f_third].std()),
                "spatial_flat": sp(g_lat[:, f_third]),
            },
            "proprio": {
                "grad_norm": float(leaves["proprio"].grad[0].norm()),
                "input_norm": float(batch["proprio"][0].float().norm()),
                "grad_per_dim": leaves["proprio"].grad[0].detach().float().tolist(),
                "input_per_dim": batch["proprio"][0].float().tolist(),
            },
            "language": {
                "grad_norm": float(leaves["t5"].grad[0].norm()),
                "input_norm": float(batch["t5_text_embeddings"][0].float().norm()),
                "grad_per_token": leaves["t5"].grad[0].detach().float().norm(dim=-1).tolist(),
                "input_shape": list(batch["t5_text_embeddings"].shape[1:]),
            },
        },
    }


def run_variant(model, batch, over, sigma_value, seed, a_idx, raw_video):
    vb = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    vb.pop("is_preprocessed", None)  # 回到 uint8，让模型自己重新归一化
    vb["video"] = over.get("video_raw", raw_video)
    if over.get("proprio") is not None:
        vb["proprio"] = over["proprio"]
    if over.get("t5") is not None:
        vb["t5_text_embeddings"] = over["t5"]
    with torch.no_grad():
        _, x0, cond = model.get_data_and_condition(vb)
        sigma, eps = fixed_sigma_eps(model, x0, cond, sigma_value, seed)
        out, _ = call_loss(model, x0, cond, eps, sigma, vb)
    return float(out["demo_sample_action_mse_loss"]), read_action_flat(out, 0, a_idx)


def load_wrist_seg(ep_meta, rel_idx, size=224, out=28):
    with h5py.File(ep_meta["file_path"], "r") as f:
        if "wrist_segmentation" not in f:
            return None
        seg = f["wrist_segmentation"][rel_idx, :, :, 0]
    s = torch.from_numpy(seg.astype(np.float32))[None, None]
    s = torch.nn.functional.interpolate(s, size=(size, size), mode="nearest")
    s = torch.nn.functional.interpolate(s, size=(out, out), mode="nearest")
    return s[0, 0].long().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--task-prefix", default="LRSCENE5")
    ap.add_argument("--episodes", default="0,1,2,3,4,5")
    ap.add_argument("--phase-fracs", default="0.3,0.6,0.85")
    ap.add_argument("--sigmas", default="20,80")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--b-offset", type=int, default=20, help="B 相供体行 = place_start + 该值")
    args = ap.parse_args()

    ds = LIBERODataset(
        data_dir=str(META),
        t5_text_embeddings_path=str(META / "t5_embeddings.pkl"),
        chunk_size=16,
        rollout_data_dir=str(DATA),
        demonstration_sampling_prob=1.0,
        treat_success_rollouts_as_demos=True,
        use_image_aug=False,
        use_stronger_image_aug=False,
    )
    eps_list = sorted(
        [(i, m) for i, m in ds.rollout_episode_metadata.items() if args.task_prefix in m["file_path"]],
        key=lambda t: t[1]["file_path"],
    )
    picks = [eps_list[int(x)] for x in args.episodes.split(",") if int(x) < len(eps_list)]
    print(f"[{args.tag}] {len(eps_list)} 个 {args.task_prefix}；选中 {len(picks)} 个 episode", flush=True)

    model, _ = load_model_from_checkpoint(
        experiment_name="tta_repair_sft",
        s3_checkpoint_dir=args.ckpt,
        config_file="cosmos_policy/config/config.py",
        load_ema_to_reg=False,
        instantiate_ema=False,
        to_device=args.device,
    )
    model.eval()
    torch.set_grad_enabled(True)
    # embedder.is_trainable 为 None（falsy）→ _forward 里 torch.no_grad，会切断 t5→crossattn_emb
    # 的图。置 True 只影响是否建图，不改变前向数值。
    for _name, _emb in model.conditioner.embedders.items():
        if _emb.is_trainable is not True:
            _emb.is_trainable = True
    print(f"[{args.tag}] 权重已载入 {args.ckpt}", flush=True)

    lang_key = "KSCENE6" if args.task_prefix != "KSCENE6" else "LRSCENE2"
    lang_swap = ds.t5_text_embeddings[OTHER_LANGS[lang_key]].squeeze().float()

    sigmas = [float(s) for s in args.sigmas.split(",")]
    fracs = [float(f) for f in args.phase_fracs.split(",")]
    records = []
    for ep_i, ep_meta in picks:
        name = Path(ep_meta["file_path"]).name.split("--")[1]
        path = ep_meta["file_path"]
        with h5py.File(path, "r") as f:
            place_start = int(f.attrs.get("place_start", 0)) or ep_meta["num_steps"]
        ret = ep_meta.get("returns")
        ret = None if ret is None else np.asarray(ret)
        b_row = min(place_start + args.b_offset, ep_meta["num_steps"] - 2)
        donor_rows = {"early": max(1, int(0.1 * place_start)), "late": max(1, int(0.9 * place_start)), "B": b_row}
        donors = {}
        for dname, drow in donor_rows.items():
            db = collate(build_sample(ds, ep_i, drow, ret), args.device)
            donors[dname] = {
                "row": drow,
                "proprio": db["proprio"].detach().clone(),
                "video_raw": db["video"],
                "astar": db["actions"].float().cpu().reshape(-1).tolist(),
            }
        for frac in fracs:
            row = max(1, min(int(round(frac * place_start)), ep_meta["num_steps"] - 2))
            for sig in sigmas:
                batch = collate(build_sample(ds, ep_i, row, ret), args.device)
                raw_video = batch["video"].clone()  # uint8
                pq = batch["proprio"].float().detach().clone().requires_grad_(True)
                pl = batch["t5_text_embeddings"].float().detach().clone().requires_grad_(True)
                batch["proprio"] = pq.to(torch.bfloat16)
                batch["t5_text_embeddings"] = pl.to(torch.bfloat16)
                try:
                    rec = probe_grad(model, batch, {"proprio": pq, "t5": pl}, sig, args.seed)
                    rec.update({"episode": name, "rel_idx": row, "place_start": place_start,
                                "sigma": sig, "seed": args.seed,
                                "astar": batch["actions"].float().cpu().reshape(-1).tolist()})
                    a_idx = rec["action_idx"]
                    astar = torch.tensor(rec["astar"])
                    base_L, base_pred = run_variant(model, batch, {}, sig, args.seed, a_idx, raw_video)
                    ab = {
                        "self_check_dL": abs(base_L - rec["L_action"]),
                        "pred": base_pred,
                        "pred_err_vs_astar": float((torch.tensor(base_pred) - astar).norm()),
                    }
                    for dname, dinfo in donors.items():
                        sv = raw_video.clone()
                        sv[:, :, WRIST_FRAMES] = dinfo["video_raw"][:, :, WRIST_FRAMES]
                        sv[:, :, THIRD_FRAMES] = dinfo["video_raw"][:, :, THIRD_FRAMES]
                        sv_w = raw_video.clone()
                        sv_w[:, :, WRIST_FRAMES] = dinfo["video_raw"][:, :, WRIST_FRAMES]
                        sv_t = raw_video.clone()
                        sv_t[:, :, THIRD_FRAMES] = dinfo["video_raw"][:, :, THIRD_FRAMES]
                        d_star = torch.tensor(dinfo["astar"])
                        for vname, over in (
                            ("proprio", {"proprio": dinfo["proprio"].to(torch.bfloat16)}),
                            ("image", {"video_raw": sv}),
                            ("image_wrist", {"video_raw": sv_w}),
                            ("image_third", {"video_raw": sv_t}),
                        ):
                            Lv, pv = run_variant(model, batch, over, sig, args.seed, a_idx, raw_video)
                            pv_t = torch.tensor(pv)
                            ab[f"{vname}_{dname}"] = {
                                "donor_row": dinfo["row"],
                                "donor_astar": dinfo["astar"],
                                "donor_astar_dist": float((d_star - astar).norm()),
                                "dL": Lv - base_L,
                                "pred_shift_norm": float((pv_t - torch.tensor(base_pred)).norm()),
                                "pred_err_vs_astar": float((pv_t - astar).norm()),
                                "pred_err_vs_donor_astar": float((pv_t - d_star).norm()),
                                "pred_shift_to_donor": float((pv_t - torch.tensor(base_pred)).dot(d_star - astar)
                                                             / max(float((d_star - astar).norm()), 1e-9)),
                                "pred": pv,
                            }
                    Lv, pv = run_variant(model, batch, {"t5": lang_swap.to(batch["t5_text_embeddings"].device,
                                                                          dtype=torch.bfloat16)[None]},
                                         sig, args.seed, a_idx, raw_video)
                    ab["language_other_task"] = {
                        "dL": Lv - base_L,
                        "pred_shift_norm": float((torch.tensor(pv) - torch.tensor(base_pred)).norm()),
                        "pred_err_vs_astar": float((torch.tensor(pv) - astar).norm()),
                        "pred": pv,
                    }
                    rec["ablation"] = ab
                    seg = load_wrist_seg(ep_meta, row)
                    if seg is not None:
                        key = "vision_wrist_f%d" % rec["latent_idx"]["wrist"]
                        m = np.array(rec["modality"][key]["spatial_flat"]).reshape(28, 28)
                        tot = m.sum() + 1e-12
                        ids, cnt = np.unique(seg, return_counts=True)
                        rec["modality"][key]["seg_mass"] = {int(i): float(m[seg == i].sum() / tot) for i in ids}
                        rec["modality"][key]["seg_area"] = {int(i): float((seg == i).mean()) for i in ids}
                    records.append(rec)
                    m = rec["modality"]
                    vk = "vision_wrist_f%d" % rec["latent_idx"]["wrist"]
                    tk = "vision_third_f%d" % rec["latent_idx"]["third"]
                    print(
                        f"  {name} row={row}/{place_start} sig={sig:g} L={rec['L_action']:.4f} "
                        f"prederr={ab['pred_err_vs_astar']:.2f} | "
                        f"gV_w={m[vk]['grad_norm']:.3g} gV_3={m[tk]['grad_norm']:.3g} "
                        f"gQ={m['proprio']['grad_norm']:.3g} gL={m['language']['grad_norm']:.3g} | "
                        f"Δa[Q]={ab['proprio_late']['pred_shift_norm']:.2f} "
                        f"Δa[img]={ab['image_late']['pred_shift_norm']:.2f} "
                        f"Δa[lang]={ab['language_other_task']['pred_shift_norm']:.2f} | "
                        f"→donor(Q)={ab['proprio_B']['pred_shift_to_donor']:+.2f} "
                        f"→donor(img)={ab['image_B']['pred_shift_to_donor']:+.2f}",
                        flush=True,
                    )
                except Exception as e:  # noqa: BLE001
                    import traceback

                    traceback.print_exc()
                    print(f"  FAILED {name} row={row} sig={sig}: {e}", flush=True)
                torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"tag": args.tag, "ckpt": args.ckpt, "records": records}, indent=2))
    print(f"[{args.tag}] 写入 {args.out}（{len(records)} 条）", flush=True)


if __name__ == "__main__":
    main()

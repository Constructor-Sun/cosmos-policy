#!/usr/bin/env python
"""调试：为什么 t5 叶子拿不到梯度；顺带测文本是否影响 L_action。"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/data1/liu/exp/counterfactual/external/cosmos-policy")
sys.path.insert(0, str(REPO))
from memory_system.tta.model import setup_offline_hf_cache

setup_offline_hf_cache()

import h5py

sys.path.insert(0, str(REPO / "tests/tta"))
from probe_action_input_reliance import (  # noqa: E402
    META, DATA, OTHER_LANGS, build_sample, collate, fixed_sigma_eps, call_loss,
    read_action, EncodeLeaf, LIBERODataset,
)
from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "training/tta_sft_merged_v2_2030.pt")

ds = LIBERODataset(data_dir=str(META), t5_text_embeddings_path=str(META / "t5_embeddings.pkl"),
                   chunk_size=16, rollout_data_dir=str(DATA), demonstration_sampling_prob=1.0,
                   treat_success_rollouts_as_demos=True, use_image_aug=False, use_stronger_image_aug=False)
eps_l = sorted([(i, m) for i, m in ds.rollout_episode_metadata.items() if "LRSCENE5" in m["file_path"]],
               key=lambda t: t[1]["file_path"])
ep_i, ep_meta = eps_l[0]
with h5py.File(ep_meta["file_path"], "r") as f:
    place_start = int(f.attrs.get("place_start", 0))
row = int(0.6 * place_start)
print(f"episode={Path(ep_meta['file_path']).name} place_start={place_start} row={row}")

model, _ = load_model_from_checkpoint(
    experiment_name="tta_repair_sft", s3_checkpoint_dir=CKPT,
    config_file="cosmos_policy/config/config.py", load_ema_to_reg=False,
    instantiate_ema=False, to_device="cuda")
model.eval()

for name, emb in model.conditioner.embedders.items():
    print(f"embedder {name}: {type(emb).__name__} is_trainable={emb.is_trainable} dropout={emb.dropout_rate}")

ret = ep_meta.get("returns")
ret = None if ret is None else np.asarray(ret)
batch = collate(build_sample(ds, ep_i, row, ret), "cuda")
pq = batch["proprio"].float().detach().clone().requires_grad_(True)
pl = batch["t5_text_embeddings"].float().detach().clone().requires_grad_(True)
batch["proprio"] = pq.to(torch.bfloat16)
batch["t5_text_embeddings"] = pl.to(torch.bfloat16)
print("t5 emb shape", batch["t5_text_embeddings"].shape)

with EncodeLeaf(model) as el:
    raw, x0, cond = model.get_data_and_condition(batch)
    print("crossattn_emb requires_grad:", cond.crossattn_emb.requires_grad,
          "grad_fn:", cond.crossattn_emb.grad_fn, "dtype:", cond.crossattn_emb.dtype,
          "is crossattn_emb pl:", cond.crossattn_emb.data_ptr() == batch["t5_text_embeddings"].data_ptr())
    sigma, eps = fixed_sigma_eps(model, x0, cond, 20.0, 0)
    out, _ = call_loss(model, x0, cond, eps, sigma, batch)
    L = out["demo_sample_action_mse_loss"]
    print("L =", float(L))
    L.backward()
    print("grad latent:", None if el.leaf.grad is None else float(el.leaf.grad.norm()))
    print("grad proprio:", None if pq.grad is None else float(pq.grad.norm()))
    print("grad t5:", None if pl.grad is None else float(pl.grad.norm()))

# 文本消融：同样的 sigma/eps，换文本
astar = batch["actions"].float().cpu().reshape(-1, 7)


def run(t5=None, proprio=None):
    vb = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    if t5 is not None:
        vb["t5_text_embeddings"] = t5
    if proprio is not None:
        vb["proprio"] = proprio
    with torch.no_grad():
        _, x0, cond = model.get_data_and_condition(vb)
        sigma, eps = fixed_sigma_eps(model, x0, cond, 20.0, 0)
        out, _ = call_loss(model, x0, cond, eps, sigma, vb)
        a = read_action(out, 0, int(vb["action_latent_idx"][0])).float().cpu()
    return float(out["demo_sample_action_mse_loss"]), a


L0, a0 = run()
print(f"baseline           L={L0:.5f}  ||a-a*||={float((a0-astar).norm()):.4f}")
Lz, az = run(t5=torch.zeros_like(batch["t5_text_embeddings"]))
print(f"t5=0               L={Lz:.5f}  dL={Lz-L0:+.5f}  ||da||={float((az-a0).norm()):.4f}")
lang = ds.t5_text_embeddings[OTHER_LANGS["KSCENE6"]].squeeze().float().to("cuda", dtype=torch.bfloat16)[None]
Lo, ao = run(t5=lang)
print(f"t5=KSCENE6 指令     L={Lo:.5f}  dL={Lo-L0:+.5f}  ||da||={float((ao-a0).norm()):.4f}")

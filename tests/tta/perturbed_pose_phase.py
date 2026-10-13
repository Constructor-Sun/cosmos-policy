#!/usr/bin/env python
"""扰动起始位姿在 pose 空间里落在哪个相位区域？（留一 episode 的 k-NN 邻域相位）"""
import sys
from pathlib import Path

import numpy as np

REPO = Path("/data1/liu/exp/counterfactual/external/cosmos-policy")
sys.path.insert(0, str(REPO))
from memory_system.tta.model import setup_offline_hf_cache  # noqa: E402

setup_offline_hf_cache()

import h5py  # noqa: E402

from cosmos_policy.datasets.libero_dataset import LIBERODataset  # noqa: E402

META = REPO / "training/tta_sft_metadata"
DATA = REPO / "training/tta_sft_success_v2"
ds = LIBERODataset(data_dir=str(META), t5_text_embeddings_path=str(META / "t5_embeddings.pkl"),
                   chunk_size=16, rollout_data_dir=str(DATA),
                   demonstration_sampling_prob=1.0, treat_success_rollouts_as_demos=True,
                   use_image_aug=False, use_stronger_image_aug=False)

eps = sorted([(i, m) for i, m in ds.rollout_episode_metadata.items() if "LRSCENE5" in m["file_path"]],
             key=lambda t: t[1]["file_path"])
data = []
for i, m in eps:
    ed = ds._load_rollout_episode_data(m)
    with h5py.File(m["file_path"], "r") as f:
        ps = int(f.attrs.get("place_start", 0))
    data.append({"q": np.asarray(ed["proprio"], float), "ps": ps,
                 "T": len(ed["proprio"]), "name": Path(m["file_path"]).name.split("--")[1]})

print("episode  | place_start/T | 起始 0-3 步位姿的 k-NN(留一)邻域：平均相位 | 其中落在 A 相(<ps/T)的比例 | 最近邻距离")
allq = np.concatenate([d["q"] for d in data])
for k, d in enumerate(data):
    Xtr = np.concatenate([x["q"] for j, x in enumerate(data) if j != k])
    tr_ps = np.concatenate([[x["ps"] / x["T"]] * len(x["q"]) for j, x in enumerate(data) if j != k])
    tr_t = np.concatenate([np.arange(len(x["q"])) / x["T"] for j, x in enumerate(data) if j != k])
    ph = []
    dmin = []
    for t in range(4):
        dist = np.linalg.norm(Xtr - d["q"][t], axis=1)
        j = np.argsort(dist)[:8]
        ph.append(np.mean(tr_t[j] < tr_ps[j]))     # 邻居里落在其自身 A 相的比例
        dmin.append(dist[j[0]])
    print(f"{d['name']:9s}| {d['ps']}/{d['T']} = {d['ps']/d['T']:.2f}  | "
          f"A 相邻居占比 = {np.mean(ph):5.1%} | 最近邻距离 = {np.mean(dmin):.3f}")

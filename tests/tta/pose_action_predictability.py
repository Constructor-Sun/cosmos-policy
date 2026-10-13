#!/usr/bin/env python
"""教师动作到底有多"由位姿决定"？——留一 episode 的 k-NN 检验。

问题：SFT 的损失只有动作帧 MSE。如果训练集里"教师的动作"几乎可以由 proprio 单独预测
（跨 episode 泛化），那么最小化这个损失并不需要模型去看图像/语言——它能学出 pose→action
的映射而在训练集内取得低损失；测试时遇到扰动位姿就会输出"位姿对应的那个动作"。

协议（归一化用训练同一套 stats，由 LIBERODataset 提供）：
  x = 当前 proprio(9) ，y = 接下来 16 步动作块(112)，逐 episode 取
  留一 episode：用其余 episode 的样本做 k-NN(k=5) 预测 ŷ，报 ||ŷ-y||
  两个对照：
    ① 均值基线（不含任何信息）：||ȳ_train - y||
    ② 相位基线（只知道"任务进行到哪一步"）：用其它 episode 中相对步数最接近的样本的 y
  另报"同 episode 内插值"（知道自己在同一条轨迹的哪一点）作为可达性上限参考。
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/data1/liu/exp/counterfactual/external/cosmos-policy")
sys.path.insert(0, str(REPO))
from memory_system.tta.model import setup_offline_hf_cache  # noqa: E402

setup_offline_hf_cache()

from cosmos_policy.datasets.libero_dataset import LIBERODataset  # noqa: E402

META = REPO / "training/tta_sft_metadata"
DATA = REPO / "training/tta_sft_success_v2"
CHUNK = 16

ds = LIBERODataset(data_dir=str(META), t5_text_embeddings_path=str(META / "t5_embeddings.pkl"),
                   chunk_size=CHUNK, rollout_data_dir=str(DATA),
                   demonstration_sampling_prob=1.0, treat_success_rollouts_as_demos=True,
                   use_image_aug=False, use_stronger_image_aug=False)

for prefix in ("LRSCENE5", "KSCENE4"):
    eps = sorted([(i, m) for i, m in ds.rollout_episode_metadata.items() if prefix in m["file_path"]],
                 key=lambda t: t[1]["file_path"])
    data = []
    for i, m in eps:
        ed = ds._load_rollout_episode_data(m)
        q = np.asarray(ed["proprio"], dtype=np.float64)      # (T,9)
        a = np.asarray(ed["actions"], dtype=np.float64)      # (T,7)
        T = len(q)
        ys, ph = [], []
        for t in range(T):
            seg = a[t:t + CHUNK]
            if len(seg) < CHUNK:
                seg = np.concatenate([seg, np.repeat(seg[-1:], CHUNK - len(seg), 0)], 0)
            ys.append(seg.reshape(-1))
            ph.append(t / max(T - 1, 1))
        data.append({"q": q, "y": np.stack(ys), "phase": np.array(ph), "name": Path(m["file_path"]).name.split("--")[1]})

    print(f"\n===== {prefix}: {len(data)} episodes, 每条 T≈{len(data[0]['q'])}")
    X = np.concatenate([d["q"] for d in data]); Y = np.concatenate([d["y"] for d in data])
    print(f"      动作块范数中位 = {np.median(np.linalg.norm(Y, axis=1)):.2f}")

    errs = {"mean": [], "phase": [], "knn_pose": [], "insider": []}
    for k, test in enumerate(data):
        tr = [d for j, d in enumerate(data) if j != k]
        Xtr = np.concatenate([d["q"] for d in tr]); Ytr = np.concatenate([d["y"] for d in tr])
        Ptr = np.concatenate([d["phase"] for d in tr])
        for t in range(0, len(test["q"]), 5):
            y = test["y"][t]; q = test["q"][t]
            d = np.linalg.norm(Xtr - q, axis=1)
            errs["mean"].append(np.linalg.norm(Ytr.mean(0) - y))
            j = np.argsort(np.abs(Ptr - test["phase"][t]))[:5]
            errs["phase"].append(np.linalg.norm(Ytr[j].mean(0) - y))
            j = np.argsort(d)[:5]
            errs["knn_pose"].append(np.linalg.norm(Ytr[j].mean(0) - y))
            j = np.argsort(d)[:5]
            errs["knn_pose"].append(np.linalg.norm(Ytr[j].mean(0) - y))
            # 同 episode 内：用同一条轨迹 ±3 步的样本（"知道自己在轨迹哪一点"）
            lo, hi = max(0, t - 3), min(len(test["y"]), t + 4)
            idx = [i for i in range(lo, hi) if i != t]
            errs["insider"].append(np.linalg.norm(test["y"][idx].mean(0) - y))
    base = np.median(errs["mean"])
    out = {k: float(np.median(v)) for k, v in errs.items()}
    print("      预测误差中位（动作单位）与相对均值基线的 R²:")
    for k in ("mean", "phase", "knn_pose", "insider"):
        r2 = 1 - (out[k] ** 2) / (base ** 2)
        print(f"        {k:10s} err={out[k]:6.2f}   R²={r2:+.2f}")

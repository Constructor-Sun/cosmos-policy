#!/usr/bin/env python
"""逐任务：教师动作里"不能被 proprio 解释"的部分有多少？与 SFT 成功率对照。

假设：这套 SFT 只能学会 pose→action 这条捷径（损失对位姿绝对灵敏度最高、位姿和视觉共用
同一组权重）。那么"某个任务学不动"应当对应于"该任务的动作里有一大块不由位姿决定"。
协议：留一 episode 的 k-NN(k=5, 位姿欧氏) 预测 16 步动作块；对照均值基线。
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path("/data1/liu/exp/counterfactual/external/cosmos-policy")
sys.path.insert(0, str(REPO))
from memory_system.tta.model import setup_offline_hf_cache  # noqa: E402

setup_offline_hf_cache()

from cosmos_policy.datasets.libero_dataset import LIBERODataset  # noqa: E402

META = REPO / "training/tta_sft_metadata"
DATA = REPO / "training/tta_sft_success_v2"
SFT = {"KSCENE3": (1, 6), "KSCENE4": (11, 20), "KSCENE6": (4, 4), "KSCENE8": (1, 2),
       "LRSCENE2": (1, 12), "LRSCENE5": (1, 20), "LRSCENE6": (3, 4)}

ds = LIBERODataset(data_dir=str(META), t5_text_embeddings_path=str(META / "t5_embeddings.pkl"), chunk_size=16,
                   rollout_data_dir=str(DATA), demonstration_sampling_prob=1.0,
                   treat_success_rollouts_as_demos=True, use_image_aug=False, use_stronger_image_aug=False)

print("%-9s %4s %8s %9s %9s %7s   %s" % ("task", "eps", "|a|", "err_mean", "err_pose", "R2_pose", "SFT 成功"))
rows = []
for task in sorted(SFT):
    eps = sorted([(i, m) for i, m in ds.rollout_episode_metadata.items() if task in Path(m["file_path"]).name],
                 key=lambda t: t[1]["file_path"])
    data = []
    for i, m in eps:
        ed = ds._load_rollout_episode_data(m)
        q = np.asarray(ed["proprio"], float)
        a = np.asarray(ed["actions"], float)
        T = len(q)
        ys = []
        for t in range(T):
            seg = a[t:t + 16]
            if len(seg) < 16:
                seg = np.concatenate([seg, np.repeat(seg[-1:], 16 - len(seg), 0)], 0)
            ys.append(seg.reshape(-1))
        data.append({"q": q, "y": np.stack(ys)})
    if len(data) < 2:
        print("%-9s %4d  (episode 数不足，跳过)" % (task, len(data)))
        continue
    em, ep, an = [], [], []
    for k, d in enumerate(data):
        Xtr = np.concatenate([x["q"] for j, x in enumerate(data) if j != k])
        Ytr = np.concatenate([x["y"] for j, x in enumerate(data) if j != k])
        for t in range(0, len(d["q"]), 5):
            y, q = d["y"][t], d["q"][t]
            j = np.argsort(np.linalg.norm(Xtr - q, axis=1))[:5]
            ep.append(np.linalg.norm(Ytr[j].mean(0) - y))
            em.append(np.linalg.norm(Ytr.mean(0) - y))
            an.append(np.linalg.norm(y))
    em, ep, an = np.median(em), np.median(ep), np.median(an)
    r2 = 1 - (ep ** 2) / (em ** 2)
    ok, tot = SFT[task]
    rows.append((task, r2, ok / tot))
    print("%-9s %4d %8.2f %9.2f %9.2f %7.3f   %d/%d = %.0f%%" % (task, len(data), an, em, ep, r2, ok, tot, 100 * ok / tot))

print()
xs = np.array([r[1] for r in rows]); ys = np.array([r[2] for r in rows])
print("R²_pose 与 SFT 成功率的 Pearson 相关 = %.2f （n=%d）" % (np.corrcoef(xs, ys)[0, 1], len(rows)))

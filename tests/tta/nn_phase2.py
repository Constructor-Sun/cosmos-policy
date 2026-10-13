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
ds = LIBERODataset(data_dir=str(META), t5_text_embeddings_path=str(META / "t5_embeddings.pkl"), chunk_size=16,
                   rollout_data_dir=str(DATA), demonstration_sampling_prob=1.0,
                   treat_success_rollouts_as_demos=True, use_image_aug=False, use_stronger_image_aug=False)
eps = sorted([(i, m) for i, m in ds.rollout_episode_metadata.items() if "LRSCENE5" in m["file_path"]],
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
    data.append({"q": q, "y": np.stack(ys), "T": T, "name": Path(m["file_path"]).name.split("--")[1]})

print("episode  | 起点相位 t/T | 邻居相位 中位 | 25/75 分位 | 邻居动作 vs 自己A起手 距离 | 邻居间离散")
for k, d in enumerate(data):
    Xtr = np.concatenate([x["q"] for j, x in enumerate(data) if j != k])
    Ytr = np.concatenate([x["y"] for j, x in enumerate(data) if j != k])
    Ph = np.concatenate([np.arange(len(x["q"])) / x["T"] for j, x in enumerate(data) if j != k])
    for t in range(2):
        dist = np.linalg.norm(Xtr - d["q"][t], axis=1)
        j = np.argsort(dist)[:8]
        nn_ph, nn_y = Ph[j], Ytr[j]
        own = d["y"][0]
        dd = np.linalg.norm(nn_y - own, axis=1)
        spread = np.linalg.norm(nn_y - nn_y.mean(0), axis=1)
        print("%-9s| %5.3f | %5.3f | %.3f/%.3f | 中位=%.2f | 离散=%.2f" % (
            d["name"], t / d["T"], np.median(nn_ph), np.percentile(nn_ph, 25), np.percentile(nn_ph, 75),
            np.median(dd), np.median(spread)))

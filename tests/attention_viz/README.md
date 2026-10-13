# Attention-location visualization for Cosmos-Policy (LIBERO-plus robotinit)

针对 68 个 robotinit 失败 case（`memory_system/pointcloud_action/results/tta_screening_meta.json`，
即 `docs/repair/why_repair_only_sft_fails.md` 所用的那批）出 attention 位置图：
DiT 生成 action 时，action latent 帧（第 4 帧）的 self-attention query 行对各观测帧
（第 2 帧腕部、第 3 帧第三人称）的注意力，head/layer 平均后归一化，JET 热力图叠加在
相机原图上。每个控制步一对 PNG。

## 用法

在 cosmospolicy conda 环境里跑（需要 `python`、torch、cv2）：

```sh
sh tests/attention_viz/run_robotinit.sh KSCENE3-init006 --gpu 3
# 指定 SFT checkpoint（对比 base 与微调模型的注意力差异）：
sh tests/attention_viz/run_robotinit.sh KSCENE3-init006 --gpu 3 --ckpt training/tta_sft_ownseg_e1_merged.pt
# 每 5 步存一张：
sh tests/attention_viz/run_robotinit.sh KSCENE3-init006 --gpu 3 --every 5
```

产物：`outputs/attn_viz/<tag>/{base|<ckpt名>}/stepNNNN_{wrist,primary}.png`，
以及完整 episode 评测记录 `.../smoke_results/`。

## 原理（与 robust_wam 那套的差异）

- 库内核（TE / flash / SDPA）不物化 attention 权重，补丁在
  `Attention.compute_attention`（`minimal_v4_dit.py`）用 post-RoPE 的 q/k 自行重算
  softmax；输出仍走原算子，模型行为不变。
- 只捕获最后 N 个 block（默认 4，`ATTN_VIZ_BLOCKS` 可调）的 self-attention、
  只算 action 帧行；keep-latest，`get_action` 返回后即为最后一个去噪步。
- 接入方式：`run_libero_smoke_test.sh` 的 `SMOKE_PYTHON_SCRIPT` 指向
  `smoke_entry.py`，后者在评测脚本 import `run_libero_eval` 之前就地包好
  `cosmos_utils.get_action`（它按名字 import，必须在它之前替换）。
- LIBERO latent 布局（state_t=9）：`[blank, proprio, wrist, primary, ACTION, future...]`；
  帧号运行时由 `get_latent_indices_from_model_config` 推导。

## 已知近似

- 若 `cfg.trained_with_image_aug`，输入有 90% 面积中心裁剪再缩放（~1.05 倍变焦），
  热力图与原图存在等比缩放级别的微小错位，定性看不受影响。
- 图像按模型输入方向输出（与 `rollouts/` 里评测视频的方向一致，不做二次翻转）；
  `*_grid.npz` 存的也是该方向的未归一化 14×14 网格。

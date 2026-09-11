# TTA 训练实现(v1,2026-09-09)

用同一测试案例的失败轨迹(rejected)与 memory 修复成功轨迹(chosen)构成偏好对,以 Diffusion-DPO 更新 Cosmos Policy 的 LoRA。历史讨论与早期方案见 `TTA_TRAINING_NOTES.md`;验证证据见 `memory_system/tta/results/VALIDATION_RECORD_20260909.md`。

代码只有四个模块 + 三个数据工具:

```
memory_system/tta/
├── dataset.py    # 每个 item = 一个 pair(chosen/rejected 各 1 个 chunk,K=1)
├── model.py      # 加载策略、LoRA 注入、共享噪声 DPO 前向、adapter 存取
├── dpo_train.py      # 参数、单卡自定义循环(无端口、无分布式)、adapter 保存
├── test_tta.py   # CPU 测试(默认)+ GPU smoke(--gpu,尚未执行)
└── tools/        # 一次性数据准备:replay_rejected / collect_chosen / build_manifest
```

## 1. 数据格式

manifest 为 JSON:`{"pairs": [{pair_id, task, init, chosen_path, rejected_path, chosen_success, rejected_success}]}`。`chosen_path` 指向修复成功的采集 episode,`rejected_path` 指向基线失败回放的 episode。

episode 文件即 eval 采集格式(`run_libero_eval.py --data_collection True` 或 `tools/replay_rejected.py` 产物):数据集 `primary_images_jpeg`(或 `primary_images`)、`wrist_images_jpeg`(或 `wrist_images`)、`actions (T,7)`、`proprio (T,9)`,属性 `success`(bool,配对标签以此为准)、`task_description`。

约定:观测在动作前——`obs[k]`/`proprio[k]` 是 `action[k]` 执行前的观测,由采集循环的写入位置保证,无法事后从 h5 验证;chunk 只取完整段(start+16 ≤ T),不足段丢弃;动作/本体归一化复用 SFT 统计(`libero_dataset_statistics.json`),不重算;图像增强 v1 关闭。

## 2. 目标公式

对一个 pair,两侧各采 K=1 个 chunk,前向拼成 [2B, ...] 的批次,**共享同一份 sigma/epsilon**——注意默认的噪声抽取是逐 batch 项独立的,`model.share_noise_across_pairs` 会显式把每个 pair 的 chosen 份噪声复制给 rejected 份(2026-09-09 评审修正):

```
E(side)   = edm_loss_per_frame[side, action_latent_idx]        # 动作帧去噪误差
delta     = E_policy(side) - E_reference(side)                  # reference = 关 LoRA、no_grad
margin    = beta * (delta_rejected.sum() - delta_chosen.sum())
loss      = -logsigmoid(margin)
```

K=1 时这是**单 chunk 偏好目标**,不是全轨迹似然。前向为 bf16(sigma/latent 必须转入模型精度,否则崩溃)。显存实测:整 pair 单前向 + backward 峰值 ~10.9GB(预算 20GB)。

## 3. 三个文件如何连接

- `dataset.py` 读 manifest,一个 item 返回 `{字段: [B=1 批次下 [2, ...]]}` 的 pair(chosen 在前),由 `libero_dataset.build_action_chunk_sample` 组装(与 LIBERODataset 逐字节同布局,golden 测试锁定)。
- `model.py::TTADPOModel` 加载策略(`load_policy_model`)、注入 LoRA(q/k/v/o/mlp,rank8)、冻结 base;`dpo_forward(batch)` 做策略前向 + 关 adapter 的 no_grad reference 前向,返回 (loss, margin, deltas);`save_adapter/load_adapter` 存取 LoRA 权重。
- `dpo_train.py` 组装以上两者:每步 loss.backward() + AdamW(仅 LoRA 参数) + step;log 每 N 步;结束时保存 adapter。GPU 选占用最少的卡,预算 20GB,不足即停。

运行:`python memory_system/tta/dpo_train.py --manifest <json> --checkpoint <pt> --dataset-stats-path <json> --t5-text-embeddings-path <pkl>`

评测接入(2026-09-09 新增接口):`run_libero_eval.py --adapter_path <adapter.pt>` —— 加载 base 后按 adapter 内的元数据注入 LoRA 并载入权重(`model.attach_adapter`),即可用现有流程验证策略成功率。

## 4. 已验证 / 未验证

模型加载配置路径与 get_model() 一致(cosmos_policy/config/config.py,2026-09-09 评审修正——旧路径找不到 policy experiment)。已验证(GPU,2026-09-09,见 VALIDATION_RECORD):LoRA 注入 280 层/11.5M 参数;零增量 adapter 开关输出等价;策略前向 bf16 可跑;DPO backward 梯度仅在 LoRA;base 无梯度;两遍法(已弃用)可行。CPU 测试:loss 符号/交换翻转/log2;pair item 形状与共享组装等价;golden 回归。

未验证(下次持 GPU 按序执行):`test_tta.py --gpu` 的 smoke(注入/共享噪声下开关等价且扰动非空/forward-backward/抛弃式 step 后仅 LoRA 变化且 base `_version` 不变/adapter 保存→重载→输出一致/峰值显存报告);然后 1 对真实 pair 的 `dpo_train.py` 短跑;adapter 保存→重载→输出改变。**在此之前不要扩大训练规模。**

数据侧未执行:`tools/replay_rejected.py`(已存 actions 回放,无 policy)与 `tools/collect_chosen.py`(修复模式 + data_collection 重采)从未跑过;首批产物先用 `tools/build_manifest.py --validate-only` 校验再入训。

# TTA LoRA Fix

影响范围：v2（61 条，2026-09-10）与 ks4（17 条，2026-09-12）两次 SFT 的 checkpoint 均为"base + 未训练 LoRA"，基于它们的一切评测结论不可作为数据或算法结论引用。

## 1. 错误的表现

- SFT 训练表面完全正常：无任何报错、loss 从 2.2 "降至" 0.014、checkpoint 正常保存、进程干净退出。
- 但评测显示模型没有被训练：ks4 的 merged checkpoint 在自己的 17 条训练 case 上仅 2/17（≈base 地板）；v2 在 68 个 case 上 10/68。
- 本质：LoRA 参数从未被训练，merged checkpoint ≈ 被轻微污染的 base。

## 2. 为什么发生、在哪里发生

发生地点：`cosmos_policy/_src/predict2/models/text2world_model.py` 的 `build_net()`。

发生机制（四步）：

1. 网络在 **meta device** 上构建（`model_init`，官方省内存流程）。
2. `add_lora()` → PEFT `inject_adapter_in_model(init_lora_weights=True)` 的标准初始化（A=kaiming、B=0）作用在 **meta 张量**上——meta 张量没有值，初始化是空操作。
3. `net.to_empty(device="cuda")` 物化：LoRA 参数变成**未初始化内存**（恰好全零）；随后的 `net.init_weights()` 只重置基座模块，**不认识 PEFT 动态注入的 `lora_A.default`/`lora_B.default`**，没有任何环节补初始化。
4. 全部 LoRA 以 **A=B=0** 进入训练。LoRA 分支 y=B·A(x) 在 A=B=0 时两个梯度同时恒为零：`∂L/∂B = up⊗A(x) = 0`，`∂L/∂A = Bᵀ·up = 0`——不可逃逸的死锁点。

附带效应：`to_empty` 的未初始化内存不保证全零。blocks 0-3 的 47 个张量分到了**复用内存页的非零残留值**，梯度连通并真实训练；其余 513 个分到零页，永久死锁。这些残留值（单元素 ~0.4，相对 delta 高达 715×）是内存残留的量级，不是训练造成的。

## 3. 证据

全部探针脚本在 `tests/tta/`：

1. **meta 注入**：spy 钩子在 `inject_adapter_in_model` 返回瞬间读参数 → `Tensor.item() cannot be called on meta tensors`——初始化作用在无值的 meta 张量上。
2. **全零起点**：`init_state_probe.py`——加载 base 权重后的训练起点，抽样层 A/B norm 全部 = 0.000000（dtype bf16）。
3. **梯度死锁**：起点一步梯度 0/560 非零；优化器状态 `optim/` 中 513/560 个张量 exp_avg_sq **精确为零**（结构性证据，非数值下溢）。
4. **训练未改变权重**：同一 run 内 iter_400 vs iter_800 的 LoRA 差异 = 恰好一个 bf16 量化台阶（0.0039）；v2 与 ks4 两个独立 run 的 LoRA 值逐数值一致到 0.1%。
5. **部分层逃逸**：blocks 0-3 的 47 个张量在训练 run 中非零（相对 delta 高达 715×），与零梯度区完全互补——对应复用内存页的非零残留。

## 4. 如何修复

根因修复（方案 A，待批准后收编）：在 `text2world_model.py` 的 `build_net()` 中 `net.init_weights()` 之后追加：若 net 含 LoRA 模块，按 PEFT 标准重初始化（A=kaiming_uniform(a=√5)、B=zeros）。

已执行的临时载体：`tests/tta/train_tta_sft_fixed.py`（复刻官方 launch 并插入重初始化；验证完成后删除）。

修复验证结果：重训后 280/280 层 LoRA 非零、最大相对 delta 2.1%（健康量级）；评测 2/17 → **8/17**，无训练数据的 held-out case 2/3。

流程守则：今后任何新训练配置，全量跑之前先做一步梯度连通性检查——`python tests/tta/grad_flow_probe.py`，期望 LoRA 梯度非零张量 560/560。

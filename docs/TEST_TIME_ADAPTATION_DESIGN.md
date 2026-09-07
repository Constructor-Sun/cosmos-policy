# Test-Time Adaptation 设计文档：Memory-Guided Preference Optimization on LIBERO-PLUS

> 目标：在不重训、不使用 simulator dense reward / privileged oracle、不引入人工的前提下，
> 让 Cosmos-Policy 在 LIBERO-PLUS 的扰动维度上于评测/部署时在线变强。
>
> 一句话：**成功轨迹收录进 memory；失败轨迹用 memory 检索到的成功经验构造"正确道路"，
> 以 (chosen, rejected) 偏好对的形式对 policy 的 flow matching action head 做 DPO 式更新。**

---

## 1. 背景与定位

- Cosmos-Policy 已在 LIBERO 四套件上达到很强的水平（`cosmos_predict2_2b_480p_libero` checkpoint），**不做 SFT**。
- LIBERO-PLUS 的六个扰动维度（objects / layout / background / camera / light / robot init）
  天然构成"memory 中没有对应经验的 OOD 状态"的受控评测面。
- TTA 场景设定：
  - 每条轨迹结束后只有**终态 success flag**（评测脚本天然输出，与 GRAPE / RFT 立场一致，不算 oracle reward）；
  - 不使用 simulator 的 dense reward、privileged state、同初始状态 reset 重采样；
  - 更新发生在评测循环内部（或评测间隙），单卡可行。

---

## 2. 核心思想

### 2.1 闭环

```
                ┌────────────────────────────────────────────┐
                │                                            │
   rollout ──→ 成功? ──是──→ 收录进 TTA memory（观测描述子+action chunk+元数据）
      ↑          │
      │          否
      │          ↓
      │    失败归因（定位轨迹中出错段）
      │          ↓
      │    memory 检索：相似状态的成功片段 → 一致性过滤 → 构造 chosen
      │          ↓
      │    (chosen=检索构造的正确道路, rejected=实际执行的失败段)
      │          ↓
      │    flow matching DPO 更新（冻结 backbone，只调 action head / LoRA）
      │          ↓
      └──── 继续评测（policy 已更新）
```

### 2.2 三个关键设计

1. **成功 → 收录**。存 observation 描述子（可复用 `memory_system/pointcloud_action/retrieval/descriptors.py`
   的描述子方案）+ 执行的 action chunk + 任务/扰动元数据。参考 Retrieve-then-Steer 的
   "progress-calibrated successful segments"思想：只存可信的成功经验。
2. **失败 → 不丢弃，而是归因 + 检索修正**。DPO 需要 (chosen, rejected) 在相近状态下配对，
   而失败状态天然没有成功对应——这是机器人 DPO 的核心难点。
   我们的解法：**用 memory 检索相似状态的成功片段充当 chosen 端**。
   GRAPE 靠大量同初始状态重采样硬凑配对，HAPO 靠人工示范修正；检索构造是第三条路。
3. **DPO on flow matching**。Cosmos-Policy 的 action 生成是 flow matching / EDM 采样
   （`cosmos_policy/modules/hybrid_edm_sde.py`）。把 Diffusion-DPO 的去噪加权偏好损失
   移植到 velocity field 回归目标上（见 §4.2）。

### 2.3 Novelty 边界

| 组件 | 已有最近邻 | 我们的差异 |
|---|---|---|
| 成功/失败轨迹做偏好优化 | GRAPE (ICLR'25) | GRAPE 是训练时迭代采样（大量 rollout 预算、需同初始状态配对）；我们是部署时在线、检索构造配对 |
| 部署时偏好优化闭环 | HAPO (NeurIPS'25) | HAPO 的 chosen 端靠**人工**示范修正；我们用 memory 自动构造 |
| 检索成功经验指导动作 | Retrieve-then-Steer (2605.10094) | 它**零参数更新**（prior 注入采样器）；我们把检索结果用作梯度监督信号 |
| Diffusion/flow matching 上的 DPO | Diffusion-DPO (2311.12908)、FlowPRO | 均为图像/训练时；部署时在线做没有先例 |
| Test-time RL | EVOLVE-VLA、T²VLA、Q-VGM | 它们需要 progress estimator / learned critic；我们不需要 reward model |

**可以主张的创新点**：(a) 检索构造 chosen 端解决 test-time 配对问题；(b) 完整的无监督信号栈
（无人工、无 reward model、无 oracle，只用终态 success flag）；(c) flow matching policy 上的
test-time preference optimization。

---

## 3. 相关工作地图（含代码状态）

| 工作 | 信号 | 更新对象 | 需人工 | 需 reward model | 代码 |
|---|---|---|---|---|---|
| [GRAPE](https://grape-vla.github.io/) (ICRA'26, arXiv 2411.19309) | 成败轨迹对 + VLM 阶段奖励 | 全模型 LoRA | 否 | 否（隐式） | ✅ [aiming-lab/GRAPE](https://github.com/aiming-lab/GRAPE)（OpenVLA） |
| [HAPO](https://www.alphaxiv.org/abs/2506.07127) (NeurIPS'25) | 人工修正示范 | action 策略 | **是** | 否 | 未确认 |
| [Retrieve-then-Steer](https://tldr.takara.ai/p/2605.10094) | 成功片段检索 | **无参数更新** | 否 | 否 | ❌ |
| [EVOLVE-VLA](https://showlab.github.io/EVOLVE-VLA/) (2512.14666) | learned progress estimator + GRPO | VLA | 否 | 是（progress model） | 空壳（coming soon） |
| [T²VLA](https://arxiv.org/html/2606.29892v1) (2606.29892) | confidence 内部奖励 | VLA | 否 | 否（自举） | 未确认 |
| [Q-VGM / Flow Matching TTR](https://luneresearch.com/papers/c77ab81f-b700-484e-921c-0857873af85e) (ICML'26) | IQL critic 梯度 → residual velocity target | velocity field（backbone 冻结） | 否 | 是（critic） | ❌ |
| [Diffusion-DPO](https://arxiv.org/abs/2311.12908) | 人工偏好对 | diffusion 全模型 | 标注期 | 否 | ✅ [SalesforceAIResearch/DiffusionDPO](https://github.com/SalesforceAIResearch/DiffusionDPO) |
| [FlowPRO](https://arxiv.org/html/2606.05468v1) (2606.05468) | reward-free 偏好（per-state） | flow matching VLA | 否 | 否 | ❌ |
| [FPO](https://flowreinforce.github.io/) (ICLR'26) / [FPO++](https://github.com/amazon-far/fpo-control) | reward → CFM loss 比值 | flow policy | 否 | 否（但需 reward） | ✅ [akanazawa/fpo](https://github.com/akanazawa/fpo)、[amazon-far/fpo-control](https://github.com/amazon-far/fpo-control) |

经典 TTA（TENT / entropy minimization / confidence maximization，见 [IJCV'24 综述](https://dl.acm.org/doi/10.1007/s11263-024-02181-w)）
面向分类任务，对 action 生成不直接适用，仅作理论背景。

---

## 4. 复用什么代码

> 总原则：**GRAPE 抄循环与数据管线，Diffusion-DPO 抄偏好 loss，FPO 系抄 flow matching 在线微调 infra；
> 底座用本仓库自己的训练栈。没有可直接整体运行的 TTA 框架，缝合约几百行。**

### 4.1 GRAPE —— 迭代循环与数据管线（读 + 改造，不整体跑）

仓库绑定 OpenVLA（自回归 action token 的 per-token logprob），其 **loss 不能用于 Cosmos-Policy**。

**抄**：
- `Data Collection/libero_data_collect.py`：rollout 收集 + 成败标注 + 结果写文件名；
- **配对约定**：chosen/rejected 一一对应、同任务、同初始状态（我们改为检索构造，但排序/过滤逻辑可复用）；
- `rlds_covert.py`：轨迹转训练格式的管线思路；
- `TPO-Train/finetune.py` 的 iterative TPO 循环结构（collect → pair → train → 重复），LoRA 配置（rank 32, lr 2e-5）。

**不抄/改造**：配对依赖同初始状态 reset（违反我们的 no-oracle 设定）→ 换成 memory 检索构造；
OpenVLA 特有的 token logprob / flash-attn / RLDS 栈 → 换成 cosmos_policy 自己的 dataset/loader。

### 4.2 Diffusion-DPO —— 偏好 loss 的参考实现（核心移植对象）

官方实现 [SalesforceAIResearch/DiffusionDPO](https://github.com/SalesforceAIResearch/DiffusionDPO)（SDXL/Pick-a-Pic）；
[diffusers LoRA 版](https://github.com/huggingface/diffusers/blob/main/examples/research_projects/diffusion_dpo/README.md) 显存更友好。

数学骨架：
```
L_DPO = -log σ( β · [ ΔL_fm(x, a_chosen) − ΔL_fm(x, a_rejected) ] )
ΔL_fm(x, a) = L_fm^policy(x, a) − L_fm^ref(x, a)
```
- `L_fm` 在原论文中是 diffusion 去噪加权损失；移植 = 换成 flow matching 的
  velocity 回归目标（对每个 (observation, action) 对采样若干噪声水平 t，回归 v 与目标的 MSE）。
- 其余结构（β、reference 项、sigmoid 外壳）原样保留。
- 工程要点：chosen/rejected 共享同一 observation 的噪声采样，否则配对噪声淹没信号。

### 4.3 FPO / FPO++ —— Plan B（flow matching 在线微调 infra）

[akanazawa/fpo](https://github.com/akanazawa/fpo)（官方，ICLR'26）、[amazon-far/fpo-control](https://github.com/amazon-far/fpo-control)（真机版）。
它们是 RL 不是 DPO，但提供了 flow matching policy 上 on-policy 微调的成熟实现
（rollout buffer、CFM loss 处理、策略更新）。若 DPO 移植受阻（如 reference model 显存），
退路是"success 当二元 reward 的 GRPO 式更新"，这两份代码可直接撑起该路线。

### 4.4 没有代码、只读论文的

FlowPRO（per-state 粒度选择可参考其论证）、Retrieve-then-Steer（一致性过滤 +
confidence-adaptive 注入强度，用于我们的检索过滤设计）、EVOLVE-VLA（horizon 课程）、
Q-VGM（residual velocity target 的构造方式，若想从 DPO 换成 critic 引导可参考）。

---

## 5. 本仓库落地方案

### 5.1 复用的现有组件

| 现有文件/模块 | 用途 |
|---|---|
| `cosmos_policy/experiments/robot/libero/run_libero_eval.py` | 评测循环骨架，TTA driver 挂载点 |
| `cosmos_policy/experiments/robot/cosmos_utils.py`（get_model / get_action） | 模型加载与推理 |
| `cosmos_policy/modules/hybrid_edm_sde.py` | flow matching / EDM 采样与训练目标，DPO loss 的移植基础 |
| `cosmos_policy/models/policy_video2world_model.py` | policy 模型（决定冻结范围） |
| `cosmos_policy/datasets/libero_dataset.py` | 微调 minibatch 构造可参考其预处理 |
| `memory_system/pointcloud_action/retrieval/descriptors.py` | TTA memory 的观测描述子 |
| `memory_system/pointcloud_action/retrieval/pointcloud_action_memory.py` | memory 存取模式参考 |
| `analyze_failure_metrics.py` 等 failure 分析 | 失败归因的现状基础 |
| LIBERO-PLUS 评测脚本（`memory_system/pointcloud_action/eval/`、`scripts/`） | 扰动维度的 rollout 来源 |

### 5.2 新增模块（建议 `cosmos_policy/tta/` 或根级 `tta/`）

```
tta/
├── memory.py            # TTA memory：schema（描述子、action chunk、success、任务/扰动元数据、时间戳）
├── retrieval_bridge.py  # 对接 memory_system 的检索；输出候选成功片段
├── pair_construction.py # 失败归因 → 检索 → 一致性过滤 → (chosen, rejected) 对
├── fm_dpo_loss.py       # flow matching DPO（Diffusion-DPO 公式 + velocity 目标）
└── tta_driver.py        # 外挂循环：每 N episodes 触发 检索→配对→更新；checkpoint 管理
```

### 5.3 更新哪些参数

- **冻结**：WAN tokenizer、video2world backbone、T5 条件通路。
- **更新**：action head（全参或 LoRA；LoRA 参考 Diffusion-DPO 的 diffusers 实现与 GRAPE 的 rank 32 配置）。
- **Reference model**：更新前的初始 action head 权重副本（很小），或直接用 LoRA 的
  "冻结底座 + 初始 adapter" 构成 reference，几乎零额外显存。
- value head 可选用途：Cosmos-Policy 自带 value 预测，可作为**内部**信号对检索候选做重排
  （不需要 oracle，且不产生梯度）——记为可探索项，非首版必需。

### 5.4 数据流

1. `tta_driver` 包装 eval loop；每个 episode 结束后取 (轨迹, success flag)。
2. 成功 → `memory.py` 入库；失败 → `pair_construction`：
   失败归因（分段，定位出错窗口）→ 检索 top-k 相似成功片段 →
   状态距离阈值 + 轨迹级一致性过滤 → 取局部片段构造 chosen。
3. 缓冲攒够 mini-batch（如 8–32 对）→ 一次 DPO 梯度步（或每 N episodes 周期性多步）。
4. 周期性 checkpoint + 定期回测原 LIBERO 四套件（防遗忘监控）。

---

## 6. 关键技术问题与对策

| # | 问题 | 对策 |
|---|---|---|
| 1 | **State misalignment**：检索片段与失败状态不完全对齐，chosen action 可能对当前状态不可行（off-policy DPO 经典失效） | 状态距离硬阈值；只取局部窗口而非整条轨迹；借用 Retrieve-then-Steer 的 trajectory-level consistency filtering |
| 2 | **失败归因**：失败轨迹并非步步皆错，整条当 rejected 会污染信号 | 分段归因（复用 failure metrics 分析）；rejected 只取归因出的出错窗口；必要时保留失败轨迹中"正确"部分为额外的 chosen 候选 |
| 3 | **粒度**：trajectory-level（GRAPE）粗、per-state（FlowPRO）细但样本少 | TTA 场景样本少，首版用 segment-level（窗口级）折中，留消融 |
| 4 | **Reference 开销** | 初始权重副本 / LoRA 底座作 reference（§5.3） |
| 5 | **遗忘与漂移**：在线更新可能伤原有能力 | 小学习率 + LoRA 限制容量 + β 调节 KL 强度 + 定期回测原 LIBERO 套件 |
| 6 | **"成功判定"的合法性** | 立场写明：仅用评测天然产出的终态 success flag（GRAPE/RFT 同款），不用 dense reward / privileged state / oracle reset |
| 7 | **早期 memory 为空**（冷启动） | 失败轨迹先只归因不入池；或允许用 LIBERO 原域成功经验作初始 memory |

---

## 7. 实施路线图

**Phase 1 — 离线验证 loss（不改 memory）**
- LIBERO-PLUS 上收一批轨迹；用同任务配对（暂借用 GRAPE 式配对，仅离线实验用）手工构造偏好对；
- 在 action head 上跑通 `fm_dpo_loss`，确认 success rate 相对 frozen policy 有反应。
- 验收：离线 DPO 后，至少一个扰动维度 success rate 提升且原 LIBERO 回测不掉超过 1–2 个点。

**Phase 2 — 换成 memory 检索构造 chosen**
- 实现 `memory.py` / `retrieval_bridge.py` / `pair_construction.py`；
- 重点验证：检索构造的 chosen 与重采样配对的 chosen，效果差距多大（这决定 novelty 成立与否）。
- 验收：无 reset 配对条件下达到接近 Phase 1 的增益。

**Phase 3 — 在线 TTA 闭环**
- `tta_driver` 接入 eval loop，在线边评边学；
- 画 success rate 随 TTA 步数曲线；监控遗忘。
- 验收：随 episode 数 success rate 单调/近单调上升；扰动维度间无互相损害。

**Plan B**（Phase 1 失败时）：改走 FPO 式 "success 当二元 reward" 的 GRPO 更新，复用 [akanazawa/fpo](https://github.com/akanazawa/fpo) 的 on-policy infra。

---

## 8. 实验设计建议

- **Baselines**：frozen policy；success-only SFT（RFT，最朴素的收录复用）；GRAPE 式 offline TPO；（可选）Retrieve-then-Steer 式零训练 guidance。
- **评测面**：LIBERO-PLUS 六个扰动维度逐维度报告（受控的 OOD 轴），加原 LIBERO 四套件做遗忘回测。
- **消融**：检索构造 vs 同初始状态配对；segment-level vs trajectory-level；LoRA rank / β；memory 容量与过滤阈值。
- **卖点**：数据效率曲线（每个扰动维度需要多少 episodes 才追回损失）——TTA 相对 SFT 的核心优势就是数据效率。

---

## 9. 参考链接

- GRAPE: https://grape-vla.github.io/ · https://github.com/aiming-lab/GRAPE · arXiv:2411.19309
- Diffusion-DPO: https://arxiv.org/abs/2311.12908 · https://github.com/SalesforceAIResearch/DiffusionDPO
- diffusers LoRA DPO: https://github.com/huggingface/diffusers/blob/main/examples/research_projects/diffusion_dpo/README.md
- HAPO: https://www.alphaxiv.org/abs/2506.07127
- Retrieve-then-Steer: arXiv:2605.10094
- EVOLVE-VLA: https://showlab.github.io/EVOLVE-VLA/
- T²VLA: https://arxiv.org/html/2606.29892v1
- Q-VGM (Flow Matching TTR): https://luneresearch.com/papers/c77ab81f-b700-484e-921c-0857873af85e
- FlowPRO: https://arxiv.org/html/2606.05468v1
- FPO: https://flowreinforce.github.io/ · https://github.com/akanazawa/fpo · arXiv:2507.21053
- FPO++: https://github.com/amazon-far/fpo-control
- TTA 综述: https://dl.acm.org/doi/10.1007/s11263-024-02181-w

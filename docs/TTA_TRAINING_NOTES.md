# Memory 引导的 TTA 参数训练：LoRA + Diffusion-DPO 实现方案

> **2026-09-09 更新**：文末新增「§11 可行性核验结论、数据现状与执行约束」一节，为当前执行口径；§1–§10 保留作背景与原始规划，冲突处以 §11 为准。

## 1. 本轮决定

目标是利用同一测试案例的失败/修复成功轨迹，更新 Cosmos Policy 的 LoRA，使模型学到修复行为。本文件是代码实施规划，不表示已实现，也不启动训练；执行修复的现状仍见 TMP.MD。

- rejected：原 policy 测试失败的轨迹。
- chosen：同一 task、variant、init 下，经 memory 修复后终态成功的轨迹，包含修补控制器实际执行的动作。
- 正负轨迹使用各自的真实观测，不做跨轨迹逐帧对齐。
- 共享 DiT 的注意力/MLP 使用 LoRA；底座、VAE、文本编码器冻结。这里“都是 LoRA”指共享网络采用 LoRA，不是全参数训练，也不把编码器加入训练。
- DPO 只使用 action latent 的去噪误差；不额外训练视频/未来状态/value 损失。
- 不再要求视频生成行为严格不变：动作和视频共享带 adapter 的 DiT，不拆双前向、不设计视频专用隔离路径。
- reference 为此次适应开始前的底座，关闭本次 LoRA 并 no_grad 计算；同一训练 run 内保持固定。以后若以已有 adapter 为 reference，需要保存固定版本，不能把持续更新的 adapter 当 reference。

## 2. 已核对的可复用代码

| 现有文件 | 已有能力 | 使用方式 |
|---|---|---|
| cosmos_policy/models/policy_text2world_model.py | CosmosPolicyDiffusionModel；get_data_and_condition、采样 sigma/epsilon、EDM loss；output_batch 中有 edm_loss_per_frame | 派生 DPO 模型，复用原 conditioning、动作 latent 布局与去噪路径 |
| cosmos_policy/_src/predict2/models/text2world_model.py | use_lora、lora_rank、add_lora；通过 PEFT inject_adapter_in_model 注入 | 复用注入，不手写 LoRA Linear |
| cosmos_policy/datasets/libero_dataset.py | LIBERODataset；图像/本体/动作归一化、latent 槽位、action chunk padding | 提取或复用样本组装逻辑，增加配对数据集 |
| cosmos_policy/datasets/dataset_common.py、dataset_utils.py | 数据组装与归一化工具 | 复用既有格式，不建立另一套动作单位 |
| cosmos_policy/scripts/train.py、cosmos_policy/trainer.py | 分布式 DataLoader、训练循环、optimizer/checkpoint 调度 | 继续作为训练入口，不复制整个 trainer |
| cosmos_policy/config/config_v2.py、config/experiment/ | 配置构建/注册 | 注册新 DPO experiment |
| cosmos_policy/_src/predict2/checkpointer/dcp.py、utils/model_loader.py | 部分 LoRA 保存/加载支持 | 核对 Policy 路径兼容性，再接入 adapter 导出与推理 |

现有 mask_loss_for_action_future_state_prediction 会依据样本类型选择不同帧，不应只打开开关就假设所有配对样本均 action-only。DPO 明确用 action_latent_idx 提取动作误差，不使用已经跨 batch 平均的 demo_sample_action_mse_loss。

## 3. 训练目标

基础方法采用 Wallace 等的 Diffusion-DPO。原方法是扩散生成的偏好优化；下面的 action-frame EDM 与轨迹聚合是本项目适配，不能称为直接算出了精确轨迹 log probability。

对轨迹 tau 中第 k 个 action chunk，给其自身观测 O_k 和动作 A_k 加噪，计算加权 EDM 动作误差 E_theta(k) 与 E_ref(k)。两模型必须使用完全相同的条件、sigma、epsilon、有效动作掩码。

```text
delta(k) = E_policy(k) - E_reference(k)
D(tau)   = sum_k delta(k)
margin   = beta * (D(rejected) - D(chosen))
loss     = -logsigmoid(margin)
```

chosen 相对 reference 的误差降低更多时，margin 增大、loss 降低。使用稳定的 logsigmoid；beta 显式配置，不照搬 SDXL 的数值，因为噪声权重、动作维度和轨迹尺度不同。

关键约束：

1. 每对轨迹得到一个偏好损失，不给每个失败 chunk 都单独打“坏动作”标签，也不把两条不同长度轨迹按索引硬配。
2. 每个 action frame 内按有效维度归约，再在轨迹内求和；长度与有效 chunk 数单独记录。若改为轨迹均值，必须标明它改变了目标，不能悄悄替换。
3. 第一版先用一对真实轨迹检验完整聚合。为显存采用 K 个 chunk 均匀采样时，使用 N/K 权重估计总分，并显式称为采样代理目标：分数估计无偏不代表经过 sigmoid 的 loss 无偏；K 是后续配置，不作为精确 DPO 宣称。
4. 完全相同且 chunk 划分也相同的前缀可去除，但先验证实际观测/动作一致。修补开始后的正负后缀不能因观测不同而丢弃。
5. 不额外添加 reward model、GRPO、视频生成损失或未讨论的正则。先把基本偏好目标接正确。

## 4. 数据入口与样本格式

新增 pair manifest，一行一对，只引用现有轨迹文件：pair_id、task/variant/init、chosen_path、rejected_path、各自起止索引、终态标签、修补段范围、数据/动作配置版本。不能仅靠文件名猜配对；chosen 必须实际成功，rejected 必须实际失败。

每条轨迹至少读取 primary/wrist 图像、proprio、实际 action 与时间映射。已有观测动作是数据前提；实施时核对字段名和逐步对应，不重建采集体系。

- observation[k] 对应 action[k] 执行前；控制器动作也需对应它执行前的真实观测。
- 动作单位、旋转表示、夹爪符号、图像朝向、resize 和归一化使用原训练约定；环境动作若经过缩放/夹爪变换，按实际接口转换，不能重复归一化。
- chunk 长度读模型配置，不硬编码“16”替代模型训练 horizon；16-step 切入约定与训练 chunk 长度是不同概念。
- 取连续实际动作组成 chunk，不穿越 episode 边界。保留修补段及其与 policy 接管的连续动作，不因动作来源不同删掉关键纠正段。
- 末尾 padding 沿用原工具，但要追踪有效动作维度；通过现有 action→latent 映射生成对应有效 mask，padding 不贡献偏好分数。若暂不实现 mask，则明确丢弃不足 horizon 的尾块并报告比例，不能把 padding 当真实动作。
- 首版冻结图像增强随机性，保证 policy/reference 条件一致。不能让后续未来图像或未来状态以可见条件泄漏给动作预测；按原 policy-mode conditioning 核对掩码。
- 固定形状批次：pair 轴、chosen/rejected 轴、chunk 轴；变长靠 mask 或固定 K 采样处理，兼容现有默认 DataLoader collate。

## 5. LoRA 和 reference 实现

建议初始配置：rank=8、alpha=16、dropout=0；作为工程起点而非最优值。target modules 先沿用已有 q_proj/k_proj/v_proj/output_proj/MLP 列表，打印实际命中模块与可训练参数量，不能静默漏挂。

Policy 与 reference 共享同一底座。现有使用 inject_adapter_in_model，不保证网络拥有 PeftModel 的 disable_adapter() 方法；实现一个小 context manager，通过实际注入的 adapter 层禁用/恢复，进入前保存状态，异常退出也恢复。不要 merge adapter 后再关闭。

同一数据处理一次得到条件和噪声，先 no_grad 算 reference，恢复 adapter 后算 policy。开启 gradient checkpointing 时，反向重算期间 adapter 必须仍启用；不能在 policy graph 未反传时把 adapter 留在禁用状态。

现有 trainer 创建 optimizer 和加载 checkpoint 的顺序需核对：只允许 LoRA 参数进 optimizer；base checkpoint 加载兼容新增 adapter keys；避免 EMA 或其他回调意外更新底座、reference。首版不维护第二份可训练/EMA reference。

保存 adapter 权重及 rank/alpha/target modules、base checkpoint 标识、数据 manifest、DPO 配置。加载到原底座后推理，不合并覆盖原权重。视频可能随共享 LoRA 改变，这是本轮接受的行为。

## 6. 新增文件与预计行数

以下是建议文件名与职责，实施前确认没有同名新文件。行数是新增/实质修改代码的估计，不含复用依赖、自动格式化和本文。

| 文件 | 职责 | 预计行数 |
|---|---|---:|
| memory_system/tta/build_preference_pairs.py | 配对、标签/时间范围检查、生成 manifest，支持只校验 | 80–130 |
| cosmos_policy/datasets/tta_preference_dataset.py | 读取正负轨迹，复用原样本组装，连续 chunk 与有效 mask | 170–250 |
| cosmos_policy/models/tta_dpo_loss.py | chunk→轨迹分数、mask、稳定 DPO loss 和统计 | 40–80 |
| cosmos_policy/models/policy_dpo_model.py | 派生 CosmosPolicyDiffusionModel；reference 开关、噪声复用、DPO training_step、梯度范围 | 180–270 |
| cosmos_policy/config/experiment/tta_dpo.py | 配对 dataset、模型类型、LoRA、optimizer 与训练配置 | 50–80 |
| scripts/train_tta_dpo.sh | 调用既有 torchrun/train.py，传数据与输出配置 | 20–40 |
| 现有文件的接口调整（见下一节） | 样本组装复用、配置注册、加载适配 | 70–140 |
| 合计：核心实现 | 不重写 trainer、VAE 或 DiT | **610–990** |
| tests/tta/test_preference_dataset.py | 配对、动作时序、chunk/padding、修补段覆盖 | 70–110 |
| tests/tta/test_dpo_loss.py | 损失符号、归约、mask、采样与变长 | 60–90 |
| tests/tta/test_dpo_reference.py | 冻结/adapter 开关、checkpoint 恢复、梯度范围 | 70–120 |
| 合计：测试 | 另计实际 GPU smoke，不复制训练器 | **200–320** |

预计总量约 810–1310 行；最小能跑版本更少，但不以省略 reference/数据验证来压行数。若现有 LoRA checkpoint/FSDP 兼容性需要修正，追加约 100–200 行；不在检查前保证精确工作量。

## 7. 现有代码具体怎么改

### 7.1 模型损失

优先不改 policy_text2world_model.py 的现有 SFT 行为。新增子类覆写 training_step，直接复用 compute_loss_with_epsilon_and_sigma 返回的 edm_loss_per_frame，以 action_latent_idx gather 每样本动作误差。需要有效动作维度 mask 时，从已返回的未归约 EDM 张量抽取 action frame，复用 latent 铺排映射；必要时只增加一个 helper，不再写一遍 denoise。

不能调用原 training_step 后只拿 batch mean 计算 DPO，因为已经丢失 pair/chunk 身份。逐帧统计含未来帧，必须显式抽取 action；不把 debug 视频误差加入 loss。

### 7.2 数据组装

LIBERODataset.__getitem__ 当前直接组装当前/未来图像、proprio、action 索引与 padding。若可按轨迹和 step 调用现有 helper，直接复用；否则提取一个共享 build_sample helper，原 dataset 与新 paired dataset 都调用，保持原返回字段不变。不要复制几百行旧 __getitem__。

### 7.3 配置与训练入口

在 config_v2/experiment 的实际注册路径增加新配置导入；模型类型切换为 DPO 子类，dataset 类型切换为配对类。继续使用 cosmos_policy.scripts.train 与 CosmosPolicyTrainer 的 optimizer、梯度累积、日志与恢复。脚本中的具体 experiment key 在注册后确定，不提供未经验证的可执行命令。

先单卡、一个 pair 跑通，再使用原分布式入口。一次 pair 的总分计算后再过 sigmoid；普通梯度累积不能替代这个步骤，不能把 microbatch loss 平均冒充全轨迹 loss。显存不足先测量，再选择已明确记录的 K-chunk 代理配置。

### 7.4 保存与加载

优先复用现有 dcp/model_loader 的 LoRA 支持；若 Eval 加载链未覆盖 adapter，仅增加可选 adapter_path 及加载调用。不改变 memory 修复入口；微调后是否调用 memory 由评测配置独立决定。

## 8. 实施顺序与验证

1. **数据先过关。** 构造一对真实失败/成功轨迹，输出样本图、动作索引与来源。验证 pair 身份、每侧 obs/action 对齐、修补段确实入训、尾块 mask。不能把已达位后的数据当成全部 chosen，漏掉纠正动作。
2. **LoRA/reference 过关。** 初始零增量 adapter 启用/禁用时输出一致；更新一步后只有 adapter 变化，底座/编码器/reference 校验不变。禁用 adapter 后恢复原输出，保存重载后恢复更新输出。
3. **损失过关。** policy=reference 时 margin≈0、loss≈log(2)，但参数梯度不应被错误切断；只降低 chosen 的相对误差应使 loss 下降，交换正负应翻转 margin。验证 padding 不参与、短长轨迹分别归约、reference 与 policy 噪声完全一致。
4. **真实单步。** 一对真实数据完成 forward/backward，梯度有限且仅在 LoRA，action loss 有效、无额外视频/value loss。记录显存与耗时，再确定 batch/K 配置；不开全量 rollout。
5. **小规模拟合。** 固定少量 pair 检查偏好 margin 能改善，adapter 可保存恢复；训练 loss 降低只说明优化接通，不等于任务提升。完整训练前检查共享前缀处理与失败轨迹长时间挣扎没有支配分数。
6. **部署验证。** 将 adapter 加载到现有 policy，关闭本次测试中的外部修复，观察模型是否学到了改进；另外记录任务成功、轨迹稳定性和视频变化，不要求视频数值不变。不自动触发新的数据收集/发布/长时间训练。

日志至少包含 pair_id、chosen/rejected 的 policy/reference action loss、轨迹长度、margin、pair accuracy、LoRA 梯度范数、学习率、显存和采样配置。第一版用已积累的配对批量更新；逐 episode 在线更新的调度属于后续扩展，不在这次训练核心里重复实现。

## 9. 已知边界

成功轨迹允许与训练示范路径不同，修补动作也可学习；但数据必须是实际执行后重新记录的观测动作。若对停顿/突变做了后处理，不能保留旧观测配新动作。轨迹质量筛选可复用现有规则，不把新轨迹优化系统作为本轮前置。

当前工作的核心是配对数据和 DPO 损失适配，不是再次改 phase 判别、t*、运动规划。已有执行结果若混淆 reached 与终态 success，训练标签取真实终态，数据来源和修补段另行核对，不凭 reached 推断 chosen。

## 10. 方法与实现参考

- [Diffusion-DPO 原论文](https://arxiv.org/abs/2311.12908)：经典扩散偏好目标。
- [作者官方实现](https://github.com/SalesforceAIResearch/DiffusionDPO)：参考噪声构造和正负相对去噪误差计算，不复制其整套 SDXL trainer。
- [Diffusers LoRA Diffusion-DPO 示例](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/diffusion_dpo)：参考 adapter/reference 管理。Cosmos 使用 EDM 和不同 latent 布局，不能直接换数据路径就运行。

## 11. 2026-09-09 修订：可行性核验结论、数据现状与执行约束

> 本节为当前执行口径；§1–§10 与本节冲突处以本节为准。本节只是计划与约束更新，不代表任何代码已实现或已验证。

### 11.1 可行性核验结论

§2 所列可复用代码已逐条对照源码核实，全部属实：

| 声称 | 核实位置 |
|---|---|
| training_step / compute_loss_with_epsilon_and_sigma / edm_loss_per_frame / action_latent_idx | cosmos_policy/models/policy_text2world_model.py:240、:310、:675、:281 |
| use_lora / add_lora / inject_adapter_in_model；无 disable_adapter，需自写 context manager | cosmos_policy/_src/predict2/models/text2world_model.py:116、:1042、:1112（§5 的判断正确） |
| checkpointer 与 adapter 加载 | cosmos_policy/_src/predict2/checkpointer/dcp.py:265、:294；cosmos_policy/_src/predict2/utils/model_loader.py 的 adapter_checkpoint_paths |
| LIBERODataset 与 hdf5 格式 | cosmos_policy/datasets/libero_dataset.py:485 起 __getitem__ 为约 250 行单体函数；抽共享 build_sample 是必要重构 |
| trainer / train 入口 | cosmos_policy/trainer.py 的 CosmosPolicyTrainer、cosmos_policy/scripts/train.py 可沿用 |

结论：工程上成立，主要缺口在数据侧，不在训练代码。

### 11.2 数据现状（修订 §4 的数据前提）

- rejected：68 条基线失败已定位（experiments/tta_census/robotinit_baseline_failures.json）；experiments/tta_phase_check/&lt;task&gt;/&lt;init&gt;/episode.h5 已存 actions (T,7) + proprio (T,9)，**无图像**；memory_system/tta/diagnose_failed.py 可确定性重放（此前已验证可复现）。
- chosen：68 次单次修复已跑完（summary JSON 记 62/68 success，TMP.MD 记 64/68，以重跑实测为准），但当时 data_collection=False，**逐步观测与实际执行动作未落盘**；现存仅汇总 JSON 与少量视频。视频无腕部图像、无 proprio、帧与动作步不对齐，不可作训练数据；对齐段动作依赖实时观测计算，没有存动作就无法离线重建。
- 因此：chosen 必须重跑采集；rejected 只需补图像。

### 11.3 修订后的数据策略（替代 §4 的采集前提）

1. rejected 补图像：用已存 actions 在 env 确定性重放取图，**不需要 policy 推理**，开销小。重放时同一遍落盘 actions/proprio/图像/终态，不拿旧轨迹配新 metadata；重放意外成功的 case 丢弃并记录。
2. chosen 重新采集：以 COSMOS_TTA_REPAIR 模式重跑修复并开 data_collection。采集设施已覆盖全部动作来源（run_libero_eval.py:1013–1015 图像/proprio、:1066 对齐控制器动作、:1367 policy 动作），h5 文件名已含 COSMOS_TTA_REPAIR_TAG，天然不覆盖旧文件。
3. 缩水与增量，68 条全量不是训练开工前置：先 1 对冒烟——repair 模式 × data_collection 从未同跑，须先验证回放前缀动作是否进采集缓冲、obs[k] 对应 action[k] 执行前、无时间缝隙；再采 8–16 对（覆盖 7 个 task 类型）供第一轮小规模拟合；其余断点续跑，边采边并入训练。
4. 时间预期：单进程顺序下每条约 2–5 分钟；16 条约 0.5–1.5 小时；68 条约 3–6 小时。
5. manifest 按重跑实测终态标签构建；运行间漂移（±2 条量级）不作特殊处理，一律以实际执行结果为准。

### 11.4 本轮新增设计决定

- batch=1 pair。显存与「整轨迹求和后再过 sigmoid」的冲突用两遍法解决：第一遍 no_grad 计算全部 chunk 的 delta 得 margin，第二遍把 sigmoid(-margin) 作为常数（detached）权重带梯度回传；梯度与精确 DPO 等价，代价是每步双倍前向。「不用梯度累积平均冒充轨迹 loss」的约束不变，且由此不再依赖大显存整批前向。
- §3 补充约束 6：chosen 与 rejected 按 chunk 索引共享同一组 sigma/epsilon（policy/reference 共享的要求不变），降低轨迹级 margin 方差；这是 Diffusion-DPO 官方实现的做法。
- §3 约束 3 的 K-chunk 采样代理：仅在两遍法实测显存不足时才启用。
- padding 采用 §4 的退路：丢弃不足 horizon 的尾块并报告比例，首版不实现 latent 级有效维度 mask。
- adapter 开关 context manager 先在小 CPU 网络（同样用 inject_adapter_in_model）上按 §8.2 验证（零增量开/关一致、禁用后恢复、异常退出恢复、checkpointing 重算期间保持启用、保存重载），再上真模型。
- pair batch 与 chunk 数均为显式配置参数：`pair_batch_size`（每优化步的 pair 数，测试默认 1，实验可改大；>1 时逐 pair 过 sigmoid 后取均值，语义不变）、`chunks_per_pair`（不设或 0 表示全量 chunk，>0 表示均匀采样 K 个并按 §3 约束 3 记为采样代理）。
- 部署收益（§8.6）不作宣称，仍是开放验证；sum-over-chunks 是代理目标而非精确轨迹似然的定位不变。

### 11.5 执行约束（2026-09-09 起生效）

- 不删除 memory_system 及任何目录的既有文件；不修改 conda、git 环境（不 commit、不建环境）。
- 单 GPU（2026-09-09 修订）：整个任务最多使用 1 块 GPU，开工时选定后不再换卡；跑前 nvidia-smi 选当前**显存占用最少**（最闲、对他人干扰最小）的一块，并核对其空闲量 ≥ 本任务预算，不足则按降级原则延后；单进程顺序执行；并发 ≤8 进程且 GPU 相关 ≤1；峰值内存 ≤150GB；按 task 分批、env 按 task 复用创建，禁止全量轨迹同时驻留内存。
- GPU 显存预算 10GB（policy 推理与真模型单步冒烟含在内）；纯渲染/抽检 ≤1GB。空闲不足或抽检失败即降级延后，不阻塞、不重试轰炸、不挤占他人。
- 分辨率沿用现有 eval/采集管线约定（预期 256；以 get_image_resize_size 与训练 dataset 配置的实测为准，确认结果写入验证记录，不为凑数改管线）。
- 不占用任何端口：训练单进程、非 torchrun，wandb 关闭。
- 只新增不覆盖：采集结果、汇总、manifest、验证记录均写新文件/新目录；不覆盖既有 tta_repair_68_summary.json、experiments/tta_phase_check/ 等。
- **代码位置（2026-09-09 追加；同日经批准放宽）**：新增代码置于 memory_system/tta/ 下；既有文件默认不改动。放宽记录：为消除并排重复，`libero_dataset.py` 完成了用户此前批准的最小接口改动——把 `__getitem__` 的样本组装体提取为模块级 `build_action_chunk_sample()`（纯提取），配对数据集改为调用同一函数，不再维护第二份组装逻辑。提取用「git HEAD 原始版 vs 重构版」同进程对照（6 样本 × 6 内容哈希 + 13 字段逐项相等）加持久 golden 回归测试（tests/test_liberodataset_refactor.py）双重锁定；原返回字段不变。experiment 配置注册一行导入暂不需要：启动器的编程式组装本身无法省略，注册不省行数。
- 训练尽可能用现成轮子：训练循环、optimizer/梯度累积、checkpoint、回调全部沿用 CosmosPolicyTrainer/ImaginaireTrainer；LoRA 注入沿用 PEFT；不重写 trainer、不手写 LoRA Linear、不另建优化器。启动器只做 config 组装（config_v2 数据类编程式构建，绕开 experiment 注册表）后交回原 trainer。
- 渲染/采集完即释放，不常驻。

### 11.7 环境与依赖（2026-09-09 核查）

- 解释器：/data1/liu/miniconda3/envs/cosmospolicy/bin/python（scripts 内既有约定 conda activate cosmospolicy；不修改 conda 环境）。
- 关键依赖已全部就位，无缺失：torch 2.7.0+cu128、peft 0.19.1（inject_adapter_in_model 可用）、transformers 4.57.1、h5py 3.13.0、numpy 2.2.6、Pillow、imageio、draccus 0.11.6、omegaconf/hydra、einops、flash-attn 2.7.3、matplotlib、scipy；cosmos_policy 可导入。
- 规则：缺少库只报告、不自行安装；本次核查无缺失项。

### 11.6 本轮执行范围与顺序

> **2026-09-09 进度**:§11.6 第 1 条(CPU 交付物)已完成并全部验证(22/22 测试);GPU 验证(仅验证、未训练)已通过——真模型 LoRA 注入 280 层/11.5M 参数、adapter 开关等价、两遍法微批梯度仅在 LoRA、训练步等价峰值 7.51GB≤10GB。证据与未验证假设清单见 `memory_system/tta/results/VALIDATION_RECORD_20260909.md` 与 `results/validate_real_model_*.json`。第 2 条(数据采集与真模型训练步集成)未开始,等待采集授权。

1. 无 GPU 时（CPU）：完成全部训练侧代码与测试，全部位于 memory_system/tta/ 下：
   - `training/`：preference_dataset.py、dpo_loss.py、dpo_model.py、adapter_toggle.py、tta_dpo_config.py（含 pair_batch_size / chunks_per_pair 参数）、train_tta_dpo.py（薄启动器，组装 config 后交回 CosmosPolicyTrainer）。
   - `data/`：build_preference_pairs.py（manifest 与校验，支持只校验）、replay_rejected.py（复用 diagnose_failed.run_one_init 重放取图落盘，不修改原文件）、collect_repair_episodes.py（批量修复采集，断点续跑、按 task 分批；冒烟核对——前缀完整、obs[k] 对应 action[k] 执行前、无时间缝——已内建为采后自动步骤）。
   - `tests/`：test_dpo_loss.py（§8.3 全部断言）、test_preference_dataset.py（含同一 hdf5 上与 LIBERODataset 输出逐字段对齐）、test_adapter_toggle.py（小 CPU 网络）。
   对真模型编写但未执行的代码，逐处标注「未验证假设」（字段名/张量布局），形成冒烟核对清单。
2. 有 GPU 后按序：1 对冒烟 → rejected 全量重放取图 + chosen 8–16 对采集 → 真模型单步 forward/backward（§8.4，含显存/耗时测量）→ 确定 K 与 beta。
3. 预期首次真模型冒烟有 1–2 轮形状/字段名级小修，属正常成本，按核对清单逐项确认。

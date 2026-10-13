# Memory guided correction + SFT：现有文件结构与复现依赖

核对日期：2026-10-13。主机：xiaogang_249。

项目根目录：/data1/liu/exp/counterfactual/external/cosmos-policy。

## 当前实验的定义

当前链路是：原始 policy 测试 → 记录失败轨迹 → 诊断失败阶段与 t* → 重放失败前缀 → Memory 引导机器人修正 → 原始 policy 继续执行 → 保存完整成功轨迹 → 同任务 P95 长度筛选 → SFT → 导出模型 → 关闭 Memory 进行 rollout 评测。

Memory 在这套实验中提供 SFT 训练轨迹。现有 SFT 评测测试的是微调后 policy 本身；SFT 后在线继续开启 Memory 是另一种实验设置，不能把现有结果当成该组合的结果。

现有数据：2519 个 baseline case，2084 成功、435 失败；对 435 个失败 case 做 Memory correction，229 修复成功；其中 190 条完整成功轨迹通过 P95 筛选，39 条成功但过长的轨迹被排除。单扰动训练目前只使用原始失败 case 的修复轨迹。

## 一、按实际路径保留文件

下面是现有结构；<job>、<data_dir> 和 <group> 是占位符，不是实际目录名称。准确路径见 memory_guided_correction_sft_inventory.json。

```text
/data1/liu/exp/counterfactual/
├── external/
│   ├── LIBERO-plus/
│   │   └── libero/libero/
│   │       ├── benchmark/              # task 注册、task_classification.json
│   │       ├── bddl_files/             # 每个 case 的任务定义
│   │       ├── init_files/             # 初始状态
│   │       └── assets/                 # 物体、纹理、机器人与场景资源
│   └── cosmos-policy/
│       ├── cosmos_policy/             # policy、dataset、config、trainer 源码
│       ├── memory_system/             # Memory 构建、检索、控制、诊断源码
│       ├── scripts/                   # rollout、correction、SFT、导出入口及依赖
│       ├── tests/tta/archived_lorafix/train_tta_sft_fixed.py
│       ├── configs/libero10_experiment_tasks.json
│       ├── skill_memory_test/libero_10/
│       │   ├── segments_ready_fixed16.json
│       │   ├── feasible_recovery_targets.pt
│       │   └── ready3d_targets.pt
│       ├── LIBERO-Cosmos-Policy/success_only/
│       │   ├── libero_10_regen/*_demo.hdf5
│       │   ├── dataset_statistics.json
│       │   └── dataset_statistics_post_norm.json
│       ├── experiments/
│       │   ├── cache/libero_plus_language_t5_libero10.pkl
│       │   ├── libero_plus/task_lists/libero_10/  # index.json + 7 类 task 清单
│       │   ├── libero_plus/cosmospolicy_full_7dims/  # baseline
│       │   ├── tta/repair_libero_plus_435/
│       │   │   ├── inputs/failed_435.json
│       │   │   ├── smoke_4/                       # 首批 4 个修复 case
│       │   │   ├── remaining_431/                 # 其余 431 个修复 case
│       │   │   └── success_task_p95/              # 190 条筛选清单及汇总
│       │   └── libero_plus/sft190_p95/<job>/epoch_XX/  # SFT rollout
│       ├── training/
│       │   ├── tta_sft_metadata/                  # 共用标准化/文本资源链接
│       │   ├── <data_dir>/                        # 每次 SFT 的训练数据
│       │   └── <job>_iter<N>_full.pt               # 导出的全参数 policy
│       └── scripts/experiments/libero_plus_sft190/
│           ├── _plans/<job>/                     # 计划、清单、代码快照、局部配置
│           └── <job>/                            # 超参数、日志、审计、评测汇总
└── checkpoints/
    ├── Cosmos-Policy-LIBERO-Predict2-2B/
    │   ├── Cosmos-Policy-LIBERO-Predict2-2B.pt
    │   ├── libero_dataset_statistics.json
    │   ├── libero_t5_embeddings.pkl
    │   └── dataset_statistics_post_norm.json
    ├── Cosmos-Predict2-2B-Video2World/
    │   ├── model-480p-16fps.pt
    │   └── tokenizer/tokenizer.pth
    ├── huggingface-hub/                         # 模型 URI 对应的缓存/链接
    └── cosmos_policy/cosmos_v2_finetune/<job>/checkpoints/
        └── iter_000xxxxxx/{model,optim,scheduler,trainer}/
```

运行环境在 /data1/liu/miniconda3/envs/cosmospolicy/，不在项目目录内。memory_guided_correction_sft_package_versions.txt 记录本次核对时的已安装包版本；这是一份版本快照，不能单独代替完整的环境安装说明或 CUDA/cuRobo 资源。

## 二、每个阶段需要什么

| 阶段 | 现有主要入口 | 必需输入 | 主要输出 |
|---|---|---|---|
| Baseline rollout | scripts/run_libero_plus_batch.py | LIBERO-plus、固定 task 清单、初始状态、原始 policy、标准化参数、T5 缓存 | run_config.json、逐 case result.json、episode HDF5、actions.npy、视频 |
| 失败 case 清单 | scripts/collect_libero_plus_failures.py | baseline result 与协议、task 身份、原始 HDF5 路径 | failed_435.json 与生成记录 |
| Memory correction | scripts/run_libero_plus_repair.py | failed_435.json、baseline run_config 与失败 HDF5、Memory 库、原始 policy | diagnosis、memory_result.json、result.json、完整修复 HDF5、actions、视频 |
| P95 成功轨迹筛选 | scripts/select_libero_plus_success_p95.py | 修复结果、同 base task 的 regenerated 成功 demo 长度 | selected/excluded manifest、summary、CSV、各扰动训练/独立/Memory 失败清单 |
| SFT 数据准备与训练 | scripts/train_libero_plus_sft190.py | 选中 manifest、原始修复 HDF5、failed_435.json、baseline 配置、标准化参数、T5 缓存、基础 policy | 数据视图、manifest/preflight、训练日志、分布式 checkpoint |
| 训练到评测调度 | scripts/run_libero_plus_sft190_pipeline.py 或对应实验副本 | 数据清单、GPU/epoch/训练模式、固定 evaluation_cases | smoke 验证、训练、导出、epoch 评测和汇总 |
| 模型导出与评测 | scripts/export_full_finetune_ckpt.py + scripts/run_libero_plus_batch.py | DCP model、共同模型资产、原始 case/seed 协议 | 单文件 policy .pt、逐 case rollout、成功率报告 |

已补充 scripts/collect_libero_plus_failures.py 与 scripts/select_libero_plus_success_p95.py。两个入口均支持自定义输入/输出、--dry-run 与明确覆盖选项；运行说明见 scripts/README_libero_plus_manifests.md。现有 435/190/39 清单已独立重建并与历史文件逐字节比较通过。

run_libero_plus_repair.py 默认只选 4 个 smoke case。跑全清单需要 --all-cases；也支持指定 task ID 与恢复已完成结果。

scripts/tta_repair_batch.py 属于旧的另一套实验入口，当前 LIBERO-plus 435-case correction 使用 run_libero_plus_repair.py。

## 三、Memory 库的组成

| 文件/目录 | 作用 | 保留范围 |
|---|---|---|
| skill_memory_test/libero_10/segments_ready_fixed16.json | 成功 demo 的阶段、顺序与 ready frame 定义 | 当前运行必需 |
| skill_memory_test/libero_10/feasible_recovery_targets.pt | 可行恢复目标、ready VAE 与 EE 状态等检索数据 | 当前运行必需 |
| skill_memory_test/libero_10/ready3d_targets.pt | 目标物体 3D 位置到 ready pose 的匹配原型 | 为复现当前 3D 选择行为必须保留；loader 本身允许不加载 |
| LIBERO-Cosmos-Policy/success_only/libero_10_regen/ | Memory 构建所用成功 demo，同时提供同任务 P95 长度依据 | 重建 Memory、重算 P95 时必需 |
| memory_system/offline/ | 分段、目标提取、recovery/ready3d 构建及后续 ready 修订 | 重建与理解 Memory 时保留 |

核心代码还包括 memory_system/tta/phase_record.py、object_query.py、execute/initial_alignment.py、execute/plan.py、execute/vla_skill_runtime.py，以及 recovery、skill_completion、规划器和几何辅助模块。建议保留整个 memory_system/。

现有构建函数包括 memory_system.offline.build_recovery.build_recovery、memory_system.offline.build_targets.build_phase_targets；Ready3D 有 scripts/build_ready3d_memory.py 入口。重建与原实验完全一致的 Memory 还需要相同 demo、分段 manifest 和 ready 修订参数；已生成的三个核心文件应作为固定版本归档。

当前主流程的默认 selector 使用上面三个文件。memory_system/pointcloud_action/ 以及 LIBERO-PRO 的 OracleReadySelector 是另外的分支；不能未经记录切换 COSMOS_SKILL_READY_MEMORY 后仍当成相同实验。

## 四、一个 case 与一个训练目录

Baseline 的 case 目录形如 experiments/libero_plus/cosmospolicy_full_7dims/<group>/episodes/taskXXXX/。

Memory correction 的 case 目录形如 experiments/tta/repair_libero_plus_435/{smoke_4,remaining_431}/<group>/taskXXXX/：

```text
taskXXXX/
├── diagnosis/
│   ├── phase_record.json            # 失败阶段与触发位置
│   ├── repair_request.json          # 修复请求
│   ├── replay_verification.json     # 初始状态/前缀重放核对
│   └── tstar.png
├── memory_result.json              # Memory 选择与执行记录
├── result.json                     # 成功与否、case 身份、轨迹路径
├── actions.npy
├── video.mp4
├── rollout_data/episode*.hdf5       # 完整轨迹，而非只存修复片段
└── eval.log
```

原始失败 HDF5 的 initial_state、actions 和 proprio 用于重放与验证。修复成功 HDF5 保存失败前缀、Memory 干预与 policy continuation，SFT 使用这条完整轨迹。当前协议使用总共 700 个动作预算，包含前缀与干预；30 个 settling 动作不计入保存轨迹长度。

训练目录的结构：

```text
training/<data_dir>/
├── manifest.json                   # case、来源、训练 HDF5、精确 instruction
├── preflight.json                  # 数量、形状、长度、清单 hash 等检查结果
├── metadata/
│   ├── dataset_statistics.json      # 基础 policy 的 action/proprio 标准化
│   ├── dataset_statistics_post_norm.json
│   └── t5_embeddings.pkl           # 该训练集精确 instruction 的 embeddings
└── rollouts/
    └── episode_data--taskXXXX--suite=libero_10--success=True--p95.hdf5
```

HDF5 中关键字段是 actions[T,7]、proprio[T,9]、primary_images_jpeg[T]、wrist_images_jpeg[T]；其他字段如 initial_state 会继承原始保存结果。success 与 task_description 等属性随文件保存。

这些训练 HDF5 的数据集使用 h5py.ExternalLink，链接目标是原始修复 HDF5 的绝对路径。只拷贝 training/<data_dir> 不能得到独立训练数据。success_task_p95/episodes/ 中的 episode.hdf5 也存在指向原修复文件的软链接。

metadata 采用原 policy 的标准化参数；不是重新计算小规模修复集的均值方差。精确 instruction 与 T5 embedding 保持一致；Language 的实际改写指令保留，不能全部替换成 clean instruction。

training/tta_sft_metadata/dataset_statistics_post_norm.json 当前是软链接，实际目标是 LIBERO-Cosmos-Policy/success_only/dataset_statistics_post_norm.json。其 t5_embeddings.pkl 链接到基础 policy 的文本缓存。

## 五、扰动与数据划分

| Perturbation | 原始失败 case / 当前 SFT 全部评测 case | P95 后训练 case | 未参与训练 case | 其中 Memory 修复失败 case | 单独 8-epoch 全参数实验 |
|---|---:|---:|---:|---:|---|
| Robot Initial | 119 | 90 | 29 | 23 | 已完成 |
| Camera | 167 | 24 | 143 | 116 | 已完成 |
| Background | 70 | 29 | 41 | 40 | 已完成 |
| Sensor Noise | 27 | 22 | 5 | 4 | 已完成 |
| Language | 15 | 8 | 7 | 5 | 尚未单独进行 |
| Light | 16 | 14 | 2 | 2 | 已完成 |
| Objects | 21 | 3 | 18 | 16 | 尚未单独进行 |
| Total | 435 | 190 | 245 | 206 | 5 类已完成 |

训练集由 Memory 修复成功且通过 P95 的 case 决定，并非按比例随机划分。未参与训练的集合 = Memory 修复失败 case + 修复成功但被 P95 排除的 case。Memory 失败集是未参与训练集合的子集。

训练集成功率指微调模型重新 rollout 这些 case 后的成功比例；不能从训练 HDF5 的 success=True 推出评测成功率为 100%。独立测试是 case 与训练清单不重叠，base task 仍可能相同。

全参数 job 对应关系：

| Perturbation | job |
|---|---|
| Robot Initial | libero_plus_robotinit90_fullft_p95_epoch8_20261010_011329 |
| Camera | libero_plus_camera24_fullft_p95_epoch8_20261010_192440 |
| Sensor Noise | libero_plus_sensor_noise22_fullft_p95_epoch8_20261012_145307 |
| Background | libero_plus_background29_fullft_p95_epoch8_20261012_220444 |
| Light | libero_plus_light14_fullft_p95_epoch8_20261013_103633 |

实际 data_directory、checkpoint_directory、导出文件和 evaluation_cases.json 的完整路径均保存在 memory_guided_correction_sft_inventory.json。

## 六、代码与配置不能只保留主脚本

除整个 cosmos_policy/ 和 memory_system/ 外，当前 SFT 还会直接使用：

- tests/tta/archived_lorafix/train_tta_sft_fixed.py：动态导入 mutate_config。即使这次不用 LoRA，也需要此配置逻辑。
- scripts/train_lrscene5_sft_only.py：提供训练 callback SFTMetrics。
- scripts/run_lrscene5_sft_only_pipeline.py：提供 checkpoint/导出验证函数。
- cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py：注册 tta_repair_sft；通用默认配置包含 LoRA，训练入口的 --full-finetune 会覆盖为全参数模式。
- scripts/run_libero_smoke_test.sh、run_libero_smoke_test.py 与 cosmos_policy/experiments/robot/：rollout 与模型/环境接入。

要精确重现旧实验，还应保留 _plans/<job>/ 下的清单和代码快照。当前 Background 的完整失败集评测由 run_background_fullft_pipeline.py 提供，Light 由 run_light_fullft_pipeline.py 提供；这些文件位于各自 _plans/<job>/ 中，公共 pipeline 目前没有这两个相同 scope。

Sensor Noise 的评测恢复代码在其 _plans/<job>/evaluation_recovery_20261012_201542/resume_completed_training_evaluation.py。

LIBERO_CONFIG_PATH 和 COSMOS_LIBERO_ROOT 需要一致地指向 LIBERO-plus。Background、Light 的局部配置在 _plans/<job>/libero_config/config.yaml；Sensor Noise 恢复评测也使用其恢复目录内的局部配置。全局 ~/.libero/config.yaml 可能被其他实验更改，归档时应保存本实验的局部配置。

本次核对发现 cosmos-policy 与 LIBERO-plus 均有本地源码修改，且多个实验入口不一定属于原始 commit。只保存 Git commit 或重新 clone 上游不能复现当前行为；需保存实际运行源码、代码快照与 diff。

## 七、checkpoint 与实验记录

```text
checkpoints/cosmos_policy/cosmos_v2_finetune/<job>/checkpoints/iter_000xxxxxx/
├── model/       # .metadata + __0_0.distcp ... __7_0.distcp
├── optim/       # optimizer 状态
├── scheduler/   # 学习率调度状态
└── trainer/     # iteration 等训练状态
```

继续训练应保留这四类状态；仅导出 policy 推理权重时需要 model/。导出的 training/<job>_iter<N>_full.pt 用于现有全参数推理流程；LoRA 实验则使用 scripts/merge_lora_ckpt.py 合并，必须同时记录 rank、alpha 和对应基础模型。

现有 rollout wrapper 检查 Video2World 的 model-480p-16fps.pt 与 tokenizer/tokenizer.pth；因此即使已有微调 policy .pt，按现有入口运行也需要这些共同资产。

每个 scripts/experiments/libero_plus_sft190/<job>/ 应保留 run.json、evaluation_cases.json、pipeline_status.json、completed.json、full_parameter_audit.json、train_metrics.jsonl、fixed_probe_metrics.jsonl、epoch_reports/、evaluation_curve.csv/json 与主要日志。评测逐 case 数据位于 experiments/libero_plus/sft190_p95/<job>/epoch_XX/。

run.json 保存实际超参数与训练模式；本轮五类使用从同一原始 policy 重新初始化的全参数 policy backbone SFT、8 epochs、8 GPUs、每卡 batch 1。仅保存配置文件中的默认值会漏掉训练入口覆盖后的设置。

## 八、整理为通用流程时的建议

现阶段优先保留上述实际路径，以免绝对链接断开。若后续统一入口，可以将配置和数据职责整理为下面的形式；以下是建议结构，目前尚未迁移或创建这些新入口：

```text
configs/memory_sft/
├── paths.yaml                     # repo、LIBERO、模型、数据与输出根目录
├── protocol.yaml                  # case 清单、seeds、700-step budget、Memory 模式
├── selection.yaml                 # success + same-task P95 规则
└── sft.yaml                       # full/LoRA、LR、epoch、batch、GPU 等
scripts/memory_sft/
└── run_pipeline.py                # 复用已有两个清单脚本，统一七类训练与评测
data/memory_sft/
├── task_lists/
├── memory/
├── baseline/
├── corrections/
└── selections/<group>/
    ├── train_manifest.json
    ├── heldout_manifest.json
    └── memory_failed_manifest.json
runs/memory_sft/<group>/<job>/      # 配置快照、训练/评测日志、汇总、权重索引
```

迁移时应统一处理 manifest 里的绝对路径、HDF5 ExternalLink、软链接及脚本固定路径。可以保留原始轨迹并重建链接，或导出为自包含 HDF5；简单重命名目录会破坏读取。

Baseline 的全部 2519 个 result.json 与 run_config.json 应保留：现有 pipeline 的完整性检查会读取全基线。若只重新执行 Memory 修复，实际重放输入是 435 个失败 HDF5。原本成功的轨迹不属于本轮单扰动 SFT 训练集。

smoke_4 与 remaining_431 的 case 文件应同时保留；只保存 remaining_431 会漏掉首批 4 个 case。现有 resume 检查还要求已完成修复 case 的视频、actions 与 HDF5 存在，因此视频不能在不改恢复检查逻辑时一概删除。

## 九、本次核对范围

- 2519 个 baseline result 均为 completed；435-case 失败清单引用的 baseline HDF5、actions 和 result 文件均存在，source_config hash 与原 run_config 一致。
- smoke_4 + remaining_431 共 435 个唯一修复 case，其中 229 成功。
- 190 条选中修复轨迹的原始 HDF5、actions、repair result、视频存在，HDF5 success/长度与选中清单一致，actions/proprio 形状正确。
- 五个已完成全参数实验共 179 个训练视图均能解析 actions 外链；90/24/22/29/14 的训练数量与 run.json 相符。
- 五个 epoch-8 DCP 的 model、optim、scheduler、trainer 均有 .metadata；epoch 1/2/4/8 的导出 policy 文件均存在。
- memory_guided_correction_sft_inventory.json 记录关键文件 hash、路径、软链接目标、包版本、实际 Git 状态和实验目录。大型权重只记录存在性与大小，本次没有重新计算全部权重 hash，也没有重新训练或 rollout。

- 两个新增清单入口通过 12 个 CPU 回归测试；重建验证输出在 experiments/tta/repair_libero_plus_435/manifest_tools_validation_20261013/，未覆盖历史实验文件。

已新增清单脚本与运行说明，未移动或清理实验数据。

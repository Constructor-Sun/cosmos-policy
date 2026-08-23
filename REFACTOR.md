# Memory System 代码迁移与 3D 扩展方案

## 1. 文档目的

当前 `bin/memory` 主要负责离线 Memory 构建，`bin/execute` 负责在线读取 Memory、执行 verifier、判断 skill completion 并触发 recovery。这些文件已经是会被 eval 导入的运行库，不再只是命令行脚本。

本文档定义：

1. 如何在不改变当前 2D 行为的前提下迁移代码；
2. 目标目录结构及每个新文件的职责；
3. 当前文件和逻辑应如何映射到新结构；
4. Phase、Feasible、Skill Completion 和 Recovery 的边界；
5. 未来加入 Main RGB-D/3D 后的代码与 Memory 组织方式。

本迁移应分为“结构迁移”和“3D 规则升级”两个可独立验证的阶段，不应在同一次改动中同时改变 import、Memory format、坐标系和判断规则。

## 2. 核心概念

`memory_system` 代表广义的 Memory 系统：

```text
Offline construction
    -> 从 task/demo 构建 Memory artifact
    -> skill_memory/libero_10/*.pt / *.json
    -> Execute 在线读取并使用 Memory
```

其中：

- `offline` 回答“Memory 如何建立”；
- `execute` 回答“Memory 如何在线使用”；
- `phase` 判断当前是否在执行正确 skill，以及是否向当前目标前进；
- `feasible` 判断是否进入当前 skill 的 ready/可执行状态；
- `skill_completion` 判断 Pick、Place、Open/Close 等 skill 的语义结果是否成立；
- `execution_monitor` 只编排 `Phase -> Feasible -> Skill Completion` 和 recovery 后的状态恢复；
- `recovery` 负责“恢复到哪里”和“如何移动过去”，不重复实现 Phase/Feasible 的错误判断。

## 3. 第一阶段：保持当前 2D 行为的目标结构

```text
cosmos_policy/
└── memory_system/
    ├── __init__.py
    ├── types.py
    ├── artifacts.py
    ├── geometry.py
    ├── skills.py
    │
    ├── offline/
    │   ├── __init__.py
    │   ├── planner.py
    │   ├── predicate_planner.py
    │   ├── label_segments.py
    │   ├── label_boundaries.py
    │   ├── build_targets.py
    │   └── build_recovery.py
    │
    └── execute/
        ├── __init__.py
        ├── plan.py
        ├── phase.py
        ├── feasible.py
        ├── execution_monitor.py
        │
        ├── skill_completion/
        │   ├── __init__.py
        │   ├── pick.py
        │   ├── place.py
        │   └── open_close.py
        │
        └── recovery/
            ├── __init__.py
            ├── retrieval.py
            ├── selectors.py
            └── controller.py
```

第一阶段不包含 `rgbd.py`，Offline 不渲染 depth，Phase/Feasible 仍使用当前 2D 规则。

### 3.1 为什么 Offline 暂时保持扁平

当前 Offline 代码只有 planner、两类 labeler 和两类 builder，没有必要立即建立 `planners/labelers/builders/` 三层子目录。先保持 6 个职责清晰的文件，当某一类稳定增长到 3 个以上独立实现时再建目录。

### 3.2 顶层共享文件

#### `types.py`

保存 Offline 和 Execute 都会使用的类型，例如：

- `MemoryKey`：task、planner step、skill、arguments、demo id 的统一 key；
- `SkillStep` / `SkillPlan`；
- `VerifierObservation`；
- `TargetGeometry`；
- Phase、Feasible、Completion 结果 dataclass；
- recovery request/target/result 的通用数据类型。

第一阶段的 `TargetGeometry` 可只有 `center_xy` 和 `bbox_xyxy`，但应预留 optional XYZ 字段，便于未来兼容 3D：

```python
@dataclass
class TargetGeometry:
    center_xy: np.ndarray | None
    bbox_xyxy: np.ndarray | None
    center_xyz_world: np.ndarray | None = None
    points_xyz_world: np.ndarray | None = None
    depth_confidence: float = 0.0
```

#### `artifacts.py`

集中处理：

- `phase_targets.pt`、wrist targets、recovery targets 的加载/保存；
- `format` / version 检查；
- v1/v2 兼容和迁移；
- 统一的 artifact path 解析。

不需要额外创建 `schemas/` 或 `stores/` 目录。

#### `geometry.py`

第一阶段只保存当前已有的通用几何：

- EEF world pose 和 rotation/quaternion 处理；
- world point 投影到 image pixel；
- 2D point/bbox/distance 计算；
- 6D pose delta 和刚体位姿工具。

Action scale 和控制收敛逻辑属于 `execute/recovery/controller.py`，不属于通用 geometry。

#### `skills.py`

保存 Offline 和 Execute 共享的 skill 声明性语义，不实现 completion 细节。例如：

```python
SKILLS = {
    "Pick": {"target_argument": "item"},
    "PlaceOn": {"target_argument": "target"},
    "PlaceIn": {"target_argument": "target"},
    "Open": {"target_argument": "target"},
    "Close": {"target_argument": "target"},
}
```

这可替代当前在 builder、monitor 和 completion 文件中重复的 `PICK_SKILLS`、`PLACE_SKILLS` 和 `approach_argument()` 分支。

## 4. Offline 文件职责与当前文件映射

### 4.1 `offline/planner.py`

来源：

- `bin/memory/libero_skill_skeleton.py`

迁移逻辑：

- 从 BDDL goal 构建 state-free `SkillPlan`；
- `SkillStep` 的 skill、arguments、depends_on、execution mode；
- open/close/turn/pick/place 的名义规划；
- 使用根目录 `skills.py` 读取 skill 定义。

### 4.2 `offline/predicate_planner.py`

来源：

- `bin/memory/libero_predicate_planner.py`

迁移逻辑：

- BDDL initial state/goal 建模；
- Unified Planning/Fast Downward action definition；
- symbolic plan 到 `SkillPlan` 的转换。

State-free planner 与 predicate planner 是两个可选计划策略，不应各自定义不兼容的 step 类型。

### 4.3 `offline/label_segments.py`

来源：

- `bin/memory/label_libero_skill_segments.py`

迁移逻辑：

- plan step 与 demo 轨迹对齐；
- skill start/end 标注；
- Pick、Place、Open/Close 的 simulator predicate 成功区间；
- `already_satisfied` 等 segment 状态。

### 4.4 `offline/label_boundaries.py`

来源：

- `bin/memory/label_libero_skill_ready_boundaries.py`

迁移逻辑：

- 在 segment 内标注 `ready_frame`；
- 保留/补充 success start/end；
- 为 Feasible 和 Recovery builder 提供统一边界。

Labeler 应逐步丰富同一份 manifest record，不要为每个事件建立彼此不兼容的 manifest 格式。

### 4.5 `offline/build_targets.py`

主要来源：

- `bin/memory/build_libero_phase_targets.py`

第一阶段迁移的逻辑：

- 恢复 MuJoCo demo state；
- 渲染 main/wrist segmentation；
- 读取 demo main/wrist RGB；
- 构建 phase RGB crop、mask、bbox、target center XY；
- 投影 EEF 得到 gripper XY；
- 构建 wrist completion/feasible templates；
- 保存当前 v1 artifact，保持输出行为不变。

第一阶段不开启 `camera_depths`，不保存 main/wrist depth。

如果未来 target builder 稳定增长为 Phase、Feasible、Completion 三个大型独立 builder，再拆分 `offline/builders/`；不在第一次迁移时提前拆分。

### 4.6 `offline/build_recovery.py`

主要来源：

- `bin/memory/build_vector_db_from_demos.py`

迁移逻辑：

- phase recovery frame 的 main VAE/EE pose；
- feasible ready frame 的 main/wrist VAE/EE pose；
- action scale 或恢复控制所需校准信息；
- recovery/feasible-recovery artifact 构建。

## 5. Execute 文件职责与当前文件映射

### 5.1 `execute/plan.py`

主要来源：

- `bin/execute/libero_phase_monitor.py` 中的 `PhaseSpec`、`load_phase_plans()` 和 plan cursor 相关逻辑。

命名为 `plan.py` 而不是 `phase_plan.py`，因为它描述的是整个 skill 序列。同一个 `SkillStep` 会先后经过 Phase、Feasible 和 Skill Completion，并不只属于 Phase。

当前 `LiberoPhaseMonitor` 的 current/next comparative monitor 如果仍有明确用途，可作为独立实验策略保留；不应仅为复用 plan loader 而与主 Execution Monitor 并存两套状态机。

### 5.2 `execute/phase.py`

主要来源：

- `bin/execute/libero_phase_verifier.py`

迁移逻辑：

- Phase target memory 选择；
- masked RGB template preparation/matching；
- 跨 demo 投票和 target cluster；
- 当前 2D gripper-target progress；
- `PHASE_OK / PHASE_ERROR / PHASE_UNKNOWN`；
- Phase-specific 阈值、时序累计和触发条件。

如果未来需要增加其他 matcher，应在同一 Phase 接口下作为 matcher strategy 添加，不再维护完整 verifier 副本。

### 5.3 `execute/feasible.py`

主要来源：

- `bin/execute/libero_feasible_region_verifier.py`

迁移逻辑：

- ready prototype 加载/选择；
- 2D normalized ready distance；
- demo votes；
- stagnation/reversal 时序证据；
- feasible latch；
- optional wrist-ready 证据；
- `FEASIBLE / NOT_FEASIBLE / FEASIBLE_UNKNOWN`；
- Feasible-specific 阈值和触发条件。

目录/文件使用 `feasible`，不使用 `feasible_region`。`region` 是当前实现方式，未来可以替换为 3D pose、learned feasibility 或接触约束。

Feasible 不需要自己的 plan。它消费 `plan.py` 提供的当前 `SkillStep`，并为该 step 查询 ready/feasible prototype。

### 5.4 `execute/skill_completion/`

主要来源：

- `bin/execute/libero_skill_completion_verifier_geo.py`

原 `bin/execute/libero_skill_completion_verifier.py` 的 VAE/DTW completion 已被替代，不迁入新结构。

拆分原则：

- `pick.py`：Pick close edge、ready anchor、EEF 移动、wrist flow/RANSAC、gripper gap、wrong-grasp 检测；
- `place.py`：PlaceOn/PlaceIn release transition、open state、target gate；
- `open_close.py`：Open/Close/Turn 的几何/状态规则；
- `__init__.py`：保存最小的 skill-to-completion-rule registry。

Skill Completion 不归入一个广义 `verifier/` 目录。Phase 和 Feasible 是通用阶段判断，Completion 是强 skill-specific 语义。

`libero_skill_completion_verifier_geo copy.py`、`.bak_closedloop` 和已被明确替代的实验副本不应迁入新 runtime package。确认历史用途后通过 Git 历史保留，不再把 copy/bak 当作源码版本管理。

### 5.5 `execute/execution_monitor.py`

来源：

- `bin/execute/libero_execution_monitor.py`

迁移后只负责：

```text
PHASE_CHECK
    -> FEASIBLE_CHECK
    -> COMPLETION_CHECK
    -> next SkillStep / PLAN_COMPLETE
```

它还负责：

- 将 Phase/Feasible/Completion 结果映射为 intervention request；
- recovery 后回到 Phase、Feasible 或 Completion；
- timeout、phase advance 和 plan complete；
- 统一调用 skill completion rule。

不应再在 monitor 中硬编码 `PLACE_SKILLS` 等 skill-specific 判断；类似“Place 允许在 Feasible 前观察 release”的能力由 `skills.py`/completion rule 声明，monitor 只读取声明。

## 6. Recovery 的拆分方式

Recovery 应按以下三层拆分：

```text
何时需要恢复
    -> phase.py / feasible.py / skill_completion/*.py

恢复到哪里
    -> recovery/selectors.py

怎样移动过去
    -> recovery/controller.py
```

### 6.1 `execute/recovery/retrieval.py`

主要来源：

- `libero_pose_recovery.py` 和 `libero_feasible_recovery.py` 中重复的基础逻辑。

抽取：

- arguments/MemoryKey 匹配；
- per-demo best candidate；
- token normalization 和 cosine similarity；
- EE pose position/rotation cluster；
- position/quaternion 平均；
- 投票数和相似度阈值。

### 6.2 `execute/recovery/selectors.py`

保留 Phase 和 Feasible 的不同策略：

| 项目 | Phase selector | Feasible selector |
|---|---|---|
| 触发来源 | `PHASE_ERROR` | `NOT_FEASIBLE` / wrong-grasp retry |
| Memory frame | recovery frame | ready frame |
| embedding | main | main + wrist |
| nearest-pose fallback | 默认无 | 支持 |
| controller z-lift | `0` | 默认 `0.02` |
| 恢复后 | 重新 Phase check | Feasible check 或 Completion |

Selector 只返回 recovery target/config，不直接运行低层 action。

### 6.3 `execute/recovery/controller.py`

主要来源：

- `bin/execute/closed_loop.py`
- 两个 recovery 文件中生成 correction action 的共享部分。

负责：

- 当前 EE pose 到目标 EE pose 的闭环靠近；
- optional z-lift；
- action scale/夹紧限制；
- 收敛、step budget 和 controller result。

Phase 和 Feasible 共享同一个“如何靠近目标 pose”控制器，不复制两套轨迹生成。

Wrong-grasp 由 `skill_completion/pick.py` 检测，但恢复目标是 ready pose，因此由 monitor 请求 Feasible selector，打开夹爪后回到 Feasible check。

## 7. 当前文件到目标文件的总映射

| 当前文件 | 目标文件 | 备注 |
|---|---|---|
| `bin/memory/libero_skill_skeleton.py` | `memory_system/offline/planner.py` | state-free SkillPlan |
| `bin/memory/libero_predicate_planner.py` | `memory_system/offline/predicate_planner.py` | predicate planner |
| `bin/memory/label_libero_skill_segments.py` | `memory_system/offline/label_segments.py` | skill segment |
| `bin/memory/label_libero_skill_ready_boundaries.py` | `memory_system/offline/label_boundaries.py` | ready/success boundary |
| `bin/memory/build_libero_phase_targets.py` | `memory_system/offline/build_targets.py` | Phase + wrist target builder；初期保持输出 |
| `bin/memory/build_vector_db_from_demos.py` | `memory_system/offline/build_recovery.py` | recovery/ready pose memory |
| `bin/execute/libero_phase_monitor.py` | `memory_system/execute/plan.py` | 主要迁移 plan types/loader |
| `bin/execute/libero_phase_verifier.py` | `memory_system/execute/phase.py` | default matcher + phase rules |
| `bin/execute/libero_feasible_region_verifier.py` | `memory_system/execute/feasible.py` | 去掉 region 命名 |
| `bin/execute/libero_skill_completion_verifier_geo.py` | `execute/skill_completion/{pick,place,open_close}.py` | 按 skill 拆分 |
| `bin/execute/libero_skill_completion_verifier.py` | 不迁移 | VAE/DTW completion 已被 geo 版本替代 |
| `bin/execute/libero_execution_monitor.py` | `memory_system/execute/execution_monitor.py` | 只保留编排和状态转移 |
| `bin/execute/libero_pose_recovery.py` | `recovery/retrieval.py` + `selectors.py` | Phase selector |
| `bin/execute/libero_feasible_recovery.py` | `recovery/retrieval.py` + `selectors.py` | Feasible selector |
| `bin/execute/closed_loop.py` | `recovery/controller.py` | 共享低层闭环控制 |
| `bin/execute/*copy.py`, `*.bak_closedloop` | 不迁移 | 由 Git 历史保留 |

`bin/detect_intervention_point.py` 是已与正式 Eval 解耦的 legacy/offline 分析工具。其 stagnation、double-empty-grasp 和 early-manifold 逻辑不迁入 Memory System；Phase、Feasible、Skill Completion 及其 recovery 是正式在线路径。

## 8. 迁移顺序

### Step 0：冻结 2D baseline

- 记录当前 eval 命令、任务、seed 和成功率；
- 保存代表性 Phase/Feasible/Completion/Recovery 日志；
- 确认现有 verifier/monitor 单测可通过；
- 增加当前缺失的 Phase matcher、artifact loader 和 recovery selector 特征测试。

### Step 1：创建 package 和共享类型

- 创建 `cosmos_policy/memory_system`；
- 先迁移 `types.py`、`artifacts.py`、`skills.py`；
- 将重复 `_arguments_key` / phase key 收敛为一个 `MemoryKey`；
- 此时不更改 `.pt` format。

### Step 2：迁移 Offline，保持 artifact 输出

- 一个文件一个文件地迁移 planner、labeler、builder；
- 使用相同输入比较迁移前后 manifest 和 `.pt` 内容；
- 保持 v1 字段、任务数、template 数和 failure/warning 统计。

### Step 3：迁移 Execute，保持 2D 规则

- 迁移 plan、phase、feasible 和 monitor；
- 将 completion 拆成 Pick/Place/OpenClose，但保持现有阈值与时序逻辑；
- 抽取 recovery retrieval/selector/controller，保持当前 Phase/Feasible 不同配置；
- 不在此阶段开启 depth。

### Step 4：更新 Eval 接线

- `run_libero_eval.py` 直接从 `cosmos_policy.memory_system` 导入；
- 移除为 `bin` 添加 `sys.path` 的 Memory System 依赖；
- 将零散参数逐步收敛到 `VerifierObservation`；
- 保持现有 JSON log/payload 字段兼容。

### Step 5：兼容入口与清理

- 迁移期间可在旧 `bin/memory`/`bin/execute` 保留薄 wrapper；
- 所有正式调用转到新 package 后再删除 wrapper；
- copy/bak 不迁移；
- 确认 baseline 不变后再开始 3D 工作。

## 9. 第一阶段 2D Memory 如何构建

```text
BDDL goal
    -> planner.py / predicate_planner.py
    -> SkillPlan
    -> label_segments.py
    -> start/end/success interval
    -> label_boundaries.py
    -> ready_frame
    -> build_targets.py
    -> RGB crop/mask/bbox/target XY/gripper XY/wrist targets
    -> build_recovery.py
    -> VAE token + recovery/ready EE 6D pose
```

第一阶段的 `build_targets.py` 只恢复 simulator state 和渲染 segmentation，不开启 Main/Wrist depth。

需要注意：这里的“2D”是指目标定位、Phase 距离和 Feasible 使用 2D image geometry。当前 Pick completion 已经使用 EEF 3D 位移，Recovery 也已使用 EE 6D pose，不应为了“纯 2D”而移除这些已有 proprioception。

## 10. 未来加入 Main RGB-D/3D

### 10.1 第一步：只在 Execute 引入 Main RGB-D

初期不改变 Offline，在 Execute 中新增：

```text
cosmos_policy/
└── memory_system/
    ├── types.py
    ├── artifacts.py
    ├── geometry.py
    ├── skills.py
    ├── offline/                 # 仍是 2D Memory construction
    └── execute/
        ├── rgbd.py                 # 新增；仅 Main RGB-D
        ├── plan.py
        ├── phase.py                # 增加 3D/hybrid progress
        ├── feasible.py             # 初期仍可保持 2D
        ├── execution_monitor.py
        ├── skill_completion/
        │   ├── pick.py             # 增加 3D lift/follow
        │   ├── place.py
        │   └── open_close.py
        └── recovery/
            ├── retrieval.py
            ├── selectors.py
            └── controller.py
```

LIBERO 环境开启：

```python
camera_names=["agentview", "robot0_eye_in_hand"]
camera_depths=[True, False]
```

- Main camera 提供 RGB-D；
- Wrist camera 仍只提供 RGB；
- EEF XYZ/quaternion 仍来自机器人状态/FK。

### 10.2 `execute/rgbd.py` 职责

`rgbd.py` 负责将当前 Main camera observation 变成可供规则使用的 3D target：

- MuJoCo normalized depth 转换为 metric depth；
- RGB/depth 对齐与 `flip_images` 坐标处理；
- 将 template match 后的 mask/bbox 映射到 depth pixel；
- 无效 depth 和离群点过滤；
- masked point cloud 与 robust 3D center；
- depth confidence；
- 调用 `geometry.py` 完成 pixel/depth/K/T 到 world XYZ 的反投影。

`geometry.py` 保持为纯几何工具，`rgbd.py` 处理传感器数据和有效性。

### 10.3 不修改 Offline 时可完成的 3D 规则

#### Phase 3D

Offline 继续提供 2D RGB template/mask；在线匹配到当前目标后，Main depth 将当前 mask 升维为 `target_xyz`，与 `robot0_eef_pos` 比较 3D progress。

```text
Offline：记住目标长什么样
Online RGB-D：计算目标现在的 3D 位置
```

#### Pick 3D Completion

在夹爪闭合时在线记录 object/EEF XYZ，后续判断：

- EEF 是否向上移动；
- 目标物体是否真正抬升；
- object displacement 是否与 EEF displacement 一致；
- object-EEF relative offset 是否稳定；
- Main depth 遮挡时是否需要回退 wrist RGB flow。

这些是从当前 grasp 时刻开始的相对在线变化，初期不要求 Offline 提供 depth。

### 10.4 Feasible 3D 为什么可能需要 Offline 3D

当前 Feasible 阈值来自成功 demo ready frame 的 2D normalized distance。如果未来希望 Feasible 继续保持“阈值由成功 demo 定义”，则需要 ready frame 的：

```text
target_xyz
eef_xyz
3D ready distance / XY-Z ready constraints
```

可选数据源：

1. 为每个 skill 手工定义固定米制阈值；
2. Offline 使用 simulator object state + EEF pose；
3. Offline 回放 ready frame 并渲染 Main depth，使用与在线一致的 RGB-D 反投影。

建议先保持 Feasible 2D，完成 Phase/Pick 的在线 3D 实验后，再决定是否引入 Offline depth。

### 10.5 如果未来 Offline 也使用 RGB-D

当 `offline/build_targets.py` 开始调用 RGB-D 能力时，将：

```text
memory_system/execute/rgbd.py
```

提升为：

```text
memory_system/rgbd.py
```

最终结构为：

```text
cosmos_policy/
└── memory_system/
    ├── types.py
    ├── artifacts.py
    ├── geometry.py
    ├── rgbd.py                    # Offline/Execute 共享
    ├── skills.py
    │
    ├── offline/
    │   ├── planner.py
    │   ├── predicate_planner.py
    │   ├── label_segments.py
    │   ├── label_boundaries.py
    │   ├── build_targets.py       # 可选生成 3D ready/target memory
    │   └── build_recovery.py
    │
    └── execute/
        ├── plan.py
        ├── phase.py
        ├── feasible.py
        ├── execution_monitor.py
        ├── skill_completion/
        │   ├── pick.py
        │   ├── place.py
        │   └── open_close.py
        └── recovery/
            ├── retrieval.py
            ├── selectors.py
            └── controller.py
```

3D artifact 应使用新 version，同时保留 XY 与 XYZ 字段：

```text
skill_memory/libero_10/v1_2d/
skill_memory/libero_10/v2_rgbd/
```

源码仍只保持一套 `memory_system`，不复制 `memory_system_2d` / `memory_system_3d` 目录。

## 11. 3D 过渡期的运行模式

可在过渡期保留：

```python
geometry_mode = "2d" | "3d" | "hybrid"
```

- `2d`：复现迁移前 baseline；
- `3d`：用于独立评估 3D 规则；
- `hybrid`：3D 可信时使用 3D，depth 无效/遮挡时回退 2D/wrist RGB。

建议升级顺序：

1. Phase 3D progress；
2. Pick 3D object lift/follow；
3. 评估是否需要 Offline depth；
4. Feasible 3D ready prototype；
5. PlaceOn/PlaceIn 分阶段 3D 规则；
6. 稳定后将 `hybrid` 设为默认，保留 2D 作为遮挡/无效 depth fallback。

## 12. 验收标准

### 结构迁移验收

- `run_libero_eval.py` 不再为 Memory System 修改 `sys.path` 并导入 `bin/execute`；
- `bin/memory` / `bin/execute` 的正式实现已迁入 `cosmos_policy.memory_system`；
- 2D artifact format 和数量不变；
- Phase/Feasible/Completion/Recovery 现有单测通过；
- 代表性 2D eval 行为和成功率没有非预期变化；
- 新 runtime package 中不包含 copy/bak 实验副本。

### 3D 扩展验收

- Main RGB/depth/mask/pixel 坐标和 EEF 投影经过可视化对齐验证；
- depth 明确转换为米制；
- Wrist depth 不是必需输入；
- Phase/Pick 在 Main depth 缺失或遮挡时返回 UNKNOWN 或回退 2D；
- 单帧 depth 噪声不会立即触发 recovery；
- 2D/3D/hybrid 模式可在同一 task/seed 上比较；
- 3D 升级不要求 policy 使用 depth，只扩展 Memory System verifier/completion 观测。

## 13. 设计结论

1. `memory_system` 作为广义 Memory 系统的正式 Python package；
2. `offline` 保留并表达原 `bin/memory` 的 Offline construction 语义；
3. `execute` 保留原 `bin/execute` 的 Online use 语义；
4. Plan 是整个 SkillPlan，不是 Phase 独有 plan；
5. Phase 和 Feasible 各自保留判断条件，Skill Completion 按 skill 拆分；
6. Phase/Feasible Recovery 共享 retrieval/controller，但保留不同 selector 和恢复后状态；
7. 第一次迁移保持 2D，Offline 不引入 depth；
8. 第一版 3D 只在 Execute 使用 Main RGB-D，优先升级 Phase 和 Pick；
9. 只有当 Feasible 需要 demo-derived 3D ready threshold 时，再考虑 Offline depth；
10. 2D 和 3D 共享一套源码目录，过渡期通过 artifact version 和 geometry mode 对比。

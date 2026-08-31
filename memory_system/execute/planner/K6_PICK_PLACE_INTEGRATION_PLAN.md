# K6 Held Object 接入实现与验证记录

## 1. 状态

本文档记录当前实现与验证状态。  
核心代码已经接入，正在做 episode 级验证和失败排查。

当前状态：

- 无 oracle 提取器已实现；
- `HeldObjectPlanner` 已接入 Pick→Place 切换；
- K6 脚本已可运行；
- 已新增四个单物体任务的统一 launcher：`scripts/run_libero10_robotinit_single_object_20.sh`；
- 支持 DEBUG 开关保存干预规划成功/失败点云图；
- Place backoff 已从单向 `0 ~ 0.08m` 改为有符号 `-0.04m ~ +0.04m`；
- 已知主要失败模式为 `no feasible backoff to ready pose`。

## 2. 固定 case

```text
suite: LIBERO-10
task: KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it
object: white_yellow_mug_1
robot initial state variant: KITCHEN_SCENE6_..._view_0_0_100_0_0_initstate_270
```

从 episode 起点运行完整流程，不从 Pick 后的保存帧直接开始。

## 3. 当前整体流程

```text
Initial Alignment
  -> Pick VLA
  -> Pick completion 触发 phase advance
  -> 当前 RGB-D 无 oracle 提取 mug
  -> 构造 hand-local HeldObjectObservation
  -> HeldObjectPlanner 规划到 Place ready pose
  -> 如果成功：执行 planner controller
  -> 恢复 Place VLA
  -> 执行 Place / Close
```

如果 `HeldObjectPlanner` 失败：

```text
记录失败原因到 log
直接回退到 Place VLA
```

## 4. 已实现文件

### 4.1 无 oracle 提取器

```text
memory_system/execute/planner/held_object/connected_component.py
```

职责：

```text
metric depth 输入
URDF robot depth filter
cuRobo robot sphere filter
EEF ROI
DBSCAN 和组件选择
world points -> cuRobo tool-local points
HeldObjectObservation 输出
```

不读取 simulator segmentation / object instance id。

### 4.2 HeldObjectPlanner

```text
memory_system/execute/planner/held_object/planner.py
```

已实现/调整：

- `surface_points` 使用 2cm 体素降采样后再做碰撞检查；
- 增加 `self.last_failure` 记录失败原因；
- 增加 `self.last_debug` 保存调试点云和冲突信息；
- 扩展 `TrajectoryConflict`，记录最近冲突点、对应 robot sphere 中心、最小间隙；
- 当 cuRobo 完全失败时，记录 `nearest_obstacle_to_target`，即 ready pose 附近最近的场景点；
- 成功时也会记录最近点信息。

### 4.3 run_libero_eval.py 修改

```text
cosmos_policy/experiments/robot/libero/run_libero_eval.py
```

增加：

- `run_episode(..., phase_transition_hook=None)`
- `run_task(..., phase_transition_hook=None)`
- 模块级 `_PHASE_TRANSITION_HOOK`
- Pick→Place 时调用 hook，成功则接入 `_alignment_controller` 执行通道
- hook 调用时传入 `episode_id`

普通评测不设置 hook 时行为不变。

### 4.4 smoke test 开关

```text
scripts/run_libero_smoke_test.sh
scripts/run_libero_smoke_test.py
```

新增环境变量：

```text
COSMOS_HELD_OBJECT=1
COSMOS_HELD_OBJECT_DEBUG=1
COSMOS_SKILL_COMPLETION_ACTIVE=1
```

- `COSMOS_HELD_OBJECT`：开启 held-object 干预；
- `COSMOS_HELD_OBJECT_DEBUG`：保存干预规划成功/失败点云图；
- `COSMOS_SKILL_COMPLETION_ACTIVE`：启用 Pick completion 检测。

### 4.5 K6 运行脚本

```text
scripts/run_libero10_k6_held_object.sh
```

基于 `scripts/run_libero10_robotinit_20.sh` 复制，保留原有参数，新增：

```text
COSMOS_HELD_OBJECT
COSMOS_HELD_OBJECT_DEBUG
COSMOS_SKILL_COMPLETION_ACTIVE
COSMOS_SKILL_COMPLETION_ITEM
```

并按照 `run_libero10_other9_robotinit_20.sh` 的方式，为 KITCHEN_SCENE6 使用正确的 robot initial state：

```text
initstate_270
```

## 5. 运行方式

```bash
cd /data1/liu/exp/counterfactual/external/cosmos-policy

conda activate cosmospolicy

COSMOS_HELD_OBJECT=1 \
COSMOS_HELD_OBJECT_DEBUG=1 \
COSMOS_SKILL_COMPLETION_ACTIVE=1 \
NUM_CASES=5 \
ROBOTINIT_GPU=7 \
sh scripts/run_libero10_k6_held_object.sh
```

## 6. 输出位置

当前输出沿用原有 robotinit 脚本路径：

```text
scripts/experiments/libero10_robotinit_single/robotinit/
```

其中包含：

```text
*_summary.json
logs/ENV_EVAL-*.txt
rollouts/...
```

held-object 相关日志通过 `log_message` 写入对应 eval log。

DEBUG 点云图输出到：

```text
tests/held_object/k6_pick_place_integration/
```

命名规则：

```text
i_planner_success_{task}_ep{episode}_multiview.png
i_planner_failure_{task}_ep{episode}_multiview.png
```

含义：

- `i_planner_success`：HeldObjectPlanner 干预规划成功；
- `i_planner_failure`：HeldObjectPlanner 干预规划失败；
- 不是最终 episode 成功/失败。

图中标注：

```text
黑色 X：最近冲突点
蓝色圆点：robot sphere 中心
黑色虚线：两者距离连线
紫色三角形：ready pose 附近最近的场景点
```

## 7. 当前验证结果与已知问题

### 7.1 单物体任务 20-case 验证

使用 GPU 5、每个任务 20 cases 的验证结果如下：

| 任务 | Place planner | 最终任务 |
|---|---:|---:|
| KITCHEN_SCENE3 moka pot | 20/20 (100%) | 20/20 (100%) |
| KITCHEN_SCENE4 black bowl | 20/20 (100%) | 19/20 (95%) |
| KITCHEN_SCENE6 yellow-white mug | 16/20 (80%) | 17/20 (85%) |
| STUDY_SCENE1 book | 17/20 (85%) | 19/20 (95%) |
| **合计** | **73/80 (91.25%)** | **75/80 (93.75%)** |

日志中 planner 结果会同时写入 stdout 和 log 文件，因此统计时已去重。

### 7.2 主要失败模式

当前失败原因主要为：

```text
no feasible backoff to ready pose
```

即直接目标以及有符号 `[-0.04m, +0.04m]` backoff 范围内都没有找到可行路径。该结果表示当前 attachment、场景点云和安全余量组成的碰撞模型下无解，不等价于真实场景绝对无解。

### 7.3 失败点可能落在桌面上

`nearest_obstacle_to_target` 是 ready pose 附近最近的场景点。  
如果 ready pose 在桌面上方，最近点通常是桌面，这不一定代表桌面就是真正失败原因。

真正原因可能包括：

- ready pose 姿态不可达；
- attachment 与场景碰撞；
- 路径中间被其他物体挡住；
- 关节限位 / 奇异。

### 7.3 URDF 计算

测试脚本中为了速度，临时使用了诊断脚本里的 GPU `WarpUrdfDepthFilter`。  
正式 `connected_component.py` 和 `HeldObjectPlanner` 默认仍与 `UrdfDepthFilter` 保持一致。

## 8. Oracle 边界

- 无 oracle 提取器：不读取 simulator segmentation。
- `HeldObjectPlanner`：不读取 simulator segmentation。
- Pick completion 检测：使用现有 `PickSkillCompletion`，可能依赖 simulator segmentation 或 timeout，这部分属于既有 skill-completion 逻辑，不属于 held-object 提取/规划输入。

## 9. 后续待办

- 根据 `i_planner_failure_*.png` 和 log 定位 `no feasible backoff` 的具体原因；
- 比较正、负 backoff 的实际选择分布及其对抬升现象的影响；
- 对 K6 和 Study 的 `no feasible backoff` episode 保存更详细的 planner attempt diagnostics；
- 稳定后清理临时调试 print / Warp 注入。

# 通用 Skill Transition 接入规范

## 1. 目标与边界

本文档定义 LIBERO-10 skill sequence 的统一衔接方式：

```text
skill completion
  -> skill effect
  -> next-phase ready pose
  -> planner routing
  -> controller
  -> next-phase VLA
```

现有 phase sequence、ready-pose memory、Pick completion、Initial Alignment、HeldObjectPlanner、CuroboPlanner 和 controller 执行通道全部复用。本次不新增运动规划算法，也不实现 Open、Close、TurnOn 的状态感知或 Place 的空间关系判断。

## 2. Completion 策略

| Skill | 策略 |
| --- | --- |
| `Pick` | 现有语义判断 + timeout fallback |
| `PlaceIn` | VLA 夹爪 closed→open + timeout fallback |
| `PlaceOn` | VLA 夹爪 closed→open + timeout fallback |
| `Open` | 现有 timeout completion |
| `Close` | 现有 timeout completion |
| `TurnOn` | 现有 timeout completion |

所有 skill completion 都必须配置 timeout。Pick 和 Place 可以在 timeout 前因语义条件满足而提前 advance；Open、Close、TurnOn 只通过 timeout advance。timeout 仅在 VLA action queue 自然耗尽的 chunk boundary 累计，并沿用 `SKILL_MAX_ACTION_CHUNKS` 的 per-skill budget。

`PlaceIn` 和 `PlaceOn` 共用 `ReleaseSkillCompletion`：

```text
WAIT_CLOSED -> WAIT_OPEN -> COMPLETED
```

具体规则：

- 使用现有 VLA action 定义 `gripper_closed = action[-1] > 0`；
- 进入 Place 时若 session 存在 `held_item`，直接认为 closed 已确认；
- 连续观察到配置数量的 VLA open frame 后 advance；
- `gripper_qpos` 只用于诊断；
- planner、Initial Alignment、fine-align action 不进入 checker；
- 若语义条件始终未满足，达到 Place 的 VLA action-chunk budget 后以 timeout advance。

completion 通过显式 registry 创建；未知 skill 不静默选择 checker。

## 3. 当前 Transition Graph

过滤 `already_satisfied` phase 后：

```text
TurnOn  -> Pick
Pick    -> PlaceIn
Pick    -> PlaceOn
PlaceIn -> Close
PlaceIn -> Pick
PlaceOn -> Pick
```

当前只有 `Pick -> PlaceIn/PlaceOn` 在 transition 时持物。

## 4. Runtime 模型

### 4.1 Episode session

每个 episode 创建独立 session：

```python
@dataclass
class SkillTransitionSession:
    task_name: str
    demo_id: str
    held_item: str | None = None
    held_observation: HeldObjectObservation | None = None
    pending_place_aligner: PlaceFineAligner | None = None
```

session 不保存 active controller。controller 仍由 `run_libero_eval.py` 独占执行。planner 实例可以复用，session 状态不得跨 episode 或保存在模块级全局变量中。

### 4.2 Skill effects

completion 后、planner routing 前应用：

```text
Pick:
  held_item = completed_phase.arguments["item"]

PlaceIn / PlaceOn:
  held_item = None
  held_observation = None
  pending_place_aligner = None

Open / Close / TurnOn:
  held state 不变
```

### 4.3 Ready-pose 查询与身份断言

Skill 的执行顺序只使用 exact demo sequence 中的原始 list 顺序。`planner_step_id` 是 phase identity 和 ready-pose join key，不用于排序。

Initial Alignment 已绑定 exact demo。中间 transition 不重新做 VAE 检索，继续调用现有 `FeasibleRecoveryMemory.select(...)`，随后在 coordinator 中筛选 selected demo，并断言返回 target 与 next phase 完全一致：

```python
candidates = memory.select(
    task_name,
    next_phase.planner_step_id,
    next_phase.skill,
    next_phase.arguments,
)
exact = [c for c in candidates if str(c["demo_id"]) == str(demo_id)]

assert len(exact) == 1
target = exact[0]
assert int(target["planner_step_id"]) == int(next_phase.planner_step_id)
assert target["skill"] == next_phase.skill
assert normalize_args(target["arguments"]) == normalize_args(next_phase.arguments)
```

断言失败时记录 phase identity mismatch 并恢复 next-phase VLA。无需修改 `memory_system/artifacts.py`，也不新增另一套 ready-pose 查询 API。

查询逻辑不修改 pose；z offset 和 local-axis backoff 由具体 planner 决定，不能统一套用 Initial Alignment 的 world-z offset。

### 4.4 Coordinator 输出

```python
@dataclass
class TransitionIntervention:
    kind: str
    controller: object | None
    step_budget: int
    gripper_action: float
    deferred_aligner: PlaceFineAligner | None = None
```

`controller=None` 且存在 `deferred_aligner` 表示 simple Place：先恢复 Place VLA，接近 ready pose 后再执行 fine alignment。

## 5. Planner Routing

路由顺序固定为：

```text
持物 + next=Place + mode=curobo
  -> HeldObjectPlanner

持物 + next=Place + mode=simple
  -> episode-local deferred PlaceFineAligner

无持物
  -> 现有 CuroboPlanner / PoseController
```

当前不存在持物进入非 Place skill 的 sequence；未来出现时再增加 attachment-aware handler。

普通移动直接复用 `memory_system/execute/curobo_planner.py`。Coordinator 只增加薄 adapter，将现有 `PlanResult` 转为 `TransitionIntervention`。

Gripper policy：

```text
Pick -> Place       closed
TurnOn -> Pick      open
Place -> Pick       open
PlaceIn -> Close    open
```

`run_libero_eval.py` 不再把所有 transition controller 的 gripper action 硬编码为 closed。

## 6. 执行与失败语义

```text
VLA completion advance
  -> 清空旧 VLA queue
  -> 应用 skill effect
  -> 取得 next phase
  -> ready-pose 查询与身份断言
  -> coordinator routing
  -> controller 或 deferred aligner
  -> intervention 结束
  -> 启动 next-phase completion window
  -> 恢复 VLA
```

若 next phase 不存在则 sequence exhausted。Planner action 不推进 phase，也不消耗 timeout chunk budget。

以下失败统一记录后恢复 next-phase VLA，不回退已经完成的 phase：

- exact target 缺失；
- held-object 提取失败；
- planner/controller 创建失败；
- controller 未在 budget 内收敛。

日志包含 episode、demo、completed/next phase、held item、planner kind 和失败原因。

## 7. 文件改动

新增生产代码：

```text
memory_system/execute/skill_completion/place.py
memory_system/execute/skill_transition.py
```

`skill_transition.py` 包含 session、context、intervention、ready-pose 身份断言和 coordinator；逻辑明显增长后再拆 package。

修改生产代码：

```text
memory_system/execute/skill_completion/__init__.py
memory_system/execute/vla_skill_runtime.py
cosmos_policy/experiments/robot/libero/run_libero_eval.py
scripts/run_libero_smoke_test.py
```

可能需要兼容修改：

```text
tests/held_object/run_k6_mug_pick_place_integration.py
```

原则上不改算法文件：

```text
memory_system/execute/curobo_planner.py
memory_system/execute/initial_alignment.py
memory_system/execute/planner/held_object/planner.py
memory_system/execute/planner/held_object/connected_component.py
memory_system/execute/planner/place_fine_aligner.py
```

## 8. 测试与验收

新增：

```text
tests/skill_completion/test_place.py
tests/test_skill_transition.py
```

修改：

```text
tests/skill_completion/test_vla_skill_runtime.py
```

必须覆盖：

- Place closed→open、无持物 open、open 抖动；
- Place 语义未完成时在 action-chunk budget 到达后 timeout advance；
- Pick、Place 和 timeout-only skill 的 budget 分别生效；
- planner action 与 completion 隔离；
- Pick 设置、Place 清除 held state；
- cuRobo Place、simple Place、无携物三种路由；
- ready-pose demo/step identity 断言成功与失败；
- gripper policy；
- `TurnOn -> Pick`、`Place -> Close`；
- 双物体 episode 连续两次 Pick→Place，第二次使用正确 item、step 和 ready pose。

验收条件：

1. 六种 skill 使用显式 completion registry。
2. 所有 skill 都配置 timeout fallback，且 planner/controller action 不累计 timeout budget。
3. Place 可由 VLA closed→open 提前 advance，也可在 budget 到达后 timeout advance。
4. transition 通知不再只允许 Pick→Place。
5. held state 和 Place aligner 为 episode-local。
6. ready pose 查询后通过 demo/step/skill/arguments 身份断言。
7. 三种 planner 路由返回统一 intervention。
8. controller 使用 intervention 指定的 gripper action。
9. planner action 不污染 completion。
10. intervention 结束后正确恢复 next-phase VLA。
11. 双物体任务能完成两次独立 Pick→Place transition。

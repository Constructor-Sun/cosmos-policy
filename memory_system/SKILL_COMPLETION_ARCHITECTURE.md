# Minimal Skill Completion Architecture

## 1. 目标

本阶段只实现一件事：限制 VLA 在每个 memory skill 上执行的 action chunks，并在当前 skill 结束后清空旧动作、推进到 memory 中记录的下一阶段。

Completion 不负责判断 LIBERO 任务在功能上是否正确。`rule` 表示观测到当前 skill 的语义结束；`timeout` 只表示 VLA 预算耗尽。两者都推进 skill，但最终正确性只由 episode 的 LIBERO/BDDL success result 判定。

设计原则：

- skill 和顺序完全来自 memory；
- planner 只负责移动到 ready pose，不执行 skill；
- VLA 真正执行 skill；
- completion 只在 VLA 执行期间激活；
- completion 可以中断 VLA，绝不控制 planner；
- Pick 使用现有保守规则，其余 skill 暂时只使用 timeout；
- 不增加 recovery、replan、Place 规则或复杂状态机。

## 2. Memory skill sequence

Runtime 暂时使用现有 `PhaseSpec`：

```text
PhaseSpec
  - planner_step_id
  - skill
  - arguments
```

Memory 是 skill 的唯一来源。Runtime 不根据动作、视觉或 BDDL 在线创建、删除、猜测或重排 skill。

每条 memory record 中 `segments` 的原始 list 顺序就是执行顺序。`planner_step_id` 只是阶段 identity 和 memory 检索键，不是排序键。

当前 `load_phase_plans()` 会按 `planner_step_id` 排序，不能直接用于 runtime continuation。需要增加一个保序 loader：

```text
(task_name, demo_id) -> tuple[PhaseSpec, ...]
```

规则只有：

- 按 `record["segments"]` 原始顺序读取；
- 跳过 `status="already_satisfied"`，但不改变其他 segment 的相对顺序；
- 使用 Initial Alignment 选中的 `demo_id` 对应的完整 sequence；
- 不跨 demo 投票，不按 step id 排序；
- 找不到合法 sequence 时关闭该 episode 的主动 continuation并保持原 baseline，不构造替代计划。

## 3. Planner、VLA 与 completion 的边界

### Planner / Alignment

Planner 只把夹爪移动到当前 skill 的 ready pose 附近，不会完成 Pick、Place 或其他 skill。

- planner 自己管理 trajectory、controller、收敛和预算；
- planner 自己决定何时结束；
- planner action 不进入 completion；
- planner steps 不计入 VLA chunk budget；
- completion 不清理或中断 planner trajectory；
- 当前 Initial Alignment 行为保持不变。

当前只有第一阶段 Initial Alignment 是正式在线路径。未来 per-skill alignment 或带物体的 Place alignment 不属于本阶段。

### VLA

VLA 在 planner 结束后真正执行当前 skill。Completion 生命周期从准备重新 query VLA 时开始，到当前 skill 以 `rule` 或 `timeout` 推进时结束。

没有 planner 的阶段可以直接开始 VLA window。未来增加 per-skill planner 时，只需把它插入“新 PhaseSpec 激活”和“VLA window 开始”之间。

### Completion

Completion 只返回 VLA 推进决定。它不直接加载 memory、不启动 planner，也不判断最终任务成功。

## 4. 最小生命周期

```text
读取 PhaseSpec[i]
        ↓
可选 planner / alignment
        ↓ planner 自己结束
清空旧 VLA queue
completion.reset()
        ↓
query VLA，执行当前 skill
        ↓
每个 VLA action 后 observe_frame(...)
        ├── rule
        │     → 清空当前 VLA queue
        │     → 推进 PhaseSpec[i+1]
        │
        └── running
              → queue 自然耗尽
              → finish_action_chunk()
                    ├── running → query 下一 chunk
                    └── timeout → 清空当前 VLA queue
                                  → 推进 PhaseSpec[i+1]
```

Completion 激活范围必须严格限定为：

```text
planner 执行中        inactive
planner 结束          reset
当前 skill 的 VLA     active
skill 推进            inactive
```

当前 shadow 从 episode 第 0 帧持续观察，只能用于只读实验，不能原样用于 active continuation。Planner telemetry 可以单独记录，但不得进入 active checker 的状态历史。

## 5. Action-chunk 语义

- 一个 chunk 是一次 VLA query 后交给当前 skill 的 action queue；
- queue 自然耗尽时调用一次 `finish_action_chunk()`；
- planner trajectory 不调用 `finish_action_chunk()`；
- rule 在 chunk 中间满足时立即清空剩余 VLA actions；
- rule 已满足后不能被同一边界覆盖成 timeout；
- 同一边界同时出现 rule 和预算耗尽时，rule 优先；
- timeout 从当前 VLA window 的第一个 chunk 开始计数，不使用 episode 绝对帧数。

## 6. 决策结果

继续使用当前最小结果：

```text
SkillDecision
  - advance
  - semantic_completed
  - reason
  - action_chunks
```

只保留三个 reason：

- `running`：继续当前 skill 的 VLA；
- `rule`：语义规则满足，`semantic_completed=True`，推进；
- `timeout`：chunk budget 用尽，`semantic_completed=False`，仍然推进。

当前策略有意将 `rule` 和 `timeout` 都映射为 `advance=True`，但日志必须保留二者区别。

## 7. Per-skill budget

采用集中、固定配置：

```python
DEFAULT_MAX_ACTION_CHUNKS = 3

SKILL_MAX_ACTION_CHUNKS = {
    "Pick": 7,
    "PlaceOn": 3,
    "PlaceIn": 3,
    "Open": 3,
    "Close": 3,
    "TurnOn": 3,
}
```

- Pick 规则已经过 replay 和 shadow 验证，最多执行 7 个 chunks，降低过早 timeout 风险；
- Place、Open、Close、TurnOn 暂时没有语义规则，执行 3 个 chunks 后推进；
- 未列出的 memory skill 使用默认 3 个 chunks；
- 本阶段不增加自适应 timeout。

## 8. Checker registry

本阶段只需要两个 completion 类型：

```text
Pick
  -> PickSkillCompletion(max_action_chunks=4)

其他 memory skill
  -> TimeoutOnlySkillCompletion(max_action_chunks=配置值)
```

`TimeoutOnlySkillCompletion._check_rule()` 永远返回 `False`。它不创建 skill，只为当前 memory `PhaseSpec` 提供统一预算。

```python
def make_completion(phase: PhaseSpec) -> TimedSkillCompletion:
    budget = SKILL_MAX_ACTION_CHUNKS.get(
        phase.skill,
        DEFAULT_MAX_ACTION_CHUNKS,
    )
    if phase.skill == "Pick":
        return PickSkillCompletion(max_action_chunks=budget)
    return TimeoutOnlySkillCompletion(max_action_chunks=budget)
```

## 9. Pick 规则

Pick 继续复用现有 `PickCompletionChecker`，不扩展规则：

1. VLA 命令夹爪闭合，并且不是空闭合；
2. 当前 `PhaseSpec.arguments["item"]` 的可见点云与 EEF 保持刚体共运动；
3. item 和 EEF 累积足够的共同垂直上升距离；
4. 完成条件连续保持配置帧数。

保留现有四个语义参数：

- `empty_closed_gap`；
- `min_lift_distance`；
- `max_rigid_error`；
- `stable_frames`。

Pick item 必须来自当前 memory `PhaseSpec`。Runtime 不跟踪其他物体，也不根据动作猜测目标。

现有 shadow 和 replay 只证明 Pick rule 能提供有用信号，不证明 phase continuation、queue interruption 或 4-chunk budget 已经完成验证。

## 10. Phase advance

Completion 类不直接操作 planner、memory 或 queue。调用层消费 `SkillDecision`：

```text
decision.advance
  -> clear current VLA action queue
  -> log current phase result
  -> deactivate current completion
  -> active_phase_index += 1
  -> expose next PhaseSpec
```

如果没有 next `PhaseSpec`，标记 memory plan exhausted，不创建新的 skill。

当前没有 per-skill planner 的阶段可以直接对新 `PhaseSpec` 调用 `completion.reset()` 并重新 query 完整任务 VLA。未来 planner 接入后，由 planner terminal event 触发这个 VLA start hook，completion 无需修改。

## 11. 日志

每个 memory skill 至少记录：

```text
episode_id
memory_demo_id
phase_index
planner_step_id
skill
arguments
vla_start_frame
vla_end_frame
action_chunks
semantic_completed
advance_reason
task_success
```

`task_success` 可在 episode 结束后通过 `episode_id` 关联。Completion 不根据结果在线修正当前 episode，只用于后续评估 rule 和 budget。

## 12. 验证

单元测试至少覆盖：

- memory segment 原始顺序被保留；
- `planner_step_id` 不改变 sequence；
- planner frame 不进入 completion；
- planner step 不增加 chunk count；
- VLA window 开始时 reset；
- Pick 在第 7 个 chunk 结束后 timeout；
- 其他 skill 在第 2 个 chunk 结束后 timeout；
- rule 优先于同边界 timeout；
- advance 清空旧 VLA queue；
- phase 切换后 checker 状态不泄漏。

Pick rule 继续使用现有 demo replay 和 simulator oracle 验证；simulator state、predicate 和 ground-truth segmentation 只能用于测试。

最终运行只比较：

```text
当前 baseline
vs
完整 minimal completion continuation
```

使用相同 task、seed 和 initial state，验收指标只看最终 LIBERO success rate 是否受到不可接受的影响，同时保留逐 episode completion 日志用于定位失败。

## 13. 当前实现差距

已经存在：

- `TimedSkillCompletion` 和 `SkillDecision`；
- `PickCompletionChecker` 和 `PickSkillCompletion`；
- Pick shadow、单元测试和 demo replay。

仍需实现：

- selected memory demo 的保序 PhaseSpec loader；
- active phase cursor；
- VLA-only completion gating；
- `TimeoutOnlySkillCompletion`；
- per-skill budget registry；
- VLA chunk boundary 接线；
- advance 时清理 queue 和推进 phase；
- per-skill summary 与最终 task success 关联。

## 14. 本阶段不做

- Place、Open、Close、TurnOn 的语义规则；
- PlaceOn / PlaceIn 功能关系；
- 带物体的 Place planner/alignment；
- per-skill planner continuation；
- Phase/Feasible verifier；
- recovery、replan 或自动纠错；
- Initial Alignment 行为修改；
- BDDL/simulator predicate 进入 runtime；
- 在线创建或重排 skill；
- 自适应 timeout 或复杂 orchestrator。

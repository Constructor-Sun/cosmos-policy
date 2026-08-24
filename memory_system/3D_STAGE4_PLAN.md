# Stage 4 / Feasible 3D 实现计划

> 本文件是 Stage 4（Feasible 3D）的独立实现计划，放在 `memory_system/3D.md` 旁边。
> 目标是在**完整保留现有 2D Feasible baseline** 的前提下，新增一条平行的 3D Feasible 路径。

---

## 1. 背景与目标

当前 Feasible 阶段使用 2D 像素距离：

```text
current_distance = ||gripper_xy - target_xy|| / bbox_diagonal
```

Stage 4 的目标是：

```text
current_distance_m = ||eef_pos - target_xyz_world||
```

用真实三维空间中的米制距离来判断是否进入 Feasible/ready 区域。

### 明确不做的内容

- 不修改现有 2D `FeasibleVerifier` 行为；
- 不删除 `memory_system/execute/feasible.py`；
- 不修改 `memory_system/offline/` 下已有文件；
- 不修改 Recovery 相关逻辑；
- 不改变现有 Phase/Feasible/Completion 状态机框架；
- 不改变默认的 8-step Feasible 检查节奏。

---

## 2. 设计原则

### 2.1 像 Phase 3D 一样新增平行实现

```text
现有 2D 路径：
phase.py        -> FeasibleVerifier(feasible.py)

新增 3D 路径：
phase3d.py      -> Feasible3DVerifier(feasible3d.py)
```

- `feasible.py` 保持原样；
- `feasible3d.py` 作为新增文件；
- 通过配置开关决定使用 2D 还是 3D Feasible。

### 2.2 复用已经确定的 3D 表示

Stage 2 / Stage 3 已经确定：

- 目标 3D 几何：物体可见表面 camera-frame Z-depth；
- 目标代表点：mask 内有效深度反投影到世界坐标后取逐坐标 median；
- 距离公式：
  ```text
  d(t) = ||eef_xyz(t) - target_xyz_world(t)||
  ```

Feasible 3D 直接复用这套表示，不重新发明 3D 几何。

### 2.3 Feasible 3D 需要 offline 3D ready memory

Phase 3D 只需要“当前帧的 3D 距离趋势”，所以不需要 offline 3D ready 数据。

Feasible 3D 需要回答：

> 对当前 skill / 目标，多少米算 ready？

因此需要从 training demo 的 `ready_frame` 构建 3D ready memory：

```text
ready_distance_m = ||eef_pos_ready - target_xyz_world_ready||
```

---

## 3. 新增 / 修改文件总览

| 文件 | 类型 | 说明 |
|---|---|---|
| `memory_system/execute/feasible3d.py` | 新增 | `Feasible3DVerifier` + 3D ready memory 类 |
| `scripts/build_ready3d_memory.py` | 新增 | 独立的 ready3d artifact 构建脚本，不改 offline 现有文件 |
| `memory_system/types.py` | 修改 | `FeasibleResult` 增加 3D 字段 |
| `memory_system/execute/phase3d.py` | 修改 | 内部 Feasible 切换为 3D，输出 `target_xyz_world` |
| `memory_system/execute/execution_monitor.py` | 修改 | 传递 `target_xyz_world`，兼容 2D/3D Feasible |
| `memory_system/execute/__init__.py` | 修改 | 导出 `Feasible3DVerifier` |
| `cosmos_policy/experiments/robot/libero/run_libero_eval.py` | 修改 | 增加 `enable_feasible_3d` 配置和日志字段 |
| `scripts/run_libero10_robotinit_20.sh` | 修改 | 增加默认开关 |
| `tests/stage4/` | 新增 | Feasible 3D 回放测试 |

---

## 4. 新增 `memory_system/execute/feasible3d.py`

### 4.1 包含内容

```text
ReadyDistance3DPrototype
ReadyDistance3DMemory
Feasible3DVerifier
```

### 4.2 ReadyDistance3DPrototype

```python
@dataclass(frozen=True)
class ReadyDistance3DPrototype:
    demo_id: str
    ready_frame: int
    target_xyz_world: np.ndarray
    eef_pos_ready: np.ndarray
    distance_m: float
```

### 4.3 ReadyDistance3DMemory

从新的 ready3d artifact 读取数据，并按 `task / planner_step_id / skill / arguments` 建立索引。

提供：

```python
select(task_name, planner_step_id, skill, arguments, exclude_demo_ids=())
```

返回 `ReadyDistance3DPrototype` 列表。

### 4.4 Feasible3DVerifier

建议继承现有 `FeasibleVerifier`，复用：

- wrist feasible 判断；
- `entered_feasible` latch 逻辑；
- `FEASIBLE / NOT_FEASIBLE / FEASIBLE_UNKNOWN` 语义；
- `reset()` 的基本结构。

核心覆盖：

```python
current_distance_m = ||eef_pos - target_xyz_world||
ready_votes = sum(current_distance_m <= proto.distance_m for proto in prototypes)
```

判断规则保持与 2D 一致：

```text
ready_votes >= min_demo_votes  -> FEASIBLE
```

默认 `min_demo_votes = 2`。

---

## 5. 新增 `scripts/build_ready3d_memory.py`

### 5.1 目的

不修改现有 offline 文件，独立生成 3D ready memory artifact。

### 5.2 输入

- LIBERO-10 training demo HDF5；
- `segments_ready_fixed16.json`；
- 现有 `phase_targets.pt`（可选，用于复用模板/目标定义）；
- 相机参数与 RGB-D 几何路径。

### 5.3 输出

```text
skill_memory_test/libero_10/ready3d_targets.pt
```

格式建议：

```text
format: "libero_ready3d_targets_v1"
```

每个 prototype 包含：

```text
task_name
planner_step_id
skill
arguments
demo_id
ready_frame
target_xyz_world
eef_pos_ready
distance_m
```

### 5.4 构建逻辑

对每个有效 segment：

1. 定位 `ready_frame`；
2. 使用与 Stage 2/3 相同的 RGB-D 路径：
   ```text
   mask -> pixel_to_world -> median target_xyz_world
   ```
3. 读取 `eef_pos[ready_frame]`；
4. 计算：
   ```text
   distance_m = ||eef_pos_ready - target_xyz_world_ready||
   ```
5. 写入 artifact。

---

## 6. 修改 `memory_system/types.py`

`FeasibleResult` 增加可选 3D 字段：

```python
current_distance_m: float | None = None
ready_distance_m: float | None = None
progress_m: float | None = None
target_xyz_world: np.ndarray | None = None
```

所有字段默认 `None`，保证 2D 结果完全兼容。

---

## 7. 修改 `memory_system/execute/phase3d.py`

### 7.1 内部 Feasible 切换

把：

```python
self.feasible = FeasibleVerifier(...)
```

改为：

```python
self.feasible = Feasible3DVerifier(...)
```

这样 Phase 的“首次进入 Feasible”和 Stage 4 的 Feasible 判断使用同一个 3D 标准。

### 7.2 输出 target_xyz_world

在 `PhaseResult.details` 中增加：

```python
details["target_xyz_world"] = target_xyz
```

供 ExecutionMonitor 传给 Feasible3D。

---

## 8. 修改 `memory_system/execute/execution_monitor.py`

### 8.1 增加状态

```python
self.target_xyz_world = None
```

### 8.2 Phase -> Feasible 传递

Phase 返回 `PHASE_DONE` 时：

```python
self.target_xyz_world = phase_result.details.get("target_xyz_world")
```

### 8.3 兼容 2D / 3D Feasible

```python
if hasattr(self.feasible_verifier, "use_3d"):
    feasible_result = self.feasible_verifier.update(
        observation,
        target_xyz_world=self.target_xyz_world,
        confidence=self.target_confidence,
    )
else:
    feasible_result = self.feasible_verifier.update(
        observation,
        self.target_xy,
        self.target_bbox,
        confidence=self.target_confidence,
    )
```

---

## 9. 修改 `cosmos_policy/experiments/robot/libero/run_libero_eval.py`

### 9.1 配置

新增：

```python
enable_feasible_3d: bool = False
```

### 9.2 创建 monitor

```python
feasible_verifier = (
    Feasible3DVerifier(...)
    if cfg.enable_feasible_3d
    else FeasibleVerifier(...)
)
```

### 9.3 日志

`_feasible_result_payload()` 增加：

```python
current_distance_m
ready_distance_m
progress_m
target_xyz_world
```

---

## 10. 修改 `scripts/run_libero10_robotinit_20.sh`

增加默认环境变量：

```bash
export COSMOS_FEASIBLE_3D=1
```

或根据实验需要设为 0。

---

## 11. 新增 `tests/stage4/`

### 11.1 目的

验证 3D Feasible 是否与 simulator Oracle 的 ready boundary 一致。

### 11.2 测试方式

- 回放 LIBERO-10 training demo；
- 在每个 skill step 的 Phase 区间内按固定 stride 计算 Feasible3D；
- 对比 simulator Oracle：
  - `ready_frame` 之前应为非 FEASIBLE；
  - `ready_frame` 附近应进入 FEASIBLE；
  - 不应频繁抖动。

### 11.3 指标

- 正确进入 Feasible 的 phase 比例；
- 平均进入时间与 `ready_frame` 的偏差；
- 3D ready distance 在不同 demo 间的稳定性；
- 与 2D Feasible 的对比结果。

---

## 12. 不改动的部分

- `memory_system/execute/feasible.py`
- `memory_system/artifacts.py`
- `memory_system/offline/*`
- Recovery 相关所有文件
- Completion 相关判断
- 现有 2D Phase/Feasible/Completion 行为

---

## 13. 实施顺序建议

1. 先写 `scripts/build_ready3d_memory.py` 并生成 `ready3d_targets.pt`；
2. 新增 `feasible3d.py`，先做单元/回放测试；
3. 修改 `phase3d.py` 内部 Feasible 为 3D；
4. 修改 `execution_monitor.py` 和 `run_libero_eval.py` 接线；
5. 跑 `tests/stage4/` 验证；
6. 跑 robotinit runtime 对比 2D/3D 成功率。

---

## 14. 待确认问题

- 3D ready 距离使用绝对米制距离，还是继续按目标尺寸归一化？
- Phase3D 是否立即切换为 Feasible3D，还是先只让 Feasible 阶段使用 3D？
- wrist 2D 条件是否保留？
- 对 `target_xyz_world` 无效/深度缺失帧如何处理？
- `ready3d_targets.pt` 的格式和存放位置是否按上面建议执行？

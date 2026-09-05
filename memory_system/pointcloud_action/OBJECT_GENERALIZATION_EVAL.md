# PointCloud Action Transfer Evaluation

当前 `pointcloud_action` 只支持原子 `Pick` skill。最多进行两个主实验，二者都固定使用同一份 LIBERO-90 memory，不使用测试集 demo 更新 memory。

测试前应先修复点云提取、轨迹连续性和 ready-motion 状态上报问题，并在两个实验中使用完全相同的 planner、检索参数和成功判定。

## 实验一：迁移到新场景和新任务组合

### 目的

测试 LIBERO-90 Pick memory 能否迁移到 LIBERO-Long（代码中的 `libero_10`）。LIBERO-10 的目标物体都在 LIBERO-90 中出现过，因此该实验主要衡量场景布局、初始状态和任务组合变化，而不衡量未见物体。

### 设置

- Memory：`memory_system/pointcloud_action/pointcloud_action_memory.pt`
- Evaluation：`LIBERO-Cosmos-Policy/success_only/libero_10_regen`
- Manifest：`skill_memory_test/libero_10/segments_ready_fixed16.json`
- 使用每个任务相同数量的 demo，建议每个任务至少 10 个有效 Pick segment。
- 从 Pick 起始帧恢复状态，完成检索、ready motion 和 Pick action replay。

### 指标

- Stable Pick Success Rate：保持抓取、抬升至少 2 cm，并稳定 20 步。
- 每个任务成功率和任务宏平均成功率。

该实验回答：在物体已经见过的情况下，skill 能否迁移到新的场景布局和长任务组合中。

## 实验二：迁移到不同物体

### 目的

测试同一份 LIBERO-90 memory 在 LIBERO-Object 上的物体迁移能力。这里采用非严格 unseen-object 设置，不额外构建 leave-one-object-out memory。

### 设置

- Memory：与实验一相同，保持冻结。
- Evaluation：`LIBERO-Cosmos-Policy/success_only/libero_object_regen`
- 为 10 个 LIBERO-Object task 生成 Pick segment manifest。
- 每个物体使用相同数量的 demo，建议每个物体至少 10 个。
- 不将 LIBERO-Object demo 或 action 加入 memory。

### 分组

- Seen：`alphabet_soup`、`butter`、`chocolate_pudding`、`cream_cheese`、`ketchup`、`milk`、`orange_juice`、`tomato_sauce`。
- Novel-like：`salad_dressing`、`bbq_sauce`。

`salad_dressing` 与 LIBERO-90 的 `new_salad_dressing` 几何近似，因此属于 near-OOD；`bbq_sauce` 是更明确的新类别。由于 novel-like 组只有两个物体，该实验只能支持有限的物体泛化结论。

### 指标

- Seen Object Stable Pick Success Rate。
- Novel-like Object Stable Pick Success Rate。
- `Object Transfer Gap = SR_seen - SR_novel_like`。
- Retrieval Coverage 和检索距离。
- 每个物体成功率和物体宏平均成功率。

该实验回答：系统能否从 LIBERO-90 中检索形状相近的 Pick memory，并迁移到不同名称或不同类别的 LIBERO-Object 物体上。

## 统一报告方式

- 两个实验都使用完整 pipeline，不使用 oracle candidate 或 `init-at-ready` 作为主结果。
- 未检索到 memory、planner 失败和系统拒绝都按端到端失败计入。
- 分别报告 oracle complete point cloud 和 visible RGB-D；如果只能选择一种，优先完成 oracle complete point cloud，以先衡量 memory/action transfer。
- 报告总 rollout 数、成功数、成功率，以及按 task/object 的宏平均结果。

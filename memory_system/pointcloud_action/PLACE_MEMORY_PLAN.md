# Place Memory：最小实现

## 目标与状态

目标是在不改动 Pick 默认行为的前提下，复用 Pick 的“检索 → 锚定 → ready motion → waypoint replay”完成
`PlaceIn` / `PlaceOn`。

当前证据：

- 原场景 source-raw 回放：405/405；
- leave-one-demo-out：505/627；
- Pick 原记忆文件不修改；
- 现有 Place `.pt` 是旧构库产物，需用当前 builder 重建后再作为正式 artifact。

405/405 只表示同一 demo XML、state 和 action 可以完整恢复，不表示跨演示或新场景成功率为 100%。

## 约束

1. Place 只使用精确 `(skill, item, target)` 检索。
2. 不删除实例编号，不归一化容器名称；`plate_1`、`plate_2` 始终是不同实例。
3. 坐标变换、ready motion 和时间索引 gripper replay 直接复用 Pick。
4. Place 只新增目的地锚定、携物闭爪、抓持补偿和 On/In 成功判定。
5. Pick 与 Place 使用独立 `.pt`，避免互相覆盖。

## 记录格式

Place 记录复用 Pick 字段，仅增加：

- `anchor_role="destination"`；
- `restore_frame`：同一 Place segment 的起点，仅供原场景诊断 fallback；
- `action_stop`：exclusive action 终点；
- `item_pose_anchor_ready`：ready 时 item 在 destination frame 中的位姿，用于抓持差异补偿。

索引约定：`action[t]` 将 `state[t]` 推进到 `state[t+1]`。manifest 的 `success_end` 是状态右边界，因此：

```text
action_stop = success_end - 1
action_sequence_raw = actions[ready_frame:action_stop]
```

首次释放帧 `r` 满足 `actions[r-1,-1] > 0` 且 `actions[r,-1] < 0`。必须保证：

```text
ready_frame < r < action_stop
```

ready 搜索只在首次释放前进行；旧 ready 若已在释放后，直接移到 `r-1`。

## 构库

统一使用 `offline/build_memory.py`：默认仍只构建 Pick；传入 `--skills PlaceIn PlaceOn` 时构建独立 Place
artifact。Place 分支只做以下差异：

1. anchor 选择 target，而不是 item；
2. ready 目标点使用演示最终放稳后的 item 位置；
3. ready 搜索截止到首次释放；
4. 保存 destination-relative EE path 和 `item_pose_anchor_ready`；
5. 保存正确的 `restore_frame` / `action_stop`。

不再维护独立的 `build_place_memory.py`。

## 检索与执行

`run_local_place` 的流程：

1. 解析当前 item 和 destination 实例；
2. 提取 destination 点云；
3. 按精确 `(skill,item,target)` 取候选；
4. 在同一精确配对内，选择与当前抓持最接近的演示；
5. 右乘抓持补偿，使当前 item 跟随演示中的 item 路径：

```text
G_demo = inverse(EE_demo_ready) @ item_demo_ready
G_now  = inverse(EE_now) @ item_now
C      = G_demo @ inverse(G_now)
EE_world[t] = destination_world @ EE_destination[t] @ C
```

6. 闭爪移动到 ready；
7. 复用 Pick 的 waypoint + time-indexed gripper replay，保留演示中的释放；
8. 张爪保持，要求 On/In 且 not-grasping 连续满足 `stable_frames`。

## 原场景完整恢复

HDF `states` 只有 MuJoCo time/qpos/qvel，不包含 OSC controller cache/goal 和
`PandaGripper.current_action`。从任意帧冷启动时必须：

1. 加载该 demo 的原始 `model_file`；只修复 asset 路径和 LIBERO 元数据别名，不改实例编号或 mesh；
2. 设置 flattened state 并 `sim.forward()`；
3. 同步 `cur_time` / `timestep`，执行 `controller.update(force=True)` 和 `reset_goal()`；
4. 将 `actions[:frame]` 的 gripper 分量送入 `format_action()`，只重建 accumulator，不推进仿真；
5. 执行 `actions[frame:action_stop]` 并检查稳定成功。

先从 release 前的 local ready 回放。403/405 可直接成功；另外两条只需回退到同一 Place segment 起点：

- `STUDY_SCENE3...right_compartment.../demo_9`：129 → 61；
- `KITCHEN_SCENE1_put_the_black_bowl_on_the_plate/demo_3`：旧 ready 178 修正为 105，再回退到 59。

该 fallback 不回放 Pick。正常端到端执行没有中途冷启动，controller/gripper runtime 会自然连续存在。

## 验收边界

重建 Place artifact 时逐条检查：

- `action_sequence_raw == source_actions[ready_frame:action_stop]`；
- ready 位于首次释放前；
- chunk 起点闭爪且内部包含 close→open；
- item/target 实例名逐字一致。

原场景回放、LOO 和端到端是三个不同指标，不互相替代。当前不为其他 skill、全新 target 或扰动场景承诺
成功率。

## 保留文件

- `offline/build_memory.py`：Pick/Place 共用 builder；
- `offline/demo_state_restore.py`：仅用于从 HDF 中途恢复；
- `eval/local_place_core.py`：Place 薄执行层；
- `retrieval/pointcloud_action_memory.py`：精确实例过滤；
- `schema.py`：兼容现有 Pick/Place artifact。

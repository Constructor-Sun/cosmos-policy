# Pick / Place 目标 3D 检索

## 目的

在 Pick 和 Place 的 ready pose 选择中，使用当前目标的 RGB-D 三维位置匹配 memory episode，避免仅沿用先前 episode 导致目标位置不一致。

这里的目标位置是实例 mask 内 RGB-D 点云在世界坐标系中的中位数：

```text
target_xyz_world = median(back_project(mask_pixels, depth))
```

memory 中对应数据保存在：

```text
skill_memory_test/libero_10/ready3d_targets.pt
```

## 检索规则

候选必须严格匹配：

- `task_name`
- `planner_step_id`
- `skill`
- `arguments`

对每个候选计算：

```text
distance = ||current_target_xyz_world - memory_target_xyz_world||₂
```

按距离升序取 top-3，并使用现有 `mean_ee_states()` 聚合三个候选的完整 6D 末端位姿。位置做算术平均；旋转转为 quaternion、统一符号后平均，再转回 rotvec。

聚合后只调用 planner 一次，不会依次尝试三个候选。不进行 XY 平移，也不根据当前目标位置修改任何单个 memory pose。

## 第一次 Pick

第一次 Pick 仍由 Initial Alignment 执行，但 episode 排序改为：

1. 根据 Pick 的 `arguments["item"]` 解析实例。
2. 从当前 instance segmentation 和 depth 得到物体世界坐标点云。
3. 计算点云中位数并检索最近 top-3。
4. 对 top-3 episodes 的 `ee_states` 求 6D 平均，作为唯一规划目标。

距离最近的第一个 demo 仍作为后续 skill sequence 的绑定 demo；top-3 平均只决定 Initial Alignment 的目标位姿。

其余 Initial Alignment 逻辑保持不变，包括：

- VAE latent 的捕获和 similarity 字段。
- ready pose 的 z-offset。
- cuRobo/controller 的调用与执行。
- planner 失败后的原有 fallback。
- 选中 demo 后的 skill sequence 绑定。

如果 ready3d、mask 或点云不可用，则回退到原来的 VAE 相似度选择。

## 后续 Pick

通过 skill transition 进入的后续 Pick 使用相同的 3D 检索。例如 Scene6 中放完白杯后抓取巧克力布丁：

1. 根据当前 Pick item 的 3D 坐标检索 top-3。
2. 对 top-3 ready poses 求 6D 平均。
3. motion planner 只规划平均位姿一次。
4. 规划失败时返回 VLA，不逐个尝试三个候选。
5. 只替换该阶段的 ready pose，不改变 session 绑定的 demo。

## Place

Place 根据 `arguments["target"]` 解析容器或 region anchor，再使用同一 RGB-D 点云中位数算法：

1. 检索当前容器位置最近的 top-3 memory episodes。
2. 对 top-3 ready poses 求 6D 平均。
3. held-object planner 只规划平均位姿一次。
4. 规划失败时返回 VLA，不逐个尝试三个候选。
5. 不改变 session 绑定的 demo。

如果当前目标不可见或没有 3D 候选，coordinator 保留原来的同-demo ready pose 回退。

## 旋转

当前检索只比较目标物体/容器的 XYZ，不比较其自身旋转。top-3 聚合会通过 `mean_ee_states()` 同时平均机械臂末端位置和 quaternion 旋转，因此 cuRobo 执行的是平均后的完整 6D 末端位姿。

暂未加入目标旋转，原因包括对称目标的方向歧义、单视角点云遮挡以及 PCA 方向的 180°翻转问题。

## 代码位置

- `memory_system/artifacts.py`
  - `Ready3DMemory`：加载并按欧氏距离排序 ready3d prototypes。
- `memory_system/execute/skill_completion/shadow.py`
  - `resolve_target_instance()`：解析 Pick item 或 Place target/region anchor。
  - `PickTargetPointCloud`：从 instance mask 和 depth 得到世界坐标点云。
- `memory_system/execute/initial_alignment.py`
  - 第一次 Pick 的 3D 优先、VAE fallback 选择。
- `memory_system/execute/skill_transition.py`
  - 后续 Pick 与 Place 的 top-3 平均、单次规划和 VLA fallback。
- `cosmos_policy/experiments/robot/libero/run_libero_eval.py`
  - 为 Initial Alignment 传入当前 observation/env，并加载 ready3d memory。
- `scripts/run_libero_smoke_test.py`
  - Pick/Place 共用的在线 3D 候选检索接线。

## 日志

第一次 Pick：

```text
[INIT ALIGN 3D] xyz=[...] top3=[('demo_x', distance_m), ...]
```

后续 Pick 和 Place：

```text
[TARGET_3D] xyz=[...] top3=[('demo_x', distance_m), ...]
[TARGET_3D] planner accepted demos=(...)
[TARGET_3D] planner failed demos=(...); resume VLA
```

## 测试

聚焦单元测试：

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_initial_alignment_3d.py \
  tests/test_skill_transition.py \
  tests/test_memory_system_artifacts.py \
  tests/test_libero_eval_verifier_wiring.py
```

当前结果：`27 passed`。

使用四张 GPU、每个 task 测试 5 个 cases：

```bash
NUM_CASES=5 GPU_IDS="0 1 2 3" sh scripts/run_libero10_all_10_held_object.sh
```

运行时需要启用 `COSMOS_SKILL_COMPLETION_ACTIVE=1`，以提供 instance segmentation 和 depth。

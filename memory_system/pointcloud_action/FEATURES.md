# PointCloud Action Memory — 功能记录

本文只记录功能，不包含测试结果/成功率。

## 1. Memory 构建

- 支持从 LIBERO-90 manifest 构建 Pick memory。
- 支持生成完整 LIBERO-90 manifest。
- Memory 中保存点云、物体坐标系、动作序列、EE 序列、controller 配置、action scale、gripper 序列等信息。
- Memory 同时保存单视角可见点云与完整物体点云：
  - `target_points_object`：单视角可见点云 key；
  - `complete_points_object`：oracle/完整物体点云 key。

## 2. Ready Frame 选择

- 支持按距离选择 ready frame。
- 会确保所选帧位于抓取接近路径上。

## 3. 检索

- 以物体点云为 key 检索 memory。
- 支持选择点云来源：
  - `visible`：使用单视角可见点云；
  - `complete`：使用 oracle 完整物体点云。
- 支持将 object-relative 的 ready pose / EE 序列映射到当前场景。

## 4. 执行 / Eval

- 支持单条 local Pick 评测。
- 支持同套件与跨套件使用。
- 支持从中间 state 恢复后同步 controller。
- 支持保存视频。
- Move-to-ready 支持两种模式：
  - `legacy`：使用原 CuroboPlanner 路径；
  - `waypoint`：使用轻量 waypoint + IK + 关节空间闭环执行。
- waypoint 模式会生成少量完整位姿 waypoint，并在 IK 关节跳变过大时插入中间完整位姿，减少诡异路径。
- 执行 joint-space 运动后会强制刷新并恢复 OSC controller，避免影响后续 Pick 动作。

## 5. 批量 Sweep

- 支持批量遍历 LIBERO-10 Pick 任务。
- 可限制每个 task 的 case 数量。
- 输出结果 JSONL。
- 可选保存视频，视频名包含 success 信息。
- 支持并行分片：
  - `--shard-id` / `--num-shards` 可将 case 均匀分到多个进程/GPU。

## 6. 主要配置

- `POINT_CLOUD_SOURCE`：检索点云来源，`visible` 或 `complete`。
- `READY_MOTION_MODE`：ready 运动模式，`legacy` 或 `waypoint`。
- `READY_MOTION_*`：waypoint 模式的抬升高度、approach 距离、IK seeds、最大关节步长、中间点数量等参数。

## 7. 主要脚本

- 构建 memory：`offline/build_memory.py`
- 生成 manifest：`offline/generate_libero90_manifest.py`
- 单条评测：`eval/eval_pointcloud_pick.py`
- 批量评测：`eval/run_libero10_pick_sweep.py`
- 8 GPU 并行：`scripts/run_pointcloud_pick_sweep_8gpu.sh`
- Oracle 完整点云 8 GPU 并行：`scripts/run_pointcloud_pick_sweep_oracle_8gpu.sh`

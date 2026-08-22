# Phase-Error Pose Recovery（阶段错误位姿恢复）

本文档记录与 verifier 配套的 **phase-error correction / pose recovery** 逻辑。

## 1. 目标

当 phase verifier 发现当前动作 chunk 明显偏离当前 phase 的目标时，不再只是记录 `should_intervene`，而是：

1. 根据当前主相机 VAE token，检索最相似的成功 demo 的恢复位姿；
2. 用一段显式 correction 动作把机械臂拉回该位姿；
3. 清空当前 policy action queue；
4. 恢复 policy 推理，从更接近 demo 的状态继续执行。

## 2. 涉及文件

- 核心逻辑：`bin/execute/libero_pose_recovery.py`
- Eval 接入：`cosmos_policy/experiments/robot/libero/run_libero_eval.py`
- 离线恢复目标：`skill_memory/libero_10/recovery_targets.pt`
- 恢复目标构建：`bin/memory/build_vector_db_from_demos.py`

## 3. 数据流

```text
Phase verifier 返回 PHASE_ERROR
        |
        v
LiberoPoseRecovery.compute(...)
   - 用当前 main-camera VAE token 检索相似 demo
   - 用当前 3D EE pose 和 demo recovery EE pose 计算 delta
   - 除以 action_scale 并均分到 correction_steps 步
        |
        v
run_libero_eval.py 设置：
   _correction_kind = "phase_pose_recovery"
   _correction_per_step = recovery.correction_per_step
   _correction_steps_remaining = recovery.correction_steps
   action_queue.clear()
        |
        v
连续执行 correction_steps 步 correction action
        |
        v
correction 结束后恢复 policy 推理
```

## 4. 关键设计

### 4.1 检索

- 输入：`task_name`, `planner_step_id`, `skill`, `arguments`, 当前 main VAE token, 当前 EE 6D pose。
- 检索目标：`recovery_targets.pt` 中每个 demo 在 `recovery_frame` 处的 `recovery_vae_main`。
- 投票：同一 demo 只保留最高相似度；需要至少 `min_demo_votes` 个 demo 且最高相似度 >= threshold 才返回结果。

### 4.2 动作单位换算

`recovery_targets.pt` 中保存了 `action_scale`，用于把世界系位移/旋转转换为 LIBERO action 单位：

```python
action_delta = world_delta / action_scale
per_step = action_delta / correction_steps
```

如果缺少 `action_scale`，代码会 fallback 到全 1，这会导致修正量过小、机械臂几乎不动。  
因此恢复目标必须包含 `action_scale`。

### 4.3 与 verifier 的 2D 坐标关系

- correction 本身在 **3D 世界坐标** 中计算，不受图像翻转影响。
- verifier 的 `gripper_xy` 是 2D 投影，必须和 phase verifier / replay 图像使用同一坐标系。
- 当前已经修正：`_project_gripper_xy` 不再对 `gripper_xy` 做 `flip_images` 行翻转，以匹配实际 verifier 图像/视频坐标。

## 5. Eval 开关

- `COSMOS_PHASE_RECOVERY=1`
- 同时需要 `COSMOS_PHASE_VERIFIER=1`

## 6. 当前已知问题

1. PlaceOn 的 phase verifier 仍使用“ gripper 到盘子中心 2D 距离单调减小”的判定，可能把“先到盘子上方再下降”的合法动作误判为 `PHASE_ERROR`。
2. 如果某个 phase 反复触发 recovery，会形成“拉回 -> 跑偏 -> 再拉回”的振荡，需要后续增加 recovery 次数上限或更合理的接近判定。
3. 目前看至少有了这个机制后在LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate任务上成功率从0提升到19/20

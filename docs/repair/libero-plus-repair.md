# LIBERO-plus memory 修复

memory 修复链路（census → 诊断 → t* → prefix 回放 → 对齐 → VLA 续跑）的三件事：
相关代码、运行命令、LIBERO-plus 全 7 维 167 条修复批次的产物路径

## 1. 相关代码

数据流：`census json` → `diagnose_failed.py`（确定性重跑失败 case，产出 `episode.h5` + `phase_record.json` + t*）→ `tta_repair_batch.py`（打包 `repair_<tag>.json`）→ `run_libero_eval.py` repair 模式（settle → 回放 0..t* → memory 对齐 → VLA 续跑）→ `outputs/<tag>.json` + episode hdf5

| 环节 | 文件 | 关键入口 / 行为 |
|---|---|---|
| census 生成 | `memory_system/tta/tools/build_census.py` | 汇总 baseline 失败 init（按扰动维度） |
| 诊断 | `memory_system/tta/diagnose_failed.py` | `load_policy()` / `run_one_init()` / `diagnose_task()`。确定性重放（同 variant/init/seed 逐位复现）：reset → settle 10 → 逐 chunk 调策略（520 步上限）→ 每步喂 `PhaseEventRecorder.observe()`。产出 `episode.h5`（actions T×7 + proprio T×9）、`phase_record.json`、`tstar.png` |
| phase / t* | `memory_system/tta/phase_record.py` | `PhaseEventRecorder`：语义规则命中推进（`memory_system/execute/skill_completion/`）、超时预算 `SKILL_MAX_ACTION_CHUNKS`（Pick 9 / PlaceOn 5 / 其余 3）、失败候选锁定（object_mismatch 等）。`classify_phase()` 归类；`compute_t_star()`：`t* = max(span.start, 事件步 − 16)`（CUT_IN_STEPS=16），首阶段无事件下限 10，无 repairable 事件 → None |
| request 打包 | `scripts/tta_repair_batch.py` | `prepare_inputs_from_diagnosis()` 扫 phase_record.json；`build_request()` 生成 `repair_<tag>.json`（task / tag / phase / t_star / actions=基线动作流）。t* 为 null 时 fallback 10 |
| 修复执行 | `cosmos_policy/experiments/robot/libero/run_libero_eval.py` | 环境变量 `COSMOS_TTA_REPAIR`（request 路径）激活。reset → settle 10 → 回放 `actions[:t*]`（帧捕获进 `_prefix_frames`）→ 主循环 `t` 从 `NUM_STEPS_WAIT`(10) 起步 → `_maybe_start_repair_alignment()` 武装：`InitialAlignmentSelector.select()`（`memory_system/execute/initial_alignment.py`，VAE/位姿相似度选 demo 段 → ready 位姿 +2cm z 偏移）→ cuRobo 规划 + `LiberoJointTrajectoryController`（`libero_joint_control.py`）逐步执行 → `finished` 后交回 VLA 重新 query → 结束写 `COSMOS_TTA_REPAIR_OUT` |

行为要点：

- memory 只提供**选段 + 目标位姿**；接管动作由规划器在线生成，不含 memory 存储的动作块。
- 落盘时 `_prefix_frames`（0..t*−1）prepend 到主循环录制：`frame_indices` 有一次编号跳变（如 `0..93, 10..529`），数组位置连续，帧内容无缺失、无重复执行。
- repair 模式与标准评测的差异仅四处：主循环 `t` 从 10 起步、多一段 prefix 回放、`frame_indices` 一次跳变、结束多写一个 repair 结果 JSON。
- Pick 完成判定（`memory_system/execute/skill_completion/pick.py`）：中位数中心 + 下 10% 分位 Z 抬升 ≥ `min_lift_distance` 两条件 AND，连续 5 帧确认；原 center-only 规则以注释保留在该文件可回滚。这是启发式 RGB-D 修正，不证明物理抓取。

## 2. 运行命令

前置：

1. 输入三件套（按扰动维度替换对应文件）：census json、meta json、summary json
   （robotinit 68 条用的是 `memory_system/pointcloud_action/failed_census_68.json`、
   `memory_system/pointcloud_action/results/tta_screening_meta.json`、
   `memory_system/pointcloud_action/results/tta_repair_68_summary.json`）。
2. 对齐 memory：`skill_memory_test/libero_10/` 下 `segments_ready_fixed16.json`、
   `feasible_recovery_targets.pt`、`ready3d_targets.pt`（缺失则全部 route_failed）。

第一步：在线诊断（每 task 一卡并行，7 task 示例）：

```bash
conda activate cosmospolicy
cd /data1/liu/exp/counterfactual/external/cosmos-policy
for spec in "0:273" "1:274" "2:270" "3:269" "4:271" "5:265" "6:267"; do
  gpu=${spec%%:*}; task=${spec##*:}
  MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  nohup python memory_system/tta/diagnose_failed.py \
    --census memory_system/pointcloud_action/failed_census_68.json \
    --out experiments/tta_phase_check_repro_68 --tasks $task \
    > diag_$task.log 2>&1 &
done
```

健康判断：日志出现 `[<id>] init NNN: success=False candidate=...` 才算稳；`t_star: None` 正常（修复时取 10）。
完成判断：`find <out> -name episode.h5 | wc -l` 等于 case 数（robotinit 68 条 = 68）。

第二步：批量修复（8 卡）。**必须等第一步录满再启动**，否则读不到 h5 直接 FileNotFoundError：

```bash
cd scripts
nohup python3 tta_repair_batch.py --gpus 0,1,2,3,4,5,6,7 \
  --result-dir ../experiments/tta_phase_check_repro_68 \
  --meta ../memory_system/pointcloud_action/results/tta_screening_meta.json \
  --summary ../memory_system/pointcloud_action/results/tta_repair_68_summary.json \
  --work-dir <绝对路径 work-dir> \
  > repair.log 2>&1 &
```

必需环境变量已写死在 `tta_repair_batch.py` 的 `run_case` 里（不能靠外部 export）：

| 环境变量 | 作用 | 缺失后果 |
|---|---|---|
| `COSMOS_INITIAL_ALIGNMENT=1` | 构建修复对齐选择器 | 全部 `route_failed` |
| `SMOKE_DATA_COLLECTION=1` | 训练 hdf5 落盘（会覆盖 `COSMOS_DATA_COLLECTION`） | 无训练数据 |
| `COSMOS_TTA_RGBD=1` | 腕部深度 + 相机参数 | h5 缺 wrist_depth/c2w/camera_K |
| `COSMOS_SKILL_COMPLETION_SHADOW=1` | 实例分割渲染 | `KeyError: robot0_eye_in_hand_segmentation_instance` |

`status` 取值：`reached` / `route_failed` / `prefix_replayed` / `aligned_pending`；单次成功率有 ±2 条运行间波动。

产物（每 case）：

| 产物 | 位置 |
|---|---|
| 修复结果 JSON | `<work-dir>/outputs/<tag>.json` |
| 修复请求 JSON | `<work-dir>/requests/<tag>.json` |
| 子进程日志 / 视频帧 | `<work-dir>/results/<tag>/logs/` |
| 训练 hdf5（全量） | `<work-dir>/results/<tag>/logs/rollout_data/episode_data--<tag>--*.hdf5` |
| 诊断 episode.h5 / phase_record.json / tstar.png | `<result-dir>/<task>/initNNN/` |

筛查（Place 倾倒 + 时长）：

```bash
python memory_system/tta/screen_repaired_place.py \
  <work-dir>/results/<tag>/logs/rollout_data/episode_data--<tag>--*.hdf5 \
  --out screen_<tag>.json [--max-actions N --max-tilt-deg 10]
```

已知坑（全部实际踩过）：

1. nohup 前必须激活 conda，否则 ModuleNotFoundError 秒死。
2. 不用 `pkill -f`（会匹配自己的命令行），先 `pgrep -af` 拿 PID 再 kill。
3. 并发渲染必须 `MUJOCO_EGL_DEVICE_ID=<卡号>`，否则 EGL 上下文挤同一设备报错。
4. episode.h5 可能被清掉：先 `find <result-dir> -name episode.h5 | wc -l` 确认，缺了重录。
5. `SMOKE_DATA_COLLECTION` 会被 smoke 脚本用来重新 export `COSMOS_DATA_COLLECTION`（默认空=清掉），repair 必须设 1。
6. meta 类型配对维度要匹配：robotinit meta 配 `Robot Initial States` + `suite_task_name`；变体 meta 配各自维度，错配报 `Task not found in suite=...`。
7. 实例分割依赖 `COSMOS_SKILL_COMPLETION_SHADOW=1`，不设则 RGBD 采集第一步 KeyError。
8. `robotinit_20.sh` 写死配对参数，变体任务不要经过它。
9. 诊断日志个别 init 复现成成功（census_mismatch）属 policy 波动，记录即可。

chosen/rejected 配对与 manifest 构建（训练侧消费本流水线产物）见 `../dpo/dpo.md` Step 1–2。

## 3. LIBERO-plus 167 条修复批次的产物

| 产物 | 路径 |
|---|---|
| 汇总表（按维度） | `experiments/tta/repair_all7dims/final_table.json` |
| 单 case 修复结果（167 个 json：status / t_star / phase / demos / target / similarity / final_ee / duration_steps / finish_status / task_success） | `experiments/tta/repair_all7dims/<维度>/outputs/<tag>.json` |
| 视频（162 个 mp4，命名 `<维度>__<tag>.mp4`） | `experiments/tta/repair_all7dims/videos/` |
| 诊断源（episode.h5 + phase_record.json + tstar.png） | `experiments/tta/phase_check_all7dims/` |

`final_table.json` 摘要：

| 维度 | trials | baseline 失败 | 修复数 | 修复成功 | repair_rate |
|---|---|---|---|---|---|
| background_textures | 200 | 36 | 39 | 23 | 0.59 |
| camera_viewpoints | 200 | 22 | 24 | 17 | 0.708 |
| language_instructions | 200 | 8 | 8 | 6 | 0.75 |
| light_conditions | 200 | 13 | 13 | 9 | 0.692 |
| objects_layout | 193 | 14 | 8 | 4 | 0.50 |
| sensor_noise | 200 | 6 | 6 | 6 | 1.0 |
| robot_initial_states | 200 | 68 | 69 | 59 | 0.855 |
| **合计** | 1393 | 167 | 167 | **124** | **0.743** |

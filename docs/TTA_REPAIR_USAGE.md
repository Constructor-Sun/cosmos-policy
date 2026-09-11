# TTA 修复流水线使用指南（2026-09-09 修订版）

功能与设计背景见 `docs/TMP.MD`。本指南描述**从零复现"基线失败 → 修复成功"全流程**的完整步骤，
已吸收 2026-09-09 robotinit 68 条全量重跑时踩过的所有坑（见文末"已知坑"）。

## 0. 全流程总览

```
基线失败 census ──> 在线诊断(重录 episode.h5 + 算 t* + 出图)
                        │  experiments/tta_phase_check_repro_68/<task>/initNNN/
                        │    episode.h5 (actions T×7 + proprio T×9)
                        │    phase_record.json / tstar.png(干预时刻图)
                        ▼
              8 卡批量修复(tta_repair_batch.py)
                        │  每 case: 前缀回放 → 对齐注入 → policy 续跑
                        │  同时落盘全量训练 hdf5(图像+腕部RGBD+逐步动作)
                        ▼
              outputs/<tag>.json (reached/route_failed + task_success)
                        ▼
              筛查(screen_repaired_place.py: 时长 + Place 倾倒)
                        ▼
              训练对构建(tools/replay_rejected.py + tools/build_manifest.py)
```

## 1. 前置条件（缺一不可）

1. conda 环境 `cosmospolicy`。**启动任何 nohup 前必须激活**——不激活进程因缺 torch 秒死。
2. 三份静态 JSON（robotinit 68 条用的就是这三个，不用重新生成）：
   - census：`memory_system/pointcloud_action/failed_census_68.json`
     （68 个失败 init：7 个 task × 各自 `fail_init_indices_abs`）；
   - meta：`memory_system/pointcloud_action/results/tta_screening_meta.json`
     （tag → task/init/census_abs_init/suite_task_name）；
   - summary：`memory_system/pointcloud_action/results/tta_repair_68_summary.json`
     （每 case 的 t\*；t\* 为 null 时脚本自动取 10）。
3. 对齐 memory：`skill_memory_test/libero_10/` 下的 `segments_ready_fixed16.json`、
   `feasible_recovery_targets.pt`、`ready3d_targets.pt`（缺失则全部 route_failed）。

## 2. 第一步：在线诊断（重新录制失败轨迹）

`memory_system/tta/diagnose_failed.py` 按 census 确定性重放每个失败 init，
录制 actions+proprio、判定候选阶段与 t\*、生成干预时刻图。**census 只有 7 个 task，
所以录制阶段最多 7 卡并行**（每卡一个 task，写同一 out 目录的不同子目录，无冲突）：

```bash
conda activate cosmospolicy
cd /data1/liu/exp/counterfactual/external/cosmos-policy

for spec in "0:273" "1:274" "2:270" "3:269" "4:271" "5:265" "6:267"; do
  gpu=${spec%%:*}; task=${spec##*:}
  MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
  nohup python memory_system/tta/diagnose_failed.py \
    --census memory_system/pointcloud_action/failed_census_68.json \
    --out experiments/tta_phase_check_repro_68 --tasks $task \
    > repro_68_task$task.log 2>&1 &
done
```

（273/274/270/269/271/265/267 是 7 个 task 的 variant_state，n_fail 合计 68。）

健康判断：日志依次出现 `loading policy...` → `[<id>] task=...` → 每条 init 一行
`[<id>] init NNN: success=False candidate=...`。**打印出 init 行才算稳**；
`UserWarning: Tight layout` 和刷屏的 `fori_loop: 0` INFO 均正常。
`t_star: None` 也正常（无闭爪事件的 case，修复时按约定取 10）。

完成判断：

```bash
pgrep -af diagnose_failed | wc -l                                  # 归零
find experiments/tta_phase_check_repro_68 -name episode.h5 | wc -l  # = 68
```

## 3. 第二步：批量修复（8 卡）

所有必需的环境变量已写死在 `scripts/tta_repair_batch.py` 的 `run_case` 里，
**不需要也不能依赖启动 shell 里的 export**：

| 环境变量 | 作用 | 不设的后果（都踩过） |
|---|---|---|
| `COSMOS_INITIAL_ALIGNMENT=1` | 构建修复对齐选择器 | 全部 `route_failed: selector_or_latent_missing` |
| `SMOKE_DATA_COLLECTION=1` | 训练 hdf5 落盘（smoke 脚本会用它**覆盖** `COSMOS_DATA_COLLECTION`） | config 显示 `data_collection=False`，无训练数据 |
| `COSMOS_TTA_RGBD=1` | 腕部深度 + 相机参数 | h5 缺 `wrist_depth/c2w/camera_K` |
| `COSMOS_SKILL_COMPLETION_SHADOW=1` | 让环境渲染实例分割（ITEM 留空即遥测关闭） | `KeyError: robot0_eye_in_hand_segmentation_instance` |

配对模式自动区分 meta 类型：`meta["task"] == base_task` 的 robotinit meta 走
`Robot Initial States` 类 + `suite_task_name`；带变体后缀的 meta（如 bg5 的
`_table_5`）走 `Background Textures` + 变体任务名。

```bash
conda activate cosmospolicy
cd /data1/liu/exp/counterfactual/external/cosmos-policy/scripts

nohup python3 tta_repair_batch.py --gpus 0,1,2,3,4,5,6,7 \
  --result-dir ../experiments/tta_phase_check_repro_68 \
  --meta ../memory_system/pointcloud_action/results/tta_screening_meta.json \
  --summary ../memory_system/pointcloud_action/results/tta_repair_68_summary.json \
  --work-dir ../experiments/tta_repair_work_repro_68 \
  > repair_repro_68.log 2>&1 &
```

**必须等第一步录满 68 条再启动**，否则 repair_batch 读不到 h5 直接
FileNotFoundError 崩掉。健康判断（启动后几分钟）：

```bash
ls ../experiments/tta_repair_work_repro_68/outputs | wc -l   # 向 68 增长
cat ../experiments/tta_repair_work_repro_68/outputs/<tag>.json  # status 应为 reached
find ../experiments/tta_repair_work_repro_68/results -name "*.hdf5" -newermt today | wc -l
```

`status` 取值：`reached`（对齐到达）/ `route_failed`（选路失败）/ `prefix_replayed`
（未触发对齐）/ `aligned_pending`。单次成功率有 ±2 条运行间波动，
robotinit 68 条历史结果 62/68（summary）～64/68（TMP.MD 复核），预期 60–64。

只跑指定 case：`--tags KSCENE3-init006,... --gpus 0`（串行补跑）。

## 4. 产物清单（每个 case）

| 产物 | 位置 |
|---|---|
| 修复结果 JSON（status/t\*/target/similarity/task_success） | `<work-dir>/outputs/<tag>.json` |
| 修复请求 JSON（phase + t\* + actions，可单 case 复用） | `<work-dir>/requests/<tag>.json` |
| 子进程日志 / 视频帧 | `<work-dir>/results/<tag>/logs/` |
| **全量训练 hdf5** | `<work-dir>/results/<tag>/logs/rollout_data/episode_data--<tag>--*.hdf5` |
| 诊断 episode.h5（actions+proprio，rejected 侧） | `<result-dir>/<task>/initNNN/episode.h5` |
| 诊断 phase_record.json（候选阶段/事件步） | `<result-dir>/<task>/initNNN/phase_record.json` |
| **干预时刻图片** | `<result-dir>/<task>/initNNN/tstar.png`（图上印 t\*/候选阶段/事件步） |

训练 hdf5 内容（`COSMOS_TTA_RGBD=1` 时的全量格式）：

- `primary_images_jpeg` / `wrist_images_jpeg` (T,)，逐帧 JPEG；
- `wrist_depth` / `wrist_segmentation` (T,256,256,1)、`wrist_c2w` (T,4,4)、`camera_K` (3,3)；
- 深度为 MuJoCo 归一化：attrs 里 `depth_metric=False` + `depth_near/depth_far`；
- `actions` (T,7) / `proprio` (T,9) / `frame_indices` (T,)——obs[k] 在 action[k] 之前；
  `chunk_size=16` 开环执行，actions+frame_indices 可无损还原每个 action chunk；
- `future_primary/wrist_images_jpeg`——policy 每次 query 的 chunk 级未来预测；
- attrs：`success`、`task_description`、`place_start`、`release_frame`（筛查用，
  -1 表示未检测到事件，可离线从 actions 重算，见 §6）。

## 5. 筛查：时长 + Place 倾倒

`memory_system/tta/screen_repaired_place.py` 用腕部 RGBD 比对 Place 开始帧与释放帧
的目标物体倾角（点云反投影世界系拟合法向，阈值默认 10°），外加 `--max-actions` 时长上限：

```bash
python memory_system/tta/screen_repaired_place.py \
  <work-dir>/results/<tag>/logs/rollout_data/episode_data--<tag>--*.hdf5 \
  --out screen_<tag>.json [--max-actions N --max-tilt-deg 10] [--target-id N]
```

`place_start/release_frame` 优先读 h5 attrs；采集侧两级自动标注：
①阶段切换钩子检测进入 PlaceIn/PlaceOn；②repair 模式不经过阶段推进，落盘时从
执行动作流的夹爪开合事件回退推导（最后一次闭合→最终张开）。`target_id` 不自动写
（不传时用整个分割 mask，对释放帧场景基本等价）。

时长/停顿/抖动筛查工具 `quality_screen.py` 源码已丢失，仅存
`memory_system/offline/__pycache__/quality_screen.cpython-310.pyc`
（可直接 import），历史输出在 `archive/traj_postfix_20260909/outputs/traj_postfix/`。

## 6. 训练对构建（chosen / rejected）

- chosen：修复 `success=True` 的 hdf5 即训练正样本（上表全量格式，
  `memory_system/tta/dataset.py` 直接消费）；
- rejected：诊断 h5 无图像，用 `memory_system/tta/tools/replay_rejected.py`
  按存储动作确定性重放取图（不跑 policy，很快）；
- 配对 manifest：`memory_system/tta/tools/build_manifest.py`（校验字段对齐、
  chosen 必须 success=True）。

## 7. 已知坑（全部实际踩过，按出现顺序）

1. **nohup 前必须激活 conda**：否则 import torch 秒死，日志只留 ModuleNotFoundError。
2. **不要用 pkill -f**：模式串会匹配自己的命令行。先 `pgrep -af` 拿 PID 再 kill。
3. **并发渲染必须 `MUJOCO_EGL_DEVICE_ID=<卡号>`**：否则 EGL 上下文挤同一设备报错。
4. **原始录制可能被清掉**：census/meta/summary 在，episode.h5 不一定在——
   永远先 `find <result-dir> -name episode.h5 | wc -l` 确认，缺了就走 §2 重录。
5. **shell 变量覆盖**：`run_libero_smoke_test.sh` 会用 `SMOKE_DATA_COLLECTION`
   重新 export `COSMOS_DATA_COLLECTION`（默认空，等于清掉）。给 repair 用必须设
   `SMOKE_DATA_COLLECTION=1`。
6. **配对维度要匹配 meta**：robotinit meta 配 `Robot Initial States` +
   `suite_task_name`；变体 meta 配各自维度。错配报
   `Task not found in suite=...`。
7. **实例分割依赖 SHADOW 开关**：不设 `COSMOS_SKILL_COMPLETION_SHADOW=1` 则
   obs 无 `robot0_eye_in_hand_segmentation_instance`，RGBD 采集第一步即 KeyError。
8. **`robotinit_20.sh` 写死配对参数**，变体任务不要经过它。
9. **诊断日志里 `census_mismatch`**：个别 init 复现成成功，属 policy 波动，
   单独记录即可。

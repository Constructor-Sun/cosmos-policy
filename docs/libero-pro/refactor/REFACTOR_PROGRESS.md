# LIBERO-PRO 重构进度（截至 2026-09-23）

对应讨论文档：`docs/libero-pro/LIBERO_PRO_REFACTOR_DISCUSSION.md`。
本文只记**已完成的事实、验证方式、实测结果与已知分歧**，不记建议。

## 1. 总览

- §2.1–§2.10 十类**都已写出**，落在新顶层包 `memory/`（8252 行，其中构库链 4538 行）。
- 新入口：`cosmos_policy/experiments/robot/libero/run_libero_eval_memory.py`（1214 行），
  与新的脚本目录 `scripts_memory/`（harness 1275 行 + runner 89 行）。
- **旧路径一行未改、未删**：`run_libero_eval.py`（2137 行）、`memory_system/`、`scripts/` 全在；
  运行时默认仍未切到新树（`scripts_memory/` 是并行入口，不是替代）。

## 2. 新树清单（`memory/`）

| 文件 | 行数 | 对应 |
|---|---|---|
| `types.py` | 187 | §3 三个数据结构 + `PhaseSpec`/`PhaseSpan`/`PhaseDecision`/`CameraParams`/`CaseSpec`/`Run`/`RunStep`/`RunEvent`/`ActionSource` |
| `retriever.py` | 183 | §2.4 MemoryRetriever |
| `run/planner.py` | 136 | §2.1 TaskPlanner（类 + 两个方法） |
| `run/scheduler.py` | 187 | §2.7 RunScheduler |
| `run/recorder.py` | 203 | §2.6 RunRecorder |
| `check/phase_check.py` | 394 | §2.2 PhaseChecker |
| `check/repair_check.py` | 146 | §2.3 RepairChecker |
| `execute/executor.py` | 424 | §2.5 InterventionExecutor |
| `execute/{curobo_planner,curobo_trajectory,surface_obstacles,urdf_depth_filter,controller}.py` | 1489 | 照搬的规划器/控制器（逐字，仅改 import；controller 与 curobo_trajectory 做过去重） |
| `adapters/environment.py` | 167 | §2.8 EnvironmentAdapter |
| `adapters/policy.py` | 198 | §2.9 PolicyAdapter |
| `build/**` | 4538 | §2.10 构库链（两个子目录逐字搬入，内部 import 已改为 `memory.build.*` 自我引用） |

非构库部分 3714 行，其中 1489 行是照搬的规划器/控制器。

## 3. 已完成、已跑通的验证（离线）

| 类 | 验证方式 | 结果 |
|---|---|---|
| TaskPlanner | 与 `plan_from_bddl`、与三个冒烟 case 的 `request.plan` 对比 | 全等 |
| PhaseChecker | 与旧 `TimedSkillCompletion` 家族逐步对拍（4 技能 × 多配置） | 0 不一致 |
| RepairChecker | 与 118 条冻结 request 逐字段对比（swap 100 + task 18） | 全等；t* 三条 fallback 规则与数据吻合 |
| MemoryRetriever | 三个库全部 73 个 (skill,args) 组的首条与旧实现对比 | 全等 |
| InterventionExecutor | 段回放（timegrip/rawactions × 接近步数）与旧 `_ReplayController` 逐帧对比 | 0 不一致 |
| RunRecorder | 合成驱动，§2.6 五条不变量（三层判定/两份 holding/chunk 边界/未采集/索引） | 逐条对上 |
| EnvironmentAdapter | 118 例的 `suite+task → bddl_file` 与 meta 对比；BDDL 指令 | 全等；task 套件 18/18 指令与"名字推导"不同 |
| PolicyAdapter | 假模型驱动控制流（seed 序列/选条/hook 装卸/latent） | 通过 |

## 4. 新入口与运行方式

- `scripts_memory/run_libero_smoke_test.py`：与旧 harness 逐字相同，唯二改动是
  ① eval 模块由 `COSMOS_EVAL_MODULE` 选择（默认 `...run_libero_eval_memory`，设成 `...run_libero_eval` 即切回旧模块）；
  ② `run_single_case.py` 的 cwd 指向 `scripts_memory/`。
- 例：
  `python scripts_memory/libero_pro/run_single_case.py <TAG> --gpu N --meta <abs meta> --requests-dir <abs dir> --out-dir <abs dir>`
  （注意：tags 是**位置参数**；三个路径必须是**绝对路径**，因为 harness 的 cwd 是 `scripts_memory/`。）
- 逐帧对拍的产物：`COSMOS_DATA_COLLECTION=1` 写出的 `episode_data--*.hdf5`（`actions`、`frame_indices`）；
  视频在 `memory/experiments/rollouts/<COSMOS_ROLLOUT_SUBDIR>/`（`scripts_memory/rollouts` 是指向它的软链；
  不设该变量时两次运行会同名覆盖）。
- **所有实验产物统一放 `memory/experiments/`**（不要在 `external/counterfactual/` 下另开目录）：
  `--out-dir` 指到 `memory/experiments/<名字>`，视频子目录用 `COSMOS_ROLLOUT_SUBDIR`。
- GPU 选择：空余 ≥10GB 即可用（按空余降序取）。

## 5. 实测结果

**纯 VLA 路径（不设 repair 变量）：逐帧完全一致**（4 次 A/B：actions 与 frame_indices 的最大绝对差 0.0）。

**修复路径（`COSMOS_TTA_REPAIR` + `TTA_PHASE_LOOP=1` + `SKILL_READY_MEMORY=timegrip`）**

- swap 冒烟三例（重构版）：`LRSCENE2-init004` 成功、`LRSCENE5-init000` 成功、`KSCENE8-init004` 失败
  （该例旧模块同样失败 → 不是回归）。
- `LRSCENE2-init004` A/B：`t*` 都 38、**首次干预都在 frame 138 结束**（一致）；
  之后阶段结束帧 138/237/353（新）vs 138/281/439（旧），采集帧 3 vs 92，两边都成功。
- task 套件 18 例：**重构版 2/18，旧模块 9/18**（8 卡并行，同 case 同参数）。

## 6. 已知分歧与根因（定位到的）

1. **"无记忆 → 交给 VLA"这条回退在重构版里丢了**（5 例）：
   旧模块在该阶段记忆库无记录时 `select()` 返回 None（`route_failed / no_memory:<skill>`）→ 不干预、VLA 跑完；
   重构版 `_arm` 里 `retriever.retrieve()` 抛出的 `KeyError` 冒到 `run_episode` 的 `except` → **episode 被中止**。
   证据：`Episode error: "no PlaceIn record for {...}"`。
   **对应补丁曾加上、随后按要求撤销**（新模块现状=无该回退）。
2. **纯 VLA 段本身也存在分歧**：`SSCENE1-init000` 两边都是 `route_failed/no_memory:PlaceIn`（都不干预），
   旧模块成功、重构版失败 → 分歧在 prefix 回放/切点状态或 VLA 段，未定位。
3. **后续阶段节奏错开**：`LRSCENE2-init004` 首次推进帧一致、之后不同；嫌疑按序为
   ① 开窗/喂判定的帧序差一帧；② 执行器预算与旧 `_alignment_steps_remaining` 的换算差一帧；③ Place 段（force_lift）收尾时机。

## 7. 有意差异与"记一行"清单

- 接口偏离文档签名：`TaskPlanner.plan(bddl_path)`（未引入 `Task`）；`MemoryRetriever.retrieve(phase)`（不吃 request/observation/env）；
  `RunScheduler.begin(*, task, phases, request)`（`CaseSpec` 曾为留白）、`apply(verdict, observation=None)` 且推进时返回决定字符串；
  `RunRecorder.record_step` 多几个关键字参数（chunk_start/control_space/collected/memory_frame/data_row）；
  `InterventionExecutor` 的段策略/预算进 config；`PolicyAdapter` 的 `capture_latent` 为构造参数。
- 新类型：`ActionSource`/`CaseSpec`（4 字段）/`Run`/`RunStep`/`RunEvent`/`ReplaySegment`/`ExecutionResult`/`MemoryCandidate`。
- `Run` 的记录器不填证据字段（交互事件步/物体匹配等仍由离线 `PhaseEventRecorder` 产出）。
- `capture_latent` 目前只按 `cfg.save_vector_db`（旧模块在修复运行里也靠首查捕获 latent 才武装；新链不依赖它）。
- 前缀帧未并入 `collected_data`（修复模式下 hdf5 会比旧模块少 prefix 行）。
- 未接：plus 的 initial alignment（`COSMOS_INITIAL_ALIGNMENT` 无请求那条）；place 精细对齐/held-object/`coordinator_factory`（按决定不做，但被开启时新模块会静默忽略）。
- `TASK_MAX_STEPS`/`validate_config`/`unnorm_key`/9 个 cfg 字段已按当前版本补齐（否则 harness 构造 cfg 即崩）。
- `CameraParams`、`PhaseSpec`、`PhaseSpan` 在新旧树各有一份（同名不同一）；`memory.build.*` 内部仍引用未搬的 `pointcloud_action.config`/`schema`/`memory_system.geometry`。
- 讨论文档（`LIBERO_PRO_REFACTOR_DISCUSSION.md`）尚未更新本轮的当场决定。

- **采集流按动作空间分，不按控制权分**（2026-09-25）：`collected_data` 的 `primary_images`/`wrist_images`/`proprio`/`actions` 现在共用一个门 `cfg.data_collection and len(action) == 7`，在动作确定之后一次提交一行；`driver.prefix_*` 在循环前种进同一批表，`frame_indices` 也按同一个门筛。此前 obs 每步都收、actions 只收 `is_vla_action` 的步，两边行数本就不一致（§5 里 LRSCENE2-init004 的「采集帧 3 vs 92」即此），且「整集无 VLA 动作」时空表 `np.stack` 直接崩。**代价：8 维关节帧被跳过，导出流的 `frame_indices` 会 t 跳号**——这些帧（开局对齐、APPROACH 的关节轨迹，单轮实测 243 次接管）仍保留在 Run 记录里并标「未采集」，只是不进 7 维训练流；要 t 连续的导出需另设方案。

## 7.1 产物现状（2026-09-23 13:37）

- 18 例两批的原始产物（`outputs/*.json`、`results/`、`gpu*.log`、hdf5）**已丢失**（目录曾被清空）；
  结论可从 `memory/experiments/rollouts/` 里的视频文件名复原：旧模块 `old_task/` 36 个文件中 9 个 `success=True`（=9/18），
  新模块 13:08–13:11 那批 18 个中 2 个（=2/18），与当时打印的逐例表一致。
- 仅 `memory/experiments/verify_fix/`（撤销回退前的 3 例验证）保留完整产物（18 个文件）。
- 如需原始 JSON/hdf5：按第 4 节重跑两批（18×2 例，8 卡约 15–20 分钟），`--out-dir` 指向 `memory/experiments/` 下。
- `memory/experiments/` 下另有 7 个空目录（`cases_task`、`cases_task_old`、`cases_new`、`cases_old`、`ab_repair`、
  `ab_repair_video`、`ab_block1`）：只是被清空后的壳，没有内容，保留是为了留痕；不需要可直接删。
- **当前新模块的代码状态 = §5 里 2/18 的那一版**：`_arm` 里没有"无记忆回退"（§6.1 的那条补丁加了又按要求撤销），
  因此 task 套件上"记忆库缺记录"的阶段仍会抛 `KeyError` 并中止该 episode。

## 8. 未做

- 冒烟三例未全过（TurnOn 例两边都失败）。
- `libero_10_swap` 全套（100 例）未跑。
- 调用点切换：批处理脚本、`tta_repair_batch.py` 等仍指旧路径；旧 eval 文件仍在被默认使用（除 `scripts_memory/`）。
- §4 的"运行时不再 import scripts/ 下模块"、真机替换点集中等条目未动。
- 13 条手工 place 记录的处理未动。

## 9. 变更记录（本轮）

| 时间 | 变更 |
|---|---|
| 13:32 前后 | `external/counterfactual/` 下我的实验目录被清空（非我的命令）；18 例两批的 JSON/hdf5/log 丢失 |
| 13:37 | 所有实验产物归置到 `memory/experiments/`；`scripts_memory/rollouts` 改为指向它的软链 |
| — | `_arm` 的"无记忆回退"补丁：加上 → 3 例验证（`route_failed` 生效但结果仍失败）→ 按要求撤销 |
| — | 本文件建立（§1–§8），随后补 §4 路径、§7.1 产物现状、§7.1 空目录说明 |

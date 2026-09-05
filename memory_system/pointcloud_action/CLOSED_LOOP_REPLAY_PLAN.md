# 闭环回放改造计划（CLOSED-LOOP REPLAY PLAN）

> 状态：待执行（本文档为执行前的完整设计与验收方案）
> 约束：**全程零删除**——不删除 memory_system/ 及其父目录下的任何文件。
> 所有改动均为"新增文件 + 原位修改（修改前备份）"，memory 数据文件（pointcloud_action_memory.pt）本计划不触碰。

---

## 0. 背景与已证事实（本计划的经验依据）

| # | 事实 | 证据 |
|---|---|---|
| F1 | oracle 点云提取的 geom 查找错位已修复；10 个 plus-only 物体云正常 | /tmp/plusonly_oracle_sanity.json |
| F2 | 记忆库 complete 云曾被旧 bug 污染，已外科手术修复（781/781 条，23/23 自匹配 d=0） | pointcloud_action_memory.pt.pre-surgical-20260905_224349 |
| F3 | 部分 chunk 截取不含提升段 → 回放+hold 无法满足 2cm 判据 | milk 合爪后 EE z 仅 +0.8mm |
| F4 | 已实装"回放后脚本化提升段"（run_local_pick `post_replay_lift_steps=12`），milk/orange_juice 转为成功 | rollouts/a_class_liftfix_videos/ |
| F5 | 回放对 ready 关节构型鲁棒：4 种构型下实现轨迹物系差仅 2-4mm | /tmp/posture_test.py 输出 |
| F6 | 抓取结果由 chunk 确定性决定：moka demo_1 4/4 构型成功，demo_0 4/4 失败（壶上抓取点相差 4.3cm） | 同上 |
| F7 | 开环回放的合爪时刻 EE 误差 ~1cm；失败 chunk 合爪后手被壶挡住、壶被推走 2cm | /tmp/trace_moka.py 逐步追踪 |
| F8 | 闭环航点跟踪存在两类卡死：P 控制器极限环（v1 milk 卡 321 步）、内部 max_steps=96 不足（GOAL_NOT_CONVERGED） | /tmp/closedloop_test*.py |

**当前基线**：coverage 34 条 = 13 通过；提升段补丁后推定 15/34。
**剩余 A 类失败（本计划目标）**：akita_black_bowl ×3、porcelain_mug、white_yellow_mug、wine_bottle、moka_pot（外加 cream_cheese libero_goal 边缘波动 1 例）。
**精度要求（来自数据）**：示教自身合爪时刻偏差 ~10mm 且成功/失败分界在 ~10mm → 目标合爪时刻 EE 误差 ≤ 5mm（严格优于示教）。

---

## 1. 目标与非目标

**目标**：让回放在合爪瞬间把手送到演示合爪位形 ±5mm 以内，并用接触检测替代时间触发，使 5 个根因 2 案例中 ≥3 个翻转为成功。

**非目标**：
- 不解决 B 类（plus-only / 无自物演示物体的跨形状迁移）——那是重试/抓取适配问题；
- 不修改记忆数据、不重建 memory；
- 不改变 open_loop 模式的任何现有行为（默认值保证）。

---

## 2. 改动清单（4 个原位修改 + 若干新增，零删除）

### 改动 1：`memory_system/execute/curobo_trajectory.py` — `WaypointPoseController` 修卡死

仅此类被闭环回放使用（ready motion 走的是 LiberoJointTrajectoryController，不受影响）。

| 项 | 现状 | 改为 |
|---|---|---|
| 航点卡死 | P 控制器极限环：误差永远在 eps 外徘徊 → index 永不前进 | 新增 `dwell_timeout=40`：单航点驻留超 40 步则强制推进，记录"强制推进时的最终误差" |
| max_steps=96 | 42 航点平均不到 2.3 步/个，必然超时 | 新参数 `max_steps=None` 时取 `len(waypoints) × dwell_timeout` |
| 仪表化 | 无 | 新增 `self.waypoint_errors`（每航点最终误差列表）与 `self.forced_advances` 计数 |

**兼容性**：新增参数全部带默认值，且 `dwell_timeout` 仅在显式传入时启用强制推进（默认 None = 行为不变）。现有调用方零影响。

备份：`cp curobo_trajectory.py curobo_trajectory.py.bak-clplan`

### 改动 2：`memory_system/pointcloud_action/eval/local_pick_core.py` — 新增闭环回放分支

`run_local_pick` 增加 `replay_mode` 参数（None → 读 config）。当为 `"closed_loop"` 时，回放段替换为：

```
1. 参考轨迹映射：ee_pose_object_sequence → 世界系 → 6D 状态序列（matrix_to_ee_states）
2. WaypointPoseController 跟踪（带改动 1 的 dwell timeout）
3. 每步合爪指令 = 进度同步（手位置在参考轨迹上的最近点所对应的 gripper_sequence 符号）
   —— v1 的 index 挂钩 bug 已排除（while 循环跳航点 / 卡死两种失效均已定位）
4. 合爪确认：越过合爪点后，连续 M=5 步 is_grasping 为 True → 视为"抓住"，
   跳过剩余航点直接进入提升段；30 步未确认 → 继续走完（最终判定如实报失败）
5. 提升段（已有）→ 稳定保持（已有）→ 判定（已有）
```

**新增结果字段**（仪表化，每次运行自带证据）：
`replay_mode`、`close_time_ee_error_mm`（合爪确认时刻手与参考合爪位形的偏差）、
`waypoint_max_final_error_mm`、`grasp_confirmed`、`grasp_confirmed_step`。

### 改动 3：`memory_system/pointcloud_action/config.py` — 模式开关

```python
REPLAY_MODE = "open_loop"   # "open_loop" | "closed_loop"（默认 open_loop，验证完成前不切换）
```

### 改动 4：新增验证脚本（不进 pipeline，放 /tmp 或 scripts/ 均可）

- `V1` 航点控制器收敛测试脚本
- `V2/V5` 复用已有仪器化脚本（trace_moka.py / diag_a_class.py 模式）

### 明确不触碰

- `pointcloud_action_memory.pt`（及任何 .pt/.bak）
- open_loop 代码路径的现有行为
- B 类相关逻辑、plus-only 相关逻辑

---

## 3. 验证方案（V1-V6，每层带量化通过标准与判决规则）

### V1 航点控制器收敛性（单元级，~10 分钟）
- 方法：moka demo_1 参考轨迹（42 航点）喂给修好的控制器，空爪、无接触，逐步记录；
- 通过：总步数 ≤ 42×40；无死锁；每航点最终误差 ≤5mm 或被记录为强制推进（≤1cm）；
- 失败含义：参考轨迹本身在该上下文不可行 → 停止本计划，转向轨迹可行性分析。

### V2 回放保真度（决定性实验，~2 分钟）
- 方法：闭环模式重跑 moka demo_1（成功）与 demo_0（失败），仪器化同 trace_moka.py；
- 通过：**合爪确认时刻 EE 误差 ≤5mm**（对照：示教 ~10mm、v2 闭环 ~10mm）；
- **判决规则**：
  - demo_0 翻转成功 → 精度假设成立 → 继续 V3；
  - demo_0 仍失败但合爪误差 ≤5mm → **精度假设证伪**（问题在抓取本身，非精度）→ 停止调控制器，转向"跨演示重试"路线；
  - demo_1 失败 → 新模式有 bug → 修复前不继续。

### V3 确定性守恒（~10 分钟）
- 同案例同 seed 跑 3 次，结果（grasp/lift/成功）必须逐项一致；
- 失败含义：引入了隐藏非确定性（渲染/EGL/物理线程），必须先消除。

### V4 无回归（~20 分钟）
- 当前全部通过案例（milk、orange_juice、alphabet_soup、black_book、butter、chocolate_pudding、ketchup、tomato_sauce、salad_dressing、cream_cheese ×2 = 11 条）在 closed_loop 模式下重跑；
- 通过：11/11 仍成功（cream_cheese libero_goal 的已知边缘波动可单独标注）；
- 失败含义：闭环模式有回归性 bug → 修复前 REPLAY_MODE 保持 open_loop。

### V5 端到端（~15 分钟）
- 5 个根因 2 案例闭环重跑；
- 判决：≥3 个翻转成功 → 精度路线成立，进入全量 coverage；≤1 个 → 按 V2 判决转向。

### V6 视频抽验
- 每个翻转案例存视频（含合爪瞬间），人工确认接触补偿生效。

---

## 4. 执行顺序、预计耗时与止损点

```
改动1 → V1 ──失败──→ 止损：轨迹可行性分析
              │通过
改动2/3 → V2 ──demo_0 仍失败(≤5mm)──→ 止损：转向跨演示重试（精度假设证伪）
              │demo_0 翻转
V3 → V4 ──回归──→ 止损：修 bug，默认不切换
              │通过
V5 → V6 → 全量 coverage 重跑（官方数字）→ 决定是否将 REPLAY_MODE 默认值切为 closed_loop
```

预计总耗时：代码改动 ~30 分钟，V1-V6 ~60 分钟，全量 coverage ~15 分钟。

---

## 5. 回退方案

- `REPLAY_MODE` 默认 `open_loop`：任何时候改回默认即恢复现状；
- `curobo_trajectory.py` / `local_pick_core.py` 修改前备份为 `*.bak-clplan`；
- 新增参数全部带向后兼容默认值（不传参 = 原行为）；
- memory 数据文件本计划不读不写（已修复版本 + 既有备份保持不变）。

## 6. 成功标准总表（需求 → 证据）

| 需求 | 证据 |
|---|---|
| 航点控制器不卡死 | V1 步数有界 + waypoint_errors 日志 |
| 合爪精度 ≤5mm | V2 close_time_ee_error_mm |
| 精度假设成立/证伪 | V2 判决规则 |
| 结果可复现 | V3 三次一致 |
| 无回归 | V4 11/11 |
| 根因 2 案例翻转 | V5 ≥3/5 + V6 视频 |
| 官方 coverage 数字 | 全量重跑 jsonl |
| 零删除约束 | 本计划全部为新增/原位修改 + .bak 备份 |

---

## 7. 无人值守执行安全协议（夜间 /goal 运行适用）

### 7.1 绝对禁令（任何情况下不执行，无例外）

1. **任何形式的删除**：`rm`、`unlink`、`shutil.rmtree`、`os.remove`、`git clean`、`find ... -delete`、覆盖式重定向到既有文件（`> 既有文件`）；
2. **任何 git 写操作**：`add` / `commit` / `checkout` / `restore` / `reset` / `stash` / `push` / `pull` / `branch` / `tag`。只允许只读的 `status` / `diff` / `log`；
3. **任何环境变更**：pip / conda 的 install/uninstall/upgrade，修改 shell 配置文件，`sudo`，杀死非本人启动的进程；
4. **超出白名单的文件修改**：只允许修改计划第 2 节列出的 4 个文件（修改前先 .bak）+ 新增文件；不触碰仓库其他目录、不触碰其他项目；
5. **覆盖既有结果**：`rollouts/` 下只使用带新后缀的新文件名。

### 7.2 资源防护（防卡死/防拖垮服务器）

1. **所有 Bash 调用带 timeout**（单次 ≤600s）；长任务后台运行 + 轮询，超时即杀（只杀自己启动的进程）；
2. **所有仿真循环步数有界**（每条 rollout ≤200 步 + 提升段 12 + 保持 20；闭环 ≤600 步）；控制器含驻留超时，无无限循环；
3. **同时只存在一个环境实例**，`env.close()` 放 finally —— 避免 EGL 上下文泄漏；
4. 闭环回放为 CPU + EGL 渲染，**不占用训练 GPU**、无大显存任务、无多进程并行需求；
5. 开工前只读检查 `df -h`（磁盘剩余）与 `nvidia-smi`（如有必要），异常则暂停并记录。

### 7.3 无人值守执行协议

1. **开工快照**（只读）：`git status --porcelain` 输出 + 待修改 4 文件的 sha256 写入 `/tmp/clplan_pre_state.json`——任何意外改动事后可检出；
2. **增量落盘**：每个 V 阶段的结果 JSONL / 日志立即写盘（沿用既有模式），中断不丢已完成部分；
3. **判决规则机械化**：V1-V6 的通过/失败判据已在第 3 节量化，无人值守时按判决规则自动停止或转向，**不做计划外的即兴操作**；
4. **阶段检查点**：上一阶段通过才进入下一阶段；到达止损点即写报告收尾；
5. **收尾报告**：运行结束（无论成败）写 `rollouts/clplan_overnight_report.md`：执行了哪些阶段、每阶段结果与判决、修改了哪些文件（含备份路径）、遗留问题。

### 7.4 残余风险（诚实声明）

1. **驱动级 EGL/图形死锁无法 100% 排除**——表现为进程卡住，被 timeout 杀掉后重试；普通用户进程卡死不影响系统健康；
2. **MuJoCo 物理发散（NaN）**——表现为该条 rollout 失败，不损坏文件或环境；
3. 磁盘写满等**环境既有问题**——开工前检查缓解，无法根治；
4. 权限系统兜底：未获批准的命令只会执行失败（fail-safe），不会被静默运行。

### 7.5 建议用户在夜跑前自行决定的两件事

1. **是否打一个 git 快照**：最稳妥的保险是夜跑前 `git add -A && git commit`（由用户自己执行，或明确授权后由我执行——默认我不做任何 git 写操作）；
2. **/goal 预算**：设置 token/时间上限，配合第 4 节的止损点，最坏情况是"报告写明停在哪一层"。

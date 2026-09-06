# Pick 记忆复用：问题、修复与遗留

> 日期：2026-09-06 ｜ 本文是唯一权威文档。目录下其他 5 份历史 MD 已移入 `archive/`（副本亦存于
> `cosmos-policy_snapshot_20260906.tar.gz`），处置（保留/删除）待维护者确认。
> 功能边界一句话：从 pointcloud action memory（LIBERO-90 演示构建的 skill chunk 库）中按物体点云
> 检索技能段并重放，在 LIBERO 单物体 coverage 基准（26 类物体 × 4 suite = 34 案例）上评测。

---

## 1. 问题

coverage 评测基线 15/34。失败分两类：
- **A 类（15 组）**：记忆库存在该物体自物演示（检索距离≈0）仍失败——执行/锚定问题；
- **B 类（11 组）**：plus-only 物体无自演示（距离 0.6–3.7）——检索/迁移问题（未处理）。

A 类最终拆解为**四个独立机制**（§2）；修复后路线 C 达 **23/34（seed 0），4 seeds 21–24（均值 22.5）**。

---

## 2. 根因与机制（现象 → 测量 → 结论）

### 2.1 锚定时机（6 组失败：akita×3、porcelain、white_yellow、wine）
reset 后立即锚定时物体尚未稳定：实测 reset 后首步弹跳/下落 12–72mm，第 3–6 步物理静止，第 12–15 步可在线确认；
coverage BDDL 的初始摆放本身有缺陷——salad/ketchup 生成于地板下方 37.7mm（穿模，接触穿透直接测量），其余部分悬空。
轨迹锚定在过期位姿上 → 合爪悬空。**修复**：ready motion 后重锚定（`REANCHOR_AFTER_READY`）或锚定前 settle
（`--settle-steps`）——两者等效（12/12 vs 12/12），只等待不重锚定无效（0/12）。

### 2.2 moka：open-loop 实现滞后（1 组失败）
open-loop 重放的指令流是实现位移的 4.4 倍，累积滞后使合爪时刻 EE 偏差 ~1cm（D6）——该 chunk 需要到位精度。
**修复**：`wp_time` 回放（航点逐点收敛跟踪 + 时间索引闭爪）。2×2 消融证明有效成分是航点收敛而非位置触发闭爪；
多 seed 下 moka 4/4 经重试翻转（SSSS）。

### 2.3 salad：侧捏 chunk × 实现滞后 × 摆放穿模的三重叠加（1 组失败）
该 chunk 是**水平侧捏**：演示合爪时 EE 在物体轴线旁 9.4cm、origin 下方 1.2cm。open-loop 实现滞后使合爪比意图
点早 ~11cm。**基线的"成功"是双重误差抵消**：旧锚点（穿模位）偏低 37.7mm + 滞后 11cm，净效果恰好落在瓶身宽处
（直径 6.4cm）→ 捏住（68 次手指接触）。修正锚定后滞后暴露：合爪落在瓶颈区（直径 2.5–3.5cm）+ 9.4cm 侧偏 →
手指合拢**零接触**（gripper_qpos 0.0011 全闭、物体位移 0）。**摆放修复无效**（D3 干预：消除穿模后 V1/V2 仍
失败）——滞后是执行器层的，与初始穿透无关。

**定点闭爪实验（进一步定位）**：将夹爪以 **0.939mm / 0.217°** 精度驱动到记录的 close_idx 位姿（重锚定坐标，
approach 收敛 43 步、0 强制推进），**静态闭爪 30 步——全程零手指接触、物体位移 0、从未 grasping**。
结论：**记录的合爪帧位姿本身不是静态可抓取位姿**——演示的抓取是"运动中合爪"（合爪指令触发后手继续沿路径
运动，手指在运动中扫过瓶身才发生接触）；任何"到达即闭爪"的重放（无论锚定正确与否）都会捏空。

### 2.4 ketchup：wp_time 的合爪/保持语义边界（1 组失败）
V1（open-loop）成功、V2（wp_time）失败；消除穿模后 V2 能举 21.4cm 但 hold 阶段失去夹持。与 salad V2 同族：
到达捏点后的闭爪/保持语义尚未解决。**路线 C 下被 V1 尝试覆盖，不阻塞。**

### 2.5 穿模与 done：两个环境事实（非执行器问题）
- 初始穿模 37.7mm 来自**上游 LIBERO floor placement/对象资产约定**（源任务环境一致），非生成器引入；
  干预证明消除它不改变任何 Pick 结论 → 不修（D3 关闭）。
- 生成 BDDL 的 goal 恒真（生成器注释明示 success 由 Pick evaluator 判定）→ **done 不可靠，harness 不依赖**
  （执行器曾依赖 done 导致误杀，已撤销）。

---

## 3. 修复决策与理由（为什么这样做）

| 决策 | 理由 |
|---|---|
| 重锚定放在 ready motion 之后，而非只做 settle | 与 settle-first 等效（12/12），且不携带调参常数（settle 时长）；等待不锚定无效 |
| waypoint 模式必须重映射参考路径 | 否则控制器忠实跟踪悬空旧路径（Run 2 9/34 → V2 21/34 的教训）；不变量测试锁定该合同 |
| wp_time = 航点跟踪 + 时间索引闭爪 | "准"（航点收敛）与"顺"（open-loop）的结构性取舍：moka 需要准，ketchup 类需要顺——失败集互补 |
| 路线 C 重试（V1 失败 → V2 重试） | 失败集互补的并集；D1 口径：最终物体状态成功（TIMEOUT 辅助成功计入，用户确认） |
| 不修初始穿透 | 上游 LIBERO 异常；干预证明不影响 Pick 结论；修了会破坏与历史数字的可比性 |
| 不依赖 done | 恒真 goal 是有意设计；曾依赖导致误杀正常回放（已撤销） |

---

## 4. 验证

| 策略 | runner 参数 | 结果 | 验证 |
|---|---|---|---|
| 基线 | `--no-reanchor` | 15/34 | seed 0 |
| Run 1 | （config 默认） | 19/34 | 1 次 |
| V1 | `--settle-steps 15 --no-reanchor` | 21/34 | 正式 runner + 诊断管线各 1 次 |
| V2 | config `REPLAY_MODE="wp_time"`（正式 runner 无此 CLI 参数；诊断脚本 p1_full_regression.py 有） | 21/34 | **3 次逐位一致** |
| **路线 C** | `--settle-steps 15 --retry-wptime` | **23/34** | seed 0 两次一致 + **4 seeds 21–24** |

多 seed 稳定性（**136 个 case-seed 对，其中 52 次触发重试、共 188 次环境执行**）：**21 例稳定成功（4/4，含全部 A 类）**、3 例翻转敏感（ketchup 3/4 + 2 例 B 类
摆放敏感成功）、10 例稳定失败（9 B + salad 0/4）。重试机制 4 seeds 一致交付 moka（×4 翻转）。
**边界**：结论限于"声明区域内 ±2.5cm 摆放抖动"（libero_object floor 场景实际仅 5–10mm，检验力有限）、单场景
单实例、单仿真环境；非任意布局，B 类迁移未测。另两条已知边界：**同进程重试存在状态残留**（重试结果非独立
进程逐位复现，veg_juice 案例）；**恒真 goal 使 done/terminated 语义不可靠**（§2.5，harness 不依赖 done）。

---

## 5. 代码位置与开关

| 关注点 | 位置 | 开关 |
|---|---|---|
| 重锚定 | `eval/local_pick_core.py::_reanchor_after_ready`（调用点在 `run_local_pick`） | `config.REANCHOR_AFTER_READY`（默认 True） |
| 路径重映射 | `::_remap_object_sequence`（纯函数；合同见 `tests/test_reanchor_invariants.py`，5 项） | 随重锚定自动生效（waypoint 模式） |
| wp_time 执行器 | `::_replay_waypoint_timegrip` | `REPLAY_MODE="wp_time"` |
| settle-first / 重试策略 | `eval/run_single_object_pick_sweep.py` | `--settle-steps` / `--no-reanchor` / `--retry-wptime` |
| 观测 | `ready_pos_error`（实际目标）、`reanchored`、`anchor_to_ready_disp_mm`、`wp_finished`、`object_pos_*` | 自动 |

回滚：运行级——关 flags 即回基线行为；代码级——`cosmos-policy_snapshot_20260906.tar.gz` 内含修复前全量文件。

---

## 6. 留待未来

1. **运动中闭爪的触发设计**（原"接触触发合爪"方向**已被定点闭爪实验否定**：到位 0.939mm + 静态闭爪 30 步
   零接触零位移——记录的 close_idx 位姿处没有接触可供触发）。可行方向：合爪指令按接近进度提前触发、
   闭爪期间继续沿路径运动（复现演示的"运动中合爪"）。适用于 salad 与 ketchup V2（hold 丢夹同族）。
   未实现；需重过全量回归。
2. **salad 4 个摆放的失败视频逐帧分析**（素材：`rollouts/multiseed_libero_object_videos/`）。
3. **B 类检索/迁移**：0/10（2 例摆放敏感成功说明非铁板一块）；根子在检索无稳健性信号。
4. **HDF5 一帧错位**（ee_states[i] ↔ states[i+1]，偏 8.7–11.9mm）：已复现未修正；当前无实害，闭环会放大。
5. **检索平局**：同物体 ESF 距离全零按文件序取 top-1。
6. **默认值切换**：config/runner 默认仍为 Run 1 语义；路线 C 需显式 flags。
7. **泛化**：多场景/多实例/更大布局扰动未测（本轮仅声明区域内 4 seeds）。

---

## 附录 A. 实验台账索引

| 实验 | 一句话结论 | 脚本（`diagnostics/`） |
|---|---|---|
| 稳定测量 | 位移集中在前 1–3 步；确认步 12–15 | `p0_stability.py` / `p0_stability_v2.py` |
| moka 2×2 | 有效成分=航点收敛，非位置触发 | `p0_moka_ablation.py` |
| 重锚定四臂 | 重锚定≡settle；只等待 0/12 | `p0_reanchor.py` |
| Run 1 / V1 / V2 / 多 seed | 19/21/21/23 及 21–24 | `p1_full_regression.py` |
| D3 摆放干预 | 消除穿模不改变 salad/ketchup 结论 | `d3_small_experiment.py/.json` |
| HDF5 对齐 | ee_states[i] ↔ states[i+1]，8.7–11.9mm | `zcode_verify_h5_alignment_cpu.py` |
| V1–V5 / 实验A·B / D4 | closed-loop 判决、自演示回放 5/5、跨演示 1/40 | `v*_*.py`、`exp_*.py`、`posture_test.py` |
| 逐案例框架诊断 | 每案例位移/ready 误差/源任务对照 | `pick_frame_diagnosis.py` 等 |

数据：`rollouts/multiseed_*.jsonl`（136 episodes）、`..._settlefirst/wptimefix/reanchor/hybrid_retry.jsonl`、
过程 log `rollouts/VERIFY_V1V2V3_20260906.log`、视频 `rollouts/*_videos/`。

## 附录 B. 历史 MD 处置清单（已移入 `archive/`，内容未修改）

| 文件 | 行数 | 原角色 |
|---|---|---|
| `archive/A_CLASS_PICK_FAILURE_ROOT_CAUSE.md` | 338 | 诊断期快照，含已修正的单因表述 |
| `archive/PICK_DIAGNOSIS_EXPERIMENTS.md` | 72 | 实验台账，事实已入 §4/附录 A |
| `archive/A_CLASS_PICK_FIX_SUMMARY.md` | ~160 | 被本文取代 |
| `archive/TEMP_P0_P1_FIX_RESULTS.md` | ~180 | 过程记录，原已标 SUPERSEDED |
| `archive/A_CLASS_PICK_CHANGES.md` | ~130 | §5 已吸收其内容 |

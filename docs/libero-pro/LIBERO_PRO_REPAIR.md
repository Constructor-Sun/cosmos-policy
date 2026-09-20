# LIBERO-PRO 上的 memory 修复

## 目标

证明「用 LIBERO-90 构建的**物体系** memory 做一次对齐」能修复 LIBERO-PRO 的 swap 失败。

## 已验证（swap / STUDY_SCENE1 / 20 trials）

| 配置 | 成功率 |
|---|---|
| baseline（不干预） | 0/20 |
| oracle + IK teleport | 19/20 |
| oracle + cuRobo/关节轨迹（**真机可复用路径**） | **20/20** |

oracle = 在 t=10（settle 后、VLA 首次推理前）把机器人对齐到
`T_world_object_current @ T_object_ee_ready`，然后交回 VLA。
物体身份由命令行 `--item` 给出，ready pose 取该物体在 memory 里的**第一条**记录（无检索）。

## 与 LIBERO-plus repair 的差异

流程一致（census → diagnose → t* → 对齐 → 交回）；只有 ③ 不同：

| | LIBERO-plus | 本方案 |
|---|---|---|
| memory | `skill_memory_test/libero_10/feasible_recovery_targets.pt` | `memory_system/pointcloud_action/pointcloud_action_memory.pt` |
| 过滤键 | `(task, planner_step_id, skill, arguments)` | `(skill, arguments)` —— LIBERO-90 任务名与 libero_10 不同，**去掉 task 层** |
| 输出位姿 | **绝对** `ee_states` | **物体系** `T_object_ee_ready`，用**实时** `T_world_object` 组合 |
| 规划 / 交回 | cuRobo + 关节轨迹 / `_begin_vla_window` | 完全一致 |

**为什么必须换物体系**：LIBERO-plus 存绝对位姿，只在「物体与 demo 同分布」时成立。
swap 把物体挪走（KITCHEN_SCENE3 实测偏 27cm），绝对位姿系统性失效——这正是它在
`objects_layout` 上 repair_rate 只有 0.50 的原因。

## t\* 的实测分布（167 次 LIBERO-plus repair）

```
min=10  中位=10  p90=64  max=252      ≤20: 54%   ≤100: 93%
```

中位数 10 正是「无整数 t\*」的 fallback，语义是「**从第一段开头修复**」。
即 **swap 场景下 repair 基本等价于开局对齐**，而这一步已在上表验证。

## 实现

沿用 `scripts/libero_pro/` 的 monkey-patch 模式，**不改仓库任何现有文件**。

| # | 改动 | 说明 |
|---|---|---|
| 1 | selector 支持 `phase` 入口 | 物体身份从 `phase.arguments["item"]` 取；`--item` 仅留给 initial_alignment 路径 |
| 2 | 检索换成 LIBERO-90 memory + 物体系 | 见下方调用；映射已由 `PointCloudSelector` 实现 |
| 3 | 手工 request（`t_star=0`, `actions=[]`）+ repair 启动脚本 | 走**真正的 repair 执行路径** |

`t_star=0, actions=[]` 时 `_maybe_start_repair_alignment` 的 prefix 回放为 0 步，`t` 仍从
`NUM_STEPS_WAIT` 起步 —— 与现在的开局对齐等价，但经过 repair 的调用链，且
`phase.arguments["item"]` 自带物体身份。

要覆盖大 t\* 的 case，另需：

| # | 改动 | 说明 |
|---|---|---|
| 4 | `diagnose_failed.py` 支持 LIBERO-PRO | 现在走 LIBERO-plus 的 `resolve_variant`；swap/task 的 BDDL 文件名与 base 相同，会解析到 base libero_10。改为由 census 直接给 suite 名 |
| 5 | baseline 日志 → census json | `{"tasks": [{"task", "variant_state", "n_fail", "fail_init_indices_abs"}]}` |

## 关键调用（零件都已存在）

```python
instance   = resolve_target_instance(env, phase.arguments, phase.skill)
points     = visible_point_cloud(env, obs, instance, resolution)
T_wo, _, _ = object_frame(env, instance)
hits       = PointCloudSelector(MEMORY).select(points, T_wo, skill="Pick", top_k=1)
ee_states  = hits[0]["ready_ee_states"]        # 世界系，直接喂 cuRobo
```

## 范围与已知限制

- **只覆盖 Pick 阶段**（LIBERO-plus 的 167 次干预里 141 次是 Pick = 84%）
- 非 Pick 阶段 memory 里没有 `T_object_ee_ready`；`pointcloud_action_memory_place.pt` 有 407 条 Place 记录，但未接入
- 物体位姿 / 实例分割仍取自**仿真真值**（`sim.data.body_xpos` / `agentview_segmentation_instance`）——
  真机需替换为位姿估计与分割网络，本阶段不涉及
- 检索：同物体的 ESF 距离可能全部平局（见 `PICK_MEMORY_REUSE.md` §6）。**改用物体系后位置已归一化，
  检索只影响抓取姿态**（物体系 ready 姿态两两夹角中位 12°、p90 24°），不构成本方案的瓶颈

## 验证

```bash
# 单任务（对照 0/20 baseline 与 20/20 oracle）
python scripts/libero_pro/oracle_ready_eval.py --item black_book_1 \
  --mode controller --suite libero_10_swap --task STUDY_SCENE1 --trials 20 ...

# 横向：10 任务 × 20 = 200 cases（与 baseline 同口径）
bash scripts/libero_pro/run_swap_oracle.sh
```

判据：`<out>/oracle_debug.log` 中 `planned steps` 计数应等于 trials，
`plan FAILED` 与 `skipping` 均为 0，否则该批结果掺入了未干预的 case。

**10 个任务全部覆盖**，其中两个需要说明：

- `KITCHEN_SCENE8_*` —— 第一阶段要抓 `moka_pot_2`，该**实例**不在 memory 里。
  靠**同型回退**命中（`moka_pot_2` → `moka_pot_1` 的记录）。这是正确行为而非权宜：
  memory 存的是物体系位姿，实例编号本不该参与匹配；真系统的点云形状检索同样天然命中同型物体。
- `KITCHEN_SCENE3_*` —— 第一阶段是 **TurnOn**，而 memory 里没有 TurnOn 记录（只有 Pick /
  PlaceIn / PlaceOn），所以对齐只能落在 Pick 的 ready pose 上，**等于跳过了开灶台**，
  TurnOn 需由 VLA 在后续补做。该任务结果需单独解读；根治办法是走 repair 流程的
  prefix 回放（让 VLA 先做 TurnOn，再在 Pick 前对齐），见上文「实现」第 4/5 项。

# LIBERO-PRO swap：oracle 对齐实验结果（临时）

> 状态：初稿，待补。日期 2026-09-17/18。
> 相关：`LIBERO_PRO_REPAIR.md`（方案）、`scripts/libero_pro/oracle_ready_eval.py`、`scripts/libero_pro/run_swap_oracle.sh`

## 1. 实验设置

在所有 10 个 `libero_10_swap` 任务上，各 20 trials（共 200），执行：

```
reset → settle 10 步 → 把机器人对齐到 T_world_object_current @ T_object_ee_ready
      → cuRobo 规划 + 关节轨迹执行 → 交回 VLA
```

memory 来自 **LIBERO-90** 构建的 `pointcloud_action_memory.pt`（物体系 ready 位姿）。
不干预的 baseline 同为 10 任务 × 20 = 200。

## 2. 结果

```
baseline  0/200
oracle   27/200

干预计数：planned 199 / FAILED 1 / skipping 1   → 数据有效（99.5% 的 case 真的干预了）
```

| 任务 | phase 序列 | 需抓 | 结果 |
|---|---|---|---|
| **STUDY_SCENE1** | Pick, PlaceIn | **1** | **20/20** |
| **LIVING_ROOM_SCENE2（奶酪）** | Pick, PlaceIn, Pick, PlaceIn | 2 | **7/20** |
| LIVING_ROOM_SCENE1 | Pick, PlaceIn, Pick, PlaceIn | 2 | 0/20 |
| LIVING_ROOM_SCENE2（汤） | Pick, PlaceIn, Pick, PlaceIn | 2 | 0/20 |
| LIVING_ROOM_SCENE5 | Pick, PlaceOn, Pick, PlaceOn | 2 | 0/20 |
| LIVING_ROOM_SCENE6 | Pick, PlaceOn, Pick, PlaceOn | 2 | 0/20 |
| KITCHEN_SCENE8 | Pick, PlaceOn, Pick, PlaceOn | 2 | 0/20 |
| KITCHEN_SCENE4 | Pick, PlaceIn, Close | 1 | 0/20 |
| KITCHEN_SCENE6 | Pick, PlaceIn, Close | 1 | 0/20 |
| KITCHEN_SCENE3 | **TurnOn**, Pick, PlaceOn | 1 | 0/20 |

**分布不是随机的，是三类原因：**

### 2.1 对齐机制在它作用的范围内是 100% 有效的

`STUDY_SCENE1`（唯一的「单 Pick + 容器不动」任务）：0/20 → **20/20**。

这是本实验最重要的正面证据：**memory 的物体系 ready pose 对齐是有效的**，且走的是真机可复用的
执行路径（cuRobo + 关节轨迹，不经 teleport）。

### 2.2 只修第一个 Pick，多 Pick 任务修不完

swap 打乱**全部**物体，而当前只在 episode 开局对齐一次。所以：

- `LIVING_ROOM_SCENE2（奶酪）`：第一个（cream cheese）放对了，第二个（butter）仍错位 → 7/20
- 其余多 Pick 任务：第二个 Pick 失败 → 0/20

### 2.3 部分 0/20 是 benchmark 的几何退化样本，不是方法的锅

详见 §3。

## 3. 发现：swap 只平移容器、不校验朝向

**swap 在桌子两端之间搬运物体，但不旋转它们。** 对「有朝向语义」的物体（微波炉、抽屉柜、灶台），
移过去之后开口方向相对于机器人可能**反向**。

### KITCHEN_SCENE6（微波炉）实测

```
微波炉 body 朝向:  base 与 swap 的 xmat 完全相同（单位矩阵）→ 未被旋转
位置:
  base   y ∈ [-0.03, +0.60]   机器人在 y = 0   → 微波炉在机器人【对侧】，开口朝向机器人 ✅
  swap   y ∈ [-0.61, +0.00]   机器人在 y = 0   → 微波炉在机器人【同侧】，开口【背对】机器人 ❌

杯子放入 heating_region 后的落点:  base z=1.108  vs  swap z=1.108   → 完全一致
```

**微波炉自身几何没问题**（杯子在里面的落点两边一模一样），变的是**机械臂能否伸进开口**——
微波炉 AABB 跨 0.61 m，挪到近端后机械臂要绕过整个箱体。

视频佐证（`rollouts/libero-pro/oracle_seed7/…mug_in_the_micr.mp4`）：
t≈60 机器人下降到杯子上方（对齐生效），t≈150 抓起杯子，t≈300 举着不动，
t≈517（episode 结束）**从未走向微波炉**。

### 受影响的其它任务

| 任务 | swap 挪了什么 | 朝向风险 |
|---|---|---|
| KITCHEN_SCENE6 | 微波炉 ↔ 瓷杯 | ❌ 已实测 |
| KITCHEN_SCENE4 | **白柜子 ↔ 酒架** | ⚠️ 抽屉面大概率同样翻转 |
| KITCHEN_SCENE8 | 灶台 ↔ moka pot | ⚠️ 灶台面朝向 |
| KITCHEN_SCENE3 | 两个小物体（锅/煎锅） | ✅ 无朝向语义 |
| LIVING_ROOM_* / STUDY_SCENE1 | 仅小物体 | ✅ |

这与另一处已有发现同类：`STUDY_SCENE1` 的 **caddy back 格**装不下杯子（内腔最窄 5.56 cm，
杯子直径约 10.3 cm），却因 `in_box()` **只检查物体原点**（`site_object.py`：
"treating the object as a point"）而仍可能判过。

> 共同模式：**扰动只做了位置层面的置换，没有校验目标几何是否仍然可行**。

## 4. 结论

**不能说「memory 修复在 swap 上只有 13.5% 有效」。** 正确的读法是：

> 对齐机制在它作用的范围内（第一个 Pick）是 **100% 有效**（STUDY_SCENE1 = 20/20）；
> 其余失败全部来自**对齐范围之外**：第二个 Pick、被挪走的容器、被跳过的 TurnOn。

本实验最有价值的产出是把问题从「memory 有没有用」精确到了
**「干预覆盖了几个阶段」**。

## 5. 下一步

| # | 事项 | 目的 |
|---|---|---|
| 1 | **per-phase 对齐**：在每个 Pick 前各对齐一次，而非只在开局 | 修 §2.2（多 Pick）——当前最大失败源 |
| 2 | 补 Place 对齐：`pointcloud_action_memory_place.pt` 有 407 条 PlaceIn/PlaceOn，未接入 | 修容器类任务的放置 |
| 3 | 走 repair 流程的 t\* + prefix 回放 | 修 KITCHEN_SCENE3 的 TurnOn 跳过问题 |
| 4 | 几何可行性判定脚本：检查容器开口方向与机器人的相对方位 | 把 §3 的退化样本从统计里剔除，并在写作中说明 |

**待办**：`LIBERO_PRO_REPAIR.md` 里的实现清单（selector 支持 `phase` 入口、
换 LIBERO-90 memory）尚未动工；本实验用的仍是命令行给 `--item` 的 oracle 形态。

## 6. 复现

```bash
# baseline
bash scripts/libero_pro/run_swap.sh

# oracle
bash scripts/libero_pro/run_swap_oracle.sh
```

产物：`experiments/liberopro/libero_10_swap_oracle_seed7/`（日志 + summary）
视频：`rollouts/libero-pro/oracle_seed7/`
判据：`oracle_debug.log` 中 `planned steps` 计数 ≈ trials，`plan FAILED` / `skipping` 应为 0。

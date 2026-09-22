# 记录选择与相位判定的修正（2026-09-22）

LIBERO-PRO `libero_10_swap` / `KITCHEN_SCENE3`（开灶台 → 抓摩卡壶 → 放灶台）上的四类修正。
任务成功率 **1/9 → 4/10**，其中 TurnOn 相位 **10/10 由规则收尾**。

## 0. 背景

现象（同一批 case 反复出现）：

- TurnOn 相位永远 `timeout semantic=False`，靠 chunk 预算耗尽才推进；
- Pick 相位 6/10 跑满 9 个 chunk 后 `timeout`，壶没有被抬起；
- Pick 一 timeout，"拿在手里"的信念为假，相位循环**主动跳过** PlaceOn 的对齐
  （`skip PlaceOn correction (Pick advanced by timeout)`），后面全部交给 VLA；
- 即使三个相位都判 SEMANTIC，任务仍可能 False（相位判定 ≠ 任务目标达成）。

历轮结果（KSCENE3，8 卡并排）：

```
原始代码                    1/9
+ TurnOn 选条               2/10
+ 规则喂入 / 预算 / Pick / Place 选条   4/10   ← 本文档记录的版本
```

## 1. TurnOn：换一条记录（`TURNON_RECORD_INDEX`）

`pointcloud_action_memory_turnon.pt` 里 40 条记录**参数完全相同**（`{target: flat_stove_1}`），
`_load_ready_pose` 只取"第一条"，而它在库里恰好是空段：

```
索引 0  段长  5  净旋转  8.6°   ← 被选中；低于完成规则阈值，永远触发不了
索引 1  段长 24  净旋转 52.7°   ← 现在用它
全部 40 条的段长: [5×14, 6,7,9,11, 24..44×22]
```

改动：`scripts/libero_pro/oracle_ready_eval.py` 增加 `TURNON_RECORD_INDEX = 1`，
TurnOn 走"收集全部精确匹配 → 取第 N 条"；日志会打印
`[ORACLE] TurnOn 取第 1 条匹配记录（共 40 条）: ...::demo_1::step1`。

## 2. TurnOn：完成规则阈值 35° → 30°

`memory_system/execute/skill_completion/turnon.py` 的 `DEFAULT_MIN_ROTATION_DEG`
原本 35°，而：

- 模拟器判"灶台开着"的判据是旋钮 hinge 转 **0.5 rad = 28.6°**（`yellow_stove.xml`）；
- 实测执行到 **32.8°**（索引 1 那条）→ 物理上已开、规则判不到。

改为 **30.0**（测试断言同步）。`tests/skill_completion/test_turnon.py` 原注释里
记录过标定区间"29–34° 时旋钮开、10 帧后 49–55°"，30 是该区间下沿。

## 3. Pick：按"合爪后抬升"选条（`PICK_MIN_LIFT`）

完成规则要求目标点云随夹爪刚性移动且 **z 抬升 ≥2cm**；而库里：

```
Pick 库 775 条含合爪记录的"合爪后抬升": 0-1cm 13.9% | 1-2cm 8.3% | 2-3cm 11.6%
                                        3-5cm 42.6% | 5-8cm 23.0% | ≥8cm 0.6%
                                        中位 3.89cm
```

即 **22.2% 的记录本身低于规则阈值**。KSCENE3 的 `moka_pot_1` 被选中的那条恰好是
20 条里抬升最小的：`KITCHEN_SCENE3_...::demo_0::step1` = **1.13 cm** ✗。

改动：新增 `_post_grasp_lift(record)`（合爪帧之后 z 的抬升），Pick 在所有精确匹配里
取"抬升 ≥ `PICK_MIN_LIFT`（0.02m）中最小的一条"，避免过度改变现状。实测选中：

```
[ORACLE] Pick 'moka_pot_1' 取抬升 2.22cm 的记录（共 20 条，阈值 2cm）:
         KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove::demo_6::step1
```

换条之后该 case 的壶从"没被抬起"变成抬起 **+5.7cm**（z 0.9661 → 1.0228），Pick 相位
由 `timeout(9 chunk)` 变为 `rule(0~6 chunk)`。

## 4. Place：按"工具轴最竖直"选条（`_place_tilt`）

`flat_stove_1_cook_region` 的 13 条 PlaceOn(moka_pot_1) 记录，ready pose 的工具 z 轴
偏离竖直从 **20° 到 63°** 不等，而"第一条"是 **38°** 的斜插姿态 ✗。

改动：新增 `_place_tilt(record)`（物体系里工具 z 轴的横向分量），Place 取最小者：

```
[ORACLE] PlaceOn 'moka_pot_1'->'flat_stove_1_cook_region' 取工具轴偏竖直 20° 的记录
         （共 13 条）: KITCHEN_SCENE3_put_the_moka_pot_on_the_stove::demo_5::step2
```

**已知限制（数据层面，不是选条能解的）**：该目标下 34 条 PlaceOn 记录的 ready pose
x 全在 **+0.103 ~ +0.267**，而旋钮在 **-0.15** 侧 → **没有一条从旋钮侧接近**，
"远离旋钮侧放不好放"这件事只能靠补数据或改姿态（镜像/旋转 ready pose）解决。
"最竖直"与"侧偏最小"在现有记录里不可兼得（最竖直 demo_5 侧偏 ~0.19；侧偏最小的
demo_4 倾角 52°）。

## 5. 规则喂入：对齐段的帧也喂给唯一判定点

原设计（代码注释）：*"it observes VLA actions only (never planner actions)"* ——
**相位判定只吃 VLA 帧**，而对齐段（记忆回放真正在动的那段）对它不可见：

- TurnOn 的整个旋转发生在对齐段 → 规则永远看不到 → 必然 timeout；
- Pick/Place 的抓取与松手若发生在对齐段 → 同样判不到；
- 只有 Pick 在少数 case 里因为 VLA 接手后又动了一段才被认出来（文档实测 TP27/FN1）。

改动（`cosmos_policy/experiments/robot/libero/run_libero_eval.py`）：

- `observe_skill_shadow(obs, action, t, feed_rule=True)`：在两处**对齐分支**把同一帧
  喂给**同一个** `VLASkillRuntime`（判定点仍然只有一处）；判定语义不变——
  `timeout` 只由 `finish_action_chunk` 累计，而对齐段不调它；
- 推进后仍走 VLA 路径同一条序列：先结束当前对齐（`_alignment_steps_remaining=min(...,1)`），
  再调 `_maybe_run_phase_transition_hook(...)`（**顺序不能反**：钩子若新武装了对齐，
  它会自己重设预算）；
- 原来的设计注释一并改写，避免后人误以为代码写错。

实测效果：TurnOn 相位由 `timeout(3 chunk)` 变为 `rule(0 chunk)`，每个 episode 省约 100 帧；
PlaceOn 也能在对齐段内被判到（0 chunk）。

## 6. 段预算系数 2 → 4（`SEGMENT_STEPS_PER_FRAME`）

`steps = 64 + max(64, 2×seg_len)` 是按记录长度给的段份额，而实测
`WaypointPoseController` 要 **~3.8 帧/waypoint**（"到位才推进"）：
23 点的 TurnOn 段只拿到 64 帧，实测只走完 **71% 位移 / 40% 旋转**。
改为 `4×` 后 TurnOn 段预算 128 → 156。注意这只是**上限**，段提前完成就提前交回。

## 7. 结果（KSCENE3 全 10 case，当前代码）

```
0 TurnOn  10/10 rule（0 chunk）           ✓ 全部修好
1 Pick     4/10 rule（0/0/0/6 chunk）
           6/10 仍 9-chunk timeout        ✗ 壶没被抬到 ≥2cm
2 PlaceOn  随 Pick 成败：Pick 判到 → 武装对齐 → 多数 rule(0 chunk)
任务        4/10（init003/005/007/008）
```

## 8. 复现与产物

```bash
# 单 case（带灶台/壶的逐帧探针）：见 /tmp/run_k3_turnon1_liu.py 的用法
COSMOS_TTA_PHASE_LOOP=1 COSMOS_SKILL_READY_MEMORY=timegrip \
  python /tmp/run_k3_turnon1_liu.py KSCENE3-init002   # LAUNCH_GPU=<g> OUT_DIR=<abs>

# 输入：experiments/tta_phase_check_pro_full10（10 task × 10 init 的录制 + 诊断）
```

产物目录（都在 `experiments/` 下）：

```
tta_repair_work_k3_all10/      本轮 10 case 结果（4/10）
tta_repair_work_k3_vertical/   4 case 验证（2/4，首次到 2/4）
tta_repair_work_k3_picklift/   换 Pick 记录（0/4，但 Pick 相位 3/4 rule）
tta_repair_work_k3_feedrule2/  规则喂入 + 钩子修好
tta_repair_work_pro_full10_v4/ 全 10 task × 10 init（41/99，本次改动之前）
```

## 9. 遗留问题

1. **Pick 仍是瓶颈**（6/10 timeout）：同样的记录、不同 init 的执行差 → 怀疑与早期在篮子上
   量到的"waypoint 冻在够不到的点"同族。可对照 TurnOn 换 `rawactions` 的做法（执行量
   21° → 47°），Pick 目前仍走 `timegrip`。
2. **相位判定 ≠ 任务达成**：init002/init006 三相位全 SEMANTIC 但仍 False，
   需要看 benchmark 自己的 `check_success`（壶是否真落在灶台上）。
3. **逐位不可复现**：同一 case 在不同轮次里结果会翻转（init005 曾 True→False→True），
   4 个 case 的对比要小心把噪声当效果。
4. **`_load_ready_pose` 的"取第一条"是通用问题**：TurnOn/Pick/Place 都踩过，
   其它任务/物体大概率还有同类未被选中的退化记录（Pick 库 22% 抬升 <2cm、
   TurnOn 库 45% 净旋转 <30°）。
5. **`frames=10->89` 这类字段的口径**在文档里未统一（相位窗口 vs VLA 窗口），
   读日志时需要按 `[INIT ALIGN] t=...` 行校准。

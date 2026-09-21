# Place 阶段的问题（临时记录）

> 2026-09-21

## 1. cuRobo

- stove（KSCENE8 的 PlaceOn）：cuRobo 对目标位姿直接报 `Start or End state in collision`，退让 4/8cm 均不可行（`no feasible goal within 0.080m backoff`）；把深度障碍点云几乎去光（`max_spheres=1`）、`robot_padding=0`、`path_safety_margin=0`，仍然失败。
- 两个"汤"任务（LRSCENE1、LRSCENE2 汤+番茄）：cuRobo 规划成功，但 `LiberoJointTrajectoryController` 执行下来手臂停在离 ready pose **200–367mm** 处，128 步用满，结束距离 = 最近距离；同一任务 10 个 init 的失败距离几乎一致。

## 2. 时间不够

- correction 预算 = `plan.correction_steps + max(64, 2×seg_len)`，篮子 place 是 **128 步**；阶段完成判定是另一套 chunk 预算。
- 30 个 case 里 **19 个用满** 128 步；成功的那批用了 **100/128**，离上限不远。

## 3. 选择容器时 ready pose 可能不对

- ready pose 是相对参照物定义的；Place 的参照物通常是区域名（`basket_1_contain_region`、`flat_stove_1_cook_region`…），需要"区域 → 锚物体"的映射。
- 建库用 `memory_system/offline/build_targets.py: match_region_anchor`，运行时用 `oracle_ready_eval._resolve_reference`（另一份实现）。两者对同一区域名若给出不同的 body，组合出的 ready pose 就会平移一个 body 偏移。
- 另外 `_load_ready_pose` 取"参数完全匹配的**第一条**"；同一条目下各条记录的 ready pose 本身相差可达几厘米。

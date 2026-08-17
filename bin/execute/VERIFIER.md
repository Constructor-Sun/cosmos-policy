# 两阶段 Verifier 设计与实现状态

Verifier 分成两个职责独立的阶段：

1. Phase verifier 判断当前是否在执行正确的 skill，并大致朝正确目标移动。
2. Feasible-region verifier 判断机械臂是否进入可以执行末端动作的可行域。

第一版不训练额外模型，也不调用 VLM。
当前 phase 的离线构建、视觉定位、方向判定、可视化和 monitor 接口已经完成。
在线切换校准、可行域 verifier 和恢复策略尚未完成。

## 一、阶段边界

Phase verifier 只负责接近目标的过程。
它不负责判断夹爪是否已经处于严格抓取或释放位姿。
Phase 由 planner step、skill 和目标角色共同定义。
软绑定的是目标身份，而不是 memory 中的固定像素坐标或机械臂绝对位姿。

当机械臂已经接触目标、抓起物体、开始放置或完成释放后，
“是否继续靠近目标”不再是可靠判据，应交给 feasible-region verifier。
因此 phase verifier 的错误不能直接触发恢复，必须先结合当前运行阶段解释。

## 二、阶段一：Phase Verifier

### 2.1 已完成：离线目标构建

实现文件：`bin/memory/build_libero_phase_targets.py`。

每个成功 segment 已知 planner step、skill arguments 和目标实例名。
构建程序恢复 demonstration state，并利用 simulator instance mask 自动提取目标区域。
不需要人工截图，也不需要 VLM 定位。

正式 sidecar 为：

`skill_memory/libero_10/phase_targets.pt`

每个模板目前保存：

- task、demo、planner step、skill 和 arguments；
- frame、目标 bbox、目标中心和夹爪二维位置；
- `crop_rgb` 和 `crop_mask`，供默认视觉相似性匹配；
- keypoints 和 descriptors，供精确几何匹配版本使用。

Simulator 信息只用于离线构建。
Test-time 不读取物体坐标、instance mask 或 simulator object state。

当前正式 memory 共 1037 个模板，构建失败为 0。
其中 1037 个模板具有视觉 crop，1006 个模板具有精确局部特征。

### 2.2 已完成：默认视觉相似性匹配

实现文件：`bin/execute/libero_phase_verifier.py`。

运行时首先根据 task、planner step、skill 和 arguments 选择对应模板。
当前 demo 可以在离线 leave-one-demo-out 评估时排除。

默认匹配流程保持简单：

1. 对 masked target crop 做少量多尺度变换；
2. 在当前 third-view 图像中做归一化模板相似性匹配；
3. 每个 demonstration 只保留最佳候选；
4. 对不同 demonstration 的候选位置进行二维聚类；
5. 至少两个 demonstration 支持同一区域时才接受目标；
6. 两个区域得分过于接近时返回 `PHASE_UNKNOWN`。

模板缩放、模糊和 mask 在 phase reset 时缓存。
在线 observation 不会重复准备同一批模板。

内部会返回一个粗粒度 `target_xy`，用于方向计算和可视化。
它不是物体真实坐标，也不提供给 policy 作为控制目标。

### 2.3 已完成：目标身份软绑定与二维方向判定

软绑定由当前 skill 和目标模板集合维持。
目标在每个 observation 中重新定位，因此允许 test-time 物体位置与 memory 不同。
这里不把第一次匹配到的像素点永久固定，因为物体可能移动或被操作。

设上一时刻夹爪位置为 `g_prev`，当前夹爪位置为 `g_now`，
当前重新定位的目标为 `target_now`，则单次进度为：

`progress = distance(g_prev, target_now) - distance(g_now, target_now)`

同一个 `target_now` 同时用于两个距离，可以避免目标中心轻微抖动直接变成进度。
正值表示大致靠近，负值表示大致远离。
当前实现使用 5 像素容差，并要求连续两次明显远离才返回错误。

阶段一输出：

- `PHASE_OK`：目标得到视觉支持，且没有连续远离证据；
- `PHASE_ERROR`：连续多个 observation 明显远离当前目标；
- `PHASE_UNKNOWN`：没有足够跨 demo 共识，或候选区域存在明显多义性。

### 2.4 已完成：精确匹配版本保留

原有 SIFT/RANSAC 几何匹配实现保存在：

`bin/execute/libero_phase_verifier_exact.py`

默认文件名仍为 `libero_phase_verifier.py`，对应新的视觉相似性实现。
两个版本读取同一个 `phase_targets.pt`，没有增加 v2 文件或新 memory 名称。

精确版可用于几何条件稳定时的对照实验，但不作为当前默认方案。

### 2.5 已完成：离线评估与可视化

实现文件：`utils/eval_libero_phase_verifier.py`。

评估采用 leave-one-demo-out：当前样本所属 demo 不参与模板匹配。
当前 10 个任务、1037 个 observation 的结果为：

- `PHASE_OK = 984`；
- `PHASE_ERROR = 37`；
- `PHASE_UNKNOWN = 16`；
- 预测目标中心落在真值 bbox 内的比例约为 93.3%。

噪声、亮度变化和小幅图像平移测试中，视觉定位结果基本稳定。
这说明当前主要问题不是普通像素噪声，而是遮挡、目标状态变化和阶段边界。

错误可视化已经分为两类：

- `outputs/phase_verifier_errors/`：只包含方向判定为 `PHASE_ERROR` 的 segment；
- `outputs/phase_verifier_mismatches/`：包含目标中心落在真值 bbox 外的 segment。

普通 `.jpg` 是 memory、start、middle、ready 四联图。
`_matches.jpg` 左侧为 masked memory crop，右侧为当前图像中的匹配框。

可使用以下参数重新生成：

```bash
python utils/eval_libero_phase_verifier.py \
    --phase-targets skill_memory/libero_10/phase_targets.pt \
    --input-dir LIBERO-Cosmos-Policy/success_only/libero_10_regen \
    --vis-dir outputs/phase_verifier_debug \
    --vis-max 20 \
    --vis-filter error \
    --show-matches
```

`--vis-filter` 支持 `all`、`error` 和 `mismatch`。

### 2.6 已完成：在线 Monitor 接口

实现文件：`bin/execute/libero_phase_monitor.py`。

Monitor 同时维护 current phase 和 next phase verifier。
它只允许 planner step 单调前进，不允许视觉噪声导致 phase 回退。
连续 observation 更支持 next target 时累计切换证据。
`PHASE_UNKNOWN` 不会被直接当作错误，也不会立即触发恢复。

当前接口已经可以完整回放数据并输出：

- current/next step；
- current/next phase result；
- switch evidence；
- 是否发生 phase switch；
- 是否出现 deviation candidate。

Monitor 的代码接口已经完成，但切换规则还没有通过真实连续 test-time 序列校准。
仅用 start/middle/ready 三个稀疏锚点要求完成全部切换并不合理，
不能把稀疏锚点回放的通过率当作在线切换准确率。

### 2.7 当前已知边界

37 个 `PHASE_ERROR` 中，31 个来自 Pick，6 个来自 PlaceOn。
代表样本中视觉目标通常匹配正确，错误发生在：

- 抓住物体后向上提起；
- 放置或释放后机械臂撤离。

这些动作在二维上确实远离目标，但属于成功轨迹。
原因是 fixed16 segment 可以覆盖接触后的操作过程。
因此这 37 个结果主要是阶段边界误报，不应通过继续放宽视觉匹配来解决。

视觉 mismatch 主要出现在：

- 机械臂或夹爪大面积遮挡目标；
- 物体已经被抓起或移动，模板外观不再完整；
- 盘子、篮子、微波炉门等大面积低纹理目标；
- 书本等细长、深色目标与机器人或容器部件相似。

当前错误说明 phase verifier 应在进入接触/末端可行域后退出主导判定。

### 2.8 真实 Eval 日志证据（2026-08-18）

以下记录来自启用 phase verifier 的 LIBERO-10 paired smoke test：

`scripts/experiments/libero10_robotinit_single/robotinit/logs/ENV_EVAL-libero_10-cosmos-2026_08_18-00_49_54--paired-robot_initial_states-3pair.txt`

对应的成功 rollout 为：

`scripts/rollouts/08-17/2026_08_18-00_49_54--with_future_img--episode=2--success=True--task=put_the_black_bowl_in_the_bottom_dr.mp4`

该任务的 planner sequence 是：

1. `Open(target=white_cabinet_1_bottom_region)`；
2. `Pick(item=akita_black_bowl_1)`；
3. `PlaceIn(item=akita_black_bowl_1, target=white_cabinet_1_bottom_region)`；
4. `Close(target=white_cabinet_1_bottom_region)`。

Episode 2 共记录 17 次 phase observation，时间点从 `t=10` 开始，之后每隔 16 个低层控制步记录一次，最后一次为 `t=266`。关键日志如下：

- `t=10`：current phase 为 step 1 `Open`，结果为 `PHASE_UNKNOWN`，内部原因是 `no_templates`；next phase 为 step 2 `Pick`。
- `t=170`：日志记录 `switched=True`、`reason=switched_to_next`，active step 从 step 1 前进到 step 2；这是整个 episode 唯一一次 phase switch。
- `t=250`：active step 仍为 step 2 `Pick`；current 和 next 均为 `PHASE_ERROR`，记录为一次 `deviation_candidate`。
- `t=266`：active step 仍为 step 2 `Pick`；next phase step 3 `PlaceIn` 为 `PHASE_OK`，`progress_px=17.66400146484375`，但 `switch_evidence=1`、`switched=False`，原因是 `moving_toward_next`。

最终 phase summary 为：

```text
[PHASE SUMMARY] {"active_step_id": 2, "deviation_candidates": 1, "episode": 2, "monitor_error": null, "observations": 17, "success": true, "switches": 1, "task_name": "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"}
```

随后 eval 保存了文件名包含 `episode=2--success=True` 的 rollout，并输出 `Success: True`。本 episode 的日志中没有任何 `active_step_id=3` 或 `active_step_id=4` 的记录。因此，视频中没有显示最终 `Close` 阶段与在线 monitor 的记录一致：环境已经判定任务成功，但 monitor 在 episode 结束时仍停留在 step 2 `Pick`。

对同一任务的 `phase_targets.pt` 进一步核对得到 90 个模板：step 2 `Pick`、step 3 `PlaceIn` 和 step 4 `Close` 各 30 个，step 1 `Open` 为 0 个；这与 `t=10` 的 `no_templates` 日志一致。

## 三、阶段一仍需完成的工作

### 3.1 接入真实 Eval Loop

将 `LiberoPhaseMonitor` 接入实际 LIBERO eval 入口。
第一步只记录结果，不修改 policy action，也不执行恢复。
建议每个 action chunk 调用一次，而不是每个低层控制步调用。

在线日志至少保存：

- task、episode、timestep 和 planner step；
- current/next target、similarity、demo votes 和 confidence；
- gripper position、progress、wrong-way count；
- switch evidence、switch reason 和最终切换时刻；
- 对应 third-view 图像。

### 3.2 用连续序列校准 Phase Switch

必须使用真实 test-time 连续 action-chunk observation 校准：

- `switch_updates`；
- `min_progress_px`；
- current 与 next progress 的比较方式；
- 短时 `PHASE_UNKNOWN` 是否保留已有 switch evidence。

不要针对离线三个稀疏锚点过拟合参数。
首先验证“是否在合理时刻切换”，恢复策略以后再实现。

### 3.3 明确 Phase Verifier 的停止条件

当 feasible-region verifier 确认已经进入可行域后，
phase verifier 不应再根据上提、释放或撤离动作返回恢复请求。

在阶段二完成前，可先采用观测模式记录这些结果，
但不能让 post-contact `PHASE_ERROR` 直接中断成功轨迹。

### 3.4 真机扰动验证

仍需在真实相机条件下测试：

- 光照和曝光变化；
- camera jitter 和轻微视角改变；
- 机械臂遮挡；
- 同类物体干扰；
- 目标旋转或被移动后的重新定位。

若现有 masked crop 相似性在这些条件下不足，
优先增加简单的时序一致性或更稳健的轻量图像 embedding，
不应立即引入 VLM 或复杂检测系统。

## 四、阶段二：Feasible-region Verifier

### 4.1 当前状态

第一版 positive-only feasible-region verifier 已实现于：

`bin/execute/libero_feasible_region_verifier.py`。

它与 phase verifier 保持独立的 memory、reset、历史和判定状态。
Eval loop 只把 phase 视觉定位产生的中性几何量（目标中心、匹配框和夹爪位置）
同时提供给阶段二，不把 phase status 作为阶段二结论。
两个阶段由同一个 `enable_phase_verifier` 开关统一启用，当前均只记录，
不修改 policy action。

### 4.2 目标

Feasible-region verifier 判断当前位姿能否接入成功末端轨迹。
Fixed16 被视为 terminal corridor，不要求定位到夹爪闭合或释放前的精确一帧。

阶段二主要回答：

- 当前是否已经进入可以 Pick、Place、Open、Close 等末端动作的区域；
- 当前是否已经完成 skill；
- policy 已表现出末端意图，但视觉是否仍明显不满足条件。

### 4.3 计划输入与触发

当前实现每个 action-chunk observation 运行一次，输入为：

- 当前 phase 的 task、planner step、skill 和 arguments；
- 当前 target center、matched bbox 和 gripper 二维位置；
- fixed16 manifest 中的 `ready_frame = tau`。

对每条成功 demo，在 tau 计算：

`ready_distance = distance(gripper, target) / bbox_diagonal`

运行时使用相同归一化距离。至少两个 demo 的 tau 距离不小于当前距离时，
输出 `FEASIBLE` 并锁存；之后的上提、释放和撤离不会退出可行域。

### 4.4 计划判定

进入可行域前，阶段二使用归一化二维 progress 记录连续停滞和折返。
仍复用 phase verifier 的 5 像素容差与连续两次证据设置，
但阶段二拥有独立计数，不读取或修改 phase verifier 的 wrong-way 状态。

当前输出：

- `FEASIBLE`：可以执行或继续 terminal action；
- `NOT_FEASIBLE`：进入可行域前连续停滞或折返；
- `FEASIBLE_UNKNOWN`：目标几何不足、warmup、正常接近或异常证据尚未连续。

`SKILL_COMPLETE` 和 recovery 尚未实现。现有
`bin/detect_intervention_point.py` 继续独立负责 command-response
和动作无响应信号，不与二维可行域状态合并。

离线 leave-one-demo-out 几何评估入口为：

```bash
python utils/eval_libero_feasible_region_verifier.py
```

当前正式 memory 的 leave-one-demo-out 顺序回放结果为：

- 共回放 350 个具有 fixed16 ready boundary 的成功 segment；
- 327/350 在 tau 或更早时已锁存 `FEASIBLE`；
- 1/350 成功 segment 被连续二维停滞判为 `NOT_FEASIBLE`，来自 PlaceIn；
- success false correction rate 约为 0.29%。

该误报当前保留用于后续真实连续 observation 校准。阶段二仍处于
observation-only 模式，因此不会据此修改 action 或执行恢复。

## 五、恢复策略

恢复策略目前未实现，也不应与 verifier 判定同时开发。
推荐顺序是：

1. 完成 phase 在线记录；
2. 校准 phase switch；
3. 实现 feasible-region verifier；
4. 确认两阶段错误信号可靠；
5. 最后设计如何回到成功轨迹。

`PHASE_UNKNOWN` 和 `FEASIBLE_UNKNOWN` 默认只表示证据不足。
它们不应自动执行恢复动作。
恢复必须建立在持续、可重复的错误证据上。

## 六、总体运行流程

1. Planner 给出当前 skill 和目标角色。
2. Phase verifier 使用 third view 软绑定目标身份。
3. 每个 action chunk 检查是否大致朝当前或下一目标移动。
4. Monitor 使用连续证据决定是否切换 phase。
5. 接近目标或出现末端 action 信号时触发 feasible-region verifier。
6. 进入可行域后，phase 方向错误不再主导决策。
7. Feasible-region verifier 判断继续末端动作、skill 完成或证据不足。
8. 在两个 verifier 的在线准确率稳定后，再启用恢复策略。

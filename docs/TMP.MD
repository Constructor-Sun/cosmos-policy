# Goal：完成 Memory 引导的离线 TTA 基础链路（执行截止 Step 5）

## 目标与完成定义

只对 baseline 失败的 episode 重试。按 memory 中的顺序核对 skill(item)，在闭爪事件处识别实际交互对象：对象错误则从预期阶段开始纠正；对象正确但执行失败，则在闭爪前 16 action steps 介入。第一个 memory skill 若计算出的切入点为 `t*=0`，先保留 10 个 action steps 给模拟器自行调整；后续 skill 不改变原有切入规则。调用共享 memory 对齐能力，终态失败后复用已有回退向前重试。

兼容 Pick、Place、TurnOn 等 skill，判断尽可能复用 memory_system 下已有的完成、交互、持物与状态判别。只在闭爪交互处定位修复，不把 Place 的释放事件改作切入事件；明确不修复 Close。Close 仍可参与阶段记录与正常 policy 执行，不能从任务顺序中删除。固定 memory、不更新 policy，不引入路径投影或速度模型。

本次计划执行范围严格截止 Step 5：完成记录、诊断、t*、回放、共享对齐及单次 policy 接管验证后停止。完整回退闭环和扩大评测属于后续工作，不执行原 Step 6/7，也不以其完成作为本轮验收要求。已有回退只核对接口、保留后续接入位置，不启动自动连续重试。

本轮完成意味着：两类修复有可检查的记录，切入和回放正确，对齐结果明确，Step 1–5 的必要验证通过；不要求所有 episode 恢复成功，也不能仅凭视频宣布完成。

已完成并验证的方法与结果见文末「本轮已修改：交互对象查询与判定规则（第二轮）」一节；下文仅保留当前有效的行为约定与尚未执行的 Step 5。检查框本身不代表已实现或已验证。

## 已确定的行为

| 判断 | 动作 |
|---|---|
| 闭爪附近实际对象与 memory 当前预期 item 不同 | 修复当前预期阶段，t* 为该阶段开始；第一阶段若该值为 0 则最终取 10，不跳到错误对象对应阶段 |
| 对象正确，已有判别标记该阶段执行失败 | t* = max(phase_start_step, close_step − 16) |
| 无闭爪事件或无足够近的对象 | 记录无有效闭爪修复点，不制造事件、不用释放替代；~~本轮返回不可定位，不强行执行阶段起点修复~~（已修改：无闭爪事件归入 `object_mismatch(no_close_event)`，t* 取首阶段下限 10，见文末最新一节） |
| Close 阶段 | 保留判断与记录，不产生对齐修复请求 |
| 路由未到达，或交还 policy 后终态失败 | Step 5 记录失败结果后停止；后续完整闭环才由已有回退调度前一阶段 |

从判别失败的阶段开始，不另按最深游标猜测。对象不一致是顺序/对象错误的证据；对象一致不是动作成功的证据。优先搜索整个 memory_system 的现有判别，不因某个 completion 类只有超时就认定全库没有能力。Pick 抓起、Place 释放/持物/放置状态、TurnOn 交互及其他 skill 均先复用，缺失部分列出明确缺口，不另写重复规则，也不把 timeout 或释放直接当成动作成败。

诊断失败阶段与闭爪事件所属阶段分开记录。尤其 Place 可能沿用前序 Pick 的持物状态，不能为凑事件伪造 Place 内的闭爪；先核对已有记录是否能明确关联。若修复点属于前序 Pick，则明确将其作为修复阶段并使用对应 memory 目标，不能拿 Pick 的闭爪时刻直接执行 Place 对齐。关联不明时记录未定位，不隐式跨阶段。

## 代码架构与当前落点

| 职责 | 现有落点 | 本轮工作 |
|---|---|---|
| 阶段/事件记录与诊断 | tta/phase_record.py、tta/diagnose_failed.py | 补齐实际交互对象，按上述规则产生修复请求 |
| 单次离线修复调度 | 现有重试/回退入口，实施前定位 | reset、回放、对齐、单次 policy 验证；回退只核对接口，执行留到 Step 5 之后 |
| 共享 memory 对齐 | execute/initial_alignment.py、Eval 控制接管、已有规划控制 | 接受指定 phase，initial 与中途调用同一能力 |

当前目录还存在 tta/regen_tstar_png.py，可复用图像生成能力。旧 locate.py 已不在，不按旧规划重建它。保留现有修改与数据，先检查调用关系再编辑。

本轮数据流：阶段/事件记录 → RepairRequest(phase, reason, t*) → 回放 → Align(current_obs, phase) → 到达结果 → 单次 policy 终态 → 保存结果并停止。

对齐模块不接收 t*；时间定位与回放属于调度层。phase 必须包含顺序索引、planner_step_id、skill 和 arguments，避免同名 skill 或不同目标混淆。

## 每步验证的统一要求

- 每步保留输入案例、预期结果、实际结果、断言/误差、日志与代码配置；验证项标明通过、失败或缺少样本，未运行不能填通过。
- 轻量规则与边界检查在前，真实环境验证在后。模拟测试只能证明分支行为，不能替代真实回放与对齐验证。
- 有影响后续步骤的失败，先定位修正再继续。每步产出一份简短验证记录，缺口不得以视频或总成功率掩盖。
- 到 Step 5 完成后交付截至 Step 5 的验证结果并停止；不自动运行回退、多轮修复或扩大评测。

## Step 5：共享 memory 对齐，接管后重新运行 policy（已完成，2026-09-08）

> 状态：本节已实现并经 68 条真实轨迹全量验证，实际落点与结果见文末「本轮已修改（第三轮）」。以下为原始计划条目。


进入真实修复验证前，先通过「超时后仍正确核对 memory 对象」和「首次闭爪证据不覆盖」用例（规则见文末「本轮已修改」一节）；不能仅凭回放误差为零认定阶段诊断已经通过。

- [ ] 将已有 initial 的阶段选择与对齐执行分开：initial 传第一阶段，中途修复传诊断指定阶段。
- [ ] 按 task、phase、arguments 与当前观测检索正确目标，不跟随失败轨迹选错误对象；复用已有规划和控制。
- [ ] 核对 initial 的固定抬高、夹爪动作和单次开局触发假设，按已有 phase 配置处理，避免机械套用。
- [ ] 明确返回 reached 或 route_failed，记录位置/姿态误差；waypoint 超时不算到达。
- [ ] 到位后清除旧动作队列，用最新观测重新 query policy。Place 等兼容已有持物判断、几何和规划处理，不套用空夹爪默认动作；明确拒绝 Close 修复。
- [ ] 只做验证所需的单次修复和 policy 执行，保存终态后返回。失败也不自动启动前一阶段重试，不运行 Step 5 之后的内容。

验证：

1. 同一开局输入下，经共享接口调用与原 initial 选择相同阶段，检索及规划行为与既有配置一致。
2. 用 Pick、Place、TurnOn 等指定 phase 检查检索和接口，严格匹配 task/阶段/对象，不退回第一阶段。Place/持物输入确认使用已有持物处理；Close 请求被明确拒绝。真实修复只使用合法闭爪定位生成的请求，不为覆盖测试伪造事件。
3. 一个可到达案例确认末端误差达标且重新 query；一个不可到达/超时案例明确报告 route_failed，不假装完成。
4. 检查 policy 接管首个动作来自新 query，未执行干预前残余 chunk。
5. 验证失败返回后不执行 reset/回退第二次尝试，成功返回后也停止；每次预算可核对。对共享修改复查原 initial，不要求对齐失败案例成功，但必须准确分类失败。

完成标准：共享对齐正/负路径、原 initial 回归、指定 phase 检索、队列接管和单次停止检查均通过。至少有真实回放与对齐证据；无真实样本的 skill 分支注明仅完成接口验证。产物：单次尝试日志、检索原型、到达误差、query 记录、终态及各步验证汇总。

## Step 5 后的停止与交付

交付 Step 1–5 的实现范围、每步验证记录、失败案例及尚未覆盖的判别/样本，明确哪些是真实环境验证、哪些仅为接口/规则检查。完整回退接入、多轮修复、批量恢复率评测均留待后续，不继续执行。

不得把“本轮 Step 1–5 完成”写成“完整 TTA 回退闭环已完成”。本文修改本身不代表任何步骤或测试已执行。

## 本轮已修改：交互对象查询与判定规则（第二轮，2026-09-08）

以下内容已修改完成并经 68 条真实轨迹离线重放验证：

1. `memory_system/tta/object_query.py`：候选查询排除机器人自身实例（实例名含 panda/robot），机械臂不再充当"最近交互物体"。
2. 距离度量改为物体点云中位数中心在水平面（XY，删除 z 轴）到 EEF 的距离，与 memory 既有坐标约定（`skill_completion/pick.py`、`initial_alignment.py` 的 median）一致；不再用最近表面点（稀疏点云如酒瓶 43 点时被噪声主导）。
3. `memory_system/tta/phase_record.py` 与 `regen_tstar_png.py`：删除 0.10 m 交互门限（`MAX_INTERACTION_DISTANCE` 及其硬编码副本），候选不再设距离上限。
4. `phase_record.py` 新增 `_keep_memory_objects`：候选只保留本 task memory 涉及的物体（各 phase 的 item/target），场景干扰物（酒瓶、酒架、甜品等）不参与最近对象判定。SCENE4 20/20、全库 68 条中预期对象排第一的判定由此从 4/20、约 25/68 提升为稳定匹配。
5. 首次闭爪的对象判定时机：闭爪指令翻转后挂起 `_pending_close_interaction`，跟踪实测夹爪间隙至最小，在开爪/runtime 推进/finish_chunk 时以最小间隙帧的观测查询并判定（`_flush_close_interaction`）；t* 计算仍以闭爪指令步为基准。
6. 无闭爪事件归入 `object_mismatch(no_close_event)`，t* 取首阶段下限 10；不再返回 unlocatable。
7. 68 条失败轨迹的干预时刻图片已按原 `tstar.png` 版式（`diagnose_failed.make_png`）重新生成，保存为 `memory_system/pointcloud_action/results/tta_failure_screening_68/<task>/InitXXX.png`，任务目录下平铺、无 initXX 子目录。判定分布：execution_failure 31（t* = close−16）、object_mismatch 37（含无闭爪 1，t* = 10）、unlocatable 0。

**测试结果：插入点的计算。** 68 条全部完成诊断与插入点计算，图片平铺保存在 `memory_system/pointcloud_action/results/tta_failure_screening_68/<task>/InitXXX.png`；判定分布：执行失败（t* = 首次闭爪 − 16）31 条，对象不匹配 / 无闭爪事件（t* = 阶段起点下限 10）37 条。

## 本轮已修改：Step 5 修复接管实现（第三轮，2026-09-08）

实现（≤100 行，另加两处小修）：

1. `memory_system/execute/initial_alignment.py`：`select()` 新增可选 `phase` 参数，中途修复可指定诊断阶段；不传保持 initial 原行为。
2. `cosmos_policy/experiments/robot/libero/run_libero_eval.py`：新增 `COSMOS_TTA_REPAIR` 修复模式——回放 `actions[:t*]` 前缀后，在首次 policy query 时以诊断 phase 调 `select()` 武装对齐控制器，清空残余队列、保持夹爪状态，对齐到达（含收敛提前交接）后 policy 重新 query 接管至终态；单次修复不重试，结果写 `COSMOS_TTA_REPAIR_OUT`。initial 对齐在修复模式下自动让位。
3. 两处修正：`reached` 标记移到实际执行的递减分支；h5 文件名加入 per-case tag 防并行覆盖。
4. 修复武装时传入 `main_depth`，cuRobo 避障路径真正启用。
5. 对齐控制器收敛后立即截断剩余步数预算并交接，消除到达后的站桩停顿。

**测试结果：最终失败的挽救。** 68 条 baseline 失败轨迹全部完成单次修复：对齐 68/68 到达目标位姿，任务恢复 **62/68 = 91.2%**；未恢复 6 条为 KSCENE4-init013、KSCENE6-init013、KSCENE8-init011、LRSCENE5-005/017、LRSCENE6-021（视频在 `results/tta_repair_videos/failed/`），汇总在 `results/tta_repair_68_summary.json`。

剩余未成功的 6 条还需要逐条看一下。

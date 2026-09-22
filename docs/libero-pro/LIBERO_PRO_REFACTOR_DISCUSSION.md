# LIBERO-PRO 重构：类与接口讨论

## 1. 总则（已确认）

### 1.1 原则

- 不新增业务功能和算法能力；允许为集中现有职责调整类、接口和数据结构，便于纠错和扩展。
- 结构优先于行为保持，允许精度暂时回退，以固定冒烟例观测；不静默删除已有执行路径，有意改变或暂未迁入的逻辑明确记录。
- 补丁迁入时提升为所属组件的显式参数，标定值作默认。
- 重构求快求优雅：以搬为主、步子小、冒烟即验收，不铺仪式。
- 失败定位到类：各类的判定输出自带原因（PhaseDecision 的 reason、ExecutionResult 的 outcome/stage、MemoryCandidate 的 match、RepairRequest 的 t_star 与 failure），统一写进按类标记的运行日志；case 失败先读日志定类，再进类里修。

### 1.2 已确认的现实

- Task：任务，关联 libero-10 及其变体。
- Case：同一 Task 下具体的初始化与扰动条件。
- Run：针对一个 Case 的一次尝试。
- t*：skill 失败取夹爪变化前 16 action step；物体选择失败取 Phase 起点。
- recovery 保留对齐和记忆段执行的混合方式。
- 按具体类讨论字段、方法、职责及现有代码归属；只记录决策和当前议题。

## 2. 类与接口（已确认）

### 2.1 TaskPlanner

职责：根据 Task 生成技能顺序，并用独立方法筛除当前状态下可跳过的步骤。

```python
class TaskPlanner(ABC):
    @abstractmethod
    def plan(self, task: Task) -> tuple[PhaseSpec, ...]:
        """根据任务实际目标，生成有序技能步骤及其参数。"""
        ...

    @abstractmethod
    def filter_satisfied(
        self, phases: tuple[PhaseSpec, ...], env,
    ) -> tuple[PhaseSpec, ...]:
        """筛除可跳过步骤，保留其余步骤顺序和 ID。"""
        ...
```

- `plan()` 承接 `plan_from_bddl()`、`plan_from_goal_state()`；demo 计划来源保留对应实现。
- `filter_satisfied()` 承接 `drop_presatisfied_turnon()` 等现有跳过逻辑。
- 输出沿用 `PhaseSpec(planner_step_id, skill, arguments)`。
- 不执行动作、不检索 memory、不计算 t*。本节为接口约定，尚未修改实现。

### 2.2 PhaseChecker

职责：判定当前 Phase 是否结束及结束原因；供诊断和修复运行共用。

```python
class PhaseChecker(ABC):
    @abstractmethod
    def reset(self, phase: PhaseSpec, *, initially_holding=False): ...

    @abstractmethod
    def observe(self, observation, action, step: int) -> PhaseDecision: ...

    @abstractmethod
    def finish_action_chunk(self, step: int) -> PhaseDecision: ...
```

- 保留现有 Pick、Place、TurnOn 判据及 chunk 预算逻辑；其他技能的现有处理也保留。
- PhaseDecision 区分：运行中、规则满足、超时；阶段结束不等于语义完成。
- 每个实际执行后的观测只喂入一次；chunk 预算按现有执行口径累计。
- 不推进阶段游标、不更新持物状态、不切换控制器。

### 2.3 RepairChecker

职责：根据失败样例与阶段记录，判断在哪里干预，生成修复请求。

```python
class RepairChecker(ABC):
    @abstractmethod
    def check(
        self, failed_run: Run, phase_records: tuple[PhaseSpan, ...],
    ) -> RepairRequest:
        ...
```

- PhaseSpan 关联 PhaseSpec，提供阶段起止位置、判定结果与证据；复用现有记录结构。
- 输入多个阶段，保留从整条失败样例中选择修复阶段的能力；不另维护阶段游标。
- skill 失败取夹爪变化前 16 action step；物体选择失败取对应 Phase 起点。
- 保留现有索引边界处理和 fallback。
- RepairRequest 包含来源 Run/轨迹、失败类型、对应 Phase、t*。
- 不回放 prefix、不检索 memory、不规划或执行动作；不承担运行中后续阶段是否继续干预的决策。

关系：调用方结合 PhaseChecker 的结果形成阶段记录，RepairChecker 读取同一份记录。

### 2.4 MemoryRetriever

职责：根据修复目标与切入后的场景，选择并排序 memory 候选。

```python
class MemoryRetriever(ABC):
    @abstractmethod
    def retrieve(
        self, request: RepairRequest, observation, env,
    ) -> list[MemoryCandidate]:
        ...
```

- observation 来自 prefix 回放到 t* 后的环境；回放由外部执行流程负责。
- 收拢现有 skill/参数过滤、精确匹配、同型回退逻辑。
- 保留 Pick 抬升、Place 倾角、TurnOn 选条等现有选择策略。
- 返回候选记录、排序及匹配信息；无匹配时明确报告。
- 不将 memory 位姿映射到当前场景，不规划、不创建控制器、不执行动作。
- 位姿映射归后续干预准备/执行部分；原 selector 中的规划与回放职责迁出。

### 2.5 InterventionExecutor

职责：管理一次干预的 APPROACH（接近）与 REPLAY（记忆执行），向上层报告执行结果。

```python
class InterventionExecutor(ABC):
    @abstractmethod
    def prepare(self, candidate, observation, env, mode): ...

    @abstractmethod
    def step(self, observation): ...

    @abstractmethod
    def observe(self, observation) -> ExecutionResult: ...

    @abstractmethod
    def close(self): ...
```

- prepare 映射 memory 位姿、准备控制器；step 返回动作，env.step 由外部统一执行。
- APPROACH 复用现有 cuRobo、关节控制器及 fallback；REPLAY 保留 waypoint/timegrip/rawactions。
- align 模式在接近后结束；回放模式继续应用 memory。两个阶段的控制器均可替换。
- observe 管理干预内部交接，区分到位、结束、超时或失败；不据此认定技能完成。
- PhaseChecker 可提前确认阶段结束；上层可调用 close 终止干预并清理、恢复控制模式。
- 干预结束不固定交回 VLA；后续可能继续 VLA、进入下一阶段干预或结束任务，由上层决定。
- 上层调度综合各类、统一阶段游标与控制权；见 §2.7 RunScheduler。

### 2.6 RunRecorder

职责：记录一次 Run 的事实，产出可供诊断、回放和训练导出的 Run。

```python
class RunRecorder(ABC):
    @abstractmethod
    def begin(self, case: CaseSpec, phases: tuple[PhaseSpec, ...]) -> None:
        """开始一次 Run，登记身份、计划与起始条件。"""
        ...

    @abstractmethod
    def record_step(self, step: int, action, source: ActionSource,
                    observation=None) -> None:
        """记录一个实际执行步的动作、来源与观测。"""
        ...

    @abstractmethod
    def record_event(self, event: RunEvent) -> None:
        """记录阶段边界、控制权交接、干预与判定。"""
        ...

    @abstractmethod
    def finalize(self, success: bool) -> Run:
        """收束记录，产出 Run。"""
        ...
```

- `record_step()` 承接逐帧动作与观测；`record_event()` 承接阶段边界、控制权交接、干预与判定；`begin()`/`finalize()` 承接 `_repair_result`、`_interventions` 的汇总。
- 输出沿用 `PhaseSpan` 与 `Run`：三层判定（执行结束 / 阶段结束 / 技能完成）、holding 两份（判断 + 事实）。
- 动作按 chunk 保留边界并记来源与控制空间；对齐段保留、标"未采集"；索引保存 记忆帧号 ↔ 执行步 ↔ 数据行 的映射。
- 不推进阶段、不改持物信念、不切换控制权、不判断阶段结束、不检索、不规划、不做位姿映射。

关系：由 Runtime 广播事件驱动；RepairChecker 与训练/TTT 导出读 Run。

### 2.7 RunScheduler

职责：持有阶段游标、持物信念与控制权，决定每一步由谁发动作，并广播事件。

```python
class RunScheduler(ABC):
    @abstractmethod
    def begin(self, case: CaseSpec, phases: tuple[PhaseSpec, ...]) -> None:
        """开始一次 Run：登记身份与阶段计划，初始化游标、信念与控制权。"""
        ...

    @abstractmethod
    def decide(self, observation) -> ActionSource:
        """返回这一步由谁发动作：VLA、某次干预，或停止。"""
        ...

    @abstractmethod
    def apply(self, verdict) -> None:
        """应用判定与决定：推进游标、更新信念、接管或收回控制权。"""
        ...

    @abstractmethod
    def finish(self, success: bool) -> None:
        """结束一次 Run。"""
        ...
```

- 持有阶段游标、持物信念与控制权；按判定推进，沿用现有规则决定后续阶段是否继续干预，管理控制权交接并广播事件。
- 分工：阶段结束判定归 PhaseChecker；失败样例诊断、修复阶段与 t* 定位归 RepairChecker；运行中后续阶段是否继续干预归 RunScheduler；执行干预归 InterventionExecutor。
- 行为沿用现状：信念只认"技能完成"层更新；首次修复使用 RepairChecker 生成的请求，后续阶段由 RunScheduler 按现有规则决定继续干预或使用 VLA，不重新诊断或计算 t*；终态为任务判定完成、步数上限、异常三种；判定可以打断干预，先结束当前对齐再交接。
- 不实现阶段完成判据、不诊断失败样例、不计算 t*、不执行动作、不做位姿映射、不记录。
- 承接 `_maybe_run_phase_transition_hook`、阶段游标推进、`_holding` 与控制器交接。

关系：接收 RepairChecker 的修复请求，应用 PhaseChecker 与 InterventionExecutor 的结果；事件广播给 RunRecorder。

### 2.8 EnvironmentAdapter

职责：集中环境创建、Case 初始化、指令读取与对象身份解析，屏蔽 suite 差异。

```python
class EnvironmentAdapter(ABC):
    @abstractmethod
    def make(self, case: CaseSpec):
        """按 Case 的套件与初始状态创建环境，返回环境与真实指令。"""
        ...

    @abstractmethod
    def target(self, phase: PhaseSpec):
        """解析该阶段的目标对象/目的地身份（名字 → 实例），保留同型回退。"""
        ...

    @abstractmethod
    def geometry(self, observation, target):
        """返回目标在当前观测下的位姿与可见点云。"""
        ...
```

- 收拢现有的多处重复：env 与套件创建、Case 初始化、指令读取、对象身份解析。
- `case` 含套件、flavor、变体、init index、seed 与真实指令；一个进程只服务一种 flavor（plus 与 PRO 两棵树不能同进程混跑）。
- 指令取 BDDL 的真实 language，不取任务名推导的旧指令。
- 承接 `resolve_target_instance`、`object_frame`、`visible_point_cloud` 与各 suite 分支。
- 不做判定、不调度、不检索、不执行、不记录；真机替换点集中在此。

关系：RunScheduler 取 Case 与观测；MemoryRetriever 与 InterventionExecutor 取目标身份与几何。

### 2.9 PolicyAdapter

职责：策略推理在 LIBERO 侧的薄入口（观测 + 指令 → 动作 chunk），模型细节不出这一类。

```python
class PolicyAdapter(ABC):
    @abstractmethod
    def act(self, observation, instruction):
        """返回一个动作 chunk；best-of-N、嵌入、去归一化与 latent 捕获都在内部。"""
        ...

    @abstractmethod
    def reset(self):
        """丢弃未执行的动作；交回控制权时调用。"""
        ...
```

- 不重写推理、不改 aloha：内部就是调用现有 `cosmos_utils.py` 的 `get_action` / `query_model_parallel` 与 AR 预测函数。
- chunk 是推理单元，也是 RunRecorder 的记录单元与 TTT 的训练单元。
- 仅 LIBERO 侧：把评测循环里的策略调用（含 latent 捕获）收进来。
- 不判定、不调度、不检索、不执行、不记录。

关系：RunScheduler 把控制权借给它；RunRecorder 记录它给出的 chunk。

### 2.10 MemoryBuilder

职责：沿用现有离线构库链，不新增抽象类。

- 分段、位姿提取、记录派生与两套库的产出流程保留，算法不改。
- 记录质量过滤归 MemoryRetriever（§2.4）；构库不判定记录可用性。
- 两套库并存（`skill_memory_test/libero_10/`、`pointcloud_action_memory*.pt`）；place 库有 13 条为手工追加，重建会覆盖。

关系：产出由 MemoryRetriever 读取。

## 3. 统一数据结构（已确认）

三个都是数据：无方法、无职责，只带字段。

```python
@dataclass(frozen=True)
class RepairRequest:
    run: Run                      # 来源 Run
    failure: str                  # 失败类型
    phase: PhaseSpec              # 要修的阶段
    t_star: int | None
    prefix_actions: np.ndarray    # actions[:t_star]
    plan: tuple[PhaseSpec, ...]   # 整条阶段计划
```

```python
@dataclass(frozen=True)
class MemoryCandidate:
    record: ...        # 记忆记录：memory_id / demo_id / 阶段身份
    match: str         # 匹配方式：精确 / 同型回退 / 几何
    rank: float        # 排序依据：相似度或距离
    target_ee: np.ndarray   # 目标位姿
    replay: ...        # 回放段：位姿序列 + 夹爪序列 + 物体物理动作序列 + ready frame
```

```python
@dataclass(frozen=True)
class ExecutionResult:
    outcome: str                # 到位 / 结束 / 超时 / 失败
    stage: str                  # 干预内部阶段：approach / replay
    steps_used: int
    final_ee: np.ndarray | None
```

- `RepairRequest` 的字段取自现有 request json（`task` / `tag` / `phase` / `plan` / `t_star` / `actions`），加上 §2.3 已要求的来源 Run 与失败类型。
- `MemoryCandidate` 的"匹配方式"承接现有的精确匹配、同型回退；"排序依据"承接相似度、距离与那几条选条规则。
- `ExecutionResult` 只管这次干预自己结束与否；技能完成属于判定（PhaseDecision），不进这里。
- `PhaseSpec` / `PhaseSpan` / `PhaseDecision` 已存在，只做字段确认，并把 `PhaseDecision` 与代码里的 `SkillDecision` 合成一个名字。

## 4. 包布局与迁移顺序（已确认）

- 新建顶层包 `memory/`，逐步承接 memory_system 的全部内容（构库链最后迁入）；两套库的数据文件不迁，仍在原路径。
- 类与文件对应（包即 memory，检索与构库留在根；其余按职责分四个子目录）：

```text
memory/
    types.py        # §3 三个数据结构 + PhaseSpec/PhaseSpan/PhaseDecision
    retriever.py    # §2.4
    build/          # §2.10 构库链，最后迁入
    run/            # §2.1 planner.py、§2.6 recorder.py、§2.7 scheduler.py
    check/          # §2.2 phase_check.py、§2.3 checker.py
    execute/        # §2.5 executor.py（现有控制器随后迁入）
    adapters/       # §2.8 environment.py、§2.9 policy.py
    tools/          # 工具脚本：请求生成、诊断、库维护等杂项
```
- 组合根是现有四个入口脚本（run_libero_eval、oracle_ready_eval CLI、run_single_case、tta_repair_batch），不设总装配类；plus 与 PRO 单进程单 flavor 由入口承载。
- 迁移期兼容：`OracleReadySelector` 降为薄门面（选条归 retriever，规划与回放归 executor）；运行时不再 import scripts/ 下模块。
- 迁移期间新旧并存：修复打在职责当前所在处（未迁的在原文件，已迁的进新类）；该职责迁移时补丁随之带走。
- 迁移顺序：① §3+§2.4+§2.5 修复链路 → ② §2.2 PhaseChecker 收拢改名 → ③ §2.6+§2.3 捆绑（§2.3 的输入 Run 由 §2.6 产出）→ ④ §2.7 收拢 run_episode，§2.8/§2.9 随之就位 → ⑤ §2.10 构库链迁入 memory/build/。
- 每步验收：冒烟三例跑通（Pick：LRSCENE2-init004、PlaceOn：LRSCENE5-init000、TurnOn：KSCENE8-init004，自 requests 随机抽取）；迁入的类过一遍设计审查——职责完善、直接、优雅；有意改变行为处在本节下记一行。不铺单独测试，验证靠冒烟跑通 + 设计审查 + 使用中按类定位。
- 复现约定：case 复现一律经 run_single_case.py（环境块已钉死，requests 即冻结的 t*/prefix）；修复输出 JSON 记录 git commit 与关键环境开关；不做逐位对比与方差标定，行为问题靠按类标记的日志在使用中发现。
- 迁移中遇到文档未定的细节（Run 格式、config 等）：当场最小决定并记一行，不预先设计。
- 已知留白与触发点：Run/ActionSource/CaseSpec（③④）、config 与 §2.7 decide 语义、合并重复对齐分支（④）、flavor 承载（§2.8）、②的判定/游标过渡形态、13 条手工记录处理（⑤）。

## 5. 补丁迁移原则（已确认）

- 通用规则直接进入所属组件：坐标变换归 InterventionExecutor，观测判定归 PhaseChecker，状态更新与控制权交接归 RunScheduler；各组件遵守同一套执行约定。
- 技能差异通过可替换策略与显式参数表达，由统一主流程调用；核心流程不按 task/suite 名称堆叠特殊分支，suite 差异集中在入口和 EnvironmentAdapter。
- 经验性补丁保留为有名称、有适用范围的策略，当前标定值作默认，不视为普遍规律。例如 TurnOn 固定选条、Pick 抬升筛选、Place 倾角排序归 MemoryRetriever；回放模式、接近 fallback 与预算归 InterventionExecutor。
- 第一轮保留已有策略，策略改进与结构迁移分别记录；数据补丁归构库/库维护流程，不伪装成在线算法规则。
- 新任务若不引入新的操作类型，应主要通过任务参数、场景绑定和记忆数据接入；检查是否需要修改核心调度流程，作为抽象边界是否清楚的判断依据。
- 策略可先用函数、配置或现有实现组合，不要求额外建立类体系；把原有 if 搬到另一个文件不算完成职责抽象。

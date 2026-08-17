# LIBERO Planner 与 Skill Memory

本文档说明 `bin/memory/` 中的高层规划、离线 demonstration 切分和
MemoryVLA-like skill memory 构建流程。

相关文件：

- `libero_predicate_planner.py`：Unified Planning + Fast Downward baseline。
- `libero_skill_skeleton.py`：不读取 initial state 的名义 skill 序列。
- `label_libero_skill_segments.py`：用 LIBERO 模拟器离线标注 skill 边界。
- `label_libero_skill_ready_boundaries.py`：只读 action，标注 ready/terminal
  边界。
- `build_vector_db_from_demos.py`：编码 observation，生成 whole-demo vector
  DB 或分段 skill memory。

代码约定：`bin/memory/` 中每个 Python 文件不得超过 370 行；接近上限时应
拆分职责，而不是继续堆叠实现。当前文件均满足该约定。

## 1. 系统位置

目标是把长任务拆成少量带对象参数的 skill：

```text
open the drawer and put the bowl inside

Open(drawer)
→ Pick(bowl)
→ PlaceIn(bowl, drawer)
```

各层职责：

```text
Planner     决定 skill 及其对象/目标
Memory      检索成功片段并提供 action sequence
Verifier    根据真实 observation dynamics 判断结果（尚未实现）
Supervisor  推进、重试、纠错或重规划（尚未实现）
```

离线流程：

```text
BDDL ──→ state-free skill skeleton
  │
HDF5 states ──→ LIBERO predicate labeling ──→ segment manifest
HDF5 actions + segment manifest ──→ ready-boundary manifest
HDF5 RGB/actions + ready manifest ──→ Cosmos encoder ──→ skill memory .pt
```

模拟器只用于离线标注。在线执行和未来真机部署不应读取 simulator object
state。

## 2. 两种规划入口

### 2.1 Predicate Planner

`libero_predicate_planner.py` 读取 BDDL 的 objects、fixtures、regions、
initial state 和 goal state，构造 Unified Planning problem，并调用 Fast
Downward 输出 grounded skill。

它支持 `hand_empty`、`holding`、`on`、`inside`、`opened` 和
`turned_on`，以及 pick、place、open/close、turn_on/off 和 push_to 等
高层 action schema。LIBERO 只提供状态谓词，因此 action 的
precondition/effect 仍由该文件定义。

该路径适合作为符号规划 baseline 或离线 oracle；它不启动 MuJoCo，也不生成
机器人轨迹，并且假设 initial state 已知。依赖为：

```bash
pip install "unified-planning[fast-downward]"
```

### 2.2 State-free Skill Skeleton

`libero_skill_skeleton.py` 只读取 goal、language 和静态对象/region 信息，
不读取 BDDL initial state，也不调用 Fast Downward。主要展开规则为：

```text
In(item, target)
  → Open(target, mode=ensure)
  → Pick(item)
  → PlaceIn(item, target)

On(item, target)
  → Pick(item)
  → PlaceOn(item, target)

Close/TurnOn/TurnOff(target)
  → 对应的 ensure skill
```

语言中包含 `push` 时，部分 `On` 目标会展开为 `PushTo`；最终 Close
目标放在 place skill 之后。`ensure` 表示当前状态未知，未来执行层可以根据
observation 判断是否跳过。每个 step 输出：

```text
step_id
skill
arguments
depends_on
execution_mode
state_assumption
```

## 3. 离线 Skill Segment 标注

### 3.1 任务来源与执行顺序

LIBERO-10 的 task name 和 language 读取
`configs/libero10_experiment_tasks.json`；同名 BDDL 提供 goal、对象、region
和环境定义，避免误用 LIBERO-plus 的扩展 benchmark task map。

多个独立目标不强制使用 BDDL goal 的排列顺序。标注器按照 demonstration 中
各 `Pick→Place` 单元的实际完成时间输出 segment，同时保留 nominal
`planner_step_id`。

### 3.2 Predicate 与 Pick

`In/On/Open/Close/TurnOn/TurnOff` 直接调用 LIBERO 原生 predicate
evaluator。LIBERO 没有 `Holding` 谓词，因此 Pick 使用：

1. robosuite contact grasp + 物体位移；
2. handle geometry 漏掉接触时，回退到物体相对 skill 起点的垂直抬升。

PlaceIn/PlaceOn 优先寻找 relation 成立且已释放的稳定帧。如果官方成功 demo
到结束仍保持夹爪接触，则以 LIBERO relation 为准，不额外要求
`not grasping`。

### 3.3 State 恢复与 terminal state

普通帧直接恢复记录的 flattened MuJoCo state，不从初始状态连续重放整条
action，以免版本差异造成累计漂移。

再生成 HDF5 保存的是 action 执行前的 state/RGB，因此最后 action 之后的
terminal state 缺失。标注器从最后一个记录 state 执行最后一个 action 一次，
用恢复出的 terminal symbolic state 完成边界判定。若成功只出现在 terminal
state：

```text
success_state_source      = replayed_terminal
success_embedding_source  = last_recorded_observation
```

由于 HDF5 没有 terminal RGB，最后一张记录 RGB 只是明确标记的近似 success
embedding。

### 3.4 批量运行与 manifest

robosuite/EGL 在同一进程反复创建环境时可能直接退出。无 `--task` 的批量
模式会为每个基础任务启动独立子进程，最后合并 manifest；`--task TASK_NAME`
用于单任务调试。

segment manifest 的核心结构：

```json
{
  "task_name": "...",
  "demo_id": "demo_0",
  "valid": true,
  "segments": [
    {
      "planner_step_id": 1,
      "skill": "Pick",
      "arguments": {"item": "moka_pot_1"},
      "start": 97,
      "end": 216,
      "success_start": 216,
      "success_end": 219
    }
  ]
}
```

### 3.5 Ready boundary（tau）

`label_libero_skill_ready_boundaries.py` 不恢复 simulator state。它在已有
`start/end` 内确定 terminal action 的第一帧 `terminal_start`；该帧 RGB
是第一个 terminal action 执行前的 ready observation。

默认 `--boundary-mode auto`：

- Pick/Open/Close 优先使用最后一次持续 close edge；
- PlaceIn/PlaceOn 优先使用最后一次持续 open edge；
- 没有可靠 gripper edge 时，从同任务 action change-point 候选中选择共有
  terminal suffix；
- 共有后缀超过两个 Cosmos action horizon（32 帧）时，保守回退到最后 16
  steps。

可选 `--boundary-mode fixed` 直接使用：

```text
terminal_start = max(start, end - 16)
```

短于 16 帧的 segment 使用完整片段。输出增加：

```text
terminal_start
ready_frame
terminal_length
boundary_method
boundary_confidence
```

`already_satisfied` 没有执行动作，因此 `terminal_start/ready_frame` 为
`null`。

## 4. Skill Memory 格式

`build_vector_db_from_demos.py` 接收可选的 `--segments-manifest`。提供
ready-boundary manifest 时，每个 `.pt` 的主要结构为：

```text
format = libero_skill_memory_v2
task_name
demo_id
segments[]
  planner_step_id
  skill
  arguments
  start/end
  skill_start_vae
  ready_vae
  success_vae
  success_state_source
  success_embedding_source
  chunks[]
    vae_video
    proprio
    action_chunk_raw
    action_chunk_normalized
    valid_length
  terminal_chunks[]
    # 字段同 chunks，第一块严格从 terminal_start 对齐
```

action chunk 长度为 16，且不会跨越 skill 边界。片段不足 16 帧时用最后一个
action 补齐，`valid_length` 保存真实长度。旧 manifest 没有
`terminal_start` 时输出 v1；v2 保留原 `chunks`，并增加
`terminal_chunks`。

不提供 `--segments-manifest` 时，builder 保持 whole-demo vector DB 行为；
因此默认直接运行并不会生成 skill memory。

## 5. 运行流程

```bash
source /data1/liu/miniconda3/etc/profile.d/conda.sh
conda activate cosmospolicy
cd /data1/liu/exp/counterfactual/external/cosmos-policy
```

第一阶段，生成 segment manifest：

```bash
python bin/memory/label_libero_skill_segments.py \
  --suite libero_10 \
  --input-dir LIBERO-Cosmos-Policy/success_only/libero_10_regen \
  --output skill_memory/libero_10/segments.json \
  --max-per-task 10
```

第二阶段，生成固定 16-step ready boundary：

```bash
python bin/memory/label_libero_skill_ready_boundaries.py \
  --input-dir LIBERO-Cosmos-Policy/success_only/libero_10_regen \
  --segments-manifest skill_memory/libero_10/segments.json \
  --boundary-mode fixed \
  --output skill_memory/libero_10/segments_ready_fixed16.json
```

如果要使用自动边界，将 `fixed` 改为 `auto`。

第三阶段，生成 skill memory：

```bash
python bin/memory/build_vector_db_from_demos.py \
  --input-dir LIBERO-Cosmos-Policy/success_only/libero_10_regen \
  --segments-manifest skill_memory/libero_10/segments_ready_fixed16.json \
  --output-dir skill_memory/libero_10 \
  --max-per-task 10
```

builder 会加载 Cosmos 2B 模型、checkpoint、T5 cache 和 dataset statistics，
需要足够 GPU 显存。

## 6. 当前边界

- simulator predicate 仅用于离线标注，不是真机运行时 oracle。
- terminal symbolic state 可以恢复，但当前 HDF5 中没有对应 terminal RGB。
- Pick 的垂直抬升 fallback 是针对离线成功 demo 的启发式边界。
- skeleton 是 nominal sequencer；Predicate Planner 也只是简化的高层模型。
- 尚未实现 runtime memory retrieval、completion/error verifier、失败恢复和
  在线重规划。
- 真机仍需解决对象检测、身份绑定和 task grounding。

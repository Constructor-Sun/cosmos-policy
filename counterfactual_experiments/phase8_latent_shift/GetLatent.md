# Latent Shift Dataset Design

本文档记录 Phase 8 latent-shift correction 的数据集设计结论。

目标是学习：

```text
latent_perturbed -> latent_shift

latent_shift = latent_clean - latent_perturbed
latent_clean_pred = latent_perturbed + latent_shift_pred
```

当前只讨论 first-chunk latent correction。本文档不引入 trajectory/mid-step 数据，也暂不扩展 `env_seed`。

## 1. Cosmos Policy、corrector 和 LIBERO-10 的关系

需要区分三个数据阶段：

1. Cosmos Policy base model 的训练；
2. latent-shift corrector 的训练；
3. corrector 的测试。

官方发布的 `Cosmos-Policy-LIBERO-Predict2-2B` base policy 使用四套 suite：

```text
libero_spatial
libero_object
libero_goal
libero_10
```

每套 suite 有 10 个 task、每个 task 50 个 demonstrations，总计：

```text
4 suites * 10 tasks * 50 demos = 2000 demos
```

因此，`libero_10` 对 base policy 不是 unseen task。

本项目的 corrector 可以使用四套 suite 的 clean/perturbed pairs，但必须保留独立的 test 条件。正确的隔离方式不是排除 `libero_10` task，而是隔离：

```text
initial state
camera parameter
policy sampling seed
```

LIBERO-10 对 corrector 的测试含义是：

```text
base task 已被 base policy 见过；
corrector 没见过 LIBERO-10 的 test initial states、test camera tuples 和 test policy seeds。
```

这属于 corrector 的跨条件泛化测试，不是整个 Cosmos Policy 的 unseen-task 测试。

## 2. Base task 数量

四套 suite 的 base task 数量为：

| Suite | Base tasks |
|---|---:|
| `libero_spatial` | 10 |
| `libero_object` | 10 |
| `libero_goal` | 10 |
| `libero_10` | 10 |
| **Total** | **40** |

前三套 suite 只有 30 个 base task，没有隐藏的额外基础任务。LIBERO-Plus 中的大量 task name 主要是 camera、layout、light 等 perturbation variants，不应当当作新的 task semantics。

## 3. Train/test initial state 必须分开

### 3.1 Base policy demonstration states

Cosmos Policy 的 demonstration regeneration 从原始 HDF5 读取：

```python
orig_actions = demo_data["actions"]
orig_states = demo_data["states"]
```

然后使用：

```python
env.reset()
env.set_init_state(orig_states[0])
```

代码见 `cosmos_policy/experiments/robot/libero/regenerate_libero_dataset.py`。

每个 task 在本地 `LIBERO-Cosmos-Policy` 数据中有 50 个 demonstration。

### 3.2 LIBERO-10 test states

官方 evaluation 使用：

```python
task_suite.get_task_init_states(task_id)
```

对应每个 task 的：

```text
<task>.pruned_init
```

第 `episode_idx` 个测试 episode 使用 `initial_states[episode_idx]`。

### 3.3 已完成的重叠检查

对本地 LIBERO-10 的 10 个 task，逐元素比较了：

```text
training demonstration states[0]
vs.
test pruned_init states
```

忽略 MuJoCo state 中的第一个时间字段后，所有 task 的重叠数量都是：

```text
train states/task = 50
test states/task  = 50
overlap           = 0
```

因此 corrector train 不能直接使用当前 collector 默认读取的 `pruned_init` 作为训练状态，否则会泄漏官方 test state distribution。

## 4. Seed 设计

`env_seed`、demonstration regeneration seed 和 policy sampling seed 是不同概念。

### 4.1 Environment seed

当前保持：

```text
env_seed = 0
```

Cosmos Policy 的环境创建使用 `env.seed(0)`。固定它是为了控制环境随机性，不代表 train/test 初始状态相同。

### 4.2 Recommended policy sampling seed split

建议使用完全不重合的 seed 集合：

```text
Train seeds = 1009, 2003, 3001, 4001
Val seed    = 5003
Test seeds  = 195, 196, 197
```

如果历史 LIBERO-10 camera test 继续使用 policy seed `7`，则 `7` 也必须加入 test-only set，不能进入 train/val。

policy seed 控制 Cosmos Policy action denoising/sampling 的随机性。clean 和 perturbed branch 必须使用同一个 policy seed，保证 pair 中唯一改变的是 camera condition。

## 5. Camera perturbation inventory

LIBERO-Plus 的 camera suffix 由五个参数组成：

```text
_view_<horizontal>_<vertical>_<scale>_<endpoint_yaw>_<endpoint_pitch>_initstate_0
```

实现位置：

- 参数解析：`../LIBERO-plus/libero/libero/envs/env_wrapper.py`
- camera 变换：各 LIBERO tabletop problem 的 `_setup_camera`

四套 suite 的 `Camera Viewpoints` 分类共有：

```text
task-camera entries = 1599
unique parameter tuples = 450
```

因此不需要重新发明一套 camera 参数。最简单的方式是从：

```text
../LIBERO-plus/libero/libero/benchmark/task_classification.json
```

提取所有 `category == "Camera Viewpoints"` 的 task names，解析出 450 个 unique five-tuples。

## 6. Camera split

camera split 的单位必须是完整的五元组，而不是 task-specific filename。

固定一个 split seed，将 450 个参数 tuple 划分为：

```text
camera_train = 360 tuples
camera_val   = 45 tuples
camera_test  = 45 tuples
```

比例为：

```text
80% / 10% / 10%
```

这样相同的 camera tuple 不会通过另一个 task 泄漏到 validation 或 test。

如果某个 task 的官方 task-specific variants 较少，允许使用全局 camera tuple 生成该 task 的 suffix。当前 LIBERO-Plus env wrapper 支持从 base BDDL 文件解析任意合法的 `_view_...` suffix。

## 7. 推荐的约 8000 条 train dataset

推荐的主训练集精确规模为：

```text
40 tasks
* 20 demonstration initial states/task
* 10 camera tuples/state
* 1 policy seed/sample, balanced over 4 train seeds
= 8000 pairs
```

即：

```text
40 * 20 * 10 = 8000
```

policy seed 不再对每个 state-camera pair 做完整笛卡尔积，而是对 8000 个 pair 均衡轮转：

```text
[1009, 2003, 3001, 4001]
```

每个 seed 大约负责 2000 个样本。

这样相比当前：

```text
30 tasks * 50 pruned states * 1 camera * 5 seeds = 7500
```

新的 8000 数据具有：

```text
40 tasks
20 train demonstration states/task
大量 camera tuples
4 个 train policy seeds
```

不会把大部分预算浪费在同一个 state-camera 组合的重复 policy seeds 上。

### 7.1 State sampling

每个 task 的 50 个 demonstration states 使用固定随机种子打乱后划分：

```text
20 states -> train
5 states  -> val
25 states -> reserved/unused
```

不能使用 `pruned_init` 作为 train state。

### 7.2 Camera assignment

每个 task/state 分配 10 个 `camera_train` tuples：

```python
camera = camera_train[(task_offset + state_index * 10 + j) % 360]
```

其中 `task_offset` 固定且可复现。

要求：

- 同一个 state 内尽量不重复 camera tuple；
- 40 个 task 合计覆盖整个 `camera_train` pool；
- 不使用有放回的无约束 `random.choice`；
- clean/perturbed pair 使用完全相同的 state、task、policy seed 和 env seed。

## 8. Validation dataset

建议额外生成一个小 validation set，约 400 条：

```text
40 tasks
* 5 held-out demonstration states/task
* 2 camera_val tuples/state
* policy seed = 5003
= 400 pairs
```

validation 与 train 至少在以下维度隔离：

```text
initial state 不重合
camera tuple 不重合
policy seed 不重合
```

总生成量为：

```text
8000 train + 400 val = 8400 pairs
```

仍属于约 8000 规模。

当前 `phase8_train_latent_shift.py` 默认按 task 做 train/val split。采用本方案后，应优先使用 manifest 中明确标注的 `split`，而不是再对所有 rows 做随机 task split。

## 9. LIBERO-10 test dataset

LIBERO-10 test 使用独立的：

```text
pruned_init states
camera_test tuples
policy seeds = 195, 196, 197
```

每个 task 的 50 个 test states 中，每个 episode 分配一个 camera_test tuple，循环覆盖 45 个 test tuples：

```text
10 tasks * 50 pruned states = 500 pairs/test seed
```

三个 official test seeds 总计：

```text
1500 pairs
```

不需要计算完整的：

```text
10 tasks * 50 states * 45 cameras * 3 seeds
```

只需让 camera tuple 在 episode 序列中均衡轮转即可。

LIBERO-10 test 的准确含义是：

```text
同一组 base tasks 上，测试新的 initial states、held-out camera tuples 和 held-out policy sampling seeds。
```

它不是 base policy 的 unseen-task test，因为官方 Cosmos Policy base model 已经使用过 LIBERO-10 clean demonstrations。

## 10. Manifest 必须记录的字段

每个 pair 的 manifest 至少需要记录：

```text
suite
base_task
task_name_clean
task_name_pert
split                  # train / val / test
state_source           # demo / pruned_init
state_index
state_hash
camera_tuple
camera_split           # train / val / test
policy_seed
env_seed
condition
sample_id
```

其中 `state_hash` 和 `camera_tuple` 用于之后检查 leakage。

建议在训练前自动检查：

```text
train_state_hash intersect val_state_hash = empty
train_state_hash intersect test_state_hash = empty

train_camera_tuple intersect val_camera_tuple = empty
train_camera_tuple intersect test_camera_tuple = empty

train_policy_seed intersect val_policy_seed = empty
train_policy_seed intersect test_policy_seed = empty
```

## 11. 评价解释

训练目标仍然是 first-chunk latent shift recovery。`hidden_recovery_vs_pert` 只能作为 latent-level 指标，最终还需要报告：

```text
latent recovery
centered R2
action latent recovery
action recovery
```

对于本方案，最重要的比较是：

```text
train: seen demonstration states / camera_train / train seeds
val:   held-out demonstration states / camera_val / val seed
test:  pruned_init / camera_test / test seeds
```

如果 test 提升，才能说明 correction mapping 对 initial-state、camera parameter 和 policy sampling randomness 都有一定泛化能力。

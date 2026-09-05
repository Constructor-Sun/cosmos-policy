# 单桌面 + 单物体局部 Pick 测试方案

## 1. 目的

增加一个最小评测场景：环境中只有一个桌面和一个目标物体，不依赖 LIBERO HDF5 demo，只测试现有 pointcloud-action memory 能否完成局部 Pick。

该测试用于回答：

> 在没有其他物体碰撞或遮挡的情况下，现有 memory 检索、move-to-ready 和动作 replay 能否稳定抓起目标物体？

## 2. 测试范围

- 首版只支持 `Pick`；
- 场景只包含桌面和一个目标物体；
- 使用 `env.reset()` 生成初始状态，并支持多个随机 seed；
- 继续使用现有 `pointcloud_action_memory.pt`，不重新训练或构建 memory；
- 默认保留 move-to-ready；
- 支持现有 `complete` 和 `visible` 点云来源；
- BDDL goal 使用 `(Held <object>)`，复用 memory_system 已有的 Held/holding 定义和接入方式，不重复实现新谓词。

## 3. 执行流程

```text
生成单桌面、单物体 BDDL
        |
        v
创建环境并按 seed reset
        |
        v
提取目标物体点云
        |
        v
从现有 memory 检索 top-1
        |
        v
move-to-ready
        |
        v
replay object-centric Pick actions
        |
        v
闭爪保持 20 步并判断稳定 Pick
```

## 4. 文件修改

### 4.1 修改 `offline/extraction.py`

给环境创建函数增加直接传入 BDDL 文件路径的能力，例如：

```python
def create_env_from_bddl(bddl_file_name: str, resolution: int = 256):
    ...
```

现有 `create_env(task_name, ...)` 的行为保持不变。

### 4.2 新增 `eval/run_single_object_pick_sweep.py`

该脚本负责全部首版功能：

- 根据固定的最小 Living Room BDDL 模板生成单物体场景；
- 模板只替换 `object_name` 和 `object_type`，不解析或修改任意复杂 BDDL；
- 在创建环境前复用现有 Held 谓词接入；
- 对每个物体和 seed 调用 `env.reset()`；
- 复用现有点云提取、检索、move-to-ready 和 Pick replay 函数；
- 执行稳定 Pick 判定；
- 将结果写入 JSONL，可选保存视频。

首版不新增 `local_pick_core.py`，不重构现有 HDF5 evaluator，也不修改 `config.py`。等其他评测或 skill 确实需要复用时再抽公共模块。

## 5. 稳定 Pick 判定

在 Pick action replay 结束后，必须实际执行 20 个闭爪保持 step，不能只等待或直接读取状态，也不能继续使用已经 `finished` 的 `PointCloudPickController`：

```python
hold_action = np.zeros(env.env.action_dim, dtype=np.float32)
hold_action[-1] = 1.0  # 当前 LIBERO 控制约定：+1 持续闭爪

for _ in range(20):
    step_result = env.step(hold_action.tolist())
    obs = step_result[0] if isinstance(step_result, tuple) else step_result
```

对于当前 7 维 OSC_POSE action，该动作就是：

```text
[0, 0, 0, 0, 0, 0, +1]
```

前 6 维保持末端当前位置和姿态，最后一维持续闭合夹爪。20 步执行完成后，同时满足以下条件才算成功：

```python
is_grasping(env.env, item) and final_pos[2] - start_pos[2] >= 0.02
```

其中 `start_pos` 在 move-to-ready 完成、Pick replay 开始前记录。

## 6. 命令行参数

首版保留必要参数：

```text
--memory               memory 路径
--objects              目标 object type 列表
--num-cases            每个物体的 seed 数量
--point-cloud-source   complete / visible
--output               JSONL 输出路径
--save-video-dir       可选视频目录
```

渲染分辨率和执行步数先沿用现有默认值。

## 7. 结果格式

每个 case 记录一行：

```json
{
  "object": "xxx",
  "seed": 0,
  "success": true,
  "memory_id": "...",
  "distance": 0.0123,
  "error": null
}
```

## 8. 实施步骤

1. 生成一个最小 BDDL，确认环境能够创建和 reset，且只有桌面与目标物体；
2. 跑一个物体、一个 seed，保存视频确认完整 Pick 流程；
3. 扩展到多个物体和多个 seed，输出 JSONL。

## 9. 验收标准

- 场景中只有桌面和一个目标物体；
- 不使用 HDF5 demo，能够通过 `env.reset()` 初始化；
- 能运行完整的检索、move-to-ready 和 Pick replay；
- Pick 后保持 20 步，仍抓持并抬升至少 2 cm；
- 每个 case 都能输出一条 JSONL 结果。

## 10. 非首版范围

- 抽取通用 `local_pick_core.py`；
- 重构已有 HDF5 或 Objects Layout evaluator；
- 为其他 skill 设计统一执行接口；
- 增加新的训练数据或重新构建 memory。

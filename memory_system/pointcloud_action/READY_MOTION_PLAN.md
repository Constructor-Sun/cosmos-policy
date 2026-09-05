# Ready Motion Plan

> 本文档中的 ready 运动构造思路借鉴自 ASPIRE：不依赖完整全局碰撞规划器，而是用“少量语义 waypoint + IK + 关节空间闭环执行”来近似避障。  
> 下文不再使用 ASPIRE 命名，统一采用抽象化描述。

## 目标

在 pointcloud_action 的局部 Pick 评测中，替换当前“使用 CuroboPlanner 移动到 ready pose”的实现。

新方案需要：

- 不再依赖 CuroboPlanner；
- 不假设起点是干净的；
- 优先实现一个简单、可快速接入的版本；
- 后续再扩展为能处理更复杂杂乱场景的版本。

---

## 方案一：最小改动（当前主要实现）

### 1. 基本思想

把 move-to-ready 看作一次“带 approach 的位姿移动”：

```text
当前位姿
  -> 先抬升/后退到安全高度
  -> 水平移动到 ready 上方/后方
  -> 走到 ready 的 approach 位姿
  -> 到达 ready
```

这一版不构建完整场景地图，也不做复杂绕行，只依赖：

- 几个手工设计的 waypoint；
- 每个 waypoint 做 IK；
- 使用上一个 IK 解作为初值，保持关节连续；
- 用关节空间闭环控制执行。

### 2. 需要新增的模块

#### 运动原语层

职责类似 ASPIRE 的 motion primitive：

- 给定目标位置和姿态，求解 IK；
- 给定目标关节角，执行关节空间移动；
- 提供一个“带 approach 的位姿移动”接口。

接口内部需要处理：

- IK 求解后恢复 sim 状态；
- 连续 IK，以上一个关节角作为初值；
- 关节空间执行，而不是用 OSC 跟踪 EE waypoint；
- 保持夹爪指令不变。

#### Ready 运动构造层

职责是生成“当前 -> ready”的 waypoint 序列。

首版生成固定模板 waypoint：

1. 从当前位置先抬升一段安全高度；
2. 在安全高度移动到 ready 上方；
3. 在 ready 上方转到 ready 姿态；
4. 下降到 ready 的 approach 位姿；
5. 到达 ready。

首版参数放在配置中，例如：

- 抬升高度；
- 安全余量；
- approach 距离；
- 最大 waypoint 数；
- 关节插值步数。

### 3. 需要修改的现有文件

- `config.py`：新增 ready 运动参数；
- `eval/eval_pointcloud_pick.py`：
  - 修改 `_move_to_ready()`，不再调用 CuroboPlanner；
  - 改为调用新的 ready 运动构造层和运动原语层；
  - 删除无碰撞保护的 `PoseController` fallback；
  - 让执行循环支持基于 observation 的关节空间 controller。

### 4. 方案一的限制

- waypoint 是固定模板，不会根据实际障碍绕行；
- 适合“桌面物体不太高、ready 上方基本可走”的场景；
- 如果起点被包围，或路径中间有高障碍，仍可能碰撞。

---

## 方案二：场景自适应构造（后续扩展）

### 1. 基本思想

在方案一的基础上，加入轻量场景几何，使 waypoint 不是固定模板，而是根据当前深度图生成。

整体流程：

```text
当前深度/点云
  -> 2.5D 场景地图
  -> 从当前杂乱位置找到安全出口
  -> 根据 ready 接近方向找到目标 gate
  -> 在安全高度做轻量路径搜索
  -> 生成少量 waypoint
  -> IK + 关节空间执行
```

### 2. 新增模块

#### 2.5D 场景地图

- 从深度图反投影得到点云；
- 去掉当前机器人自身点；
- 体素化；
- 对每个 XY 网格记录最高障碍；
- 做膨胀，补偿手臂/夹爪半径和安全余量。

#### 安全出口搜索

- 不假设起点干净；
- 从当前 EE 附近搜索局部逃逸方向；
- 优先竖直抬升；
- 如果上方不安全，沿障碍较少方向斜向上退；
- 如果当前位置在容器/抽屉内，沿开口方向后退。

#### 目标 gate 搜索

- 不假设 ready 上方一定空旷；
- 根据 ready 的接近方向生成 gate；
- 如果默认 gate 被挡，绕 ready 搜索多个候选方向；
- 选择一条从 gate 到 ready 最干净的进入通道。

#### 轻量路径搜索

- 在 2.5D 地图上做 2D/2.5D 路径搜索；
- 例如 A* 或 Dijkstra；
- 目标是绕开高障碍，而不是完整 7-DoF 规划；
- 输出少量粗粒度 XY 路径点。

#### Waypoint 提升与 IK

- 把粗路径点提升为带高度的 3D waypoint；
- 在安全高度完成姿态过渡；
- 在 gate 附近再转成 ready 姿态；
- 对每个 waypoint 做连续 IK；
- 如果 IK 失败，尝试 backoff、插点或换路径。

#### 轻量轨迹校验

- 对生成后的关节路径做插值；
- 用机器人 FK 生成 swept spheres；
- 与场景点云做冲突检查；
- 发现冲突后重新生成局部 waypoint 或绕行。

### 3. 方案二的优势

- 不假设起点干净；
- 不假设 ready 上方空旷；
- 能处理路径中间存在高障碍的情况；
- 仍然保持“少量 waypoint + IK + 关节执行”的整体风格。

### 4. 方案二的工作量

- 新增场景地图、路径搜索、安全出口、目标 gate 等模块；
- 代码量明显大于方案一；
- 适合在方案一跑通后逐步加入。

---

## 当前实现范围

当前优先实现 **方案一**：

1. 完成最小可运行版本；
2. 验证 move-to-ready 不再依赖 CuroboPlanner；
3. 验证基本关节空间执行链路；
4. 保留方案二所需的分层结构，后续在同一个抽象框架内扩展场景自适应逻辑。

---

## 参考结构

```text
memory_system/pointcloud_action/
  execute/
    motion_primitives.py       # 运动原语层：IK / 关节移动 / 带 approach 的位姿移动
    ready_motion_planner.py    # ready waypoint 构造层：首版固定模板，后续场景自适应
    scene_map.py               # 后续：2.5D 场景地图
    joint_path.py              # waypoint -> 关节轨迹
  config.py                    # ready 运动参数
  eval/eval_pointcloud_pick.py # 接入新的 move-to-ready
```

> 说明：上面只给出建议的模块职责和文件划分；具体类名、接口名在实现时按项目风格再定，不在此处固定。

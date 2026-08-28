# URDF 深度过滤

`urdf_depth_filter.py` 用当前关节角摆放 URDF 机器人模型，依据相机内外参渲染一张“只有机器人”的深度图，再删除真实深度中与机器人深度一致的像素。

## 当前状态

该模块已作为默认关闭的可选功能接入 initial-alignment 规划链路。启用后采用 hybrid 流程：

```text
原始 depth
  -> URDF depth filter
  -> points_from_depth()
  -> cuRobo collision spheres 兜底过滤
  -> surface sampling
  -> cuRobo planning
```

现有 `filter_robot_points()` 没有被替换。URDF 加载或渲染失败时会记录 warning，并回退到原有 spheres 流程。关闭开关时，点云和规划路径保持原来的处理方式。

通过 smoke launcher 启用：

```bash
COSMOS_URDF_ROBOT_FILTER=1 ROBOTINIT_GPU=6 \
sh scripts/run_libero10_robotinit_20.sh
```

实现只依赖当前环境已有的 `yourdfpy`、`trimesh` 和 `rtree`。它通过 CPU 射线与 URDF 三角网格求交，不依赖 MuJoCo segmentation，也不需要 RGB 分割或 OpenGL 渲染器。

## 输入与输出

主要接口是 `UrdfDepthFilter.filter()`：

- `observed_depth`：memory system 使用的 canonical-space 公制深度，即已经按当前约定上下翻转的深度。
- `camera_params`：现有 `CameraParams`，其中 `K` 和 `T_c2w` 仍是 render-space 相机参数。
- `joint_positions`：关节名到弧度值的字典，或者关节值数组。
- `joint_names`：当数组顺序不是 URDF 默认 actuated-joint 顺序时显式传入。
- `robot_base_pose`：URDF base 在世界坐标系中的位姿，格式为 4x4 矩阵或 `[x, y, z, qx, qy, qz, qw]`。

框架接入时，`run_libero_eval.py` 将 `robot0_gripper_qpos` 经 `InitialAlignmentSelector` 传给 `CuroboPlanner`。LIBERO 两个 finger slide 的符号相反，而 Franka URDF 通过 joint axis 表达方向，因此 planner 对两个值取绝对值并限制在 `[0, 0.04]`，再与 7 个 arm joints 组成 9 关节 URDF configuration。缺少夹爪观测时使用 `[0.04, 0.04]` 作为回退值。

返回 `UrdfDepthFilterResult`：

- `filtered_depth`：机器人像素被置为 `invalid_depth`（默认 0）的深度。
- `robot_depth`：渲染得到的 canonical-space 机器人深度。
- `robot_mask`：实际删除的像素 mask。
- `removed_pixel_count`：删除像素数，便于记录和排查。

## 判定规则

只有观测深度和渲染深度都有效，并满足下式时才删除像素：

```text
abs(observed_depth - robot_depth)
    <= abs_tolerance + relative_tolerance * robot_depth
```

默认绝对容差为 8 mm，相对容差为 0.5%。这些值需要用 LIBERO 图像或真机标定数据验证。若一个物体明显位于机器人前方，其观测深度与后方机器人深度不一致，因此会被保留。

## 最小用法

```python
from memory_system.execute.urdf_depth_filter import (
    UrdfDepthFilter,
    UrdfDepthFilterConfig,
)

robot_filter = UrdfDepthFilter(
    UrdfDepthFilterConfig(urdf_path="/path/to/franka.urdf")
)

result = robot_filter.filter(
    depth,
    camera_params=camera_params,
    joint_positions=joint_positions,
    joint_names=joint_names,
    robot_base_pose=robot_base_pose,
)
depth_without_robot = result.filtered_depth
```

URDF 引用的 mesh 必须能被解析；如果使用 `package://` 且默认解析规则找不到资源，可以在构造器中传入 `filename_handler`。默认使用 visual geometry，因为它更接近深度相机看到的机器人轮廓；也可以设置 `use_collision_geometry=True` 做更粗糙但更轻量的实验。

## 已完成验证

MuJoCo segmentation 只在测试中作为真值，不进入实际过滤流程。episode 1/2 的逐像素对比结果：

| Episode | 方法 | 机械臂残留 | 删除召回率 | 环境误删 |
|---|---|---:|---:|---:|
| 1 | spheres | 567 | 86.0% | 12 |
| 1 | URDF | 240 | 94.1% | 12 |
| 2 | spheres | 638 | 84.1% | 14 |
| 2 | URDF | 249 | 93.8% | 15 |

测试脚本和产物：

```text
tests/initial_alignment_investigation/compare_urdf_robot_point_filter.py
tests/initial_alignment_investigation/urdf_vs_spheres_robot_filter_ep1_ep2.png
tests/initial_alignment_investigation/urdf_vs_spheres_robot_filter_ep1_ep2_3d.png
tests/initial_alignment_investigation/urdf_vs_spheres_robot_filter_ep1_ep2.csv
```

episode 2 的框架集成检查中，URDF 删除了 3810 个深度像素，hybrid 后进入 surface sampling 的点数为 61657；cuRobo 成功返回 `PlanResult` 和 41 个 waypoint，没有触发回退。

## 后续验证

1. 对相同起点、目标和规划配置比较 episode 1/2 的完整规划路径长度。
2. 运行 20 个 initial states，确认成功 episode 没有发生回归。
3. 分解 URDF 残留点：区分“URDF 没有射线命中”和“命中但深度误差超过容差”，并按 MuJoCo geom name 统计。
4. 单独测量 CPU URDF 渲染耗时。

真机迁移时接口不变，只需提供同步关节角、经过标定的相机参数、机器人 base 位姿，以及与实体机器人一致的 URDF/mesh。

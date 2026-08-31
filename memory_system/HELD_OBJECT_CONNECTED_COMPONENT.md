# Held Object：GPU URDF 与无 Oracle 连通体提取

本文档记录 frame 110 上最终采用的 held-object 点云提取诊断流程。该流程的提取输入兼容真机；simulator segmentation 只在提取完成后用于评分和可视化，不参与机器人删除、连通体聚类或组件选择。

## 最终流程

```text
metric RGB-D depth
  -> GPU URDF depth rendering
  -> observed/rendered depth consistency mask
  -> cuRobo robot collision-sphere filter
  -> EEF 周围 0.25 m ROI
  -> 对全部 remaining points 做 3-D DBSCAN
  -> 选择平均 EEF 距离最近的有效组件
  -> held-object 点云
```

提取过程中没有使用：

```text
simulator instance segmentation
simulator geom segmentation
target object ID
robot geom ID
物体真值位姿
```

## GPU URDF 机器人删除

诊断脚本使用 `WarpUrdfDepthFilter`：

1. 根据当前 7 个 arm joints 和 2 个 gripper joints 设置 Franka URDF。
2. 将 posed URDF visual mesh 上传到 NVIDIA Warp。
3. 使用 `wp.mesh_query_ray()` 在 CUDA 上并行渲染机器人公制 z-depth。
4. 只有 observed depth 与 rendered robot depth 满足下式时才删除该像素：

```text
abs(observed_depth - robot_depth)
    <= 0.008 + 0.005 * robot_depth
```

5. 将过滤后的深度反投影为世界点云。
6. 再使用 cuRobo 当前关节状态生成的 robot collision spheres 做一次兜底删除；默认 `robot_padding = 0.015 m`。

本次运行通过：

```bash
CUDA_VISIBLE_DEVICES=3 \
/data1/liu/miniconda3/envs/cosmospolicy/bin/python \
tests/held_object/visualize_held_object_after_robot_removal.py
```

Warp 使用逻辑 GPU `cuda:0`，对应上述命令暴露的物理 GPU 3。

## 连通体提取

GPU URDF 与 sphere filter 完成后，直接对所有 remaining points 运行连通体提取。不会先用 simulator oracle 删除 residual robot。

当前参数：

```text
frame                 = 110
EEF ROI radius        = 0.25 m
DBSCAN eps            = 0.025 m
DBSCAN min_samples    = 10
minimum cluster size  = 20 points
```

对 ROI 内所有点运行 DBSCAN 后，忽略 noise 和小于 20 点的组件，再选择平均 EEF 距离最小的组件作为 held object。

本次候选组件中，被选组件为：

```text
held points           = 1439
mean distance to EEF  = 0.0714 m
```

## Simulator segmentation 的使用边界

组件已经选定并生成 `is_held` 后，脚本才读取 simulator segmentation，用途仅限：

- 计算 target precision / recall。
- 统计橙色组件中混入的 residual robot 点。
- 将未被选中的 residual robot 点标成红色，便于检查过滤质量。

segmentation 结果不会反向修改 `is_held`，也不会重新选择 DBSCAN 组件。

本次事后评分：

```text
target precision              = 0.9673
target recall                 = 0.9858
robot points inside held      = 9
unselected residual robot     = 50
total residual robot          = 59
other unselected points       = 38278
```

其中 9 个 residual robot 点已经被无 oracle 的提取器选进 held component，因此在图中仍按算法输出显示为橙色；其余 50 个 residual robot 点显示为红色。

## 可视化颜色

```text
橙色：无 oracle 提取器选中的 held-object component
红色：未被选中、仅由事后 oracle 标出的 residual robot
灰色：其他未选中的点
```

图中的橙色始终表示提取器真实选择的组件，不会为了让结果更干净而用 simulator 标签删点。

## 当前代码与输出

诊断脚本：

```text
tests/held_object/visualize_held_object_after_robot_removal.py
tests/held_object/visualize_libero10_held_object_cases.py
```

最终输出：

```text
tests/held_object/libero10_gpu_urdf_no_oracle/*_multiview.png
tests/held_object/libero10_gpu_urdf_no_oracle/summary.csv
```

## 真机输入边界

提取所需输入为：

```text
实时 RGB-D
相机内外参
实时机械臂与夹爪关节角
实时 EEF 位姿
机器人 base 位姿
与实体机器人一致的 URDF 和 mesh
```

该诊断脚本本身尚未接入 held-object planner runtime；这里验证的是 GPU URDF 机器人删除和无 oracle 连通体提取的数据路径。

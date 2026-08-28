# Initial Alignment 问题调查记录

## 背景

任务：

```text
put the black bowl in the bottom drawer of the cabinet and close it
```

原始失败视频：

```text
scripts/rollouts/08-26/2026_08_26-16_19_03--with_future_img--episode=2--success=False--task=put_the_black_bowl_in_the_bottom_dr.mp4
```

现象：

- episode 2 在 initial alignment 后没有正确到达 memory 位置；
- 从视频看，机械臂停在柜子/瓶子附近，而不是 memory 对应的位置；
- 同一批 20 个 episode 中大多数成功，少数失败；
- 重新运行后 episode 2 仍复现类似错误。

## 相关代码与文件

### 诊断脚本（tests/initial_alignment_investigation/ 下）

```text
tests/initial_alignment_investigation/diagnose_initial_alignment_path.py
tests/initial_alignment_investigation/visualize_initial_alignment_pointcloud.py
tests/initial_alignment_investigation/visualize_initial_alignment_pointcloud_overlay.py
tests/initial_alignment_investigation/visualize_initial_alignment_pointcloud_labeled.py
tests/initial_alignment_investigation/visualize_initial_alignment_pointcloud_with_path.py
tests/initial_alignment_investigation/visualize_initial_alignment_pointcloud_with_path_views.py
tests/initial_alignment_investigation/visualize_initial_alignment_pointcloud_with_path_cropped_views.py
tests/initial_alignment_investigation/visualize_robot_point_filter.py
tests/initial_alignment_investigation/analyze_robot_point_filter_across_episodes.py
tests/initial_alignment_investigation/analyze_pointcloud_other_names.py
tests/initial_alignment_investigation/visualize_initial_alignment_real_trajectory.py
tests/initial_alignment_investigation/analyze_real_trajectory_bottle.py
```

### 生产代码改动

```text
cosmos_policy/experiments/robot/libero/run_libero_eval.py
scripts/run_libero10_robotinit_20.sh
```

`run_libero_eval.py` 中新增了由环境变量控制的逐步行日志：

```text
COSMOS_DEBUG_INIT_ALIGN=1
```

开启后，每个 initial alignment correction step 会输出：

```text
[INIT_ALIGN_DEBUG] {"episode":..., "t":..., "correction_step_index":..., "waypoint_index":..., "waypoint":..., "eef_before":..., "action":..., "eef_after":..., "steps_remaining":...}
```

`run_libero10_robotinit_20.sh` 中默认开启：

```text
COSMOS_DEBUG_INIT_ALIGN=${COSMOS_DEBUG_INIT_ALIGN:-1}
```

## 已观察到的现象

### 1. 所有 episode 选择同一个 memory

日志显示所有 episode 都选中：

```text
demo=('demo_13',)
target=[-0.0588200204, -0.0857331678, 0.9660174251, 2.6380581856, -1.9302700758, 0.2832267880]
```

因此不是 memory 检索选择不同导致的问题。

### 2. 日志显示 correction 完成，但实际未收敛

原始日志：

```text
[INIT ALIGN] t=10: target=... steps=48
[CORRECTION] t=57: initial_align correction finished; status=completed; resuming policy
```

但真实 rollout 日志显示：

- episode 1 correction 后离 memory 约 `0.1715 m`；
- episode 2 correction 后离 memory 约 `0.2973 m`。

`status=completed` 并不代表已经到达 memory。

### 3. 机械臂点没有完全扣除

通过 MuJoCo segmentation 统计：

- 每个 episode 的原始点云中都有约 4000 个机械臂点；
- `filter_robot_points()` 只删除其中约 3300～3700 个；
- 每个 episode 仍残留约 500～800 个机械臂点。

20 个 episode 的残留量：

```text
init= 0 success=True  robot_total= 4042 robot_removed= 3475 robot_remaining=  567
init= 1 success=False robot_total= 4007 robot_removed= 3369 robot_remaining=  638
init= 2 success=True  robot_total= 4019 robot_removed= 3337 robot_remaining=  682
init= 3 success=True  robot_total= 4206 robot_removed= 3508 robot_remaining=  698
init= 4 success=True  robot_total= 3954 robot_removed= 3173 robot_remaining=  781
init= 5 success=True  robot_total= 4227 robot_removed= 3557 robot_remaining=  670
init= 6 success=True  robot_total= 4191 robot_removed= 3403 robot_remaining=  788
init= 7 success=False robot_total= 4170 robot_removed= 3640 robot_remaining=  530
init= 8 success=True  robot_total= 4088 robot_removed= 3361 robot_remaining=  727
init= 9 success=False robot_total= 4166 robot_removed= 3493 robot_remaining=  673
init=10 success=False robot_total= 3997 robot_removed= 3305 robot_remaining=  692
init=11 success=True  robot_total= 4266 robot_removed= 3754 robot_remaining=  512
init=12 success=True  robot_total= 4115 robot_removed= 3516 robot_remaining=  599
init=13 success=True  robot_total= 4128 robot_removed= 3451 robot_remaining=  677
init=14 success=True  robot_total= 4081 robot_removed= 3336 robot_remaining=  745
init=15 success=True  robot_total= 4193 robot_removed= 3639 robot_remaining=  554
init=16 success=True  robot_total= 4027 robot_removed= 3269 robot_remaining=  758
init=17 success=True  robot_total= 4142 robot_removed= 3518 robot_remaining=  624
init=18 success=True  robot_total= 4279 robot_removed= 3687 robot_remaining=  592
init=19 success=False robot_total= 4162 robot_removed= 3500 robot_remaining=  662
```

结论：机械臂点残留是系统性问题，成功和失败 episode 都有，不是失败 episode 独有的原因。

### 4. 点云中 `other` 类别的主要构成

通过 geom name 统计：

```text
15249  None
 5183  wine_rack_1_g0
 1674  wine_rack_1_g6
 1447  mount0_pedestal_vis
  330  wine_bottle_1_g0
  210  wine_rack_1_g3
   20  wine_bottle_1_g1
   17  wine_bottle_1_g2
    8  mount0_torso_vis
```

即 `other` 主要是无名称背景点、酒架、酒瓶、机器人底座等。

### 5. 真实轨迹与瓶子关系

使用真实日志重建轨迹后，统计路径与瓶子的关系：

#### episode 1（成功）

```text
bottle=[-0.151, 0.060, 0.899]
logged_waypoint_path:
  min_dxy_to_bottle=0.0221
  max_z_rel_bottle=0.2749
  n_over_bottle=20/20
  end_dxy_to_bottle=0.1073
  end_z_rel_bottle=0.2227
```

episode 1 的路径水平方向非常靠近瓶子中心，并且高度在瓶子上方，看起来是“从瓶子上方跨越”。

#### episode 2（失败）

```text
bottle=[-0.171, 0.057, 0.899]
logged_waypoint_path:
  min_dxy_to_bottle=0.0425
  max_z_rel_bottle=0.2475
  n_over_bottle=17/17
  end_dxy_to_bottle=0.1245
  end_z_rel_bottle=0.0870
```

episode 2 的水平方向离瓶子中心更远，路径更可能是“绕过瓶子”而不是“从瓶子上方跨越”。

实际 EEF 轨迹与各自 waypoint 路径基本一致，说明差异主要来自规划阶段，而不是执行阶段。

## 总结

- memory 选择不是 episode 差异的来源。
- 机械臂点没有完全扣除，但这是所有 episode 都存在的系统性问题。
- 真实日志显示 correction 结束后并没有真正到达 memory。
- episode 1 与 episode 2 的规划路径在瓶子附近明显不同：
  - episode 1 的路径更靠近瓶子正上方；
  - episode 2 的路径更偏向绕开瓶子。
- 最终 episode 2 停在瓶子这一侧，无法到达 memory 位置。

## 未知

- 为什么 episode 2 的 Curobo 规划会选择“不跨越瓶子”的路径。
- 是初始关节构型导致“跨越瓶子”的路径运动学不可行，还是点云中瓶子附近的障碍导致该路径被认为不可行。
- 机械臂残留点是否在瓶子附近对规划产生了实质影响。
- 真实日志只记录了实际访问到的 waypoint，未记录 Curobo 的完整规划 waypoint，无法确认规划器是否尝试过“跨越瓶子”的路径但失败。
- 日志中 `INIT_ALIGN_DEBUG` 从 `t=11` 开始，前两步没有记录，真实轨迹缺少最开始的 2 步。
- 当前复现使用的代码与原始 rollout 时的代码是否完全一致尚不确定。

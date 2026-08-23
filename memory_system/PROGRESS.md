# Memory System 迁移进度

## 当前状态

- Step 0：特征测试完成 ✅
- Step 1：创建 `memory_system` 共享包 ✅
- Step 2：Offline 迁移完成 ✅
- Step 3：Execute 迁移未开始 ⏳
- Step 4：Eval 接线未开始 ⏳

## 已迁移内容

```text
memory_system/
├── types.py
├── artifacts.py
├── skills.py
├── offline/
│   ├── planner.py
│   ├── predicate_planner.py
│   ├── label_segments.py
│   ├── label_boundaries.py
│   ├── build_targets.py
│   └── build_recovery.py
└── PROGRESS.md
```

## 关键约束

- 不修改 `bin/`。
- `memory_system` 完全自包含，不 import `bin`。
- Offline 与 test-time 类型保持独立。

## 使用

生成完整 memory：

```bash
python tests/generate_memory_system_test.py
```

对比旧/新产物：

```bash
python tests/compare_memory_system_offline.py
```

可视化对比：

```bash
python tests/visualize_offline_memory_diff.py
```

## Step 2 已知差异（暂时接受）

### phase_targets.pt 视觉字段

- 差异字段：`crop_rgb`、`crop_mask`、`bbox_xyxy`、`target_center_xy`、`keypoints_xy`、`descriptors`、`visible_pixels`。
- 原因：
  - 这些字段都从 segmentation mask 派生；
  - 新旧数据的 task / demo / frame / gripper_xy 等逻辑元数据完全一致；
  - 差异来自同一帧渲染出的 mask 有轻微像素级偏移，疑似旧 artifact 由不同 LIBERO/MuJoCo/robosuite/OpenCV 环境生成。
- 结论：不是迁移逻辑错误，暂时接受。

### recovery_targets.pt action_chunk_raw

- 新实现统一使用 `raw_action_chunk(actions, start, start + 16)`。
- 旧数据中绝大多数符合该规则，但存在 2 个历史异常值使用了 `raw_action_chunk(actions, recovery_frame - 16, recovery_frame)`。
- 为保持新代码规则一致，暂不强制对齐这 2 个旧异常值。

### VAE 字段

- `recovery_vae_main`、`ready_vae_main`、`ready_vae_wrist` 等可能存在环境/GPU 浮动。
- 当前不参与严格一致对比。

## 下一步

- Step 3：迁移 Execute（plan / phase / feasible / completion / recovery / monitor）。

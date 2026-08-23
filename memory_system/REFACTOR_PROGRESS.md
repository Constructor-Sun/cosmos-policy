# Memory System 迁移进度

## 当前状态

- Step 0：特征测试完成 ✅
- Step 1：创建 `memory_system` 共享包 ✅
- Step 2：Offline 迁移完成 ✅
- Step 3：Execute 迁移完成 ✅
  - 已迁移 plan / phase / feasible / completion / recovery / monitor 到 `memory_system/execute`
  - 已添加新旧 execute 对照测试 `tests/test_memory_system_execute_consistency.py`
  - 验收通过：`tests/test_memory_system_execute_consistency.py` 7 passed，相关原有测试 41 passed
  - `memory_system/execute` 不依赖 `bin`，Step 3 阶段未修改 `bin/execute`
- Step 4：Eval 接线完成 ✅
  - `run_libero_eval.py` 已切换到 `memory_system.execute`
  - 已删除 `bin` 的 `sys.path` 依赖
  - 运行时 memory 默认使用 `skill_memory_test/libero_10`
  - verifier/recovery 调用统一使用 `VerifierObservation`
  - 更新 `tests/test_libero_eval_verifier_wiring.py` 锁定新接线
  - 验收通过：相关测试 42 passed
  - robotinit phase+feasible 实测：10/10 成功，FEASIBLE RECOVERY 触发 12 次，WRONG GRASP 触发 2 次
- Step 5：兼容入口与清理完成 ✅
  - 当前正式调用已全部走 `memory_system`
  - `utils/eval_libero_phase_verifier.py` 和 `utils/eval_libero_feasible_region_verifier.py` 已切换到 `memory_system`
  - 旧 `bin/memory` / `bin/execute` 已从工作区删除（Git 历史仍保留）
  - 依赖旧实现的旧测试已移除，后续按功能重写
  - 当前保留测试：`test_memory_system_artifacts.py` + `test_libero_eval_verifier_wiring.py`，12 passed
  - 最终回归确认通过：robotinit phase+feasible 再次运行 10/10 成功

## 已迁移内容

主迁移/重构文档：`REFACTOR.md`

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
├── execute/
│   ├── __init__.py
│   ├── plan.py
│   ├── phase.py
│   ├── feasible.py
│   ├── execution_monitor.py
│   ├── skill_completion/
│   │   ├── __init__.py
│   │   ├── _common.py
│   │   ├── pick.py
│   │   ├── place.py
│   │   └── open_close.py
│   └── recovery/
│       ├── __init__.py
│       ├── retrieval.py
│       ├── selectors.py
│       └── controller.py
└── REFACTOR_PROGRESS.md
```

## 关键约束

- 不修改 `bin/`。
- `memory_system` 完全自包含，不 import `bin`。
- Offline 与 test-time 类型保持独立。
- Step 3 只迁移 Execute 到新 package，不切换 Eval 接线。

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

Execute 新旧对照测试：

```bash
python -m pytest -q tests/test_memory_system_execute_consistency.py
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

- 当前迁移（Step 0 ~ Step 5）已全部完成 ✅
- 最终回归确认已通过：robotinit phase+feasible 10/10
- 后续主要方向：
  - 继续 3D / RGB-D 扩展
- 后续待办：
  - 按功能重写旧的 offline / execute / recovery / completion 测试

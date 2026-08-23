# Stage 1: Depth Correctness（测试专用，不参与 test-time）

对应 `memory_system/3D.md` 的 **Stage 1**。目标：确认从 LIBERO RGB-D observation
得到的**米制 depth 正确**，为后续 Phase/Pick 的 3D 规则打好地基。

## 设计边界

- **正式运行逻辑只消费米制 depth + 相机参数 + 机器人本体状态**，不读取 simulator
  object state（真机兼容约束，见 3D.md §3）。
- 本目录全部是 **test-only** 代码：`harness.py` / `conftest.py` / 各 `test_*.py` /
  `visualize_depth.py` 只用于测试和可视化，**任何正式 runtime 代码不得 import 本目录**。
- 模拟器内容（`sim.render`、`sim.model` 的 znear/zfar/extent、`robot0_eef_pos`）
  在本阶段只作为测试参考，不进入被测试的转换路径。

## 三个检查

| 文件 | 检查 | 做法 | 参照 |
|---|---|---|---|
| `test_depth_plumbing.py` | obs 键/形状/数值 | `obs["agentview_depth"]` == `sim.render(depth=True)`（同渲染器，只查接线） | sim.render |
| `test_depth_conversion.py` | 米制转换正确性 | implied-near 一致性：从每个像素反解 near，必须恒等于模型 `znear×extent` | sim.model near/far |
| `test_depth_flip.py` | flip 对齐 | A) flip_depth == np.flipud（精确）；B) flip 感知反投影 == 未 flip 反投影（精确）；C) EEF 世界点投影→flip→反投影→再投影，落在同一 flip 像素（±2px） | 自洽 + EEF |

## 运行

```bash
conda activate cosmospolicy
cd /data1/liu/exp/counterfactual/external/cosmos-policy
pytest tests/stage1_depth/ -v
```

可视化（生成深度图到 `tests/stage1_depth/artifacts/`）：

```bash
python tests/stage1_depth/visualize_depth.py \
    --task KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
```

## flip 约定：什么时候 flip、什么时候不 flip（核心文档）

**两个像素空间**：

- **render space（raw）**：`obs["agentview_image"]` 的原始方向（GL 约定，上下颠倒）。
  相机几何（K、T_w2c、T_c2w、`project_points_from_world_to_camera` 输出）全部在此空间。
- **canonical space（flipped）**：对 raw 图做 `np.flipud` 后的方向（校正后，视觉正确）。
  memory system 的**存储与匹配空间**：离线模板、target 中心、bbox、`gripper_xy`、
  在线 `third_view_rgb` 都在这里。

两空间只差一行镜像：

```text
row_canonical = H - 1 - row_render        # forward：world -> canonical 像素
row_render    = H - 1 - row_canonical     # backward：canonical 像素 -> 相机几何
```

**何时 flip（实测确认的事实）**：

| 环节 | flip？ | 说明 |
|---|---|---|
| 数据源头：HDF5 demo 图 | 已 flip（存的就是校正方向） | 实测：模板 crop 与 HDF5 原图 corr=1.0；HDF5 图与 flipped 渲染 corr 0.68、与 raw 渲染仅 0.36 |
| 离线构建模板（`build_targets.py`, flip_images=True） | 模板 crop 直接切 HDF5 图（已 flip）；mask 用 `flipud(seg)` → 产物在 canonical 空间 | 离线产物 = canonical |
| 离线 `gripper_xy` | **flip**：`world→pixel` 投影（raw）后镜像 row（`row = H-1-row`） | 实测：模板 gripper_xy == EEF 的 flipped 投影 |
| 在线 `third_view_rgb`（`prepare_observation`） | **flip**：`np.flipud(raw obs)` | 与模板同空间，phase 匹配一致 |
| 在线 `_project_gripper_xy` | **当前不 flip**（`run_libero_eval.py` NOTE 明确不要） | ← 现状：raw，与 canonical 不一致（见下） |
| 反投影 pixel→world（RGB-D 未来 / 本测试） | **先 unflip row 再进相机几何** | `harness.back_project_flipped` 即此约定 |

**forward / backward 两条规则（唯一约定）**：

```text
forward  (world -> canonical pixel):  用 T_w2c 投影到 raw (row, col)，
                                      然后 row_canonical = H-1-row（flip 时）

backward (canonical pixel -> world):  depth 在 canonical 像素 (r, c) 采样，
                                      先 row_render = H-1-r 还原，
                                      再用 K + T_c2w 做针孔反投影
```

**当前已知的记账不一致（保持现状，不修改）**：

在线 `_project_gripper_xy` 返回 raw 坐标，而它下游对比的模板/bbox 在 canonical 空间。
经确认：recovery 主路径（VAE/EE pose 检索）与 phase 主判断（模板匹配共识）不依赖该
像素信号，辅助信号（phase progress/wrong_way、completion inside-bbox）带容差，
因此**系统行为正常、测试通过**。按决策保持现状；若未来统一，应让在线 gripper 与
离线一致（投影后镜像 row），但**未经授权不得修改生产代码**。

## 关键发现（本阶段实测确认）

1. **depth obs 与米制转换**
   - `obs["agentview_depth"]`：`(256, 256, 1)` float32，归一化 `[0.984, 0.995]`，
     等于 `sim.render("agentview", depth=True)` 的原始渲染（IMAGE_CONVENTION=opengl，无翻转）。
   - 米制转换 `get_real_depth_map`（near = znear×extent = 0.011831,
     far = zfar×extent = 591.55）在 1.0~2.3 m 深度范围内反解出的 near **恒等于** 模型值
     （相对误差 ~1e-6），即米制转换在数学上精确正确。
   - agentview fovy = 45°；`camera_depths=[True, False]`（main depth，wrist 保持 RGB）。

2. **环境限制：同一进程内不得同时存活两个 LIBERO env（EGL 上下文串扰）**
   - 实测：创建第二个 env 后，第一个 env 的**整个渲染管线**（含 `sim.render` 和
     `regenerate_obs_from_state` 的 sensor 路径）都会返回错误 depth。
   - 本目录 fixture 强制"同一时刻只有一个 env"（function-scoped，用完即关）。
   - **对生产代码的提醒**：任何路径都不要同时持有两个 LIBERO env（如 eval 按任务创建
     env 时，确认旧 env 已 close）。生产 eval 单 env 复用不受影响。

## 容差说明

- 5 mm 的逐像素精度断言属于 **Stage 2**（目标 3D 几何 vs simulator Oracle）。
- Stage 1 的 conversion 检查把米制值验证到 ~1e-6 相对误差（换算成 1 m 处 ~1 µm），
  已为 Stage 2 的 5 mm 断言提供前提。
- EEF 检查是**像素级**一致性（±2px）：depth 在夹爪像素处测的是手指表面，与腕部
  EEF 帧沿射线有数 cm 偏差，因此不做 3D 距离断言。

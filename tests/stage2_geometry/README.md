# Stage 2：物体可见表面深度测量

本目录是 **test-only** 的 Stage 2 深度校验实现与结果记录，不参与生产逻辑。

## 定义

对于像素 \((u,v)\)，深度定义为该像素第一处可见表面在相机坐标系中的 \(Z\)：

\[
D(u,v)=Z_{\text{camera}}
\]

它不是：

- 物体 world-frame 的 \(z\)；
- 相机到物体中心的欧氏距离；
- simulator object state 中的位置；
- 整个物体唯一的一个标量。

“一个物体的深度”使用物体可见 mask 对应的 masked depth map 表示；若需要标量，则使用 eroded mask 上的 median。

## Simulator Oracle

- RGB-D 路径输出米制深度 \(D_{\text{rgbd}}(u,v)\)。
- Test Oracle 使用 MuJoCo `mj_ray()` 从 simulator geometry 获取该 pixel 第一处表面交点。
- 将交点转换到相机坐标后取 \(Z\) 作为 \(D_{\text{oracle}}(u,v)\)。
- 在 eroded object mask 上比较逐像素误差。
- 不使用 `get_real_depth_map()` 的输出同时作为被测结果和 Oracle。
- 比较前保证 RGB、Depth、mask 使用相同分辨率、pixel convention 和 flip 方式。

## 支持的 skill

| skill | 测量对象 |
|---|---|
| Pick | `arguments.item` |
| PlaceIn | `arguments.item` + 可解析的 `arguments.target` |
| PlaceOn | `arguments.item` + 可解析的 `arguments.target` |
| TurnOn | `arguments.target` |
| Close | 可解析的 `arguments.target` |

`target` 如果是 region，会尽量映射到底层实例，例如：

- `basket_1_contain_region` → `basket_1`
- `white_cabinet_1_bottom_region` → `white_cabinet_1`
- `microwave_1_heating_region` → `microwave_1`
- `desk_caddy_1_back_contain_region` → `desk_caddy_1`
- `flat_stove_1_cook_region` → `flat_stove_1`

无法映射到 segmentable instance 的抽象 region（如 `living_room_table_plate_right_region`）不按物体深度测量。

## 代码

- `harness.py`：环境、相机、mask、`mj_ray()` oracle 等测试辅助。
- `measure_object_depth.py`：采样并计算逐像素深度误差。
- `results_object_depth.json`：当前结果。

运行：

```bash
conda activate cosmospolicy
cd /data1/liu/exp/counterfactual/external/cosmos-policy
python tests/stage2_geometry/measure_object_depth.py --max-demos 2
```

## 当前结果

### 总体

```text
n_samples = 299
n_ok      = 298
n_failed  = 1

median of medians: 1.15 mm
median of P90:     4.01 mm
max P90:         192.71 mm
```

唯一 failure：

```text
Pick / chocolate_pudding_1 / demo_0 / frame=110
reason: mask_too_small
```

这不是深度误差，而是该帧 mask 像素太少，未参与比较。

### 按 skill 的 ok 样本数

```text
Pick     95
PlaceIn 108
PlaceOn  77
TurnOn    6
Close    12
```

### by-object 结果

| object | n | median | P90 | max P90 | 状态 |
|---|---:|---:|---:|---:|---|
| KITCHEN::akita_black_bowl_1 | 12 | 0.78 mm | 3.39 mm | 4.37 mm | PASS |
| KITCHEN::flat_stove_1 | 24 | 1.23 mm | 1.47 mm | 1.51 mm | PASS |
| KITCHEN::microwave_1 | 12 | 1.55 mm | 182.22 mm | 192.71 mm | FAIL |
| KITCHEN::moka_pot_1 | 24 | 3.79 mm | 11.41 mm | 13.46 mm | PASS |
| KITCHEN::moka_pot_2 | 12 | 3.64 mm | 11.94 mm | 12.45 mm | PASS |
| KITCHEN::white_cabinet_1 | 12 | 1.30 mm | 1.95 mm | 2.00 mm | PASS |
| KITCHEN::white_yellow_mug_1 | 12 | 3.66 mm | 6.46 mm | 6.86 mm | PASS |
| LIVING_ROOM::alphabet_soup_1 | 24 | 0.68 mm | 1.61 mm | 4.85 mm | PASS |
| LIVING_ROOM::basket_1 | 36 | 0.95 mm | 7.49 mm | 9.10 mm | PASS |
| LIVING_ROOM::butter_1 | 12 | 1.12 mm | 1.27 mm | 2.18 mm | PASS |
| LIVING_ROOM::chocolate_pudding_1 | 10 | 1.18 mm | 1.41 mm | 1.50 mm | PASS |
| LIVING_ROOM::cream_cheese_1 | 24 | 1.25 mm | 1.58 mm | 2.67 mm | PASS |
| LIVING_ROOM::plate_1 | 12 | 1.03 mm | 3.90 mm | 4.33 mm | PASS |
| LIVING_ROOM::plate_2 | 6 | 1.77 mm | 5.98 mm | 6.97 mm | PASS |
| LIVING_ROOM::porcelain_mug_1 | 24 | 0.90 mm | 5.53 mm | 7.22 mm | PASS |
| LIVING_ROOM::tomato_sauce_1 | 12 | 0.73 mm | 1.97 mm | 2.75 mm | PASS |
| LIVING_ROOM::white_yellow_mug_1 | 12 | 3.64 mm | 5.57 mm | 6.08 mm | PASS |
| STUDY::black_book_1 | 12 | 1.06 mm | 1.39 mm | 15.65 mm | PASS |
| STUDY::desk_caddy_1 | 6 | 1.05 mm | 2.21 mm | 2.24 mm | PASS |

### 说明

- `microwave_1` 的 P90 明显偏高，需要单独确认是否由复杂/透明/内部结构物体的 mask 或 oracle 定义导致。
- 除该异常外，其余物体在 `median < 5mm, P90 < 2cm` 的判定下均通过。

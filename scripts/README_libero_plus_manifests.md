# LIBERO-plus：失败清单与 Memory 成功轨迹 P95 筛选

在 cosmos-policy 项目根目录运行，使用 cosmospolicy Python 环境。两个入口只读取已有 rollout，不加载模型、不启动 GPU。

## 1. 从 baseline 生成失败清单

```bash
python scripts/collect_libero_plus_failures.py
```

默认读取 experiments/libero_plus/cosmospolicy_full_7dims/run_config.json 及其计划中的逐 case result.json，输出：

```text
experiments/tta/repair_libero_plus_435/inputs/
├── failed_435.json
└── failed_435_generation.json
```

筛选条件是 status == completed 且 success 为 Boolean false。脚本校验 task ID/name/group/category、fingerprint、固定 seeds 与协议，补充 base_task、clean_language 和 baseline_result，并确认失败 case 的 HDF5、actions、视频存在。原始 instruction 保留，Language 改写不会被 clean_language 替换。

计划中未完成或发生运行错误的 case 默认导致失败退出；显式 --allow-incomplete 可导出已经完成的失败 case，输出会标记 incomplete，错误 case 不计入失败数。

## 2. 生成 P95 筛选与各扰动划分

```bash
python scripts/select_libero_plus_success_p95.py
```

默认读取上面的失败清单，合并 smoke_4/ 与 remaining_431/ 的逐 case 修复结果；使用 LIBERO-Cosmos-Policy/success_only/libero_10_regen/*_demo.hdf5 计算同 base task 的动作长度 P95，numpy percentile 的 method 固定为 linear。

只选完成且修复成功的完整轨迹，并要求 full action_steps <= 同任务 P95。长度包含 baseline 前缀、Memory 干预和 policy continuation，排除 settling；不按 t_star 截成修复片段。脚本检查成功属性、动作/proprio/图像长度、有限值、actions.npy 与 HDF5 一致，以及修复协议和 case 身份。

默认输出：

```text
experiments/tta/repair_libero_plus_435/success_task_p95/
├── selected_manifest.json          # 通过 P95 的成功轨迹，可传给 SFT
├── excluded_manifest.json          # 修复成功但超过 P95
├── summary.json                    # 各基础任务与扰动的统计
├── task_summary.csv
├── generation.json                 # 输入路径/hash、完整性与筛选数量
└── by_category/<group>/
    ├── training_manifest.json      # 该扰动选中的完整成功轨迹
    ├── heldout_manifest.json       # 该扰动原始失败 case 中未参与训练者
    ├── memory_failed_manifest.json # 已完成修复、但修复仍失败者
    └── evaluation_cases.json       # 全部原始失败 case，标记 in_training
```

heldout = 原始失败 case − training，包含 P95 排除者和 Memory 修复失败者；memory_failed 是 heldout 的子集。完整输入下，training 与 heldout 不重叠且覆盖全部原始失败 case。没有进行比例随机划分。

selected_manifest.json 与 training_manifest.json 使用现有 SFT 兼容字段。运行单扰动 SFT 时传入相应 training_manifest.json，并使用匹配的 --expected-cases。Sensor Noise 等现有 scope 需要的 evaluation_subset 也会生成。Background、Light 的完整失败集评测仍使用各自实验 pipeline；此脚本不修改公共训练调度器的 scope。

evaluation_cases.json 可以作为 scripts/run_libero_plus_batch.py 的 --task-ids-file。完整训练数据仍位于原修复 HDF5；新清单不复制轨迹，也不创建 episodes 软链接视图。

默认要求所有失败 case 的修复均已完成。显式 --allow-incomplete 会跳过缺失/running/error 修复；它们仍属于未参与训练集合，但不能计入 memory_failed。重复修复 case 会报错，不会自动选择某次尝试。

## 3. 新数据或独立重建目录

```bash
python scripts/collect_libero_plus_failures.py \
  --baseline-root /path/to/baseline \
  --output /path/to/output/inputs/failed.json

python scripts/select_libero_plus_success_p95.py \
  --failure-manifest /path/to/output/inputs/failed.json \
  --repair-runs /path/to/repair_run_1 /path/to/repair_run_2 \
  --demo-dir /path/to/success_demos \
  --output-dir /path/to/output/success_task_p95
```

--base-tasks 可覆盖 baseline 配置中的基础任务映射。新的 baseline 接入现有 SFT 时，需同时对齐训练脚本仍固定读取的 experiments/tta/repair_libero_plus_435/inputs/failed_435.json；不要用旧基线清单训练新基线修复数据。

两个脚本均支持 --dry-run，只核对输入和输出统计，不写文件。等价的已有 JSON 保留原始字节及 hash；不同的已有输出默认拒绝覆盖。需要更新时使用新输出路径，或明确加 --overwrite。开始写入前会检查全部输出路径，每个文件通过临时文件替换。

## 4. 已验证的历史重建

2026-10-13，在 manifest_tools_validation_20261013/ 下独立重建：

- Baseline：2519 个已完成 case，失败清单 435 个。
- Memory：229 成功、206 失败。
- P95：保留 190，排除 39；失败/选中/排除清单与历史文件逐字节一致。
- summary.json 的 JSON 内容、task_summary.csv 与历史结果一致。
- 各扰动 training 数量：Robot Initial 90、Camera 24、Background 29、Sensor Noise 22、Language 8、Light 14、Objects 3。

验证报告：experiments/tta/repair_libero_plus_435/manifest_tools_validation_20261013/validation_report.json。验证没有覆盖历史实验文件。

CPU 回归测试：

```bash
python -m unittest discover -s tests -p test_libero_plus_manifest_tools.py -v
```

覆盖运行错误与真实失败的区分、协议变更、重复 case、缺失轨迹、分数 P95 边界、完整轨迹长度、保存动作一致性、集合划分、输出冲突等情况。

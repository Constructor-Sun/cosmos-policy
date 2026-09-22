# LIBERO-PRO `_task` 分支的兼容步骤（2026-09-22 · 第 2 版）

目标：用当前这套修复流程（记录选择 + 相位判定 + 记忆回放）去测 `libero_10_task` 分支。

结论：**流程本身变体无关；task 分支缺的不是代码，而是三份"同口径输入"**——census（含 t\*）
与两组记忆记录。指令、T5、阶段计划、身份链都已就绪。task 与 swap 各走各的目录，**不合并结果**
（census 顶部带 `flavor`/`suite`，两条分支天然分流）。

> 第 2 版说明：第 1 版（同日早些时候）的三条判断经逐条核对已与当前工作树不符，见 §7 勘误。
> 本文按当前代码重写。标 ★ 的是本次新核实的事实。

## 1. 已经就绪、不需要动的部分

### 1.1 ★ 指令（eval → env）：benchmark 自己读 BDDL

`grab_language_from_filename(x, folder)` 在传了 folder 时**先读 BDDL 的 `(:language)`**
（`LIBERO-PRO/libero/libero/benchmark/__init__.py:46-53`），而 `_make_benchmark` 正是传 folder 的
（同文件 `:162`）。用 §3 的 env 实测：

```
libero_10      -> 'turn on the stove and put the moka pot on it'
libero_10_swap -> 'turn on the stove and put the moka pot on it'
libero_10_task -> 'turn on the stove and put the pan on it'
```

`COSMOS_SMOKE_PAIR_CLEAN_LANGUAGE` 只喂 paired 模式的 synthetic `clean_task`
（`run_libero_smoke_test.py:945/:968`），只在 clean 条件用；PRO 修复跑的是
`SMOKE_ONLY_CONDITION=perturb`，clean 分支被跳过 —— 这条 env 在修复流程里**没有读取点**。
`oracle_ready_eval._patch_get_libero_env(language_from_bddl=True)` 仍然有效（定义 `:139`，
调用 `:895-896`，只在 oracle 入口），但在 benchmark 已读 BDDL 之后属于重复保险。

### 1.2 ★ T5：已补齐，且 PRO 路径本来就是 strict

策略目录 `checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl` 现有 **74 条**，
`libero_10` / `libero_10_swap` / `libero_10_task` 各 10 条指令 **10/10 命中**
（2026-09-20 已跑过 `scripts/libero_pro/precompute_t5.py`，旁边有 `.backup`）。

PRO（`pert_category` 为空）会自动取 strict（`run_libero_smoke_test.py:1035-1041`）：启动时先用
policy dir 的 pkl 做 T5 key 预检，缺失直接
`RuntimeError("Missing strict language perturbation T5 embeddings…")`（`:1071-1091`）。所以
`instruction_mode == "base"` 那条"静默换成 base 指令嵌入"的路径在 PRO 上**到不了**；
`COSMOS_SMOKE_INSTRUCTION_MODE` 不必手设（`SMOKE_T5_FALLBACK_TO_BASE` 是 shell 的正常默认项，
有默认值 true）。

### 1.3 ★ 身份链：census 决定分支

census json 顶部带 `flavor` / `suite`；`diagnose_failed.py` 的 `flavor == "pro"` 分支用记录里的
suite 建 benchmark（`:260-265`，缺 suite 直接报错 `:426`）；`tta_repair_batch.py` 的 pro 分支
`pair_suite = meta["suite"]`、`pert_category = ""`（`:243-263`）。pro 分支还会自带
`COSMOS_LIBERO_ROOT` / `LIBERO_CONFIG_PATH` / `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`。

## 2. 必须新产生的三份输入

### 2.1 census json（本次手写）

`experiments/tta_census/libero_pro_task_full10.json`：

- 顶层 `flavor: "pro"`、`suite: "libero_10_task"`
- 9 条 task，每条 `n_fail: 10`、`fail_init_indices_abs: [0..9]`、`task_name_perturbed` 与 `task` 同名
- `variant_state` 只是记账（pro 分支在 `run_case` 里把 condition 固定成 `pro_variant`）
- **共 90 个 case**，与 swap 的"每任务取前 10 个 init"约定一致（swap census 就是 `init000..init009`）

数据来源：`experiments/liberopro/libero_10_task_seed7`（2026-09-16 基线，20 trials/task，19/200）。
9 个任务是 0/20，所以前十个 init 全是失败；KSCENE8 见 §6。该基线是 `run_libero_eval` 直跑
（`initial_states_path='DEFAULT'`、`deterministic_reset=False`、`enable_initial_alignment=False`），
eval 循环是 `initial_states[episode_idx]`（`run_libero_eval.py:1835`）→ 日志里 `episode=N`
（1-based）对应 **init index N-1**。

### 2.2 census 目录（phase_record + episode.h5 + t\*）

```bash
CENSUS=experiments/tta_census/libero_pro_task_full10.json \
OUT=experiments/tta_phase_check_pro_task_full10 \
sh scripts/libero_pro/run_diagnose_smoke.sh
```

每个 case 产出 `phase_record.json`（带 `flavor`/`suite`/`bddl_file`）、`episode.h5`、`tstar.png`。
注意 `episode.h5` 是 **diagnose 重跑时**写的（`diagnose_failed.py:357`），不是基线产出的
（基线那轮 `data_collection=False`）。t\* 也在这时候定：有事件则用事件帧，没有则回退 10。

### 2.3 记忆记录：LIVING_ROOM_SCENE5 的两个缺口（已补）

LRSCENE5 是 10 条变体里唯一"交换关系"的一条：两个杯子的盘子对调，于是需要
`white_yellow_mug_1→plate_1` 和 `porcelain_mug_1→plate_2`，而构库（LIBERO-90）只有反方向。

已用 `scripts/libero_pro/add_place_records.py` **同杯子换盘子**追加 13 条（6 + 7），
只改 `arguments['target']` 与 `memory_id`；原文件备份为
`pointcloud_action_memory_place.pt.backup`（407 → 420 条）。

为什么换盘子是保帧的：这两条记录的 `anchor_role = destination`，`T_object_ee_ready` 是相对
**目的地盘子**的位姿；`plate_1 plate_2 - plate` 同类型、同一资产（稳定扫描物体，27.4cm 圆盘、
旋转对称），两盘只差位置。item 不动，所以不引入任何物体几何差异
（对比：`porcelain_mug` 与 `white_yellow_mug` 是两套网格，8322/16644 顶点面 vs 4227/8454，
高度差约 1cm —— 换 item 的复制才会有这个误差）。

运行时真正读取的字段是 `arguments`（全等匹配）、`T_object_ee_ready` 与回放段
（`ee_pose_object_sequence` / `gripper_sequence` / `ready_frame` / `segment_end`）；
`T_world_object_anchor`、`target_points_*`、`complete_points_*` 由运行时 `object_frame` 重算或
只用于离线诊断。`RECORD_FIELDS` 只是字段清单，没有校验调用点。

⚠️ place 库是 `memory_system/offline/build_recovery.py` 的产物，重建会覆盖这 13 条，
届时重跑 `scripts/libero_pro/add_place_records.py`。

## 3. 起跑环境（照抄，缺一项行为就会变）

```bash
export PATH=/data1/liu/miniconda3/envs/cosmospolicy/bin:$PATH     # shell 里 exec python
export COSMOS_LIBERO_ROOT=/data1/liu/exp/counterfactual/external/LIBERO-PRO
export LIBERO_CONFIG_PATH=$COSMOS_LIBERO_ROOT/configs/libero_pro
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1                        # PRO init-state 是 legacy pickle
export COSMOS_INITIAL_ALIGNMENT=1 COSMOS_TTA_RGBD=1 COSMOS_SKILL_COMPLETION_SHADOW=1
export COSMOS_TTA_PHASE_LOOP=1                                   # ← 漏了会静默退回"单次修复"
export COSMOS_SKILL_READY_MEMORY=timegrip                        # 对齐 + 记忆段回放
export COSMOS_DATA_COLLECTION=1 SMOKE_DATA_COLLECTION=1
export GPU_ID=<g> MUJOCO_EGL_DEVICE_ID=<g>                       # smoke 脚本据此设 CUDA_VISIBLE_DEVICES
```

- `COSMOS_TTA_PHASE_LOOP` 必须由外层 export：`tta_repair_batch.run_case` 的 env 块
  （`:265-297`）**不设**这一项；不设时 `_phase_loop=False`（`run_libero_eval.py:494-495`），
  只有 request 里没有 `plan` 才会打印提示（`:511`），否则静默退化成单次修复。
- `COSMOS_SKILL_READY_MEMORY` 在 `tta_repair_batch.py:253` 的默认是 `align`（只对齐）；
  `timegrip` = 对齐 + 回放记忆段。
- `SMOKE_PAIR_*`、`COSMOS_INIT_STATE_OFFSET`、`COSMOS_TTA_REPAIR{,_OUT,_TAG}` 不在上面的手抄
  列表里，由 §4 的入口（batch 或 runner）按 census/meta 设置。

## 4. 起跑方式

- 批量：每个 GPU 一条链

  ```bash
  python scripts/tta_repair_batch.py --gpus <g> --tags <子集> \
      --work-dir <绝对路径> --result-dir <census 目录> --diagnosis-root <census 目录>
  ```

  `--tags` 非空时它**不会**自己分流（`tta_repair_batch.py:334-347` 只跑 `chains[gpus[0]]`），
  要按 GPU 各起一个。`--diagnosis-root` 会重建 meta/summary/requests，suite 取自 phase_record。

- 单 case（调试）：`scripts/libero_pro/run_single_case.py`（原 `/tmp/run_k3_turnon1_liu.py`，
  已搬进仓库并参数化：suite 优先取 meta 的 `suite` 字段，meta / requests / out 都是参数）：

  ```bash
  python scripts/libero_pro/run_single_case.py --tags KSCENE3-init006 --gpu 3 \
      --meta <diagnosis_meta.json> --requests-dir <requests> --out-dir <绝对路径>
  ```

## 5. 跑起来应该看到的判据

```
[PHASE_LOOP] plan [[...]]                                  # 阶段计划来自 BDDL
[ORACLE] TurnOn 取第 1 条匹配记录（共 40 条）               # 选条生效
[ORACLE] Pick 'xxx' 取抬升 x.xxcm 的记录（共 N 条，阈值 2cm）
[ORACLE] PlaceOn 'xxx'->'yyy' 取工具轴偏竖直 xx° 的记录
[SKILL_COMPLETION] advance phase=k reason=rule semantic=True chunks=0（或 during alignment）
```

`reason=timeout` 连片出现、或 strict 模式抛 "T5 embedding is missing"，说明 §1.1/§1.2 的前提被破坏。
缺记忆记录时不是报错而是跳过：`select` 捕获 KeyError，记 `last_skip_reason=no_memory:<skill>`、
该相位交回 VLA（`oracle_ready_eval.py:669-678`）。

## 6. task 分支的两个已知点

**KSCENE8 —— 本分支直接跳过。** 它 20 次里有 19 次成功（唯一失败在 init index 10），但
不能算作"流程解决了 task 变体"的证据：

- 它的变体只是**删掉**一个 goal 谓词（base 要求两个壶，变体只要 `moka_pot_1` 一个），
  是 10 条里唯一不做"目标物替换"的；其余 9 条全 0/20。
- `Turnon flat_stove_1` 在该场景开局即成立（`memory_system/execute/plan.py:118` 的
  "KITCHEN_SCENE8 starts with the stove lit"，运行时有 `drop_presatisfied_turnon` 丢弃该相位）。
- ★ **它的 language 左右标注是反的**：BDDL 写 `put the left moka pot on the stove`，但 goal 要的
  `moka_pot_1` 初始化在 `kitchen_table_moka_pot_right_init_region`；LIBERO-90 的官方对应任务
  （相同 init / goal / obj_of_interest）写的是 `put the right moka pot on the stove`。即
  `LIBERO-PRO/libero_ood/ood_task.yaml` 的 KSCENE8 两条候选左右对调。eval 喂给 VLA 的正是
  BDDL 的 language，所以该变体下指令与判定目标不是同一个壶。要修就得改 yaml 并重新生成
  BDDL + init + 补 T5。

**LRSCENE5 —— 已补记录**（见 §2.3）。补完后两个 PlaceOn 相位都能命中记录。

**init 口径**：基线（`initial_states[episode_idx]`）与修复侧（`COSMOS_INIT_STATE_OFFSET` →
`states[offset:offset+num_trials]`，`run_libero_smoke_test.py:805-814`）取的是**同一个 init
文件、同一序号**，所以 index 一致 = 初始状态一致。唯一差别是基线 `deterministic_reset=False`、
修复侧默认 `True` + seed 0（`run_libero_eval.py:477-481` 只在 `env.reset()` 前设全局种子，
`set_init_state` 在其后），场景由 init state 决定，不影响对齐。

## 7. 勘误（第 1 版 → 第 2 版）

| 第 1 版的说法 | 核实结果 |
|---|---|
| "`_task` 的 BDDL 文件名与 base 相同，**benchmark 会推出 base 的旧指令**"（§1.1，列为"最容易漏"） | 不成立。benchmark 已从 BDDL 读 `(:language)`（实测 `libero_10_task` → 'put the pan on it'）；`COSMOS_SMOKE_PAIR_CLEAN_LANGUAGE` 只喂 clean 条件，而修复跑 perturb-only |
| "新指令缺嵌入时会**静默换成 base 指令的嵌入**"，建议手设 strict + 关回退 | PRO 自动 strict（`pert_category` 为空）+ 启动预检，静默回退分支到不了；T5 也已 10/10 命中 |
| §5 表格"`diagnose_failed.py` / `tta_repair_batch.py` 仍按 LIBERO-plus 解析" | 两者都已有 `flavor=="pro"` 分支（`diagnose_failed.py:260-265`、`tta_repair_batch.py:243-263`），且 `prepare_inputs_from_diagnosis` 从 phase_record 读 `suite/flavor/bddl_file`（`:110-116`） |
| §1.3"census 必须走语言路径，否则 meta 里的 `clean_language` 是旧指令" | 机制说反了：`clean_language` 是 `_clean_language()` 从**任务名**拼的（`tta_repair_batch.py:47-51`），与 census 无关；只是它在 perturb-only + strict 流程里没有读取点，所以不产生后果 |
| 未提及 | `run_task.sh`（task 评测入口）、`perturbation.py` + `libero_ood/ood_task.yaml`（变体生成器）、`experiments/liberopro/libero_10_task_seed7`（现成基线）—— 见 §8 |

## 8. 合成链路与相关产物

- **变体生成**：`LIBERO-PRO/perturbation.py` 的 `TaskPerturbator` 读
  `LIBERO-PRO/libero_ood/ood_task.yaml`（10 条任务，每条给 language → goal + obj_of_interest），
  替换 `(:language)` / `(:goal)` / `(:obj_of_interest)`；再由 `EvalEnvCreator` 调
  `notebooks/generate_init_states.py` 生成初始状态。注意 `__main__` 被注释掉，入口是
  `create_env(configs)`。产物已经在库：`LIBERO-Pro/bddl_files/libero_10_task`、
  `LIBERO-Pro/init_files/libero_10_task`（各 10 个）。
- **task 评测入口**：`scripts/libero_pro/run_task.sh`（`S=libero_10_task`，20 trials/task，seed 7）。
- **基线**：`experiments/liberopro/libero_10_task_seed7/summary.txt` = **19/200 = 0.095**
  （逐任务：KSCENE8 19/20，其余 9 个 0/20）。
- **本次产物**：`experiments/tta_census/libero_pro_task_full10.json`（90 case）、
  `experiments/tta_phase_check_pro_task_full10/`（census 目录）。
- **swap 侧对照物**（不要与 task 混跑、不要合并结果）：
  `experiments/tta_phase_check_pro_full10/`（100 case，`init000..init009`）、
  `experiments/tta_repair_work_pro_full10_v4/`、`experiments/tta_repair_work_k3_all10/`。

## 9. 已知坑（清单）

| 坑 | 表现 | 处置 |
|---|---|---|
| 漏 `COSMOS_TTA_PHASE_LOOP` | 静默退化成单次修复（曾白跑一轮） | 起跑前 `env \| grep COSMOS_` 核对 |
| LIBERO-plus 与 LIBERO-PRO 同进程混跑 | `get_benchmark_dict()` 解析到错误 suite | 别混跑；`LIBERO_CONFIG_PATH` 与 `PYTHONPATH` 一致 |
| `--tags` 非空时多卡 | 只跑 `chains[gpus[0]]`，其余 tag 不执行 | 每个 GPU 各起一条链 |
| GPU 显存被别人占 | 随机 OOM（10×10 那轮丢了一个 case，41/99） | 起跑前 `nvidia-smi` |
| place 库重建 | 覆盖 §2.3 追加的 13 条记录 | 重建后重跑 `scripts/libero_pro/add_place_records.py` |

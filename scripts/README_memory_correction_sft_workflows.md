# LIBERO-plus / LIBERO-PRO：测试、失败轨迹、Memory 修复与 SFT

核对：2026-10-13，xiaogang_249 的当前工作树。本文提供现有入口和命令，不启动实验。

**当前可运行范围：LIBERO-plus 默认数据路径上的全链路已具备；PRO 的 baseline 采集与历史 census 修复已有入口，但从新 baseline 自动提取全部失败、生成连续训练轨迹、接入全参数 SFT，还需要适配。** 以下区分现有命令与尚待补齐的部分。

微调示例沿用此前设置：全参数 policy backbone、8 GPU、8 epochs。Plus 示例将七类筛选后的成功修复轨迹合并训练一个模型；按扰动单独训练时换成各自 training_manifest.json 与数量。

## 0. 公共环境与基础资产

```bash
ssh xiaogang_249
cd /data1/liu/exp/counterfactual/external/cosmos-policy
export PATH=/data1/liu/miniconda3/envs/cosmospolicy/bin:$PATH
export MG_REPO="$PWD"
export MG_PLUS_ROOT=/data1/liu/exp/counterfactual/external/LIBERO-plus
export MG_PRO_ROOT=/data1/liu/exp/counterfactual/external/LIBERO-PRO
export MG_POLICY_DIR=/data1/liu/exp/counterfactual/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B
export HF_HUB_CACHE=/data1/liu/exp/counterfactual/checkpoints/huggingface-hub
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export MG_STAMP=$(date +%Y%m%d_%H%M%S)
```

环境已有 Torch/CUDA、TransformerEngine、Megatron、MuJoCo、robosuite、h5py、NumPy、cuRobo 等依赖。模型路径之外还需要 Video2World 的 model-480p-16fps.pt、tokenizer/tokenizer.pth 及其离线 HF 缓存布局。PRO 还需 legacy init-state 加载设置；重算 T5 时另外需要 text_encoder 资源。

## A. LIBERO-plus：七种干扰

七类为 camera_viewpoints、robot_initial_states、language_instructions、light_conditions、background_textures、sensor_noise、objects_layout。固定清单在 experiments/libero_plus/task_lists/libero_10/，共 2519 个 case。

### A1. 运行 policy 测试并保存失败轨迹

先给 Plus 使用独立的 LIBERO 配置，避免全局 ~/.libero/config.yaml 指向 PRO：

```bash
export COSMOS_LIBERO_ROOT="$MG_PLUS_ROOT"
export LIBERO_CONFIG_PATH="$MG_REPO/configs/runtime/libero_plus"
export PYTHONPATH="$MG_PLUS_ROOT:$MG_REPO"
python - <<'PY'
import json, os
from pathlib import Path
lib = Path(os.environ['MG_PLUS_ROOT']) / 'libero/libero'
out = Path(os.environ['LIBERO_CONFIG_PATH'])
out.mkdir(parents=True, exist_ok=True)
(out / 'config.yaml').write_text(json.dumps({
    'benchmark_root': str(lib), 'bddl_files': str(lib / 'bddl_files'),
    'init_states': str(lib / 'init_files'), 'assets': str(lib / 'assets'),
    'datasets': str(lib.parent / 'datasets'),
}, indent=2) + '\n')
PY
unset COSMOS_TTA_REPAIR COSMOS_TTA_REPAIR_OUT COSMOS_TTA_REPAIR_TAG COSMOS_INIT_STATE_OFFSET
export COSMOS_INITIAL_ALIGNMENT=0 COSMOS_TTA_PHASE_LOOP=0 COSMOS_SKILL_READY_MEMORY=0

python scripts/run_libero_plus_batch.py \
  --gpus 0 1 2 3 4 5 6 7 \
  --save-episode-data --resume
```

默认输出 experiments/libero_plus/cosmospolicy_full_7dims/，每个 case 有 result.json、actions.npy、video.mp4 与 rollout_data/*.hdf5。HDF5 保存 initial_state、actions、proprio 与图像，后续修复会重放原失败前缀。

已有 2519 个完整 baseline 结果时，直接跳过这一步。--resume 要求配置与 fingerprint 一致；发生版本不一致时应使用新 baseline 输出目录，并同步后述训练脚本的固定路径，不能覆盖旧数据假装同一实验。

Plus 协议由 run_config 固定：action_seed=42、environment_seed=10000、openwam RNG、settle=30、总动作预算=700、init_state_index=0、每 case 一次 trial。

### A2. 提取真实失败清单

```bash
python scripts/collect_libero_plus_failures.py
```

输出 experiments/tta/repair_libero_plus_435/inputs/failed_435.json。只有 completed 且 success=false 的 case 被收录；运行错误不当成任务失败。旧数据重建结果为 435 个。

### A3. 为七类失败 case 生成 Memory 修复轨迹

```bash
export MG_PLUS_REPAIR_DIR="$MG_REPO/experiments/tta/plus7_repair_$MG_STAMP"
python scripts/run_libero_plus_repair.py \
  --manifest "$MG_REPO/experiments/tta/repair_libero_plus_435/inputs/failed_435.json" \
  --work-dir "$MG_PLUS_REPAIR_DIR" \
  --all-cases --gpus 0 1 2 3 4 5 6 7 --resume
```

--all-cases 必须带：默认只运行 4 个 smoke case。此入口自行设置 Plus 修复所需开关；不要把 PRO 的 timegrip 多阶段模式混进来。默认 Plus 修复是诊断触发的一次干预。

每个 case 保存 diagnosis/{phase_record,repair_request,replay_verification}.json、memory_result.json、result.json、actions.npy、视频、完整 episode HDF5。完整轨迹包含 baseline 前缀、Memory 干预与 policy 继续执行。

### A4. 筛选成功轨迹和 P95，生成训练/测试划分

```bash
export MG_PLUS_SELECT_DIR="$MG_REPO/experiments/tta/plus7_selection_$MG_STAMP"
python scripts/select_libero_plus_success_p95.py \
  --repair-runs "$MG_PLUS_REPAIR_DIR" \
  --output-dir "$MG_PLUS_SELECT_DIR"
```

若直接用已完成的历史修复，替换上一条命令为：

```bash
python scripts/select_libero_plus_success_p95.py \
  --repair-runs \
    "$MG_REPO/experiments/tta/repair_libero_plus_435/smoke_4" \
    "$MG_REPO/experiments/tta/repair_libero_plus_435/remaining_431" \
  --output-dir "$MG_PLUS_SELECT_DIR"
```

输出 selected_manifest.json、excluded_manifest.json、summary.json、task_summary.csv 与 by_category/<group>/{training_manifest,heldout_manifest,memory_failed_manifest,evaluation_cases}.json。历史数据：229 修复成功，190 保留、39 超过 P95。

### A5. 用七类轨迹共同进行全参数 SFT

```bash
MG_PLUS_N=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_cases"])' \
  "$MG_PLUS_SELECT_DIR/selected_manifest.json")
python scripts/run_libero_plus_sft190_pipeline.py \
  --job-name "plus7_memory_fullft_$MG_STAMP" \
  --manifest "$MG_PLUS_SELECT_DIR/selected_manifest.json" \
  --expected-cases "$MG_PLUS_N" \
  --data-root "$MG_REPO/training/tta_sft_plus7_$MG_STAMP" \
  --full-finetune --epochs 8 --eval-epochs 1 2 4 8 \
  --evaluation-scope all --gpus 0 1 2 3 4 5 6 7
```

此 pipeline 包含数据准备、2-step smoke、训练、DCP 导出和 policy-only rollout 评测。--evaluation-scope all 会重新测 2519 个 case。轨迹数量从清单读取，不假设新一轮仍恰好 190。

默认训练初始化来自原 CosmosPolicy checkpoint，每卡 batch=1，全局 batch=8、LR=1e-5，epoch 按 dataset 动作步数计算。评测关闭 Memory；若希望微调后仍在线启用 Memory，需定义另一套评测配置。

训练脚本固定读取 experiments/tta/repair_libero_plus_435/inputs/failed_435.json，pipeline 固定读取 experiments/libero_plus/cosmospolicy_full_7dims/。本节基于这套标准路径。仅更换 --manifest 不足以接入另一个 baseline 或 PRO；需要将这些依赖参数化。

按扰动单独训练时，可使用 by_category/<group>/training_manifest.json；Robot Initial / Camera / Sensor Noise 对应现有公共 scope。Background / Light 的完整失败集评测仍在各自 _plans/<job>/run_*_fullft_pipeline.py 中，Language / Objects 尚无公共单类 scope。统一七类调度还需整理这些 scope。

## B. LIBERO-PRO：task + swap

### B1. PRO 环境与真实指令

```bash
export COSMOS_LIBERO_ROOT="$MG_PRO_ROOT"
export LIBERO_CONFIG_PATH="$MG_PRO_ROOT/configs/libero_pro"
export PYTHONPATH="$MG_PRO_ROOT:$MG_REPO"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python scripts/libero_pro/precompute_t5.py \
  --suite libero_10_task --suite libero_10_swap --dry-run
```

若显示缺失 embeddings，再去掉 --dry-run 计算。该脚本会备份并更新 policy 的 T5 缓存。task 变体的真实指令来自 BDDL，不能用文件名或 clean instruction 代替。

### B2. 采集两个 suite 的 baseline HDF5

```bash
unset COSMOS_TTA_REPAIR COSMOS_TTA_REPAIR_OUT COSMOS_TTA_REPAIR_TAG COSMOS_INIT_STATE_OFFSET
export COSMOS_INITIAL_ALIGNMENT=0 COSMOS_TTA_PHASE_LOOP=0 COSMOS_SKILL_READY_MEMORY=0

SUITE=libero_10_task NUM_TRIALS_PER_TASK=20 SEED=7 \
  GPU_IDS="0 1 2 3 4 5 6 7" \
  OUTPUT_ROOT="$MG_REPO/experiments/liberopro/task_collect_$MG_STAMP" \
  bash scripts/libero_pro/run_swap_collect.sh

SUITE=libero_10_swap NUM_TRIALS_PER_TASK=20 SEED=7 \
  GPU_IDS="0 1 2 3 4 5 6 7" \
  OUTPUT_ROOT="$MG_REPO/experiments/liberopro/swap_collect_$MG_STAMP" \
  bash scripts/libero_pro/run_swap_collect.sh
```

虽然文件名是 run_swap_collect.sh，SUITE 参数允许运行 task。它显式传 --data_collection True，HDF5 位于各自 OUTPUT_ROOT/local_logs/rollout_data/。普通 run_task.sh / run_swap.sh 没有开启数据采集，不满足“得到原始失败轨迹”的要求。

两个 suite 分开保存；典型规模为每 suite 10 tasks × 20 init，共 200 trials。PRO 当前 baseline 使用 legacy RNG、seed=7、10 settling、520-step 上限，与 Plus 的协议不同。

### B3. 现有历史 census 的诊断与修复命令

**下列命令重现已准备的历史子集，不会自动读取 B2 的新 baseline。**

- libero_pro_swap_full10.json：10 tasks × init 0–9，共 100 个 case。
- libero_pro_task_full10.json：9 tasks × init 0–9，共 90 个 case；KITCHEN_SCENE8 未包含。因此不能称为 task 全部失败 case。

```bash
export SMOKE_RNG_MODE=legacy SMOKE_DETERMINISTIC_RESET=false
export COSMOS_TTA_PHASE_LOOP=1 COSMOS_SKILL_READY_MEMORY=timegrip
export COSMOS_INITIAL_ALIGNMENT=1 COSMOS_TTA_RGBD=1 COSMOS_SKILL_COMPLETION_SHADOW=1
export COSMOS_DATA_COLLECTION=1 SMOKE_DATA_COLLECTION=1
export MG_PRO_SWAP_DIAG="$MG_REPO/experiments/tta/pro_swap_diag_$MG_STAMP"
export MG_PRO_TASK_DIAG="$MG_REPO/experiments/tta/pro_task_diag_$MG_STAMP"

CENSUS="$MG_REPO/experiments/tta_census/libero_pro_swap_full10.json" \
  OUT="$MG_PRO_SWAP_DIAG" bash scripts/libero_pro/run_diagnose_smoke.sh
CENSUS="$MG_REPO/experiments/tta_census/libero_pro_task_full10.json" \
  OUT="$MG_PRO_TASK_DIAG" bash scripts/libero_pro/run_diagnose_smoke.sh

python scripts/tta_repair_batch.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --diagnosis-root "$MG_PRO_SWAP_DIAG" --result-dir "$MG_PRO_SWAP_DIAG" \
  --work-dir "$MG_REPO/experiments/tta/pro_swap_repair_$MG_STAMP"

python scripts/tta_repair_batch.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --diagnosis-root "$MG_PRO_TASK_DIAG" --result-dir "$MG_PRO_TASK_DIAG" \
  --work-dir "$MG_REPO/experiments/tta/pro_task_repair_$MG_STAMP"
```

PRO 的 --gpus 使用逗号字符串；Plus 新入口使用空格分隔整数。work-dir 使用绝对路径，因为 repair 子进程在 scripts/ 下运行。不要加 --tags 后误以为会自动使用全部 GPU：旧 batch 的指定 tags 路径只运行第一条链。

显式 SMOKE_RNG_MODE=legacy 很关键：公共 smoke shell 默认 openwam，会把 seed 切成 action_seed=42；这里应匹配历史 PRO 的 seed=7。timegrip 与 PHASE_LOOP=1 指定多阶段对齐及 Memory 回放。

诊断产物：<task>/initNNN/{phase_record.json,episode.h5,tstar.png}；修复产物：outputs/<tag>.json、requests/repair_<tag>.json、results/<tag>/ 下的日志/HDF5。suite、BDDL、实际 instruction、init 和 seed 都要一致。

诊断入口当前是重跑 policy，不是重放 B2 保存的动作；出现 census_mismatch 时不能继续把该 case 当作已验证原失败。完整新数据流程应先修正这个连接。

### B4. 接入 PRO 全参数 SFT 前的实际缺口

| 必须补齐的接口 | 现状与要求 |
|---|---|
| 新 baseline → PRO 失败清单/census | 现有 build_census.py 读取 smoke episodes.json，不能直接接 B2 的 evaluator HDF5 输出，且不产生 PRO 顶层 flavor/suite。需按 (suite,task,init) 提取真实失败，排除运行错误并保留原轨迹路径。 |
| 已保存失败 → 同 case 诊断/修复 | 现诊断会重新推理；应重放原 baseline 动作并验证状态，保留 suite/真实指令/seeds/初始状态。 |
| 关节干预 → 可训练完整轨迹 | 当前 PRO repair 的 8 维关节执行帧没有写入 7 维训练流，检查到的成功 HDF5 有 frame_indices 跳号。需完整记录所有执行步，再定义/验证向 policy 7 维动作空间的转换；不能把删帧后的序列当成连续完整轨迹。 |
| PRO/Plus → 通用 SFT 数据准备 | 当前 train_libero_plus_sft190.py 固定读取 Plus 435-case 清单，并要求 task_name 是 base_task 的扰动后缀。PRO task/swap 与 base 同名，suite 的任务 ID 还会重复，需要支持 suite-aware case 身份和真实 instruction。 |
| 通用全参数训练与评测 | 可复用 CosmosPolicy trainer、dataset 与 checkpoint export；需要将基础模型、失败清单、数据根、suite 和评测清单参数化。PRO 的 P95 来源/规则也应单独记录，不能未经定义套用 Plus 的专用 selector。 |

因此当前不提供一个冒充已经接通的 PRO SFT 命令。将上面的接口补齐后，PRO task、PRO swap 和 Plus 七类可以分别训练，也可以按明确 manifest 合并训练。

## C. 必须保留的代码与资产

| 范围 | 代码/文件 |
|---|---|
| policy / SFT 底层 | 整个 cosmos_policy/，尤其 config/config.py、config/experiment/cosmos_policy_experiment_configs.py、datasets/libero_dataset.py、models/、scripts/train.py、_src/ trainer/optimizer/checkpoint |
| 通用 rollout | scripts/run_libero_smoke_test.sh、run_libero_smoke_test.py；cosmos_policy/experiments/robot/{cosmos_utils.py,libero/run_libero_eval.py,libero/libero_utils.py} |
| Plus 流水线 | run_libero_plus_batch.py、collect_libero_plus_failures.py、run_libero_plus_repair.py、select_libero_plus_success_p95.py、train_libero_plus_sft190.py、run_libero_plus_sft190_pipeline.py |
| 当前 SFT 的间接依赖 | tests/tta/archived_lorafix/train_tta_sft_fixed.py；scripts/train_lrscene5_sft_only.py；scripts/run_lrscene5_sft_only_pipeline.py |
| 导出 | scripts/export_full_finetune_ckpt.py；LoRA 可选路径还需 merge_lora_ckpt.py |
| Memory / 诊断 / 执行 | 整个 memory_system/：tta/phase_record.py、object_query.py、diagnose_failed.py、execute/、pointcloud_action/ 等 |
| PRO 入口 | scripts/libero_pro/{run_swap_collect.sh,run_diagnose_smoke.sh,precompute_t5.py,oracle_ready_eval.py}、scripts/tta_repair_batch.py；单例调试可用 run_single_case.py |
| Plus Memory | skill_memory_test/libero_10/{segments_ready_fixed16.json,feasible_recovery_targets.pt,ready3d_targets.pt} |
| PRO Memory | memory_system/pointcloud_action/{pointcloud_action_memory.pt,pointcloud_action_memory_place.pt,pointcloud_action_memory_turnon.pt} |
| 任务与环境资源 | ../LIBERO-plus/；../LIBERO-PRO/（含 LIBERO-Pro/bddl_files、init_files、configs/libero_pro 与 assets）；Plus 固定 task_lists、configs/libero10_experiment_tasks.json |
| 模型与文本/标准化 | ../../checkpoints/ 中的原 policy、Video2World、tokenizer、policy stats、T5 缓存；experiments/cache/libero_plus_language_t5_libero10.pkl；dataset_statistics_post_norm.json 的实际目标 |
| Memory/P95 重建输入 | LIBERO-Cosmos-Policy/success_only/libero_10_regen/；重建 PRO Memory 还需 libero_90 demo、libero90_segments_ready_full.json 等分段清单与对应构库代码 |
| 精确复现已完成单扰动实验 | scripts/experiments/libero_plus_sft190/_plans/<job>/ 的代码与配置快照、训练/评测 manifests、run.json、原始修复 HDF5 |

训练 HDF5 使用绝对 ExternalLink，目录移动时要重建链接或导出自包含 HDF5。继续训练需保留 model/optim/scheduler/trainer 分布式状态；仅推理用导出的 .pt 和共同模型资产。

## D. 本次核对与范围

36 项关键代码/数据路径均存在；所有列出的主要 Python 入口可解析。Plus 失败/P95 入口此前已通过 12 个回归测试，并逐字节重现 435 / 190 / 39 清单。本次仅整理命令与依赖，没有重新运行 baseline、Memory correction 或 SFT，也没有修复上述 PRO 接口。

# docs 索引

按主题分目录，各目录平级，每目录一个主文件。新文档放进对应主题目录，不要堆在根目录。

| 目录 | 主题 | 主文件 |
|---|---|---|
| `repair/` | memory 修复链路（census → 诊断 → t* → 对齐 → VLA） | `libero-plus-repair.md`（§1 代码地图、§2 运行命令、§3 LIBERO-plus 167 条批次产物）；`why_repair_only_sft_fails.md`（为什么模仿修复动作的 SFT 会失败） |
| `sft/` | TTA SFT 训练 | `sft.md`（①文献 ②目的 ③算法构成 ④相关代码 ⑤实验结果，含 LoRA 死初始化 bug） |
| `dpo/` | TTA DPO 训练 | `dpo.md`（①文献 ②目的与判决矩阵 ③算法构成与命令 ④相关文档 ⑤实验结果） |
| `libero-pro/` | LIBERO-PRO swap 上的 memory 修复 | `LIBERO_PRO_SWAP_RESULTS.md`（200 case 结果）、`LIBERO_PRO_REPAIR.md`（方案与实现清单） |
| `eval-tasks/` | 评测任务构造 | `unseen_object.md`（unseen 物体替换） |

根目录 `AGENT_SKILLS.md`：agent 操作守则，与项目内容无关。

## 备注

- 判决矩阵现行版在 `dpo/dpo.md` §2；`sft.md` §2 保留的是 SFT 的五条局限结论。
- 环境坑按侧归档：repair 坑在 `repair/libero-plus-repair.md` §2，SFT 训练坑在 `sft/sft.md` §3，DPO 坑在 `dpo/dpo.md` §3。

# MonoDGP 阶段归档：2026-09-08

## 已完成与未完成边界

- 已合入并推送的分数精度修复：`128e6b5`。内存评测和文本导出保留原始分数；框几何仍两位小数，候选和NMS逻辑不变。
- 本次收尾纳入的训练修复：legacy MixUp 抽到当前图片自身时拒绝并重抽，仍占用原有50次尝试预算。unified_v2原先已排除自身。
- Exp59已经完成250轮，本次归档没有重新训练或复评。
- **worker随机种子修复仅完成临时验证，未修改正式 `lib/helpers/dataloader_helper.py`。** 后续是否合入、是否跑完整实验仍由用户决定。
- 用户明确决定不复评Exp55。本档保留评测精度差异，不将Exp59和旧日志的差值归因为排除自混合的净收益。

## Exp59 最终结果

基于Exp47、MixUp触发概率0.3，两张图混合权重各0.5。方法配置与Exp55相同，新增排除自供体。BS16、seed444、4 workers、CUDA预取和严格确定性；从头250轮，沿用backbone预训练初始化，不加载实验checkpoint。E10–110每10轮验证，E120–250每轮验证。

| 指标：无NMS Car 3D AP_R40@0.7 | Easy | Moderate | Hard |
|---|---:|---:|---:|
| best E178，结束后加载best复评 | 33.2437 | 24.8539 | 21.4664 |

- 精确best Moderate：`24.853887981569496`。
- 启动：2026-09-07 16:07；结束：2026-09-08 03:14（北京时间），约11小时07分。
- `runner_exit / manifest_exit / train_exit / tee_exit` 均为0；SwanLab最终 `Upload complete: 994570 records`、`SWANLAB_FINISHED`。
- 训练中曾有SSL上传重试和终端日志跳过，未中断训练；最终上传完成不代表被跳过的终端文本必然补齐。
- [SwanLab kddn6yf6](https://swanlab.cn/@Grapymage/MonoDGP/runs/kddn6yf6)。历史tmux `monodgp_exp59`、训练PID `4180570`，已结束。
- 输出：`outputs/V2-0059_实验59_MixUp概率0.3排除自身/`；best SHA256：`f03d3e587601d30722af4aebbb26fce1ddf157d75698a5302b58733bd3445060`。
- 实际运行基线 `128e6b5` 加当时未提交修复；原始manifest保留当时状态，不追溯改写。
- 入口：`tools/run_exp55.sh`，通过 `MONODGP_FORMAL_CONFIG` 和 `MONODGP_FORMAL_OUTPUT` 指向Exp59，由tmux承载。准确Python命令见 [run_manifest.txt](exp59/run_manifest.txt)。启动器拒绝覆盖已有输出，不应直接重跑到旧目录。

## 受控诊断结果

### 1. Car-only 候选检查

Exp47 E247，同一次推理覆盖3769张。历史50×3展平top50全部选中Car，共188450个候选，没有其他类别挤占。Car-only与历史候选对照的框和AP一致；不能把旧缓存24.2528与正式24.2354的差异归因为Car-only收益。

无NMS Moderate `24.23544760548706`，NMS0.8 `24.47757496170476`。见 [car_only/result.json](car_only/result.json)。

### 2. 分数精度

Exp47 E247，同一次推理，仅替换送入评测器的分数精度，几何仍两位小数。NMS0.8保留框由原始分数计算一次，两组固定相同候选和顺序。

| 方式 | 两位小数 Moderate | 原始精度 Moderate |
|---|---:|---:|
| 无NMS | 24.235448 | 24.306298 |
| NMS0.8 | 24.477575 | 24.544726 |

有小幅影响，但不能解释历史质量头实验数个AP点的下降。见 [score_precision/result.json](score_precision/result.json)。

### 3. 自混合重复GT

16张真实有有效Car图片，每张2个随机种子，只强制触发MixUp及选中自身。32次都产生重复有效GT，共118对，均通过 `mask_2d`，监督字段逐位一致。

按全量3712张、真实P2/尺寸/容量及最多50次重抽规则计算，触发概率0.3时自混合期望0.826张/轮，占0.02225%；含无有效Car情形，不能当作重复有效GT的历史实测次数。重抽也会改变后续增强随机序列，因此单次Exp59不能单独证明修复的稳定收益。

临时诊断第一次在附加metadata JSON保存时因NumPy int64失败；修复类型转换后同条件重跑一次成功。归档的 `exitcode=1` 是首次失败，`rerun1.exitcode=0` 是成功重跑。见 [self_mixup/result.json](self_mixup/result.json)。

### 4. worker 随机初始化：仅验证，未合入

正式worker回调以 `np.random.get_state()[1][0] + worker_id` 再设种子，覆盖PyTorch已设置的状态。当前环境按原始初始化顺序测试，四个worker固定落在2147483648至2147483651。候选临时回调只读记录PyTorch提供的状态，不设种子、不消耗随机数。

| 验证 | 每次范围 | 结果 |
|---|---|---|
| 两次独立短训练 | 2个短轮次×8批，BS16 | 所有逐批字节哈希一致；4 workers跨运行一致、跨轮初始状态不同 |
| 两次独立完整单轮 | 3712张，每张恰好一次，232批，BS16 | 输入、顺序、GT、全部loss、梯度、更新后模型与优化器状态的逐批字节哈希全部一致 |

全轮平均loss两次均为 `45.585532385727454`。每次耗时366.51/365.44秒，包含逐批GPU到CPU拷贝和哈希检查，**不是正常训练性能指标**。测试均沿用Exp59配置、4 workers、严格确定性和CUDA预取，不上传SwanLab，不做AP评测。优化器包装导致的scheduler警告来自诊断插桩，两次均正常执行了232次参数更新。

这支持所测流程的可复现性，不是完整250轮或跨软硬件环境的证明，也不是与旧worker初始化逐位一致的声明。

见 [短测试结果](worker_short/result.json)、[完整一轮结果](worker_full/result.json)。A/B.json保存逐批证据，manifest保存环境和配置。

## 归档文件与使用限制

- `car_only/`、`score_precision/`、`self_mixup/`、`worker_short/`、`worker_full/`：本次诊断的源码快照、结果和运行收据。源路径分别为 `/tmp/monodgp-car-only-GFRarh`、`/tmp/monodgp-score-precision-ZvlC1G`、`/tmp/monodgp-self-mixup-6tqGHJ`、`/tmp/monodgp-worker-repro-OII7ks`、`/tmp/monodgp-worker-full-TVvCQh`。
- 这些脚本是**原执行现场快照，不是通用入口**，含当时绝对路径和旧行为断言；不要原样重跑覆盖历史输出。重跑须另建输出目录，并明确选择当时源码/评测精度。例如self_mixup脚本记录的是修复前行为，当前代码会拒绝自身。
- `exp59/`：启动环境、解析配置、成功退出收据。checkpoint、数据集、原始预测缓存及大型训练日志不上传Git，仍留在原输出目录。
- 本次同时纳入Exp55/55R/55R2/56/57/58/59配置和已有运行脚本。R/R2配置保留历史故障来源说明，不表示本次重新引入硬件排障代码。
- `tools/diagnose_exp47_ap_attribution.py`、`tools/diagnose_exp47_top1_tp_promotion.py` 是历史缓存oracle分析源码；它们依赖本地缓存和当时评测精度，不能据当前代码直接承诺重现旧数值。
- `tools/run_exp54_frozen_rank_probe.py` 为历史源码，依赖已缺失的 `/tmp/exp47_tp07_frozen_probe.py`，**当前不能独立复跑**。本次仅保存源码，没有补造依赖或重新验证其科学结论。

## 交接

本次归档前仓库测试：`pytest tests -q`，125 passed、16 skipped、1 warning（旧优化器接口弃用警告），在默认无GPU可见性的测试环境执行；不将跳过的CUDA测试算通过。另对归档JSON、Python源码语法及A/B逐批证据做一致性校验。GPU完整单轮验证结果见上文独立记录。

1. 当前已完成训练结果以Exp59上述best为准，用户决定不做Exp55统一精度复评。
2. 若继续worker随机初始化方向，先由用户确认将已验证方案合入，再决定正式训练；不要把本次测试当作已经修改训练实现。
3. 新实验均使用批准的 `.venv-cu129`，持久tmux、环境记录和独立输出目录；无已批准待启动队列。

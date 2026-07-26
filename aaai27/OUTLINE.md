# AAAI-27 论文大纲与代码映射

## 建议定位

工作标题：**Reliability-Aware Biological Joint Moment Estimation for Risk-Sensitive Task-Agnostic Exoskeleton Assistance**。

一句话主线：现有 task-agnostic 外骨骼控制解决“跨任务怎么连续估计并助力”，domain adaptation 解决“昂贵标签不够时怎么训练”，本文解决二者部署后仍未覆盖的“传感器或任务分布异常时，模型如何知道自己不可靠并安全降级”。

建议将稿件定位为可靠机器学习与具身 AI/可穿戴机器人交叉工作，而不是单纯的外骨骼应用论文。AAAI 审稿叙事应突出三个 AI 问题：异构故障的可检测性、分布偏移下的可靠性标定、预测可靠性向控制决策的转换。

## 两篇 baseline 的准确角色

1. **Nature 2024** 是直接任务 baseline：定义 25 通道可穿戴输入、hip/knee biological moment 目标、TCN 主干和 moment-to-torque 控制映射。仓库中的官方 checkpoint 可作 secondary reference。
2. **Science Robotics 2025** 是互补问题 baseline：它用 CycleGAN/U-Net 和 simulated-sensor stepping-stone domain 解决 device-specific 标签稀缺，但当前仓库没有实现其 domain-adaptation 网络。因此正文应把它作为强相关方法与未来可组合模块，不能把它写成已运行的直接实验 baseline。
3. 公平的主要实验 baseline 应是仓库已有的 `det_noaug`：与本文模型使用完全相同 participant/task split、容量和训练预算的确定性 TCN。官方 Nature checkpoint 的原训练划分不匹配，只能辅助报告，不能承担显著性结论。

## 推荐贡献点

1. **可靠性多任务 TCN**：共享 causal TCN 同时输出 hip/knee moment mean、heteroscedastic log variance、fault logit、denoising reconstruction 和 short-horizon forecast。
2. **按故障机理组织的多路检测**：学习信号之外加入 staleness、cross-modal coherence、drift 等 causal statistics；在 validation 上做 q90 归一化和最多三路信号的子集选择。
3. **utility-aware safety gate**：风险超过 deadband 后连续衰减助力；仅用 validation clean/OOD-clean 和 validation faults 选择 deadband/softness，约束 clean aligned torque，并优化 fault wrong-direction torque。
4. **三重泛化协议**：unseen participants、held-out tasks、held-out fault generators；最后用控制相关指标而非只用 RMSE 评价。

## 正文结构与篇幅预算（主内容目标 7 页）

| 部分 | 建议页数 | 核心内容 |
|---|---:|---|
| Abstract + Introduction | 1.1 | 部署可靠性缺口、研究问题、三项贡献 |
| Related Work | 0.65 | task-agnostic control、domain adaptation、uncertainty/OOD |
| Problem Formulation | 0.35 | corrupted sequence、moment estimate、risk gate、三类目标 |
| Method | 1.55 | multi-head TCN、fault curriculum、detectability taxonomy、gate |
| Experimental Protocol | 1.15 | participant/task split、baseline、faults、metrics、statistics |
| Results | 1.55 | accuracy/calibration、fault AUROC、safety-utility、ablation/stress |
| Discussion/Limitations/Ethics/Conclusion | 0.65 | 适用边界、无真机结论、组合 Science Robotics 路线 |

参考文献不计入上述正文预算；最终以 AAAI-27 track 的明确 page-limit 通知为准。

## 方法与代码映射

| 论文模块 | 代码位置 | 写作注意 |
|---|---|---|
| 输入/标签与 profile | `reliability/features.py` | human 为 25 通道；action profiles 是 E10 消融 |
| participant/task split | `reliability/nature_dataset.py`、`run_reliability_experiment.py` | 强调 participant-disjoint；heldout 默认 4 个 task family |
| multi-head TCN | `reliability/model.py` | forecast 默认 stop-gradient；MC dropout 与 ensemble 均支持 |
| 合成故障 | `reliability/faults.py` | burst/partial/jitter 是 held-out variants，不进入训练 faults |
| NLL/BCE/recon/forecast loss | `run_reliability_experiment.py` | 最终补充所有主实验的 lambda 和采样策略 |
| risk signals 与 calibration | `reliability/metrics.py` | detector fusion 与 command-time gate risk 是两个相关但不同的路径 |
| validation-only selector | `run_reliability_experiment.py` | 信号子集、q90 refs、gate softness/deadband 均不能查看 test |
| torque replay | `reliability/metrics.py` | wrong direction、peak wrong、aligned torque、jerk |
| actuator replay proxy | `reliability/closed_loop.py` | 只能称 proxy，不能推断真实 metabolic reduction |
| 完整实验矩阵 | `scripts/reliability_paper_v2.sh` | main、ablations、baseline、stress、ensemble、action、LOSO |
| AAAI 表格 | `scripts/build_aaai_tables.py` | 生成 detection 和 safety 两张核心证据表 |

## 关键实验问题

- **RQ1：可靠性训练是否牺牲 clean accuracy？** 同划分 `det_noaug` 对比 proposed，分别报告 ID/OOD clean RMSE、R2、NLL、coverage ECE。
- **RQ2：不同风险信号能否覆盖不同故障？** 报每路 AUROC 和冻结后的 fused AUROC，不能只报 best-of-test。
- **RQ3：能否泛化到未见故障生成器？** 单独标注 packet-loss burst、partial-channel loss、delay jitter 为 held-out。
- **RQ4：检测是否真的转化为更安全的助力？** 报 gated vs ungated wrong-direction、peak wrong、aligned retention、jerk；clean ID/OOD 必须一起报告。
- **RQ5：哪些模块贡献最大？** `det_noaug → det_aug → prob_aug → prob_aug_recon → prob_aug_recon_fc` 顺序消融。
- **RQ6：结论是否稳定？** 主表三 seeds；资源允许时追加 LOSO。stress 曲线至少跨主模型 seeds 聚合。

## 推荐图表

1. **Figure 1：方法总览**。输入 → causal TCN 多头 → 显式时序检测 → validation calibration → risk gate → torque。
2. **Figure 2：代表性故障时序**。clean、sensor delay 或 burst loss 下 moment、risk、gate、ungated/gated torque 同轴对齐。
3. **Figure 3：故障强度曲线/Pareto 曲线**。横轴 severity 或 retained assistance，纵轴 wrong-direction reduction。
4. **Table 1：accuracy + calibration**。proposed 对 fair deterministic baseline，ID/OOD clean。
5. **Table 2：fault detection**。每路与 fused AUROC，明确 held-out generator。
6. **Table 3：safety-utility**。ungated/gated、clean/fault、ID/OOD。
7. **Table 4：attribution ablation**。若正文空间不足，保留紧凑版，完整表放允许的 supplementary material。

## 提交前必须处理的证据缺口

1. 当前仓库只有较早的 `reports/reliability_human` 与 smoke 报告，和现行 v2 模型/评测路径不完全一致，不能直接填最终表。
2. 需要完成 `scripts/reproduce_aaai.sh core`，最好再完成 LOSO；运行后用 `summarize_reliability_suite.py` 与 `build_aaai_tables.py` 生成聚合表。
3. `collect_reliability_outputs` 当前可见一次重复追加 `fault_logit` 的代码路径，应在正式跑全套前审计，否则可能导致评测 tensor 数量不一致。本文骨架没有假定该问题已经修复。
4. detector fused score 使用 validation-selected channel subset；command gate 当前不包含 coherence/drift，正文已按代码区分，后续若改实现必须同步论文公式和消融。
5. 最终统计检验尚未在代码中固定。建议对三 seeds 做 paired difference 并报告置信区间；LOSO 完整后再考虑以 subject 为单位的非参数或 mixed-effects 分析。
6. 必须在完整结果出来后替换所有 `[TBD]`，删除正文中的内部 TODO 注释，并核对结论没有把 offline actuator proxy 写成真实人体代谢收益。

## 可选的更强扩展

若时间足够，可把 Science Robotics 2025 的 domain-adapted estimator 接入相同 reliability layer，形成 2×2 设计：supervised vs domain-adapted，ungated vs reliability-gated。这会直接证明“数据效率”和“部署可靠性”正交且可组合，但属于当前仓库以外的新实现，不应在现稿中虚构。

# 论文大纲工作文件 (PAPER_OUTLINE.md)

> 方法论借鉴 kgraph57/paper-writer-skill 的 IMRAD 工作流（story-arc + 写作顺序 + 用户确认门），
> 模板/reporting-guideline 部分不套用（该 skill 为临床医学设计，与本工程方法论文不符）。
> 内容锚点见 [REFERENCES.md]；实验数据见 reports/v2_paper_suite/ 与各 reliability_report.json。

---

## 0. 论文元信息 (Phase 0)

- **工作标题**: Reliability-Aware Task-Agnostic Exoskeleton Control: Sensor-Fault Detection, Torque-Level Harm Evaluation, and a Fault-Detectability Taxonomy
- **论文类型**: Methodology / engineering（IMRAD，实验已完成）
- **目标会议**: AAAI（首选，中稿目标）/ IEEE TNSRE（备选，工程医学方向）
- **一句话研究问题**: 当传感器发生故障时，学习型（task-agnostic 生物力矩）外骨骼控制器会对使用者施加什么样的错误辅助，能否在线检测并安全门控，哪些故障在当前观测下难以辨识？
- **现有关键数据/图表**:
  - 检测 AUROC（residual/forecast/staleness/causal cross-IMU coherence 等，按故障类型 × ID/OOD）→ reports/v2_refreshed_suite/main_detection_aggregate.tsv
  - 门控前后力矩伤害指标（wrong-dir / peak-wrong / retained / jerk）→ 各 report JSON
  - LOSO 15 折泛化 → loso 汇总
  - E5 归因消融（det_noaug→prob_aug_recon_fc）→ 精度/鲁棒性权衡
  - 严重度扫描（packet_loss@0.05/0.15/0.30、sensor_delay@5/10/20）+ held-out challenge faults（packet_loss_burst / packet_loss_partial / sensor_delay_jitter）
  - 闭环净力矩 RMS 降低（dynamics-in-the-loop）

## 0.1 待确认的期刊要求（写作前 WebSearch 落实）
- [ ] TNSRE 字数 / 摘要格式 / 图表上限 / 引用风格（IEEE 编号）
- [ ] 是否需要 AI 使用披露
- [ ] TRIPOD+AI（ML 预测模型报告规范）是否值得自愿对标 → 可加分

---

## 1. 核心 STORY ARC（一句话串起来，所有章节服从它）

> **背景**：学习型外骨骼控制（Molinaro task-agnostic 力矩估计）很强，但默认输入干净。
> **缺口**：传感器故障下它会怎样？已有的不确定性门控工作（Tourk2025）只处理 OOD *动作*，明确未碰传感器故障；且所有先前工作只用分类指标，无人量化"错误辅助对人腿的物理后果"。
> **我们的方法**：(a) 系统的传感器故障注入 benchmark；(b) 力矩级生物力学伤害评估框架；(c) 多信号故障检测 + validation-selected utility gate；(d) 故障可检测性分类学（含诚实的 near-chance 负结果）。
> **发现**：结构/时序故障可检测并安全门控（高严重度 packet_loss@0.30 的 gate 与 wrong-dir 归零）；平滑统计漂移型（imu_bias）在当前观测与 detector family 下接近随机；安全有可量化的精度代价。
> **意义**：把"可靠性"从口号变成可测量、可复现、对接 ASTM 安全标准的评估框架。

---

## 2. 章节大纲（Phase 2，按 READING 顺序排列；写作按下方 §6 顺序）

### I. Introduction
- P1 背景：外骨骼效益 + 数据驱动/task-agnostic 控制崛起 [Molinaro2024, Scherpereel2025, Siviy2022, Zhang2017]
- P2 问题：学习型控制器 "garbage-in-garbage-out"，传感器故障/OOD 下仍输出 → 错误辅助有真实安全后果 [ASTM-F3578, Wu2024, Wu2023]
- P3 缺口（关键差异化段）：
  - 已有不确定性门控（Tourk2025）只做 OOD *动作*，明确未做传感器故障
  - 鲁棒化工作（Hsu2024 贝叶斯）针对瞬时混乱，非传感器故障谱
  - 所有先前工作用分类指标，无力矩级伤害度量
- P4 贡献声明（见 §3）
- P5 论文组织

### II. Related Work
- 2.1 学习型外骨骼控制 [Molinaro2024, Scherpereel2025, Shetty2025, Kang2020, Medrano2023]
- 2.2 不确定性估计与异常检测 [Lakshmin2017, Gal2016, TCN-AE, Ens-aug-calib]
- 2.3 可穿戴/外骨骼故障容错 [Hsu2024, EMG-SFTM, Prosthesis-VAE-SOM, Exo-FMEA, myocontrol-fail]
- 2.4 外骨骼安全评估与扰动 [ASTM-F3578/F3583, Wu2024, 扰动-稳定性文献]
- 2.5 仿真验证外骨骼控制 [Luo2024, Dembia, Exo-plore] ← 为我们的 sim-only 评估铺垫合法性
- **★ 显式对照段**：us vs Tourk2025 差异表（故障类型/伤害度量/门控/控制器/检测信号/验证）

### III. Methods
- 3.1 任务与基础控制器（复现 Molinaro task-agnostic TCN 生物力矩；Camargo 数据集 [Camargo2021]）
- 3.2 概率 TCN：mean/logvar + 重构头 + forecast 头（detach）+ 故障头
- 3.3 传感器故障模型（注入）：insole_missing / encoder_dropout / packet_loss / sensor_delay / imu_bias；另设 held-out challenge variants（packet_loss_burst / packet_loss_partial / sensor_delay_jitter）检验检测器是否泛化到未见故障生成器。引 Wu2024 固定错误率范式为先例
- 3.4 多信号故障检测：aleatoric / 重构残差 / forecast残差 / staleness / drift / deep-ensemble epistemic；新增仅使用当前及历史帧的 32-frame cross-IMU difference coherence，专门检测固定与 jittered sensor delay
- 3.5 ★ 力矩级伤害评估框架：wrong-direction-ratio / peak-wrong-torque / retained-aligned-torque / jerk —— 定义 + 对接 ASTM/扰动文献的现实效度论证
- 3.6 Validation-selected utility gate：val 选择 softness + deadband，同时约束 clean retained torque 与 validation-fault wrong-direction，val_ood 只用于域偏移 deadband；coherence 与 drift 当前仅作检测通道。这里的“校准”指决策阈值选择，不等同于输出方差 calibration
- 3.7 闭环 dynamics-in-the-loop 评估（净力矩 RMS / 做功 / fight-fraction）[Luo2024 合法性]
- 3.8 实验设计：LOSO 15 折、多种子、E5 归因消融、动作 profile、严重度扫描

### IV. Results
- 4.1 干净基线 + Nature 对照（精度/校准）
- 4.2 故障检测 AUROC（按类型 × 信号 × ID/OOD）→ 分类学雏形
- 4.3 门控安全收益（力矩伤害指标，前后对比；packet_loss 完全门控）
- 4.4 ★ 故障可检测性分类学（结构缺失→residual/staleness、冻结/丢包→staleness、跨 IMU 异步→causal coherence、平滑 DC 漂移→不可检测）含 imu_bias 负结果
- 4.5 E5 归因：精度 vs 鲁棒性权衡（aug +0.035、recon +0.016；引 Ens-aug-calib）
- 4.6 LOSO 泛化 + 闭环净力矩降低
- 4.7 严重度扫描与 held-out challenge fault 泛化（训练故障生成器 vs 测试故障生成器分离）

### V. Discussion
- 5.1 主要发现回到 story arc
- 5.2 与 Tourk2025 / Hsu2024 的关系（互补，不竞争）
- 5.3 安全-精度权衡的工程含义
- 5.4 分类学的可推广性（哪类信号抓哪类故障）

### VI. Limitations & Future Work
- 纯仿真/无硬件（[Luo2024] 路线合法 + 诚实承诺硬件验证）
- 伤害指标是离线代理（[Wu2024] 建立人体-代价链，我们的代理待硬件校准）
- imu_bias 在当前观测、注入幅度和候选 detector family 下 near-chance；这是 empirical limitation，不宣称普遍不可检测
- OOD-clean 精度落后（域偏移 = 未来工作，对接 Scherpereel）
- 输出方差校准具有域依赖性：seed-7 validation-selected temperature scaling 在冻结 test ID/OOD 上均未优于 identity，因此不集成、不宣称跨域 calibrated uncertainty
- 一个统一融合分数不能同时达到所有 specialist 的 AUROC；ECDF-Fisher 因显著损伤 OOD burst detection 被拒绝，正文以故障×信号矩阵为主要证据
- 标准故障强度下 gate 的 wrong-direction 改善很小（seed-7 最大绝对改善约 0.004）；强安全收益必须由三种子 severity curves 支撑，否则只报告 trade-off，不写普遍 benefit claim

### VII. Conclusion

---

## 3. 贡献声明（草拟，待打磨）

1. **首个传感器故障可靠性 benchmark**：在 task-agnostic 学习型外骨骼控制上系统注入并评估 5 类传感器故障 × 严重度梯度（先前工作仅 OOD 动作，明确未覆盖传感器故障）。
2. **力矩级生物力学伤害评估框架**：wrong-direction / peak-wrong-torque / retained-aligned-torque / jerk，将"错误辅助的物理后果"量化为对接 ASTM-F3578 的离线指标（先前工作仅用分类 AUROC/F1）。
3. **多信号检测 + utility-aware 安全门控**：验证集选择 residual/staleness/coherence 检测策略，并独立选择 gate softness/deadband；在 clean 保留有用辅助约束下对可检测故障安全降级。coherence 目前只支撑检测 claim，不扩张为 gate claim。
4. **故障可检测性分类学**：刻画哪类故障由哪种信号可检测（结构破坏型/时序冻结型/跨模态异步型/平滑统计漂移型），含诚实的 near-chance 负结果（imu_bias），并用多种子检测评估与 LOSO 泛化实验分别验证稳定性。

---

## 4. 关键图表规划（Phase 2.5，正文前先定）

- **Fig.1** 系统总览：基础控制器 ‖ 故障注入 ‖ 多信号检测 ‖ 校准门控 ‖ 力矩回放评估（一张图讲清 pipeline）
- **Fig.2** 故障注入示意 + 各故障对输入信号的影响（直观说明 5 类故障差异）
- **★Table.I** 故障检测 AUROC 矩阵（故障类型 × 检测信号 × ID/OOD）← 分类学的核心证据
- **★Table.II** 门控前后力矩伤害指标（按故障类型）← 安全收益的核心证据
- **Fig.3** 严重度扫描曲线（packet_loss / sensor_delay 单调性）
- **Fig.4** E5 归因：精度 vs 鲁棒性权衡（消融阶梯）
- **Table.III** LOSO 15 折汇总（RMSE/R²/门控/闭环降低，mean±std）
- **★Table.IV / Fig.5** us vs Tourk2025 差异化对照表
- **Fig.6** 闭环净力矩 RMS 降低（dynamics-in-the-loop）

## 5. 审稿人攻击点 → 防御映射（写作时确保覆盖）

| 攻击点 | 防御位置 | 弹药 |
|---|---|---|
| 创新性（不就是 UQ 门控？） | Intro P3 + RelatedWork 对照段 | 重定位：故障 benchmark + 伤害框架 + 分类学；Tourk2025 只做 OOD 动作 |
| 没硬件 | Methods 3.7 + Limitations | [Luo2024] Nature 仿真路线合法 + 诚实 future work |
| 伤害指标自定义 | Methods 3.5 | [ASTM-F3578] 标准 + [Wu2024] 人体代价 |
| 精度变差了 | Results 4.5 | E5 归因 + [Ens-aug-calib] 已知现象，框成权衡 |
| n 太小 | Results 4.6 | LOSO 15 折 + 多种子 |
| imu_bias 没修好 | Results 4.4 + Limitations | 报告当前设定下的 near-chance 负结果和适用范围，不外推为普遍不可能 |
| calibrated uncertainty 是否成立 | Results 4.1 + Limitations | 报 identity coverage；validation-only temperature scaling 的冻结测试负结果，不做跨域校准强 claim |
| fused AUROC 为什么低于 specialist | Results 4.2 + Discussion | 贡献是故障可检测性矩阵与 validation-frozen policy，不声称单一融合对每类故障最优 |

---

## 6. 写作顺序（Phase 3，借鉴 skill 的"非阅读顺序"最佳实践）

1. **Methods + Results 配对**（先写，互相 1:1 对应；数据已就绪）
2. **Intro P3（缺口）+ Conclusion 配对**
3. **Discussion**
4. **Intro P1-2（背景）**
5. **Related Work**
6. **Abstract**（最后）
7. **Title** 定稿

> 每完成一步设用户确认门，再进入下一步。

---

## 7. 状态
- [x] 必要性/创新性排查、最近邻对照（Tourk2025 全文）、伤害锚点、引文库（REFERENCES.md）
- [x] 大纲建立（本文件）
- [ ] **AAAI 下一步**：完成三种子 stress 刷新 → 锁定 Table-I/II 列定义与 claim boundary → 起草 Methods+Results；校准和 ECDF 融合保留为负结果，不再消耗主实验预算

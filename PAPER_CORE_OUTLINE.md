# 核心大纲 (PAPER_CORE_OUTLINE.md)

> 一页看清全篇逻辑。详版见 [PAPER_OUTLINE.md]，文献见 [REFERENCES.md]，数据见 reports/v2_paper_suite/。

---

## 一句话定位
学习型外骨骼控制器默认输入干净；本文系统研究**传感器故障下它会施加什么错误辅助、能否在线检测与安全门控、哪些故障在现有观测下难以辨识**，并提出一套**力矩级生物力学伤害评估框架**。

## 标题（暂定）
Reliability-Aware Task-Agnostic Exoskeleton Control: Sensor-Fault Detection, Torque-Level Harm Evaluation, and a Fault-Detectability Taxonomy

目标 venue：AAAI 优先；论文叙事按 AI benchmark / uncertainty-and-risk modeling / calibrated decision policy 组织。

---

## 逻辑主线（背景 → 缺口 → 方法 → 发现 → 意义）

1. **背景**：数据驱动 task-agnostic 外骨骼控制（Molinaro 力矩估计）性能强、正在走向真实世界。
2. **缺口**：
   - 它默认传感器输入干净，故障下仍会输出 → "garbage-in-garbage-out" → 错误辅助有真实安全后果（ASTM 标准、人体研究均证实）。
   - 最近邻工作（Tourk2025 不确定性门控）**只处理 OOD 动作，明确未碰传感器故障**。
   - 所有先前工作**只用分类指标（AUROC/F1），无人量化错误辅助对人腿的物理后果**。
3. **方法（4 块）**：故障注入 benchmark + 力矩伤害框架 + 多信号检测与校准门控 + 故障可检测性分类学。
4. **发现**：结构/时序故障可检测并安全门控；平滑统计漂移在当前观测与故障幅度下接近不可辨识；安全有可量化的精度代价。
5. **意义**：把"可靠性"从口号变成可测量、可复现、对接安全标准的评估框架。

---

## 四条核心贡献
1. **首个传感器故障可靠性 benchmark**——5 类故障 × 严重度梯度，注入到 task-agnostic 学习型控制器。
2. **力矩级生物力学伤害评估框架**——wrong-direction / peak-wrong-torque / retained-aligned-torque / jerk，对接 ASTM-F3578。
3. **多信号输入完整性检测 + utility-aware 安全门控**——验证集选择 residual/staleness/coherence 紧凑检测策略；门控策略独立地在 clean 保留辅助约束下选择并冻结，避免把检测校准与输出方差校准混为一谈。
4. **故障可检测性分类学**——哪类信号抓哪类故障，含诚实的 near-chance 负结果（imu_bias）。

---

## 关键结果（已跑完，支撑上述贡献）
- **检测（三种子）**：staleness 检测 packet loss（ID/OOD AUROC 0.937/0.958）及 held-out burst（0.918/0.923）；因果 32-frame cross-IMU coherence 检测固定 delay（0.953/0.989）和 held-out jitter（0.939/0.964）。coherence 当前仅是检测通道，不声称已进入 gate。
- **统一融合的边界**：冻结的 q90-max 融合在 delay 上低于 specialist（固定 delay ID/OOD 0.878/0.774）；验证集选择的 ECDF-Fisher 虽使 seed-7 测试平均 AUROC +0.046，但使 OOD burst -0.194，故不合入主方法。
- **门控安全**：seed-7 高严重度 packet_loss@0.30 的 gate 与 wrong-direction 均归零；三种子 clean gate 保留模型未门控力矩约 0.98。
- **门控边界**：标准故障强度下 seed-7 最大 wrong-direction 绝对改善仅 0.004；完成三种子严重度曲线前，只把 gate 表述为 severity-adaptive safety trade-off，不宣称所有故障下都有显著收益。
- **泛化**：LOSO 15 折 test_id/clean RMSE 0.20±0.02、R² 0.71，闭环净力矩 RMS 降低 ~9%。
- **权衡**：E5 归因——精度代价来自数据增强(+0.035)与重构头(+0.016)，非 forecast 头；框成安全-精度权衡。
- **诚实负结果**：imu_bias 检测约 0.53；在当前传感器观测、注入幅度与候选检测器族下接近随机。只报告 empirical non-detectability，不宣称普遍不可能。
- **校准边界**：seed-7 未缩放输出在 test ID/OOD 的 coverage ECE 为 0.015/0.038；验证集选择的 per-channel temperature 反而变为 0.082/0.045。论文只称 gate 是 validation-selected，不声称输出不确定性已跨域校准。
- **证据补齐中**：严重度曲线已有单 seed，三种子刷新已进入空闲 GPU 调度队列；完成前不写多种子 stress 或普遍 gate-benefit claim。

---

## 章节骨架（IMRAD）
- **I. Intro** — 背景 / 问题（GIGO+安全后果）/ 缺口（Tourk2025 只做 OOD 动作 + 无力矩度量）/ 贡献 / 组织
- **II. Related Work** — 学习型控制｜UQ&异常检测｜可穿戴故障容错｜安全评估&扰动｜仿真验证 + **us-vs-Tourk2025 对照**
- **III. Methods** — 基础控制器复现｜概率TCN(+recon/forecast/fault头)｜故障注入模型｜多信号检测｜**力矩伤害框架**｜validation-selected utility gate｜闭环评估｜实验设计(LOSO/多种子/消融)
- **IV. Results** — 干净基线｜检测AUROC矩阵｜门控安全收益｜**可检测性分类学**｜E5权衡｜LOSO+闭环｜严重度扫描
- **V. Discussion** — 回到主线｜与Tourk2025/Hsu2024互补｜安全-精度工程含义｜分类学可推广性
- **VI. Limitations** — 纯仿真无硬件｜伤害指标是离线代理｜imu_bias不可检测｜统一融合低于 specialist｜输出校准跨域不稳定｜OOD精度落后
- **VII. Conclusion**

---

## 核心图表（两张证据表是全篇支柱）
- **★Table-I** 故障检测 AUROC 矩阵（故障 × 检测信号 × ID/OOD × base/challenge generator）→ 分类学证据
- **★Table-II** 门控前后力矩伤害指标（按故障）→ 安全收益证据
- **★Table-IV** us-vs-Tourk2025 差异化对照
- Fig-1 系统总览 pipeline｜Fig-3 严重度扫描曲线｜Fig-4 E5 权衡阶梯｜Table-III LOSO 汇总｜Fig-6 闭环净力矩降低

---

## 审稿防御速查
| 攻击 | 防御 |
|---|---|
| 不就是 UQ 门控？ | 重定位：故障benchmark+伤害框架+分类学；Tourk2025 只做 OOD 动作 |
| 没硬件 | Luo2024(Nature)仿真路线合法 + 诚实 future work |
| 伤害指标自定义 | ASTM-F3578 标准 + Wu2024 人体代价 |
| 精度变差 | E5 归因 + Ens-aug-calib 已知现象，框成权衡 |
| n 太小 | LOSO 15 折 + 多种子 |
| imu_bias 没修好 | 报告当前观测与 detector family 下的 near-chance 负结果，不外推为普遍不可能 |
| 为什么不做温度缩放？ | validation-only scaling 在冻结 test ID/OOD 上同时变差，保留 identity 并如实报告域依赖校准 |
| 为什么融合低于单通道？ | 正文以故障×信号矩阵支撑可检测性分类学；不声称一个标量融合对所有故障最优 |

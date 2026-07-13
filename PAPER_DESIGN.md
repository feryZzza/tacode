# 论文整体设计 / Paper Design Document

> 状态：历史设计文档，包含尚未被最终证据支持的早期目标。AAAI 当前 claim、数值与限制以 `PAPER_CORE_OUTLINE.md` 和 `reports/v2_refreshed_suite/` 为准；不要直接从本文件摘取结果。

> Reliability-Aware Task-Agnostic Exoskeleton Control
> 基于不确定性与输入完整性感知的任务无关外骨骼安全控制

本文件梳理重组后的论文思路、系统架构、与已有代码的映射，以及实验矩阵。
目标档次：IEEE TNSRE / RAL / T-RO（扎实方法学论文，纯离线 + 轻量动力学在环，无真机）。

---

## 0. 一句话定位

第一篇（Nature 2024）解决"能否任务无关辅助"，第二篇（SciRobotics 2025）解决"能否低成本训练"。
**本文解决两篇都回避的第三个问题：当估计不可靠、传感器异常、动作未见时，控制器如何知情并安全降级，而不在 clean 条件下损失有用辅助。**

我们不声称提升 RMSE，而是提出并验证一个**可靠性-安全评估与控制框架**：
probabilistic moment estimator + input-integrity monitor → calibrated risk → risk-gated torque mapping，
并用**离线 torque-replay 风险指标 + 轻量动力学在环代理收益**量化"安全 vs 收益"的帕累托权衡。

---

## 1. 重组后的核心卖点（Contributions）

旧版本的问题：把 "fault detection" 和 "OOD-aware safety" 当成卖点，但实验数据（AUROC≈0.5）反而证伪。
重组后，卖点收敛到**真正立得住、且现有数据支持或可补齐**的四点：

**C1（主，方法学）—— 安全导向的离线评估协议。**
对任务无关力矩控制，提出一套不依赖真机的安全风险指标：
wrong-direction ratio、peak wrong torque、retained aligned torque、command jerk，
并配一个轻量动力学在环代理（actuator 一阶响应 + 饱和 + 关节功/代谢代理），
把"估计误差"翻译成"辅助安全性与潜在收益"。这是与两篇原文最大的差异化。

**C2（主，模型）—— 输入完整性感知的可靠性头。**
在概率 TCN 上增加 denoising/重构残差头：残差能量度量"输入是否偏离训练分布"。
这修复了旧版 packet_loss / sensor_delay / imu_bias 检测 AUROC≈0.5 的根本缺陷
（这些故障对单帧统计无损，但在"可预测性/重构残差"域可分）。
风险信号 = f(aleatoric σ, 重构残差, 监督 fault logit) 的标定融合。

**C3（主，控制）—— 净收益为正的风险门控控制律。**
τ = g(risk)·k·M̂(t−d)，其中 g 在 clean（低 risk）时 →1（不误伤有用辅助），
仅在 risk 超过 validation 标定的 deadband 后才降级。
门控阈值**仅在验证集选择并冻结**，test 只报一个数（消除"测试集调参"质疑）。
报告 wrong-torque↓ 与 retained-aligned-torque 的帕累托前沿，证明存在帕累托改进点。

**C4（评估严谨性）—— 归因清晰的鲁棒性 + 跨被试泛化。**
通过消融解耦"鲁棒性来自数据增强 vs 来自概率/残差头"；
用 leave-one-subject-out 替代 n=2 test；
用同数据划分自训确定性 TCN 作为无泄漏 baseline（官方 checkpoint 仅作次要对比并标注泄漏）。

> 不再写进卖点（除非补齐到达标）：纯监督 fault logit 检测、单点 logvar 的 OOD。
> 这些降级为"消融对照"，用来反衬 C2 的残差头与 C4 的 ensemble。

---

## 2. 与源论文的关系与新颖性论证

| 维度 | Nature 2024 | SciRobotics 2025 | 本文 |
| --- | --- | --- | --- |
| 控制依据 | biological joint moment | 同左 | 同左 + 估计**可信度** |
| 训练范式 | 大量目标设备标签 | 域自适应降成本 | 不变（沿用第一篇数据） |
| 模型输出 | 点估计 moment | 点估计 moment | moment + σ + 输入完整性残差 |
| 控制映射 | 固定 scale/delay/filter/clamp | 同左 | **风险门控**、净收益为正 |
| 安全降级 | 无 | 无 | **有**（知情降级 + deadband） |
| 评估 | RMSE/R²/代谢（真机） | RMSE/R² | 安全风险指标 + 动力学在环代理（离线） |
| 故障/OOD | 未处理 | 未处理 | 系统注入 + 可检测性 + 标定 |

新颖性站位：本文是"安全可靠性"这一缺失环节的补全，方法学贡献（C1 评估协议、C2 残差可靠性、C3 净收益门控）独立于具体设备，可迁移到任意 moment-based 辅助控制器。

---

## 3. 系统架构

```
                          外骨骼传感器时序 x[B,C,T]
                                   │
                    ┌──────────────┴──────────────┐
                    │   共享 TCN 时序编码器 (Nature-capacity)   │
                    └──────────────┬──────────────┘
             ┌──────────┬──────────┼──────────┬───────────┐
             ▼          ▼          ▼          ▼           ▼
        mean_head  logvar_head  fault_head  recon_head  (MC-dropout/ensemble)
         M̂hip,knee   σhip,knee   p_fault    x̂ 重构      → epistemic σ_e
             │          │          │          │
             │          │          │     residual r = ‖x − x̂‖（按模态）
             │          └────┬─────┴────┬─────┘
             │               ▼          ▼
             │        ┌─────────────────────────┐
             │        │ 风险标定融合 calibrate_risk │
             │        │ risk = h(σ_a, σ_e, r, p_f) │
             │        └────────────┬────────────┘
             ▼                     ▼
   ┌──────────────────┐   ┌──────────────────────────────┐
   │ moment_to_torque │   │ risk_gate g(risk; soft*, dead*)│  soft*,dead* 仅由 val 选定
   │ scale/delay/LPF  │   └───────────────┬──────────────┘
   │ /clamp (Nature)  │                   │
   └────────┬─────────┘                   │
            └───────────────┬─────────────┘
                            ▼
                   τ_exo = g·k·φ(M̂(t−d))
                            │
              ┌─────────────┴──────────────┐
              ▼                            ▼
   离线 torque-replay 安全指标       轻量动力学在环代理 closed_loop.py
   (wrong dir / peak / retained /   net moment = M_bio − τ_applied
    jerk)                           actuator 一阶+饱和; 关节功/代谢代理
```

模块职责（粗体为本次新增/改造）：
- `reliability/model.py`：加 **recon_head**；支持 **MC-dropout 推理**；deep-ensemble 由多 seed checkpoint 组合。
- `reliability/metrics.py`：加 **calibrate_risk**（多信号融合 + 温度/分位标定）；门控加 **deadband**；保留 torque-replay 指标。
- `reliability/closed_loop.py`（**新文件**）：动力学在环代理收益。
- `run_reliability_experiment.py`：加 **训练模式开关**（det-no-aug / det-aug / prob-aug / prob-aug-recon）、**LOSO 模式**、**val-only 门控选择**、**ensemble/MC 评估**、closed-loop 指标输出。
- `summarize_reliability_suite.py`：已能聚合 seed/ablation；扩展读取 closed-loop 与 LOSO 字段。

---

## 4. 实验矩阵（每个缺陷 → 对应实验 → 对应卖点）

| # | 旧版缺陷（被现状证伪） | 新实验 | 支撑卖点 |
| --- | --- | --- | --- |
| E1 | packet_loss/sensor_delay AUROC=0.50 | recon 残差头 vs 监督 logit 的检测 AUROC 对比（按故障类型） | C2 |
| E2 | OOD AUROC≈0.56 | deep ensemble + MC-dropout 的 epistemic σ 做 OOD，目标 >0.7 | C2 |
| E3 | 门控 clean 上 retained delta 负（误伤） | deadband 门控；clean 上 g≈1；报 retained delta≈0 | C3 |
| E4 | softness 测试集事后挑 | val 选 soft*/dead* 并冻结，test 单值；附录敏感性 | C3 |
| E5 | 鲁棒性归因不清（混了增强） | 4 路训练消融：det-noaug/det-aug/prob-aug/prob-aug-recon | C4 |
| E6 | baseline 泄漏测试集 | 同划分自训确定性 TCN；官方 checkpoint 仅次要对比 | C4 |
| E7 | n=2 test | leave-one-subject-out（23 被试），报均值±CI + 配对显著性 | C4 |
| E8 | 纯 RMSE，无部署价值 | 动力学在环代理：辅助下 net 关节功/代谢代理变化曲线 | C1 |
| E9 | 已有但保留 | 故障强度 stress 曲线（packet_loss@*, sensor_delay@*） | C1/C2 |
| E10 | 已有但保留 | action 输入消融（desired/measured/interaction） | 次要发现 |

主结果表（论文 Table 1 雏形）：每个 scenario × {det baseline, prob, prob+recon+gate}，
报 RMSE / R² / wrong-dir / retained / closed-loop benefit，均为 LOSO 均值±CI。

帕累托图（论文 Fig.）：横轴 retained aligned torque，纵轴 wrong-direction ratio，
沿 softness/deadband 网格画曲线，标出 val 选定点，证明帕累托改进。

校准图：reliability diagram（coverage vs nominal）+ risk-error 散点。

### 4.1 评估口径的纪律（避免审稿人质疑）
- 任何超参（softness、deadband、risk 融合温度）一律 **val 选定、test 冻结**。
- LOSO：每个 fold 留 1 被试做 test，其余按被试分 train/val；阈值在该 fold 的 val 上定。
- 显著性：被试级配对（gated vs ungated、prob vs det），Wilcoxon signed-rank，报 effect size。
- 公平性：消融与 baseline 共享数据划分、窗口、归一化统计、epoch 预算。

---

## 5. 论文结构（草拟）

1. Introduction — 任务无关控制已可行、可低成本训练，但缺安全可靠性环；本文补全。
2. Related Work — task-agnostic exo control；uncertainty in regression（ensemble/evidential/heteroscedastic）；
   OOD/anomaly detection for time series；safe assistive control / fail-safe。
3. Problem & Safety Metrics（C1）— 定义 wrong-dir/peak/retained/jerk + 动力学在环代理收益。
4. Method —
   4.1 probabilistic moment estimator（heteroscedastic NLL）
   4.2 input-integrity residual head（C2）
   4.3 calibrated risk fusion + epistemic via ensemble/MC（C2）
   4.4 risk-gated torque mapping with deadband（C3）
5. Experiments —
   5.1 setup（dataset、LOSO、splits、baselines）
   5.2 estimation accuracy & calibration
   5.3 fault & OOD detectability（E1/E2）
   5.4 safety gating & Pareto（E3/E4）
   5.5 ablations & attribution（E5/E6）
   5.6 stress curves & action inputs（E9/E10）
   5.7 closed-loop proxy benefit（E8）
6. Discussion — 局限（无真机、合成故障、单设备）、与第二篇 DA 的衔接（future work）。
7. Conclusion。

## 6. 实施顺序（与 TaskList 对应）

1. closed_loop.py 动力学在环代理（C1 的部署价值环）。
2. model.py recon 头 + MC-dropout（C2）。
3. metrics.py calibrate_risk + deadband 门控（C2/C3）。
4. run_reliability_experiment.py：训练模式开关 + val-only 门控 + ensemble/MC 评估 + closed-loop 输出 + 自训 det baseline（E5/E6）。
5. LOSO 模式（E7）。
6. summarizer 扩展（closed-loop / LOSO 字段）。
7. smoke 验证 → 后台全量。

## 7. 现实风险与对策
- 残差头可能对 imu_bias（缓漂）仍弱：用一步预测残差 + 多尺度窗口残差兜底；若仍弱，如实报告并归为 limitation。
- LOSO×4 训练模式×多 seed 计算量大：先单 seed 跑通 LOSO 主结果，消融用固定划分，seed 重复只在主 profile 上做。
- 动力学在环代理是"代理"非真机：明确写明假设（被动人体模型、固定刚度、无适应），只做相对比较不做绝对代谢声明。
- 门控净收益若仍为负：回退到"安全-收益帕累托"叙事（存在可选工作点），而非"严格优于固定辅助"。

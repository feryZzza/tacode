# 第一版大纲

## 定位
baseline中的学习型外骨骼控制器默认输入干净；本文系统研究传感器故障下它会施加什么错误辅助、能否在线检测与安全门控、哪些故障本质不可检测，并提出一套力矩级生物力学伤害评估框架。

## Introduction and Related Work
> 学习型控制｜UQ门控+异常检测｜穿戴故障容错｜安全评估+扰动｜仿真验证 + 用us-vs-Tourk2025 对照
1. **背景**：数据驱动 task-agnostic 外骨骼控制（Nature2024 baseline）性能强、正在走向真实世界。
2. **缺口**：
   - 它默认传感器输入干净，故障下仍会输出 → "garbage-in-garbage-out" → 错误辅助有真实安全后果（ASTM 标准、人体研究均证实）。
   - 最近邻工作（Tourk2025 不确定性门控）只处理 OOD 动作，明确未碰传感器故障。
   - 所有先前工作只用分类指标，没有量化错误辅助对人体的伤害情况。
3. **方法**：故障注入 benchmark + 力矩伤害框架 + 多信号检测与校准门控 + 故障可检测性分类学。
4. **发现**：结构/时序故障可检测并安全门控；统计漂移型本质不可检测；安全有可量化的精度代价。
5. **意义**：把"可靠性"从口号变成可测量、可复现、对接安全标准的评估框架。

## Method
> baseline的基础上的整体闭环评估风险｜概率TCN模型改进(+recon/forecast/fault头)｜故障注入｜控制力矩伤害框架｜门控校准
1. **首个传感器故障可靠性 benchmark**——5 类故障 × 严重度梯度，注入到 task-agnostic 学习型控制器。
2. **力矩级生物力学伤害评估框架**——wrong-direction / peak-wrong-torque / retained-aligned-torque / jerk，对接 ASTM-F3578。
3. **多信号检测 + 校准安全门控**——残差/forecast/staleness/drift/ensemble 融合，clean 零误抑制下门控住 packet_loss/sensor_delay。
4. **故障可检测性分类学**——哪类信号抓哪类故障，含诚实的不可检测负结果（imu_bias）。

## Experience
packet_loss 门控→0、wrong-direction 归零；clean 门控保持
LOSO的15折测试的test_id/clean RMSE 0.20±0.02、R² 0.71，闭环净力矩 RMS 降低 ~9%

## ？？
- 纯仿真无硬件
- OOD精度落后

## ？
- UQ门控的重复性工作？
> 重定位：故障benchmark+伤害框架+分类学；Tourk2025 只做 OOD 动作
- 硬件？
> Luo2024(Nature)仿真路线合法 + 诚实 future work

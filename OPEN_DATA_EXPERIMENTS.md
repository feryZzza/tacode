# 新版公开数据实验路线与执行清单

## 1. 最合适的论文定位

当前最稳妥的主线不是继续扩大“合成故障 + 安全门控”，也不是声称实现了完整
外骨骼控制器，而是做一个严格的跨数据集研究：

> 面向可穿戴外骨骼生物关节力矩估计，在不同设备、传感器配置、被试和地形之间，
> 研究源域直接泛化、无标签目标域适应、少量目标被试适应和传感器缺失鲁棒性。

这条路线直接使用已有三个公开数据集，不需要真机和新志愿者。论文贡献应限定为
`offline estimator / transferable representation / benchmark protocol`，不能写成真人闭环
控制效果或安全性验证。

## 2. 数据统一规则

| 数据集 | 被试/本地 locomotion trial-leg | 输入 | 标签处理 |
|---|---:|---|---|
| Nature 2024 | 22 / 1500 | 双侧髋编码器、腿段 IMU；扩展配置含足底 | 髋屈曲力矩取负，膝保持；Nm/kg |
| Dryad | 34 / 10256 | 双侧髋编码器、股 IMU、骨盆 IMU | 原数据已是伸展为正、Nm/kg |
| Camargo | 22 / 3147 | 右侧髋测角、躯干/股/胫/足 IMU | OpenSim 髋取负、膝保持，Nm 除体重 |

统一单位为：角度 rad、角速度/陀螺 rad/s、加速度 m/s²、力 N/kg、力矩
Nm/kg。左右腿样本被映射到同一右腿参考系。所有 NaN 都保留为逐通道掩码，绝不
用“两个关节必须同时有效”来丢弃单侧有效标签。

默认 `minimal` 八通道是三个数据集真实共有的传感器集合：髋角/角速度 + 股部
六轴 IMU。`hip_imu` 和 `all` 用于异构传感器/缺失掩码实验，不应与 minimal
结果混在同一主表中。

## 3. 必须回答的研究问题

1. **RQ1，跨人泛化：** 单数据集训练与三数据集合并训练，哪个在严格被试留出上更好？
2. **RQ2，跨设备零样本：** 留出一个完整数据集，仅用另外两个数据集训练，误差增加多少？
3. **RQ3，无标签域适应：** 只使用若干目标域被试的传感器、不使用其力矩标签，DANN
   能否优于 source-only？适应被试与测试被试必须分离。
4. **RQ4，标签效率：** 使用 1/2/4 名目标域被试标签时，性能怎样逼近 target-only
   上限？
5. **RQ5，传感器异构与故障：** 显式缺失掩码 + 传感器组 dropout 是否提升缺少
   骨盆/胫/足 IMU 时的稳定性？

## 4. 主实验与基线

所有结果至少使用 `seed=7,17,27`，报告被试级 bootstrap 置信区间；最终汇总时不能
把时间点当成独立统计样本。

### A. 数据集内和 pooled 上限

- 每个数据集单独训练，`protocol=pooled, alignment=none`；
- 三数据集 pooled，`minimal + alignment=none`；
- 三数据集 pooled，`minimal + DANN`；
- 三数据集 pooled，`hip_imu + mask + sensor dropout`。

### B. Leave-one-dataset-out

对 Nature、Dryad、Camargo 依次作为 target：

- source-only：`adaptation-participants=0, alignment=none`；
- UDA：`adaptation-participants=4, target-supervision=none, alignment=dann`；
- few-shot：`adaptation-participants=1/2/4, target-supervision=labels`；
- target-only supervised 是参考上限，不与无标签方法混称公平基线。

### C. 消融

- `alignment`: none vs DANN；
- `sensor-dropout`: 0 vs 0.15；
- 输入：minimal vs hip_imu；
- 数据：单源 vs 双源；
- Camargo 标签变换做一次独立波形审计，确认 OpenSim 的 hip_flexion 为屈曲正、
  knee_angle 为伸展正；不得为了降低 RMSE 在测试集上选择符号。

## 5. 评估要求

主指标为 hip/knee RMSE（Nm/kg），同时报告 MAE、R²。每张主表都应包括：

- 每个目标数据集的结果，而非只给三者混合平均；
- 每个 locomotion mode 的结果；
- 训练/适应/验证/测试被试列表；
- 标签有效率与每个关节的有效样本数；
- 参数量、因果历史长度、推理延迟和训练预算。

建议新增论文级统计脚本时采用“每名测试被试先算一个指标，再跨被试比较/重采样”。
当前 JSON 中的 sample-weighted 指标用于训练监控，不能替代最终被试级显著性检验。

## 6. 执行顺序

- [x] 核验三个本地数据集的目录、被试、trial 与文件可读性。
- [x] 建立统一 schema、单位、符号、体重归一化和缺失掩码。
- [x] 实现严格被试隔离的 pooled、source-only、UDA、few-shot 划分。
- [x] 实现 mask-aware causal TCN、DANN、domain-balanced sampler、sensor dropout。
- [x] 实现配置/划分/checkpoint/指标落盘和真实三数据集 smoke test。
- [ ] 运行三个 seed 的单数据集 baseline。
- [ ] 运行 pooled minimal baseline 与 DANN。
- [ ] 对三个目标域运行 source-only、UDA、1/2/4-person few-shot。
- [ ] 完成输入、alignment、sensor-dropout 消融。
- [ ] 增加被试级 bootstrap / paired permutation 统计。
- [ ] 绘制跨域误差、标签效率曲线、传感器缺失鲁棒性图。
- [ ] 根据结果决定投稿定位；在主结果完成前不预设“域适应一定有效”。

## 7. 结果判据与止损点

值得继续写成论文的最低证据是：在至少两个 target dataset 上，无标签或少标签方法相对
source-only 有一致提升，并且不是由被试泄漏、标签符号或样本量加权造成。若 DANN 不稳定，
优先尝试 CORAL/MMD 或冻结 encoder 后做轻量 adapter，而不是继续增加网络规模。

若跨数据集结果始终显著差于单域模型，这仍可整理成公开 benchmark/负结果，但应把贡献
改成“跨设备传感器与标签标准化带来的系统误差分析”，不再强调新控制算法。

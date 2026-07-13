# 实体机实验规划与采购清单
> Reliability-Aware Exoskeleton Control —— 硬件验证台

定位（由 2026-06-14 决策确定）：**单关节起步（膝或踝）· 尽量买现成平台 · 目标是验证可靠性框架，而非复刻完整双侧助力外骨骼。**

本文件做三件事：(1) 拆解前人两篇源论文 + 数据集用的实体机设备；(2) 给出一个最小但能站得住的单关节测试台架构；(3) 列出可直接下单的采购清单 + 实验里程碑 + 预算时间线。

---

## 0. 先想清楚：实体机要为这篇论文证明什么

论文当前的全部卖点都是**纯离线 / 仿真**得到的（见 [PAPER_DESIGN.md](PAPER_DESIGN.md) 的 C1–C4）：
- C1 力矩回放伤害指标（wrong-direction / peak-wrong / retained / jerk）
- C2 输入完整性可靠性头（残差 + forecast + staleness + drift 检测）
- C3 净收益为正的风险门控律
- C4 LOSO 泛化 + 归因消融

实体机**不需要**重新证明全部，只需补上仿真最被质疑的一环。最有性价比的目标是回答三个问题：

1. **检测在真实传感器噪声下还成立吗？** —— staleness 让 packet_loss/sensor_delay 检测 AUROC→1.0，但那是在合成信号上。真实 IMU/编码器/insole 的抖动、时漂、丢包是否会制造假阳性？
2. **门控真的能在执行器上压住有害力矩吗？** —— 仿真里 gate→0 切断 packet_loss 的力矩；真机上要看作动器实际输出的力矩/jerk 是否随 gate 下降。
3. **clean 条件下不误伤辅助吗？** —— deadband 在真实噪声下是否仍保持 mean_gate≈1。

> **关键洞察（决定清单规模）：** 你的 6 类故障里，packet_loss / sensor_delay / imu_bias 本质是**数字信号腐蚀**，可以直接把 `reliability/faults.py` 挂在真实传感器数据流上软件注入——不需要任何额外硬件。只有 encoder_dropout / insole_missing 适合做**物理断连**对照实验。所以实体台的硬件重心是：**一条真实的传感链 + 一个能测真实力矩的单关节闭环**，而不是机械复杂度。这把成本从"造外骨骼"降到"搭一个带传感的单关节测试台"。

---

## 1. 前人实体机设备分析

下表是从两篇源论文 Methods 和数据集逐字提取的设备（已核对 PDF 原文）。

### 1.1 Nature 2024（Molinaro et al.）——「X Moonshot Factory」髋+膝外骨骼

| 子系统 | 具体型号 / 规格 | 来源原文 |
| --- | --- | --- |
| 作动器 | **T-Motor AK80-9**（quasi-direct-drive，准直驱），4 个（双侧髋+膝），峰值力矩限幅 **15 N·m/关节** | "Compact quasi-direct drive actuators (AK80-9 T-Motor)…up to 15 N m" |
| 主控 | **Raspberry Pi 4B**，跑 **55 Hz** 控制环，管 CAN 总线 + 蓝牙 + 本地存数据 | "A Raspberry Pi 4B served as the primary onboard computer…control loop at 55 Hz" |
| 推理协处理器 | **NVIDIA Jetson Nano**（5 V/2 A），以太网与 RPi 异步 TCP/IP 连接，全程板载推理 | "machine learning coprocessor (NVIDIA Jetson Nano)…fully onboard" |
| IMU | **6× OpenIMU（OpenIMUA）** 六轴，装在小腿/大腿支杆，经 CAN 与 RPi 通信 | "Six-axis IMUs (OpenIMUA) mounted to the shank and thigh struts…via CAN" |
| 关节编码器 | 髋/膝编码器（随 AK80-9 集成），编码器速度经二阶 10 Hz Butterworth 滤波 | "joint encoders on the hips and knees…Encoder velocity lowpass filtered…10 Hz" |
| 压力鞋垫 | **Moticon** 无线压感鞋垫，测垂直 GRF + 压心(COP)，内嵌六轴 IMU，蓝牙 + 纽扣电池供电 | "Pressure-sensitive insoles (Moticon)…embedded six-axis IMU…Bluetooth…coin-cell" |
| 电源 | 2× **DeWalt 20 V / 3 Ah 电钻电池**并联，约 2 h 连续行走 | "two 20 V, 3 Ah drill batteries (DeWalt) connected in parallel" |
| 结构 | 水刀切割**碳纤维**大腿支杆 + **3D 打印尼龙**小腿/护腰，零拉伸织物 + Velcro/快拆，总重约 **7 kg** | "waterjet-cut carbon fibre…3D printed nylon…roughly 7 kg" |
| 通信 | CAN 总线（作动器+IMU）、蓝牙（鞋垫）、WiFi（连笔记本做可视化） | "managed CAN Bus and Bluetooth…interfaced through WiFi with a laptop" |
| 控制律 | scale(髋20%/膝15%) → delay(髋125ms/膝75ms 含滤波) → 二阶10Hz Butterworth → clamp 15N·m | Methods "Real-time joint moment estimation" |

### 1.2 训练数据真值采集（实验室设备，非穿戴）

| 设备 | 用途 | 来源 |
| --- | --- | --- |
| 光学动捕 | 采人体运动学，做 OpenSim 逆动力学算关节力矩真值 | "optical motion capture…OpenSim inverse dynamics" |
| 高保真力板 / 测力跑步机 | 测 GRF，逆动力学输入 | "high-fidelity force plates…GRFs" |
| OpenSim | 逆动力学求髋/膝力矩真值标签 | "Standard OpenSim inverse dynamics…ground-truth labels" |

### 1.3 SciRobotics 2025（Scherpereel et al.）——**同一台外骨骼**

复用 1.1 的 X Moonshot Factory 设备（4× AK80-9、RPi、Jetson Nano、OpenIMU、同款控制律），新增：
- **代谢测量**：金属代谢车（间接量热，Cosmed 级），跑步机 5° 坡 / 1.25 m/s；负重任务用 25 lb(11.3 kg) 壶铃 + 节拍器 10 bpm。
- 真值采集设备同 1.2（Vicon 动捕 + Bertec 测力跑步机，来自同实验室）。

### 1.4 你的训练数据集（Camargo 2021，决定真值标定口径）

- **Vicon** 光学动捕 + **Bertec** 测力跑步机（嵌入式力板）
- **Delsys** sEMG，2000 Hz（与其余信号不同采样率）
- 22 名健康成人；平地/跑步机行走、上下楼梯、上下坡及过渡

> 含义：如果你**只复现传感链做检测验证**，不需要 Vicon/Bertec/Delsys（那些是为算力矩真值用的）。只有当你想在真机上算**生物力矩真值**做精度对照时，才需要动捕+力板，那会把预算推高一个数量级——下面的方案默认**不走真值路线**，用"作动器自身力矩传感"作相对参照。

来源：[Camargo dataset (EPIC GT)](https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/) · [PubMed 33677231](https://pubmed.ncbi.nlm.nih.gov/33677231/) · [LocoLab data](https://web.eecs.umich.edu/locolab/data.html)

---

## 2. 推荐架构：单关节可靠性验证台

目标：**用最少硬件，让 `faults.py` + `metrics.py` + 风险门控在真实传感器闭环上跑起来，并测到执行器侧的真实力矩/jerk。**

```
        ┌─────────────── 真实传感链 (study 的"输入完整性"对象) ───────────────┐
        │  关节编码器 + 大腿/小腿 IMU + 足底压感 (单腿)                          │
        └───────────────┬──────────────────────────────────────────────┘
                        │ CAN / serial, 200 Hz 上采样到与训练一致
                        ▼
            ┌────────────────────────────┐
            │  实时主控 (RPi 5 / Jetson)   │  ← faults.py 在此软件注入
            │  采集 → [故障注入] → TCN推理  │     (packet_loss/delay/imu_bias)
            │  → calibrate_risk → gate g   │
            └───────────────┬────────────┘
                            │ 力矩命令 τ = g·k·M̂(t-d)
                            ▼
            ┌────────────────────────────┐
            │  单关节准直驱作动器 (膝或踝)   │  ← 内置力矩/电流反馈 = 真实力矩观测
            │  AK80-9 或商用 ExoBoot/HEBI  │
            └───────────────┬────────────┘
                            ▼
        ┌─────────────── 负载端 (选其一) ───────────────┐
        │  A. 台架测功 (benchtop): 作动器对弹簧/惯量盘/   │  ← 先跑通、最安全、可量化
        │     力矩传感器，无人体——验证 gate↔τ 因果        │
        │  B. 人体单腿穿戴: 1 名受试者绑膝/踝，测交互力矩  │  ← 需伦理审批，二阶段再做
        └────────────────────────────────────────────┘
```

两阶段策略（强烈建议）：
- **阶段 1（台架，无人体）**：作动器接惯量盘/弹簧 + 串联力矩传感器。注入故障，验证"gate 下降 → 实测输出力矩下降、jerk 下降"，clean 时 gate≈1 输出不衰减。**这一步就足以支撑论文一张"真机闭环验证"图**，且无需伦理审批、风险最低。
- **阶段 2（人体单腿，可选）**：1–3 名受试者穿单关节，测人-机交互力矩与主观反馈。需 IRB/伦理审批（按 Wu&Stirling [Wu2024] 的 imperfect-algorithm 范式设计，已在 REFERENCES.md 锚定）。

---

## 3. 采购清单（两套方案对照）

下面给"买现成平台"（你的倾向）和"自研机械"两条线，每条都标了**单关节膝/踝**起步的最小配置。价格为 2026 年量级估算（人民币 / 美元），实际以询价为准。

### 方案 A —— 买现成平台（推荐，上手快）

| 模块 | 推荐选型 | 数量 | 估算单价 | 备注 |
| --- | --- | --- | --- | --- |
| **执行器（带力矩反馈）** | T-Motor **AK80-9** (准直驱, 内置编码器+电流环) | 1 | ¥1,800–2,500 | 和源论文同款，复现性最佳；力矩由电流估算 |
| └ 备选(更省事) | **HEBI X-Series** 模块化作动器 (内置力矩传感+API) | 1 | $1,500–3,000 | 自带高层 API+力矩闭环，开发最快；贵 |
| └ 备选(踝, 整机) | **Dephy ExoBoot** 单只 (踝助力整机, 已验证平台) | 1 | $5,000–10,000+ | Tourk2025/Shetty2025 同平台；省机械但贵、偏踝 |
| **CAN 接口** | CANable / PEAK PCAN-USB 或 Jetson 自带 CAN | 1 | ¥150–800 | AK80-9 走 CAN |
| **实时主控** | Raspberry Pi 5（或 RPi 4B 对齐源论文 55 Hz） | 1 | ¥600 | |
| **推理单元** | NVIDIA **Jetson Orin Nano**（替代已停产 Jetson Nano） | 1 | ¥1,500 | TCN 板载推理；也可先用主控 CPU 跑 |
| **大腿/小腿 IMU** | 六轴 IMU（如 LSM6DSOX / 商用 OpenIMU 替代）×2 | 2 | ¥100–600/个 | 对齐源论文 thigh+shank IMU |
| **足底压感** | **Moticon OpenGo** 鞋垫（源论文同款，含 IMU+BLE） | 1 对 | $3,000–6,000 | 贵；台架阶段可先省略，软件置零做 insole_missing |
| └ 省钱替代 | FSR 压力阵列 + 自制采集 | 1 | ¥300 | 精度低，仅作 insole 通道占位 |
| **串联力矩传感器** | 旋转扭矩传感器（量程 ±20 N·m，台架真值） | 1 | ¥2,000–5,000 | **阶段1关键件**：独立测真实输出力矩，不靠电流估算 |
| **负载端** | 惯量盘 / 扭簧 + 台架夹具（铝型材） | 1 套 | ¥500–1,500 | 模拟肢体惯量+刚度 |
| **电源** | 24 V 电源（台架）或 DeWalt 20V 电池×2（移动） | 1 | ¥300–600 | |
| **急停 + 安全** | 硬件急停开关 + 力矩软限幅 + 看门狗 | 1 套 | ¥200 | **必备**，安全关键 |
| **线缆/连接器/杂项** | CAN 线、电源线、3D 打印支架 | — | ¥500 | |

**方案 A 台架最小配置合计（不含 Moticon/ExoBoot）：约 ¥8,000–15,000**
含 Moticon 鞋垫：+ $3k–6k；改用 ExoBoot 整机路线：$8k–15k 级。

### 方案 B —— 自研机械 + 采购核心件（便宜但工程量大）

在方案 A 基础上，把"现成整机/HEBI"换成自研：
- 结构：3D 打印尼龙 + 铝型材 / 碳板（对齐源论文，可水刀外协），单关节支架 ¥800–2,000。
- 作动器仍用 AK80-9（自研也绕不开优质准直驱）。
- 省下整机平台费，但增加机械设计、装配、标定 2–4 周工时。
- 适合：有机械/嵌入式人手、要长期迭代、预算紧。

> **我的建议**：按你"买现成 + 单关节 + 验证框架"的定位，走**方案 A 的台架最小配置（AK80-9 + 串联扭矩传感器 + 惯量盘 + RPi/Jetson + 2 IMU）**，鞋垫先用软件置零占位。这套 ¥1 万级、2–3 周能跑通闭环，直接产出论文需要的"真机验证"图，且无伦理审批门槛。HEBI 是"不差钱要最快"的升级项，ExoBoot 仅当你想转向踝关节真人助力时再考虑。

---

## 4. 实验里程碑（与论文图表对应）

| 阶段 | 任务 | 产出 | 对应论文卖点 |
| --- | --- | --- | --- |
| M0 | 采购 + 台架装配 + 急停/软限幅 + 看门狗 | 安全可运行的单关节闭环 | 基础设施 |
| M1 | 真实传感链采集 → 离线喂入现有 TCN，比对仿真信号分布 | "真实 vs 合成信号"分布图；检测器假阳性率 | C2 鲁棒性外部效度 |
| M2 | 软件注入 packet_loss/sensor_delay/imu_bias，台架闭环 | gate↓ → 实测 τ↓ / jerk↓ 曲线；clean gate≈1 | C2+C3 真机验证 |
| M3 | 物理断连 encoder/insole（拔线）对照 | 残差头在真实断连下的检测 AUROC | C2 结构故障 |
| M4（可选） | 人体单腿穿戴（IRB 后）, 1–3 人 | 交互力矩 + 主观安全反馈 | C1 伤害指标外部锚定 |

M1–M3 全部**无需人体**，是性价比最高的论文增量。M4 是加分项，按 Wu&Stirling 人体故障注入范式设计，需伦理审批。

---

## 5. 风险与对策

- **安全（最高优先级）**：单关节准直驱在故障注入下可能输出错误力矩。**硬件急停 + 软件力矩限幅(≤源论文 15 N·m，台架阶段建议更低 5–8 N·m) + 看门狗(丢帧即归零) 三重保护**，且阶段 1 全程无人体。
- **真值缺失**：无 Vicon/力板时不能算生物力矩绝对精度。对策：阶段 1 只做**相对比较**（gate on/off、clean/fault 的输出力矩对比），不声称绝对代谢/力矩精度——这与你仿真论文的"相对安全收益"叙事一致。
- **Moticon 太贵**：台架阶段 insole 通道软件置零，正好天然实现 insole_missing 故障；只有要复现完整输入时才买。
- **AK80-9 力矩靠电流估算有误差**：故 BOM 里单列**串联扭矩传感器**作台架真值，¥2k–5k 是关键投资。
- **Jetson Nano 停产**：用 Jetson Orin Nano 替代；或先用主控 CPU 跑 TCN（你的模型不大，55 Hz 可达）。

---

## 6. 下一步

1. 确认走方案 A 台架最小配置 → 我可以把 BOM 拆成**带链接/货期的询价表**（T-Motor、扭矩传感器、IMU 国内外渠道）。
2. 阶段 1 不碰人体、无伦理门槛，可立即启动采购。
3. 软件侧：现有 `reliability/faults.py` / `metrics.py` / 门控**几乎可直接复用**——只需写一个实时采集→注入→推理→发力矩命令的 bridge（替换仿真 `closed_loop.py` 的数据源为真实 CAN 流）。这部分我可以先把接口设计出来。

> 需要我把第 3 节做成**可下单的询价表**（含具体卖家/型号链接/货期），还是先把**实时 bridge 的软件接口**草拟出来？

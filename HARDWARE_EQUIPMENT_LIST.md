# 实体机实验设备清单汇总
> Reliability-Aware Exoskeleton Control —— 参考文献设备 + 本项目规划设备

本文件分两部分：
- **第一部分**：从相关文献中逐字提取的实体机（硬件）实验设备（已核对原文 PDF / 全文 HTML）。
- **第二部分**：本项目规划的设备清单（单关节台架，买现成为主）。

详细的实验规划、里程碑、风险对策见 [HARDWARE_EXPERIMENT_PLAN.md](HARDWARE_EXPERIMENT_PLAN.md)。

数据来源：两篇源论文 PDF（`paper/*.pdf`）+ 引用清单中真正做了实体机实验的工作全文 + 数据集页面。**纯仿真/无硬件的引用文章不列入本表**。

---

# 第一部分：参考文献的实体机设备

## 总览对照表

| 文献 | 关节 | 执行器/整机 | 主控+推理 | IMU | 其他传感 | 控制频率 | 真值设备 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Molinaro 2024 (Nature)** | 髋+膝双侧 | 4× T-Motor AK80-9 (15N·m) | RPi 4B + Jetson Nano | 6× OpenIMU 六轴 | 关节编码器、Moticon鞋垫 | 55 Hz | Vicon+力板+OpenSim |
| **Scherpereel 2025 (SciRobotics)** | 髋+膝双侧 | 同上(同一台设备) | RPi + Jetson Nano | 同上 | 同上 + Cosmed代谢车 | 同上 | Vicon+Bertec |
| **Tourk 2025 (Uncertainty-Aware)** | 踝双侧 | Dephy ExoBoot EB60 | RPi + Jetson Orin Nano | 每boot内置IMU+踝编码器 | — | 105 Hz | Bertec测力跑步机+动捕 |
| **Weigend 2025 (post-stroke)** | 踝单侧 | Harvard定制刚性踝外骨骼 | RPi 5 + 定制MCU | 3× Movella MTi-3 (100Hz) | 2× 力传感器(load cell)、踝编码器 | 100Hz估计/1000Hz底层 | Qualisys动捕(200Hz)+Bertec(2000Hz) |
| **Camargo 2021 (数据集)** | 全下肢(采集) | 无外骨骼(纯采集) | — | 躯干/大腿/小腿/足 IMU | 4× 关节角度计、11肌群EMG | — | Vicon动捕+Bertec跑步机+Delsys EMG(2000Hz)+力板 |

---

## 1.1 Molinaro 2024 — Nature（源论文 1，X Moonshot Factory 髋+膝外骨骼）

| 子系统 | 型号/规格 | 原文依据 |
| --- | --- | --- |
| 执行器 | **T-Motor AK80-9** 准直驱(quasi-direct-drive)，4个(双侧髋+膝)，峰值限幅 **15 N·m/关节** | "Compact quasi-direct drive actuators (AK80-9 T-Motor)…up to 15 N m" |
| 主控 | **Raspberry Pi 4B**，**55 Hz** 控制环，管 CAN+蓝牙+存数据 | "Raspberry Pi 4B…control loop at 55 Hz" |
| 推理协处理器 | **NVIDIA Jetson Nano**(5V/2A)，以太网 TCP/IP 接 RPi，全板载 | "machine learning coprocessor (NVIDIA Jetson Nano)…fully onboard" |
| IMU | **6× OpenIMU(OpenIMUA)** 六轴，大腿/小腿支杆，CAN 通信 | "Six-axis IMUs (OpenIMUA)…via CAN" |
| 关节编码器 | 髋/膝编码器(随AK80-9)，速度二阶10Hz Butterworth滤波 | "joint encoders on the hips and knees" |
| 压力鞋垫 | **Moticon** 无线压感，测垂直GRF+COP，内嵌六轴IMU，蓝牙+纽扣电池 | "Pressure-sensitive insoles (Moticon)…embedded six-axis IMU…Bluetooth" |
| 电源 | 2× **DeWalt 20V/3Ah** 电钻电池并联，约2h续航 | "two 20 V, 3 Ah drill batteries (DeWalt)" |
| 结构 | 水刀切**碳纤维**大腿支杆 + **3D打印尼龙**护腰/小腿，零拉伸织物，总重约 **7 kg** | "waterjet-cut carbon fibre…3D printed nylon…roughly 7 kg" |
| 真值采集 | 光学动捕 + 高保真力板 + OpenSim 逆动力学算髋/膝力矩 | "optical motion capture…force plates…OpenSim inverse dynamics" |

控制律：scale(髋20%/膝15%) → delay(髋125ms/膝75ms含滤波) → 二阶10Hz Butterworth → clamp 15N·m。

## 1.2 Scherpereel 2025 — Science Robotics（源论文 2，复用同一台设备）

- 设备与 1.1 完全相同（4× AK80-9、RPi、Jetson Nano、OpenIMU、Moticon、同控制律）。
- 新增**代谢测量**：间接量热代谢车（Cosmed 级），跑步机 5°坡/1.25 m/s；负重任务 25 lb(11.3kg) 壶铃 + 节拍器 10 bpm。
- 真值采集设备同源实验室（Vicon 动捕 + Bertec 测力跑步机）。
- 明确为双侧 4 执行器："The device has four actuators (AK80-9 T-Motor), two at the hips and two at the knees for bilateral sagittal plane assistance."

## 1.3 Tourk 2025 — Uncertainty-Aware Ankle（★ 最近邻对照，硬件验证）

| 子系统 | 型号/规格 | 原文依据 |
| --- | --- | --- |
| 整机 | **Dephy ExoBoot EB60**(Dephy Inc, Maynard MA)，踝外骨骼，双侧 | "EB60; Dephy Inc, Maynard MA" |
| 执行器 | ExoBoot 板载执行器（故障时 unspool 至零阻抗） | "the exoboot actuators unspooled, providing zero impedance" |
| 推理 | **NVIDIA Jetson Orin Nano**（跑不确定性估计+步态相位） | "Jetson Orin Nano" |
| 中介计算 | **Raspberry Pi**（收传感→处理→发力矩命令，型号未指明) | — |
| 传感 | 每只boot的 IMU + 踝编码器；16 通道双侧(accel/gyro xyz、踝角、踝速×2) | "Sensors include IMUs and ankle encoders on each boot" |
| 控制频率 | 端到端 **105 Hz**（数据窗 175 Hz） | "operates at 105 Hz" |
| 真值 | **Bertec** 测力跑步机算步态相位真值；Northeastern 动捕实验室 | "Force-Instrumented Treadmill (Bertec)" |
| 验证规模 | 训练 n=9，离线 n=3，在线 n=1 新被试户外 | — |

> 对你最重要：这是把"集成不确定性+OOD门控"做到**真机在线**的工作，但他们**只测 OOD 动作、不测传感器故障**，且用分类指标(F1/Youden-J)无力矩伤害度量、硬阈值门控。是你论文必须显著对照差异化的对象。

## 1.4 Weigend 2025 — Post-stroke Ankle（IROS，临床单侧踝）

| 子系统 | 型号/规格 | 原文依据 |
| --- | --- | --- |
| 整机 | Harvard 定制便携刚性单侧踝外骨骼(ref[18])，远端执行机构双向跖屈/背屈助力 | "custom…portable rigid unilateral ankle exoskeleton" |
| 主控 | **Raspberry Pi 5**(Cambridge UK) 跑实时力矩估计 + 腰包内定制MCU + 锂电 + 急停 | "Raspberry Pi 5…custom MCU, lithium-ion battery, and emergency stop…waist pack" |
| IMU | **3× Movella MTi-3**(Nevada USA)，100 Hz，足/小腿/大腿 | "MTi-3, Movella Inc…100 Hz…three" |
| 力传感 | 外骨骼内 **2× 力传感器(load cell)** + 踝编码器 | "two load cells…ankle-joint encoder" |
| 控制频率 | 力矩估计经CAN @ **100 Hz**；底层力矩闭环 **1000 Hz** | "CAN Bus at 100 Hz…closed loop (1000 Hz)" |
| 真值 | **Qualisys** 动捕(200Hz) + **Bertec** 分带跑步机GRF(2000Hz) | "Qualisys…200 Hz…Bertec…2000 Hz" |
| 规模 | 训练 4 中风+6 健康预训练，实时演示 1 中风 | — |

> 价值：示范了**单侧踝 + load cell 力矩观测 + 急停腰包**的最小可行临床穿戴方案——和你"单关节起步、需真实力矩观测、安全第一"的思路高度吻合，可作为单关节安全设计的范本。

## 1.5 Camargo 2021 — 数据集（你的训练数据真值来源，非穿戴外骨骼）

| 设备 | 规格 | 用途 |
| --- | --- | --- |
| 光学动捕 | **Vicon**(改良 Plugin-Gait 标记配置) | 运动学 |
| 测力跑步机 | **Bertec** 仪器化跑步机 + 嵌入力板(地面/坡/楼梯) | GRF/动力学 |
| EMG | **Delsys**，2000 Hz，11 肌群 | 肌电 |
| 关节角度计 | 4 DOF：踝矢状/踝额状/膝矢状/髋矢状 | 关节角 |
| IMU | 躯干/大腿/小腿/足，accel+gyro xyz | 穿戴运动 |
| 处理 | OpenSim + MoCapTools(MATLAB) 逆运动学/动力学 | 力矩真值 |
| 规模 | 22 名健康成人；跑步机/平地/坡/楼梯 + 过渡 | — |

> 含义：动捕/力板/EMG 都是为**算力矩真值**用的实验室设备。你若只复现传感链做检测验证，**不需要**这些；只有要在真机上算生物力矩绝对精度对照时才需要——本项目默认不走真值路线（见第二部分理由）。

---
<!-- PART2 -->

# 第二部分：本项目规划设备清单

定位（2026-06-14 决策）：**单关节起步(膝或踝) · 尽量买现成 · 目标是验证可靠性框架而非复刻整机。**

设计依据（呼应第一部分）：
- 执行器选型对齐 **1.1 的 AK80-9**（复现性最佳，和源论文同款）。
- 单关节安全方案借鉴 **1.4 Weigend** 的 load cell 力矩观测 + 急停。
- 计算栈对齐 **1.1/1.3** 的 RPi + Jetson Orin Nano（Jetson Nano 已停产）。
- **关键省钱点**：packet_loss/sensor_delay/imu_bias 是数字信号腐蚀，用现有 `reliability/faults.py` 在真实数据流上**软件注入**，无需额外硬件；只有 encoder/insole 断连做物理对照。故硬件重心是"一条真实传感链 + 带力矩传感的单关节闭环"，不是机械复杂度。
- **不买** Vicon/Bertec/Delsys：阶段 1 只做相对比较(gate on/off、clean/fault)，不声称生物力矩绝对精度——与仿真论文的"相对安全收益"叙事一致。

## 2.1 采购清单 — 方案 A：买现成平台（推荐）

| 模块 | 推荐选型 | 数量 | 估算单价 | 对应参考 / 备注 |
| --- | --- | --- | --- | --- |
| **执行器(带力矩反馈)** | T-Motor **AK80-9** | 1 | ¥1,800–2,500 | 同 1.1，复现性最佳，力矩由电流估算 |
| └ 备选(最省事) | **HEBI X-Series** 模块化(内置力矩传感+API) | 1 | $1,500–3,000 | 自带力矩闭环+高层API，开发最快 |
| └ 备选(踝整机) | **Dephy ExoBoot** 单只 | 1 | $5,000–10,000+ | 同 1.3 Tourk 平台；省机械但贵、偏踝 |
| **CAN 接口** | CANable / PEAK PCAN-USB 或 Jetson 自带 | 1 | ¥150–800 | AK80-9 走 CAN |
| **实时主控** | Raspberry Pi 5(或 4B 对齐源论文55Hz) | 1 | ¥600 | 同 1.1/1.4 |
| **推理单元** | NVIDIA **Jetson Orin Nano** | 1 | ¥1,500 | 替代停产的 Jetson Nano；同 1.3 |
| **大腿/小腿 IMU** | 六轴 IMU(LSM6DSOX 或 Movella MTi-3) ×2 | 2 | ¥100–600/个 | 对齐 1.1 thigh+shank；MTi-3 同 1.4 |
| **足底压感** | Moticon OpenGo(同 1.1) | 1对 | $3,000–6,000 | 贵；台架阶段可软件置零省略 |
| └ 省钱替代 | FSR 压力阵列 + 自制采集 | 1 | ¥300 | 精度低，仅占位 insole 通道 |
| **串联力矩传感器** | 旋转扭矩传感器(±20 N·m) | 1 | ¥2,000–5,000 | **阶段1关键件**：独立测真实输出力矩(借鉴1.4 load cell思路) |
| **负载端** | 惯量盘/扭簧 + 铝型材台架夹具 | 1套 | ¥500–1,500 | 模拟肢体惯量+刚度 |
| **电源** | 24V电源(台架)或 DeWalt 20V×2(移动) | 1 | ¥300–600 | 移动方案同 1.1 |
| **急停+安全** | 硬件急停 + 力矩软限幅 + 看门狗 | 1套 | ¥200 | **必备**(借鉴 1.4 急停腰包) |
| **线缆/连接器/杂项** | CAN线、电源线、3D打印支架 | — | ¥500 | |

**方案 A 台架最小配置合计（不含 Moticon/ExoBoot）：约 ¥8,000–15,000**
含 Moticon：+ $3k–6k；改 ExoBoot 整机路线：$8k–15k 级。

## 2.2 采购清单 — 方案 B：自研机械 + 采购核心件

在方案 A 基础上把"现成整机/HEBI"换成自研：
- 结构：3D 打印尼龙 + 铝型材/碳板（对齐 1.1，水刀可外协），单关节支架 ¥800–2,000。
- 执行器仍用 AK80-9（自研也绕不开优质准直驱）。
- 省整机平台费，但增机械设计/装配/标定 2–4 周工时。
- 适合：有机械/嵌入式人手、长期迭代、预算紧。

## 2.3 推荐配置（结论）

走**方案 A 台架最小配置**：AK80-9 + 串联扭矩传感器 + 惯量盘 + RPi/Jetson Orin Nano + 2× IMU，鞋垫先软件置零占位。

- 成本 **¥1 万级**，2–3 周跑通闭环。
- 直接产出论文的"真机闭环验证"图（gate↓→实测τ↓/jerk↓，clean gate≈1）。
- **阶段 1 全程无人体 → 无伦理审批门槛**。
- HEBI 是"不差钱要最快"的升级项；ExoBoot 仅当转向踝关节真人助力时再考虑。

## 2.4 两阶段实验（与论文图表对应）

| 阶段 | 内容 | 人体 | 伦理 |
| --- | --- | --- | --- |
| **阶段 1（台架）** | 作动器接惯量盘/弹簧+扭矩传感器，软件注入故障，验证 gate↔τ 因果 | 无 | 不需要 |
| **阶段 2（人体单腿，可选）** | 1–3 名受试者穿单关节，测交互力矩+主观反馈（按 Wu&Stirling 故障注入范式） | 有 | 需 IRB |

完整里程碑 M0–M4、风险对策、软件 bridge 设计见 [HARDWARE_EXPERIMENT_PLAN.md](HARDWARE_EXPERIMENT_PLAN.md)。

---

## 文献链接

- Molinaro 2024: [Nature 10.1038/s41586-024-08157-7](https://www.nature.com/articles/s41586-024-08157-7)
- Scherpereel 2025: [Science Robotics eads8652](https://www.science.org/doi/10.1126/scirobotics.ads8652)
- Tourk 2025: [arXiv:2508.21221](https://arxiv.org/abs/2508.21221)
- Weigend 2025: [arXiv:2508.00691](https://arxiv.org/abs/2508.00691)
- Camargo 2021: [EPIC GT](https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/) · [PubMed 33677231](https://pubmed.ncbi.nlm.nih.gov/33677231/)

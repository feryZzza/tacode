# 参考文献库 (Reliability-Aware Exoskeleton Control 论文)

按论文章节组织。标 ★ = 必引核心；标 ◯ = 推荐补充；标 ⚙ = 方法/技术基础。
当前已有的两篇源论文 = [Molinaro2024], [Scherpereel2025]。目标 40-55 篇，下面约 45 篇。

---

## 1. 学习型外骨骼控制（直接背景 / 我们 build-on 的对象）

- ★ **[Molinaro2024]** D. Molinaro, K. Scherpereel, E. Schonhaut, G. Evangelopoulos, M. Shepherd, A. Young. "Task-agnostic exoskeleton control via biological joint moment estimation." *Nature* 635(8038):337-344, 2024. — 源论文1，我们复现/扩展的 TCN 生物力矩控制器。
- ★ **[Scherpereel2025]** K. Scherpereel, D. Molinaro, M. Gombolay, M. Shepherd, C. Carrasquillo, O. Inan, A. Young. "Deep domain adaptation eliminates costly data required for task-agnostic wearable robotic control." *Science Robotics* 10(108):eads8652, 2025. — 源论文2，域适应/泛化。
- ◯ **[Molinaro2022]** D. Molinaro, I. Kang, J. Camargo, M. Gombolay, A. Young. "Subject-Independent, Biological Hip Moment Estimation During Multimodal Overground Ambulation Using Deep Learning." *IEEE Trans. Medical Robotics and Bionics* 4(1):219-229, 2022. — 力矩估计前身工作。
- ◯ **[Shetty2025]** P. Shetty, J. Menezes, S. Song, A. Young, M. Shepherd. "Ankle Exoskeleton Control via Data-Driven Gait Estimation for Walking, Running, and Inclines." *IEEE RA-L* 10(6):5855-5862, 2025. — 数据驱动踝控制。
- ◯ **[Shepherd2022]** M. Shepherd, D. Molinaro, G. Sawicki, A. Young. "Deep Learning Enables Exoboot Control to Augment Variable-Speed Walking." *IEEE RA-L* 7(2):3571-3577, 2022.
- ⚙ **[Kang2020]** I. Kang, P. Kunapuli, A. Young. "Real-Time Neural Network-Based Gait Phase Estimation Using a Robotic Hip Exoskeleton." *IEEE Trans. Medical Robotics and Bionics* 2(1):28-37, 2020. — TCN 步态相位估计。
- ◯ **[Medrano2023]** R. Medrano, G. Thomas, C. Keais, E. Rouse, R. Gregg. "Real-time gait phase and task estimation for controlling a powered ankle exoskeleton on extremely uneven terrain." *IEEE Trans. Robotics* 39(3):2170-2182, 2023.
- ◯ **[Divekar2024]** N. Divekar et al. "A versatile knee exoskeleton mitigates quadriceps fatigue in lifting, lowering, and carrying tasks." *Science Robotics* 9(94):eadr8282, 2024.

## 2. 人在环优化 / 外骨骼效益（动机：为什么外骨骼重要）

- ◯ **[Zhang2017]** J. Zhang et al. "Human-in-the-loop optimization of exoskeleton assistance during walking." *Science* 356(6344):1280-1284, 2017.
- ◯ **[Slade2022]** P. Slade, M. Kochenderfer, S. Delp, S. Collins. "Personalizing exoskeleton assistance while walking in the real world." *Nature* 610(7931):277-282, 2022.
- ◯ **[Ding2018]** Y. Ding, M. Kim, S. Kuindersma, C. Walsh. "Human-in-the-loop optimization of hip assistance with a soft exosuit during walking." *Science Robotics* 3(15):eaar5438, 2018.
- ◯ **[Siviy2022]** C. Siviy et al. "Opportunities and challenges in the development of exoskeletons for locomotor assistance." *Nature Biomedical Engineering* 7(4):456-472, 2022. — 综述，强动机引用。

## 3. ★ 最近邻 / 必须对照差异化（不确定性门控外骨骼）

- ★★ **[Tourk2025]** F. Tourk, B. Galoaa, S. Shajan, A. Young, M. Everett, M. Shepherd. "Uncertainty-Aware Ankle Exoskeleton Control." arXiv:2508.21221, 2025. — **最强对照**。集成方差不确定性 + 门控脱离，硬件验证。我们的差异化对象：他们只做 OOD 动作不做传感器故障，用分类指标无力矩伤害度量，硬阈值门控。必须在 Related Work 显著对照。
- ◯ **[Hsu2024]** T.-W. Hsu, R. Gregg, G. Thomas. "Robustification of bayesian-inference-based gait estimation for lower-limb wearable robots." *IEEE RA-L* 9(3):2104-2111, 2024. (亦 TNSRE/PMC10831317) — **第二近邻**：可穿戴机器人从"瞬时混乱"中恢复的鲁棒化。和我们的可靠性主题最接近，必引差异化（他们贝叶斯滤波鲁棒化 vs 我们故障检测+门控）。

## 4. ★ 传感器故障 / 容错（我们的核心 wedge — 此前论文最缺，现已补齐）

- ★ **[Bayesian-robust]** 见 [Hsu2024] 上条。
- ★ **[EMG-SFTM]** "A real-time, practical sensor fault-tolerant module for robust EMG pattern recognition." *J. NeuroEng. Rehabil.* 12:11, 2015. — 可穿戴 EMG 传感器故障检测+自恢复。直接先例：故障容错在可穿戴控制里是真问题。
- ★ **[Prosthesis-VAE-SOM]** Z. Zhu et al. "Using a VAE-SOM architecture for anomaly detection of flexible sensors in limb prosthesis." *J. Industrial Information Integration* 35:100490, 2023. — 假肢柔性传感器异常检测（2508.21221 的 ref [35]）。直接相关：可穿戴假肢的传感器异常监测。
- ◯ **[Graceful-degrade]** "Versatile graceful degradation framework for bio-inspired proprioception with redundant soft sensors." *Frontiers Robotics & AI* 11:1504651, 2024. — 冗余传感器优雅降级。可对照（我们无冗余、靠检测+门控）。
- ◯ **[Quadruped-corr]** "Improvement of fault tolerance of quadruped robots by detecting correlation anomalies in sensor signals." *Artificial Life and Robotics*, 2024. — 跨通道相关性异常检测（和我们的 staleness/跨通道一致性思路呼应）。
- ◯ **[Exo-FMEA]** "Model-based Fault Injection Experiments for the Safety Analysis of Exoskeleton System." arXiv:2101.01283. — 经典 FMEA 式外骨骼故障注入安全分析（非学习型）。对照：我们做学习型控制器的故障注入评估。
- ◯ **[Myocontrol-fail]** "Automatic Detection of Myocontrol Failures Based upon Situational Context Information." arXiv:1906.11564 / *Frontiers Neurorobotics* 13:68, 2019. — 假肢肌电控制失效检测，"学习何时预测变得不可靠"——和我们门控哲学一致。
- ◯ **[Soft-muscle-fault]** "Fault Detection and Response for Safe Control of Artificial Muscles in Soft Robots." NSF par 10654847. — 形式化安全条件下的执行器故障检测。

## 5. ⚙ 不确定性量化 / 异常检测（方法基础）

- ⚙★ **[Lakshmin2017]** B. Lakshminarayanan, A. Pritzel, C. Blundell. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS 2017 / arXiv:1612.01474. — Deep ensembles 奠基文献，我们 epistemic 通道的依据。必引。
- ⚙ **[Gal2016]** Y. Gal, Z. Ghahramani. "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." ICML 2016. — MC-dropout 奠基（我们对照过的方法）。
- ⚙ **[Loquercio2020]** A. Loquercio, M. Segù, D. Scaramuzza. "A General Framework for Uncertainty Estimation in Deep Learning." *IEEE RA-L* 5(2):3153-3160, 2020. — 机器人 UQ 通用框架（2508 ref [30]）。
- ◯ **[Ens-aug-calib]** "Combining Ensembles and Data Augmentation can Harm your Calibration." arXiv:2010.09875. — **对我们尤其相关**：集成+数据增强可能损害校准，正好讨论我们的 aug 与校准/精度权衡。
- ⚙ **[TCN-AE]** M. Thill, W. Konen, H. Wang, T. Bäck. "Temporal convolutional autoencoder for unsupervised anomaly detection in time series." *Applied Soft Computing* 112:107756, 2021. — TCN 自编码器时序异常检测，我们 recon 头的方法依据。
- ◯ **[TS-anomaly-review]** K. Choi, J. Yi, C. Park, S. Yoon. "Deep Learning for Anomaly Detection in Time-Series Data: Review, Analysis, and Guidelines." *IEEE Access* 9:120043-120065, 2021. — 时序异常检测综述。
- ◯ **[Deep-ens-MTS]** A. Iqbal et al. "Anomaly detection in multivariate time series data using deep ensemble models." *PLOS ONE* 19(6):e0303890, 2024.
- ◯ **[Robot-safety-UQ]** "Well-calibrated uncertainty quantification in neural networks for barriers-based robot safety." arXiv:2407.00616, 2024. — 安全关键机器人的校准 UQ。
- ◯ **[Fail-safe-DL]** "Fail-Safe Execution of Deep Learning based Systems through Uncertainty Monitoring." arXiv:2102.00902. — 用不确定性监督触发安全模式（和我们门控同构）。

## 6. ★ 安全度量外部锚点（force-replay 伤害框架的现实效度 — 关键弹药）

- ★ **[ASTM-F3578]** ASTM F3578-22. "Standard Test Method for Evaluating Exoskeleton Fall Risk due to Stumbling." 2022. — **正式标准**。领域已承认"扰动下外骨骼行为"是标准化安全危害。我们的力矩回放伤害度量 = 对同一关切的离线操作化。
- ◯ **[ASTM-F3583]** ASTM F3583-22. "Standard Test Method for Beam Traversal Safety." 2022. — 配套外骨骼平衡安全标准。
- ★ **[Wu2024]** M. Wu, L. Stirling. "Emergent gait strategies defined by cluster analysis when using imperfect exoskeleton algorithms." *IEEE RA-L*, 2024. DOI 10.1109/LRA.2024.3366010. — **人体研究**：注入控制器错误（5 个固定错误率至 10%），用户代偿耗额外能量。我们故障注入范式的人体先例 + 错误辅助有真实生物力学代价的证据。
- ◯ **[Wu2023]** L. Stirling, M. Wu, X. Peng. "Impact of imperfect exoskeleton algorithms on step characteristics, task performance, and perception." *IEEE/RSJ IROS* 2023:4088-4093. — 移除 2-5% 步态辅助即降低信任（2508 ref [22]）。锚定"罕见故障也重要"。
- ◯ **[Stirling2024-trust]** L. Stirling, M. Wu, X. Peng. "Measuring Trust for Exoskeleton Systems." arXiv:2407.07200, 2024. — 外骨骼信任维度。
- ◯ **[Perturb-stability]** "The effects of unexpected mechanical perturbations during treadmill walking on spatiotemporal gait parameters and dynamic stability measures." *PLOS ONE* 13(4):e0195902, 2018. — 意外扰动降低稳定裕度。锚定"反向力矩=破坏稳定的扰动"。
- ◯ **[Trip-work]** "Lower extremity joint power and work during recovery following trip-induced perturbations." 2023 (pubmed 37703781). — 绊倒恢复需 +27%正功/+28%负功。量化扰动代价。
- ◯ **[Perturb-system]** "A novel system for introducing precisely-controlled, unanticipated gait perturbations for the study of stumble recovery." *J. NeuroEng. Rehabil.* 16:1, 2019. — 精确受控步态扰动实验方法。
- ◯ **[Exo-balance-react]** "Exoskeletons need to react faster than physiological responses to improve standing balance." *Science Robotics*, 2020. — 外骨骼时序对平衡的影响。

## 7. ◯ 数据集 / 评估方法

- ★ **[Camargo2021-data]** J. Camargo, A. Ramanathan, W. Flanagan, A. Young. "A comprehensive, open-source dataset of lower limb biomechanics in multiple conditions of stairs, ramps, and level-ground ambulation and transitions." *Journal of Biomechanics* 119:110320, 2021. (PMID 33677231) — **我们用的数据集**，必引。
- ◯ **[Camargo2021-ml]** J. Camargo et al. "A Machine Learning Strategy for Locomotion Classification and Parameter Estimation using Fusion of Wearable Sensors." *IEEE TBME*, 2021. — 同组方法论文。
- ◯ **[Youden1950]** W. Youden. "Index for rating diagnostic tests." *Cancer* 3(1):32-35, 1950. — 若用 J-statistic / AUROC 评估检测器可引（对照 2508 的指标选择）。

## 8. ◯ 神经力学模型外骨骼（区分我们的数据驱动路线 vs 模型驱动）

- ◯ **[Durandau2022]** G. Durandau, W. Rampeltshammer, H. van der Kooij, M. Sartori. "Neuromechanical model-based adaptive control of bilateral ankle exoskeletons." *IEEE Trans. Robotics* 38(3):1380-1394, 2022.
- ◯ **[Firouzi2025]** V. Firouzi et al. "Biomechanical models in the lower-limb exoskeletons development: A review." *J. NeuroEng. Rehabil.* 22(1):12, 2025. — 综述。

## 9. ★ 仿真验证外骨骼控制（支撑 closed_loop.py 的方法论合法性 — 关键辩护）

我们的 dynamics-in-the-loop 闭环代理"无硬件评估"会被审稿人质疑。这一节确立"在仿真里评估/学习外骨骼控制、无需人体实验"是已被顶刊接受的合法路线。

- ★ **[Luo2024]** S. Luo et al. "Experiment-free exoskeleton assistance via learning in simulation." *Nature* 630:353-359, 2024 (NCSU; 亦 PMC11344585). — **最强辩护锚点**：完全在仿真里学外骨骼控制策略、零人体实验，后续才硬件验证。Nature 级别确立此路线合法性。我们闭环代理用于"故障下安全评估"是同一范式的应用。
- ◯ **[Dembia2020]** C. Dembia, A. Bianco, A. Falisse, J. Hicks, S. Delp. "Simulating ideal assistive devices to reduce the metabolic cost of walking with heavy loads." / "Toward predicting exoskeleton-assisted gait" (OpenSim Moco). — 仿真预测外骨骼辅助下的代谢/关节负荷，离线评估辅助效果的先例。
- ◯ **[Exo-run-sim]** "Simulation-based biomechanical assessment of unpowered exoskeletons for running." *Scientific Reports* 11:9637, 2021. — OpenSim 92-肌肉 29-DOF 仿真评估外骨骼，无硬件。
- ◯ **[NMS-hipexo2026]** "Learning Hip Exoskeleton Control Policy via Predictive Neuromusculoskeletal Simulation." arXiv:2603.04166, 2026. — 神经肌肉骨骼仿真学髋外骨骼策略（仿真减肌肉激活 3.4%/关节功 7.0%）。活跃前沿佐证。
- ◯ **[Exo-plore2026]** "Exploring Exoskeleton Control space through Human-Aligned Simulation (Exo-plore)." arXiv:2601.22550, 2026. — DRL 神经力学仿真优化髋外骨骼、无人体实验。引 Molinaro。对照：同为仿真评估，目标不同（它优化代谢，我们评估故障安全）。
- ◯ **[RL-bio-moment2026]** "Exoskeleton Control through Learning to Reduce Biological Joint Moments in Simulations." arXiv:2603.07629, 2026. — 物理 RL 在仿真里学降低生物关节力矩的辅助。和我们"生物力矩"目标直接呼应。
- ◯ **[Embodied-sim2026]** "Embodied Human Simulation for Quantitative Design and Analysis of Interactive Robotics." arXiv:2603.09218, 2026. — 全身肌肉骨骼模型作人体动力学代理，优化人-外骨骼交互。佐证"人体仿真代理"是设计/分析工具。

---

## 待补 / 验证 TODO
- [ ] 所有 arXiv 条目核对是否已正式发表（用正式出处替换 preprint）。
- [ ] ASTM 两条获取正式条款号 + 年份（全文付费墙）。
- [ ] [Wu2024] 与 [Wu2023] 确认是否同一工作的会议→期刊版本，避免重复引。
- [ ] 第5节 UQ 基础按目标期刊调节数量（TNSRE 偏临床可酌减纯 ML 引用）。
- [x] ~~补 closed-loop / 动力学仿真验证方法的引用~~ → 已补为第9节（[Luo2024] 为核心辩护）。
- [ ] [Dembia2020] 核对确切出处（OpenSim Moco 论文 vs Dembia PLoS Comput Biol 2020），可能是两篇。
- [ ] 2026 年初的 arXiv 仿真预印本（NMS-hipexo/RL-bio-moment/Embodied）择优保留 1-2 篇即可，避免堆砌未发表预印本。

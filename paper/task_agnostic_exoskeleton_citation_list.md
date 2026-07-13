# Task-agnostic exoskeleton control via biological joint moment estimation 被引文章详细列表

检索日期：2026-06-07

说明：该列表基于 Nature cited-by、Semantic Scholar、arXiv/期刊页面和部分数据库条目整理；不是 Google Scholar 全量结果。

## 被引用原文

- **题名**：Task-agnostic exoskeleton control via biological joint moment estimation
- **作者**：Dean D. Molinaro, Keaton L. Scherpereel, Ethan B. Schonhaut, Georgios Evangelopoulos, Max K. Shepherd, Aaron J. Young
- **期刊**：Nature 635, 337–344 (2024)
- **DOI**：10.1038/s41586-024-08157-7
- **发表日期**：Published: 13 November 2024；Version of record: 13 November 2024
- **核心思想**：用深度神经网络实时估计髋/膝 biological joint moments，并以此作为任务无关外骨骼控制信号，使外骨骼在周期、过渡和非结构化任务中提供辅助。

## 详细 list

### 1. Deep-Learning Control of Lower-Limb Exoskeletons via Simplified Therapist Input

- **作者**：Lorenzo Vianello, Clément Lhoste, Emek Barış Küçüktabak, Matthew R. Short, Levi J. Hargrove, José L. Pons Rovira
- **年份/状态**：2025，ICORR；同时有 arXiv:2412.07959 预印本
- **标题或摘要总结**：提出一种面向康复外骨骼的三步数据驱动控制流程：先用近期传感器数据概率推断步态/运动状态，再允许治疗师在界面中修改这些状态特征，最后根据修改后的特征和预测不确定性生成目标关节姿态与弹簧-阻尼模型刚度。
- **与 Nature 原文的关系**：与 Nature 原文同属“从预定义任务控制走向数据驱动/任务泛化控制”的路线，但它更偏临床康复交互，把治疗师输入纳入闭环。
- **汇报价值/可被追问点**：可作为“任务无关控制如何进入康复场景”的后续工作；答辩时可对比 biological moment 作为控制中间量 vs therapist-adjusted locomotion features。
- **被引关系证据**：Semantic Scholar citation list；arXiv 页面
- **检索链接**：https://arxiv.org/abs/2412.07959

### 2. Uncertainty-Aware Ankle Exoskeleton Control

- **作者**：Fatima Mumtaza Tourk, Bishoy Galoaa, Sanat Shajan, Aaron J. Young, Michael Everett, Max K. Shepherd
- **年份/状态**：2025，arXiv:2508.21221 预印本；作者单位新闻称后续发表于 Robotics，但正式卷期/DOI 需继续核验
- **标题或摘要总结**：提出不确定性感知踝关节外骨骼控制框架，用 uncertainty estimator 判断当前动作是否属于训练分布。在线实验中，系统可在 in-distribution 与 out-of-distribution 任务之间自动开启/关闭辅助，报告 F1=89.2。
- **与 Nature 原文的关系**：直接补 Nature 原文的安全短板：Nature 证明 biological moment controller 能跨任务辅助，但未知动作、异常传感器或高动态动作下需要 OOD/uncertainty 机制。
- **汇报价值/可被追问点**：非常适合汇报中作为“原文不足与 follow-up 方向”：从 task-agnostic control 进一步走向 safety-aware / uncertainty-aware control。
- **被引关系证据**：Semantic Scholar citation list；arXiv 页面
- **检索链接**：https://arxiv.org/abs/2508.21221

### 3. Exoskeleton Control through Learning to Reduce Biological Joint Moments in Simulations

- **作者**：Zihang You, Xianlian Zhou
- **年份/状态**：2026，arXiv:2603.07629 预印本
- **标题或摘要总结**：用强化学习在仿真中学习外骨骼辅助策略，目标是降低 biological joint moments；同时构建验证流程，用开源步态数据集检查仿真训练出的控制网络与 biological joint moments 的一致性。
- **与 Nature 原文的关系**：Nature 原文是“估计 biological moment → 在线辅助”；这篇转向“在仿真中学习降低 biological moment 的控制策略”，重点是减少真实人体实验和探索 sim-to-real 可行性。
- **汇报价值/可被追问点**：适合放入“仿真学习 / experiment-free / sim-to-real 外骨骼控制”分支。
- **被引关系证据**：Semantic Scholar citation list；arXiv 页面
- **检索链接**：https://arxiv.org/abs/2603.07629

### 4. Optimized Mappings from Biological Hip Moment Estimates to Exoskeleton Torque can Personalize Assistance Across Users and Generalize Across Tasks

- **作者**：Justine C. Powell, Ethan B. Schonhaut, Dean D. Molinaro, Aaron J. Young
- **年份/状态**：2025，bioRxiv / 预印本线索；需以正式页面二次核验
- **标题或摘要总结**：研究如何把 biological hip moment estimates 映射成实际外骨骼扭矩。核心问题不是“能不能估计 moment”，而是“moment 到 assistance torque 的比例、偏置、时序和个体化映射如何优化”。
- **与 Nature 原文的关系**：这是 Nature 原文的直接延伸：原文证明 biological moment 可作为 task-agnostic 控制信号；这篇进一步讨论映射函数如何个性化并跨任务泛化。
- **汇报价值/可被追问点**：很适合作为答辩中的关键 follow-up：原文控制律是否只是简单比例映射？个体差异如何处理？映射优化能否比固定规则更好？
- **被引关系证据**：Semantic Scholar citation list；bioRxiv 线索
- **检索链接**：https://www.semanticscholar.org/paper/Task-agnostic-exoskeleton-control-via-biological-Molinaro-Scherpereel/f337d1553ae42795c7bf8711049b8bceea3d18ab

### 5. Motion Adaptation Across Users and Tasks for Exoskeletons via Meta-Learning

- **作者**：Muyuan Ma, Long Cheng, Lijun Han, Xiuze Xia, Houcheng Li
- **年份/状态**：2025，arXiv:2509.13736 预印本
- **标题或摘要总结**：提出 meta-imitation learning 框架，利用公开 RGB 视频和动作捕捉数据提取全身关键点，在仿真中重定向并训练任务特定网络，使外骨骼可对新用户和未见任务快速适应。
- **与 Nature 原文的关系**：与 Nature 原文共同关注跨任务/跨用户泛化，但它使用元学习和 imitation learning，而不是直接估计 biological joint moment。
- **汇报价值/可被追问点**：适合放在“泛化控制算法路线对比”：moment-estimation paradigm vs meta-learning adaptation。
- **被引关系证据**：arXiv 正文参考文献中明确列出 Molinaro et al., Nature 635, 337–344, 2024
- **检索链接**：https://arxiv.org/abs/2509.13736

### 6. A Simulation-based Lower-Limb Exoskeleton Control Method

- **作者**：Yizhe Zhou, Chunjie Chen 等
- **年份/状态**：2025，IEEE ROBIO；DOI 线索：10.1109/ROBIO66223.2025.11377519
- **标题或摘要总结**：提出基于仿真的下肢外骨骼控制学习方法，用 dynamic-aware simulation-based learning 减少真实数据采集时间与成本。
- **与 Nature 原文的关系**：与 Nature 原文的问题背景一致：真实人体/外骨骼数据采集昂贵，跨任务控制需要更高效的训练与验证方式。
- **汇报价值/可被追问点**：可作为“数据采集成本”短板的后续解决方向之一。
- **被引关系证据**：Semantic Scholar citation list；ResearchGate/会议条目线索
- **检索链接**：https://www.semanticscholar.org/paper/A-Simulation-based-Lower-Limb-Exoskeleton-Control-Zhou-Chen/33fb18f62fcd5051688f80dd54de574153b16197

### 7. Ankle Exoskeleton Control via Data-Driven Gait Estimation for Walking, Running, and Inclines

- **作者**：P. R. Shetty, J. A. Menezes, Seungmoon Song, Aaron J. Young, Max K. Shepherd
- **年份/状态**：2025，IEEE Robotics and Automation Letters
- **标题或摘要总结**：面向踝关节外骨骼，使用多头网络估计步行、跑步、坡道等任务下的步态变量，从而使踝外骨骼在多个 locomotion tasks 之间自适应。
- **与 Nature 原文的关系**：Nature 原文估计 hip/knee biological moments；这篇更偏用 gait estimation 作为踝外骨骼控制中间量，是同一大方向下的不同中间表征选择。
- **汇报价值/可被追问点**：可用于比较“估计 biological moment”与“估计 gait state / gait phase / task variables”的优劣。
- **被引关系证据**：Semantic Scholar citation list
- **检索链接**：https://www.semanticscholar.org/paper/Task-agnostic-exoskeleton-control-via-biological-Molinaro-Scherpereel/f337d1553ae42795c7bf8711049b8bceea3d18ab

### 8. Joint moment estimation for hip exoskeleton control: A generalized moment feature generation method

- **作者**：Yuanwen Zhang, Jingfeng Xiong, Haolan Xian, Chuheng Chen, Xinxing Chen, Chenglong Fu, Yuquan Leng
- **年份/状态**：2025，Biomimetic Intelligence and Robotics；相关 arXiv 题名为 Fast Hip Joint Moment Estimation with A General Moment Feature Generation Method
- **标题或摘要总结**：提出 generalized moment feature (GMF) 表征和 GRU 估计器，用髋关节运动学快速估计 hip joint moment，目标是降低计算时间并提高个体泛化能力。
- **与 Nature 原文的关系**：与 Nature 原文共享“joint moment estimation 可服务于外骨骼控制”的基本假设，但它聚焦髋关节、特征生成和推理速度。
- **汇报价值/可被追问点**：适合用来回答“moment estimation 本身还有哪些技术改进空间”：传感简化、个体泛化、实时性。
- **被引关系证据**：Semantic Scholar citation list；arXiv 页面
- **检索链接**：https://arxiv.org/abs/2410.00462

### 9. Optimizing Locomotor Task Sets for Training a Biological Joint Moment Estimator

- **作者**：Jimin An, Changseob Song, Eni Halilaj, Inseung Kang
- **年份/状态**：2025，ICORR，pp.1518–1523；DOI 线索：10.1109/ICORR66766.2025.11063074
- **标题或摘要总结**：研究训练 biological joint moment estimator 时应采集哪些 locomotor tasks。通过任务集优化寻找最小但代表性强的训练任务集合，从而降低数据采集负担并尽量保持模型性能。
- **与 Nature 原文的关系**：Nature 原文训练了覆盖大量任务的数据驱动估计器；这篇直接追问“是否必须采这么多任务？哪些任务最有训练价值？”
- **汇报价值/可被追问点**：非常适合用于讲 Nature 原文的数据成本问题，是较强相关的 follow-up。
- **被引关系证据**：Semantic Scholar citation list；DBLP/ResearchGate/ICORR 条目
- **检索链接**：https://dblp.org/rec/conf/icorr/AnSHK25

### 10. Towards Data-Driven Adaptive Exoskeleton Assistance for Post-stroke Gait

- **作者**：Fabian C. Weigend, Dabin K. Choe, Santiago Canete, Conor J. Walsh
- **年份/状态**：2025，IROS；arXiv:2508.00691；DOI 线索：10.1109/IROS60139.2025.11246935
- **标题或摘要总结**：面向中风后步态，训练多任务 TCN 从 IMU 数据估计踝关节扭矩，并在一名 post-stroke participant 上展示实时感知、估计和驱动的可行性。
- **与 Nature 原文的关系**：Nature 原文主要在健康成年人任务泛化上验证；这篇把 data-driven adaptive exoskeleton assistance 推向神经运动障碍人群。
- **汇报价值/可被追问点**：适合用于讨论临床迁移难点：患者异质性、异常步态、数据不足、安全性。
- **被引关系证据**：Semantic Scholar citation list；arXiv 页面；IROS 条目线索
- **检索链接**：https://arxiv.org/abs/2508.00691

### 11. Customizable Task-Agnostic Exoskeleton Control for Targeted Neuromuscular Assistance: Case Series

- **作者**：Nikhil V. Divekar, Alicia Baxter, Robert D. Gregg
- **年份/状态**：2025，IEEE Open Journal of Engineering in Medicine and Biology，6:564–569；DOI 线索见 PubMed
- **标题或摘要总结**：提出可定制的 task-agnostic bilateral knee exoskeleton controller，用于面向特定 neuromuscular deficits 的辅助。该工作从“通用任务无关控制”走向“针对个体神经肌肉缺陷的定制控制”。
- **与 Nature 原文的关系**：与 Nature 原文共享 task-agnostic control 思想，但应用目标更偏康复/临床病例。
- **汇报价值/可被追问点**：适合放入“从健康人群泛化到病理人群个性化”的引用分支。
- **被引关系证据**：Nature cited-by/PMC/PubMed 线索；论文参考文献包含 Molinaro et al. Nature 2024
- **检索链接**：https://pubmed.ncbi.nlm.nih.gov/41221439/

### 12. Deep domain adaptation eliminates costly data required for task-agnostic wearable robotic control

- **作者**：Keaton L. Scherpereel, Matthew C. Gombolay, Max K. Shepherd, Carlos A. Carrasquillo, Omer T. Inan, Aaron J. Young
- **年份/状态**：2025，Science Robotics，10(108):eads8652
- **标题或摘要总结**：提出 deep domain adaptation framework，用开源数据、仿真传感器和少量/无标签外骨骼数据替代昂贵的设备特定标注数据，目标是降低 task-agnostic wearable robotic control 的训练成本。
- **与 Nature 原文的关系**：这是 Nature 原文团队/方向的关键后续：原文证明 moment-based task-agnostic control 有效；这篇进一步解决训练数据昂贵的问题。
- **汇报价值/可被追问点**：汇报中应重点讲。它正面回应 Nature 原文最容易被问到的弱点：数据量、标注成本、跨设备泛化。
- **被引关系证据**：Science Robotics 页面；相关实验室新闻
- **检索链接**：https://www.science.org/doi/10.1126/scirobotics.ads8652

### 13. Task-Agnostic Exoskeleton Control Supports Elderly Joint Energetics during Hip-Intensive Tasks

- **作者**：Jiefu Zhang, Nikhil V. Divekar, Chandramouli Krishnan, Robert D. Gregg
- **年份/状态**：2026，arXiv:2603.22580 预印本
- **标题或摘要总结**：在 8 名老年人上验证 task-agnostic hip exoskeleton controller，任务包括平地行走、坡道上行、爬楼梯、坐站转换等。摘要报告平均 cosine similarity 0.89，髋关节 sagittal biological positive work 降低 24.7%。
- **与 Nature 原文的关系**：这是 Nature 原文思想在老年人 hip-intensive tasks 中的直接延展，关注老年人移动能力和髋关节动力学支持。
- **汇报价值/可被追问点**：适合用于说明 Nature 思路正在向具体人群和日常任务集迁移。
- **被引关系证据**：arXiv 正文参考文献中明确列出 Molinaro et al., Nature 635, 337–344, 2024
- **检索链接**：https://arxiv.org/abs/2603.22580

### 14. Design and evaluation of a passive knee-ankle exoskeleton for walking and squatting: a musculoskeletal simulation study

- **作者**：Lizhen Zhang, Mengxiang Zhu, Bo Jiang 等
- **年份/状态**：2026，Medical & Biological Engineering & Computing
- **标题或摘要总结**：通过肌骨仿真设计并评估被动膝-踝外骨骼在 walking 和 squatting 任务中的作用，更偏结构设计和仿真评价，而非深度学习控制器。
- **与 Nature 原文的关系**：Nature cited-by 页面列出该文引用了 Nature 原文；它不是直接 follow biological moment estimator，而是从外骨骼设计/仿真角度引用 task-agnostic 控制背景。
- **汇报价值/可被追问点**：可作为“外骨骼结构设计领域也引用该控制思想”的外围引用。
- **被引关系证据**：Nature article cited-by 列表
- **检索链接**：https://www.nature.com/articles/s41586-024-08157-7

### 15. Enhanced gastrocnemius-mimicking lower limb powered exoskeleton robot

- **作者**：Tianchi Chen, Zhi Liu, Ye He 等
- **年份/状态**：2025，Journal of NeuroEngineering and Rehabilitation
- **标题或摘要总结**：设计增强型、模仿腓肠肌功能的下肢动力外骨骼机器人，属于生物启发式下肢外骨骼结构/驱动设计。
- **与 Nature 原文的关系**：Nature cited-by 页面列出该文引用了 Nature 原文；它更多借用外骨骼辅助和人体动力学控制背景。
- **汇报价值/可被追问点**：相关度中等，可作为“生物启发式硬件设计”分支。
- **被引关系证据**：Nature article cited-by 列表
- **检索链接**：https://www.nature.com/articles/s41586-024-08157-7

### 16. Interaction-based rapid heuristic optimization of exoskeleton assistance during walking

- **作者**：Jianyu Chen, Weihao Yin, Jianquan Ding, Jiaqi Han, Lihai Zhang, Jianda Han, Juanjuan Zhang
- **年份/状态**：2025 在线发表；Communications Engineering 5, Article 19 (2026)；DOI:10.1038/s44172-025-00574-4
- **标题或摘要总结**：提出 interaction-based rapid heuristic optimization 方法，在约 2 分钟内优化外骨骼辅助，比已有方法快 16 倍；踝外骨骼实验中降低肌肉活动 36.8%、代谢成本 20.4%。
- **与 Nature 原文的关系**：引用 Nature 原文作为 task-agnostic / joint-moment-based assistance 的代表工作，但这篇更偏快速个性化优化。
- **汇报价值/可被追问点**：可用于比较两类范式：moment-estimation-based autonomous control vs human-response-based rapid optimization。
- **被引关系证据**：Nature cited-by 列表；Communications Engineering 正文参考文献中列出 Molinaro et al. Nature 635, 337–344
- **检索链接**：https://www.nature.com/articles/s44172-025-00574-4

### 17. Comprehensive human locomotion and electromyography dataset: Gait120

- **作者**：Junyo Boo, Dongwook Seo, Minseung Kim, Seungbum Koo
- **年份/状态**：2025，Scientific Data 12, Article 1023；DOI:10.1038/s41597-025-05391-0
- **标题或摘要总结**：发布 Gait120 大规模人体 locomotion 与 EMG 数据集，包含 120 名受试者的运动捕捉、关节角和肌电数据。
- **与 Nature 原文的关系**：Nature cited-by 页面列出该数据集论文引用了 Nature 原文；其意义在于为数据驱动 biomechanical estimation 和外骨骼控制提供更大规模数据资源。
- **汇报价值/可被追问点**：适合放入“数据集与训练资源”分支，说明后续研究正在补数据瓶颈。
- **被引关系证据**：Nature cited-by 列表；Scientific Data 页面
- **检索链接**：https://www.nature.com/articles/s41597-025-05391-0

### 18. The perceptual and biomechanical effects of scaling back exosuit assistance to changing task demands

- **作者**：Jinwon Chung, D. Adam Quirk, Jason M. Cherin, Dennis Friedrich, Daekyum Kim, Conor J. Walsh
- **年份/状态**：2025，Scientific Reports 15, Article 10929；DOI:10.1038/s41598-025-94726-3
- **标题或摘要总结**：研究 back exosuit 在不同任务需求下缩放辅助强度对主观感知和生物力学指标的影响。实验比较非辅助控制器和多种自适应控制器，指出辅助过强/过弱都会影响用户感受。
- **与 Nature 原文的关系**：Nature cited-by 页面列出该文引用了 Nature 原文；它从“任务需求变化时辅助如何缩放”角度补充外骨骼控制问题。
- **汇报价值/可被追问点**：适合讨论“task adaptation 不只是降低代谢，也要考虑用户感知和舒适性”。
- **被引关系证据**：Nature cited-by 列表；Scientific Reports 页面
- **检索链接**：https://www.nature.com/articles/s41598-025-94726-3

### 19. Musculoskeletal Motion Imitation for Learning Personalized Exoskeleton Control Policy in Impaired Gait

- **作者**：Itak Choi, Ilseung Park, Eni Halilaj, Inseung Kang
- **年份/状态**：2026，arXiv:2604.09431 预印本
- **标题或摘要总结**：结合生理可信的肌骨仿真和强化学习，学习可个性化的外骨骼控制策略，既面向健康步态，也面向 impaired gait 模型。
- **与 Nature 原文的关系**：正文明确评价 Molinaro 等提出的 task-agnostic paradigm，并在参考文献中列出 Nature 原文；该文进一步指出原路线在临床人群和个性化方面仍有限。
- **汇报价值/可被追问点**：适合放入“临床/病理步态 + 仿真强化学习”的后续路线。
- **被引关系证据**：arXiv HTML 正文与参考文献明确列出 Molinaro et al. Nature 2024
- **检索链接**：https://arxiv.org/abs/2604.09431

### 20. Ensuring Interaction Safety in Multitask Exoskeleton Control: A Simulation-Trained Variable Impedance Framework

- **作者**：Muyuan Ma, Houcheng Li, Haotian Zhai, Lijun Han, Xinpan Meng, Xiuze Xia, Long Cheng
- **年份/状态**：2026，arXiv:2606.06370 预印本
- **标题或摘要总结**：提出仿真训练的 variable impedance framework，目标是提高 multitask exoskeleton control 中的人机交互安全性。
- **与 Nature 原文的关系**：参考文献中明确列出 Nature 原文。与 Uncertainty-Aware Ankle Exoskeleton Control 类似，它从 safety 角度回应任务泛化控制在真实交互中的风险。
- **汇报价值/可被追问点**：这是很新的安全性 follow-up，可作为“interaction safety / impedance control”方向补充。
- **被引关系证据**：arXiv HTML 参考文献明确列出 Molinaro et al. Nature 635, 337–344, 2024
- **检索链接**：https://arxiv.org/html/2606.06370v1

## 汇报时最值得重点读的 6 篇

- **Uncertainty-Aware Ankle Exoskeleton Control**：补 Nature 原文的安全性/OOD 缺口，最适合讲“本文不足与后续方向”。
- **Deep domain adaptation eliminates costly data required for task-agnostic wearable robotic control**：补数据昂贵和跨设备泛化问题，且为 Science Robotics 正式论文。
- **Optimizing Locomotor Task Sets for Training a Biological Joint Moment Estimator**：直接回答训练 biological moment estimator 需要采哪些任务。
- **Optimized Mappings from Biological Hip Moment Estimates to Exoskeleton Torque...**：直接讨论 moment-to-torque mapping 和个体化辅助。
- **Task-Agnostic Exoskeleton Control Supports Elderly Joint Energetics...**：展示 Nature 思想向老年人和 hip-intensive tasks 迁移。
- **Musculoskeletal Motion Imitation for Learning Personalized Exoskeleton Control Policy in Impaired Gait**：将 task-agnostic 控制放入 impaired gait + 肌骨仿真 + 强化学习背景。

## 主要检索来源

- **Nature 原文页面**：https://www.nature.com/articles/s41586-024-08157-7
- **Semantic Scholar 引用列表**：https://www.semanticscholar.org/paper/Task-agnostic-exoskeleton-control-via-biological-Molinaro-Scherpereel/f337d1553ae42795c7bf8711049b8bceea3d18ab
- **PubMed 原文记录**：https://pubmed.ncbi.nlm.nih.gov/39537888/
- **Uncertainty-Aware Ankle Exoskeleton Control**：https://arxiv.org/abs/2508.21221
- **Deep-Learning Control of Lower-Limb Exoskeletons via Simplified Therapist Input**：https://arxiv.org/abs/2412.07959
- **Exoskeleton Control through Learning to Reduce Biological Joint Moments in Simulations**：https://arxiv.org/abs/2603.07629
- **Motion Adaptation Across Users and Tasks for Exoskeletons via Meta-Learning**：https://arxiv.org/abs/2509.13736
- **Task-Agnostic Exoskeleton Control Supports Elderly Joint Energetics...**：https://arxiv.org/abs/2603.22580
- **Musculoskeletal Motion Imitation...**：https://arxiv.org/abs/2604.09431
- **Ensuring Interaction Safety in Multitask Exoskeleton Control**：https://arxiv.org/html/2606.06370v1
- **Science Robotics: Deep domain adaptation...**：https://www.science.org/doi/10.1126/scirobotics.ads8652
- **Communications Engineering: Interaction-based rapid heuristic optimization...**：https://www.nature.com/articles/s44172-025-00574-4
- **Scientific Data: Gait120**：https://www.nature.com/articles/s41597-025-05391-0
- **Scientific Reports: scaling back exosuit assistance...**：https://www.nature.com/articles/s41598-025-94726-3

# AAAI-27 投稿前待补实验清单（服务器执行版）

本文件只回答一个问题：**上服务器后要跑哪些命令，才能把 `main.tex` 里现在的数字全部变成可复算、可写进正文的结果。**
数据字段口径、拷回清单、验收标准见 `DATA_REQUIREMENTS.md`；本文件是它的执行侧补充，不重复字段定义。

## 0. 总览

| # | 任务 | 类型 | 作业数 | 是否需要 GPU 训练 | 优先级 |
|---|---|---|---|---|---|
| A | 三个主模型按修正后的 `forecast_residual` 重新评测（顺带补 `stuck_imu`） | eval | 3 | 否，复用 checkpoint | P0 必做 |
| B | 故障强度 stress 重新评测 | eval | 3 | 否，复用主 checkpoint | P0 必做 |
| C | 消融第 5 阶段 `prob_aug_recon_fc` 重新评测 | eval | 1 | 否，复用 checkpoint | P0 必做 |
| D | 汇总 + 建表 + gap report | 汇总 | 1 | 否 | P0 必做 |
| E | LOSO 15 折（主安全结论的置信区间） | train | 15 | 是 | P1 |
| F | action-input 消融 4 个 profile | train | 4 | 是 | P1 |
| G | 五个消融模式补到 seeds 7/13/23 | train | 10（增量） | 是 | P2 |
| H | deep-ensemble 评测 | eval | 1 | 否 | P2 |
| I | 逐 timestep 时序导出（画故障时序图用） | eval + 少量代码 | 1 | 否 | P3 可选 |

关键判断：**主模型不需要重训。** `forecast_residual` 的修正只改变残差的发布时刻（现在在目标到达的 `t+h` 发布），预测头的训练目标没有变，所以受影响的只有评测期指标。受影响的 run 是**带预测头（`use_forecast=True`，即 `prob_aug_recon_fc`）的那些**：主模型、消融第 5 阶段、action 四个 profile、LOSO、stress、ensemble。`det_noaug` / `det_aug` / `prob_aug` / `prob_aug_recon` 没有 forecast 通道，指标不受影响，**不用动**（含三个 paired deterministic baseline）。

## 1. 前置检查（不占 GPU，先做完）

```bash
cd ~/tacode        # 服务器上的仓库路径
git pull           # 必须包含 reliability/metrics.py 的 forecast_residual 修正
git rev-parse HEAD
```

确认修正已生效，再花 GPU：

```bash
conda run --no-capture-output -n pytorch python -m unittest tests.test_forecast_residual -v
```

两个 case 都要通过（`horizon=1` 时 `u=0` 位置为 0，`horizon=2` 时前两点为 0，长度不变）。**不通过就不要开始评测**，否则拿到的还是旧口径数字。

盘点现有产物，决定 A–H 里哪几项能走「复用 checkpoint」快路：

```bash
for d in reports/v2_*; do
  [ -d "$d" ] || continue
  printf '%-46s json=%-3s ckpt=%s\n' "$d" \
    "$([ -s "$d/reliability_report.json" ] && echo yes || echo no)" \
    "$([ -f "$d/reliability_tcn_best.pt" ] && echo yes || echo no)"
done
```

再看每份报告的实际配置（用来确认不是 smoke、不是 `limit_trials>0`）：

```bash
conda run -n pytorch python - <<'PY'
import json, pathlib
for p in sorted(pathlib.Path("reports").glob("v2_*/reliability_report.json")):
    a = json.load(p.open())["args"]
    print(f"{p.parent.name:44s} mode={a['training_mode']:18s} seed={a['seed']:<3} "
          f"profile={a['input_profile']:16s} epochs={a['epochs']:<3} limit={a['limit_trials']}")
PY
```

如果某个 `prob_aug_recon_fc` 目录只有 `.pt` 没有 json，也没关系——A/C 只需要 checkpoint。
如果连 checkpoint 都没有，该项就退化为「需要重训」，按 E/F/G 的训练命令跑。

## 2. 任务 A（P0）：主模型重新评测 + 补 `stuck_imu`

现在正文写「六种训练期故障」（`main.tex:151`：insole_missing / encoder_dropout / imu_bias / packet_loss / **stuck_imu** / sensor_delay），但 `CORE_FAULTS` 和 `build_aaai_tables.py` 的 `FAULTS` 都没有 `stuck_imu`，所以现有报告根本没有 `test_id/stuck_imu` 这一行。这一步同时解决两件事：换成修正后的 forecast 口径，并把 `stuck_imu` 补进故障场景。

现成的 `scripts/refresh_v2_evaluation.sh` 已经做了「复用主 checkpoint 重新评测」，但它的 `--fault-scenarios` 是硬编码的、不含 `stuck_imu`。**直接改脚本第 53 行**，把 `imu_bias,packet_loss,` 后面加上 `stuck_imu,`：

```text
--fault-scenarios clean,insole_missing,encoder_dropout,imu_bias,packet_loss,stuck_imu,packet_loss_burst,packet_loss_partial,sensor_delay,sensor_delay_jitter \
```

然后跑（`GPUS` 换成你实际空闲的卡号，三个 seed 各占一张，并行）：

```bash
GPUS="3 4 5" SEEDS="7 13 23" \
  bash scripts/refresh_v2_evaluation.sh 2>&1 | tee logs/refresh_main_$(date +%m%d_%H%M).log
```

产出：`reports/v2_refreshed_main_seed{7,13,23}/reliability_report.json`。脚本内部已经串了 `summarize_reliability_report.py` 和 `scripts/merge_refreshed_report.py`，所以旧的 `reports/v2_main_seed*/reliability_report.json` 会被合并更新而不是覆盖丢失。

跑完立即验证 `stuck_imu` 真的进去了：

```bash
conda run -n pytorch python - <<'PY'
import json, pathlib
need = ["test_id/stuck_imu", "test_ood/stuck_imu",
        "fault_detection/test_id/stuck_imu", "fault_detection/test_ood/stuck_imu"]
for p in sorted(pathlib.Path("reports").glob("v2_refreshed_main_seed*/reliability_report.json")):
    r = json.load(p.open())["results"]
    miss = [k for k in need if k not in r]
    print(p.parent.name, "OK" if not miss else f"MISSING {miss}")
PY
```

**如果 A 因为任何原因跑不动**（比如 checkpoint 丢了、又不想重训 3×20 epoch），退路是把正文改成只声称实际评测到的故障：`main.tex:151` 的 "one of six faults" 改成五种并删掉 stuck IMU，`main_zh.tex:96` 同步改。这是降级方案，不是首选。

`stuck_imu` 进了报告之后，还要让建表脚本认它——见任务 D。

## 3. 任务 B（P0）：stress 曲线重新评测

正文里最大强度那两个数（`packet_loss@0.30`、`sensor_delay@20` 的 fused AUROC、retention、wrong-direction change）也是旧 forecast 口径，必须重算。stress 本身就是 eval-only，复用主 checkpoint：

```bash
STRESS_SKIP_COMPLETED=0 \
STRESS_SEEDS="7 13 23" \
PARALLEL=1 GPUS="3 4 5" \
  bash scripts/reliability_paper_v2.sh stress 2>&1 | tee logs/stress_$(date +%m%d_%H%M).log
```

`STRESS_SKIP_COMPLETED=0` 是关键：默认是 `1`，会因为 json 已存在而跳过，于是你拿到的还是旧数字。

产出：`reports/v2_refreshed_stress_seed{7,13,23}/reliability_report.json`，每份要含 11 个带 `@` 的场景（`packet_loss@{0.05,0.15,0.30}`、`packet_loss_burst@{0.05,0.15}`、`packet_loss_partial@0.15`、`sensor_delay@{5,10,20}`、`sensor_delay_jitter@{10,20}`）。

## 4. 任务 C（P0）：消融第 5 阶段重新评测

五个消融模式里只有 `prob_aug_recon_fc` 带预测头，所以只有它受 forecast 修正影响。但它正好是消融序列的终点，正文的「+ forecasting」那一行差值（AUROC $-0.020$、retention 到 $0.798$）依赖它，必须重算，否则第 4→5 阶段的差值是两套口径相减，无效。

单个 eval，直接调 `run_reliability_experiment.py`（复用消融目录里的 checkpoint）：

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n pytorch python run_reliability_experiment.py \
  --mode eval --device cuda \
  --data-root /home/zfy/dataset/tcn/Parsed \
  --checkpoint reports/v2_ablation_prob_aug_recon_fc_seed7/reliability_tcn_best.pt \
  --output-dir reports/v2_ablation_prob_aug_recon_fc_seed7_refreshed \
  --input-profile human --training-mode prob_aug_recon_fc \
  --batch-size 64 --num-workers 12 --prefetch-factor 4 --amp bf16 \
  --window-size 768 --stride 384 \
  --max-windows-per-trial 4 --min-valid-fraction 0.5 \
  --heldout-tasks jump,cutting,lift_weight,lunges \
  --limit-trials 0 \
  --fault-scenarios clean,insole_missing,encoder_dropout,imu_bias,packet_loss,stuck_imu,packet_loss_burst,packet_loss_partial,sensor_delay,sensor_delay_jitter \
  --eval-ignore-history 248 --mc-samples 10 \
  --select-gate-on-val --gate-softness 0.5 --gate-deadband 0.1 \
  --gate-validation-faults insole_missing,encoder_dropout,packet_loss,sensor_delay \
  --ood-deadband-quantile 0.9 --gate-max-deadband 2.0 \
  --gate-softness-grid 0.25,0.5,1.0,2.0,4.0,8.0 \
  --log-interval 20 --progress-style line --seed 7 \
  2>&1 | tee logs/ablation_fc_refresh_$(date +%m%d_%H%M).log
```

注意：这里我用了新目录 `..._refreshed` 而不是原地覆盖，方便对比。`summarize_reliability_suite.py` 的 `classify_run_family` 按 `v2_ablation_` 前缀识别，新目录名仍以 `v2_ablation_` 开头，所以会被正确归到 ablation 家族。但同一模式会出现两份报告（旧 + 新），汇总时会被当成 2 个样本平均——**汇总前把旧那份挪走**：

```bash
mkdir -p reports/_stale_pre_forecast_fix
mv reports/v2_ablation_prob_aug_recon_fc_seed7/reliability_report.json \
   reports/_stale_pre_forecast_fix/v2_ablation_prob_aug_recon_fc_seed7.json
```

前四个阶段（`det_noaug`/`det_aug`/`prob_aug`/`prob_aug_recon`）**保持原样，不要重算**，它们没有 forecast 通道。

## 5. 任务 D（P0）：汇总、建表、gap report

`stuck_imu` 现在在报告里了，但两个下游脚本的默认故障列表还不含它，会**静默丢弃**这一列（不报错，只是表里没有）。要改两处：

1. `summarize_reliability_suite.py:21` 的 `CORE_SCENARIOS`：在 `test_id/packet_loss` 后加 `"test_id/stuck_imu",`，在 `test_ood/packet_loss` 后加 `"test_ood/stuck_imu",`。
2. `scripts/build_aaai_tables.py:15` 的 `FAULTS`：在 `"packet_loss",` 后加 `"stuck_imu",`。`CHALLENGE_FAULTS` 不动——`stuck_imu` 是训练期见过的故障，不属于"未见生成器"那一组，这个区分直接决定正文里 $0.842/0.778$ 那两个宏平均的成分，不要搞混。

改完直接跑封装脚本（它 glob `reports/v2_*/reliability_report.json` 全量重算）：

```bash
scripts/reproduce_aaai.sh summary
```

第 1 处也可以不改源码：`--scenarios` 有 CLI 开关，但 `run_summary` 不透传，所以要绕过封装脚本手工调一次。二选一，不要两样都做。

```bash
conda run -n pytorch python summarize_reliability_suite.py \
  --output-dir reports/v2_paper_suite \
  --primary-report reports/v2_main_seed7/reliability_report.json \
  --scenarios "$(python3 - <<'PY'
splits = ("test_id", "test_ood")
faults = ["clean","insole_missing","encoder_dropout","imu_bias","packet_loss","stuck_imu",
          "packet_loss_burst","packet_loss_partial","sensor_delay","sensor_delay_jitter"]
print(",".join(f"{s}/{f}" for s in splits for f in faults))
PY
)" \
  reports/v2_*/reliability_report.json
```

`build_aaai_tables.py` 的 `FAULTS` 是模块级常量、没有 CLI 开关，只能改源码。改完跑：

```bash
conda run -n pytorch python scripts/build_aaai_tables.py \
  --suite-dir reports/v2_paper_suite \
  --output-dir reports/v2_paper_suite

sed -n '1,240p' reports/v2_paper_suite/paper_gap_report.md
```

`build_aaai_tables.py` 对缺行是 `KeyError` 硬失败（不是跳过），所以它一旦跑通，就等于验证了 10 个故障 × 2 个 split 的检测行和控制行都齐全——这本身就是一道验收。

`run_summary` 会把 `reports/v2_*/reliability_report.json` 全部吃进去，所以刚才那步「把旧 json 挪到 `_stale_pre_forecast_fix/`」很重要——那个目录不以 `v2_` 开头，不会被 glob 命中。

gap report 里不应再出现缺主 seed、缺 paired baseline、缺 stress 曲线的提示。如果提示 `stress seeds < 3`，说明任务 B 没跑全。

## 6. 任务 E（P1）：LOSO 15 折 —— 为主结论补置信区间

这是目前论文**方法论上最需要补**的一项。摘要的主结果是「反向力矩平均降低约 0.2 个百分点」，现在只有 3 个 seed 的均值/标准差，seed 波动衡量的是训练随机性，不是被试间差异。审稿人会问：0.2 个百分点在被试间是否稳定、是否跨过 0。15 折 LOSO 正好给出 15 个独立被试上的效应量，可以算 95% bootstrap CI 和「改善被试比例」。

15 折全是完整训练（每折 20 epoch），是本清单里最重的任务，建议单独一晚跑：

```bash
PARALLEL=1 WAIT_FOR_GPU=1 \
CORE_FAULTS="clean,insole_missing,encoder_dropout,imu_bias,packet_loss,stuck_imu,packet_loss_burst,packet_loss_partial,sensor_delay,sensor_delay_jitter" \
LOSO_SUBJECTS="BT01 BT02 BT03 BT06 BT07 BT08 BT09 BT10 BT11 BT12 BT13 BT14 BT15 BT16 BT17" \
  nohup bash scripts/reliability_paper_v2.sh loso \
  > logs/loso_$(date +%m%d_%H%M).log 2>&1 &
```

`CORE_FAULTS` 覆盖是必须的：`train_run` 用它作为 `--fault-scenarios`，默认值不含 `stuck_imu`，否则 15 折都缺这一列，跟主结果口径不一致。任务 F、G 同理，命令里都加上这一行。

调度说明：`train_run` 会把 15 个作业推进队列，由 `run_queue` 分发到空闲 GPU（`GPU_FREE_MEM_MB=2000` 且 `GPU_MAX_UTIL<=10` 才算空闲）。想固定用某几张卡就加 `GPUS="1 2 6"`。每折的 val 被试取列表里的前一个（`BT01` 的 val 是 `BT17`，环形），这个规则由脚本决定，不要手改，否则折间不可比。

产出：`reports/v2_loso_BT01/…` 共 15 个目录。跑完统计效应量：

```bash
conda run -n pytorch python - <<'PY'
import json, pathlib, random, statistics
FAULTS = ["insole_missing","encoder_dropout","imu_bias","packet_loss","stuck_imu",
          "sensor_delay","packet_loss_burst","packet_loss_partial","sensor_delay_jitter"]
eff, ret = [], []
for p in sorted(pathlib.Path("reports").glob("v2_loso_*/reliability_report.json")):
    r = json.load(p.open())["results"]
    d = [r[k]["baseline_wrong_direction_ratio"] - r[k]["gated_wrong_direction_ratio"]
         for s in ("test_id","test_ood") for f in FAULTS
         if (k := f"{s}/{f}") in r]
    c = [r[k]["gated_retained_aligned_torque"] / r[k]["baseline_retained_aligned_torque"]
         for s in ("test_id","test_ood") if (k := f"{s}/clean") in r]
    if not d or not c:
        print(f"{p.parent.name:20s} SKIP (缺 clean 或全部故障场景)"); continue
    eff.append(statistics.mean(d)); ret.append(statistics.mean(c))
    print(f"{p.parent.name:20s} effect={statistics.mean(d):+.5f} clean_ret={statistics.mean(c):.3f}")
assert len(eff) >= 10, f"只有 {len(eff)} 折可用，CI 不可信，先补齐 LOSO"
random.seed(0)
bs = sorted(statistics.mean(random.choices(eff, k=len(eff))) for _ in range(10000))
print(f"\nn={len(eff)}  mean={statistics.mean(eff):+.5f}  median={statistics.median(eff):+.5f}")
print(f"95% bootstrap CI = [{bs[250]:+.5f}, {bs[9750]:+.5f}]")
print(f"improved folds = {sum(e > 0 for e in eff)}/{len(eff)}")
print(f"clean retention mean = {statistics.mean(ret):.3f}")
PY
```

这段输出直接填 `main.tex:282` 那条 TODO（"Add confidence intervals ... after the final result refresh"）。注意 LOSO 用的是 `ABLATION_SEED=7` 单 seed，所以 CI 描述的是被试间变异，正文要写清这一点，不要和 3-seed 的 SD 混淆。

## 7. 任务 F（P1）：action-input 消融 —— 现在正文只有承诺没有结果

`main.tex:374` 写了「action-input ablation 进一步检验 desired / measured / execution / interaction 力矩是否提升检测或形成设备专用捷径」，然后**一个数字都没给**。这是审稿人最容易抓的一处：提到的实验必须有结果，否则删掉这句。

四个 profile，单 seed 7，训练类任务：

```bash
PARALLEL=1 WAIT_FOR_GPU=1 \
CORE_FAULTS="clean,insole_missing,encoder_dropout,imu_bias,packet_loss,stuck_imu,packet_loss_burst,packet_loss_partial,sensor_delay,sensor_delay_jitter" \
  nohup bash scripts/reliability_paper_v2.sh action \
  > logs/action_$(date +%m%d_%H%M).log 2>&1 &
```

产出：`reports/v2_action_human_{desired,measured,execution,interaction}_seed7/`。四个 profile 都是 `prob_aug_recon_fc`，所以天然就是修正后的 forecast 口径，不用二次评测。

跑完直接读 `reports/v2_paper_suite/core_aggregate_by_profile.tsv` 和 `detection_aggregate_by_profile.tsv`——汇总脚本本来就按 `input_profile` 分组聚合，四个 profile 会自动出现在这两张表里，不用另写脚本。`summarize_reliability_suite.py:673` 还会检查这四个 profile 是否齐全，缺哪个会在 gap report 里点名。

对比口径：以 `reports/v2_main_seed7`（`input_profile=human`，25 通道）为参照，看 macro fused AUROC 是否提升、clean OOD RMSE 是否变化。**结论方向不重要，写清楚就行**——如果 action 通道明显提升检测，那是「设备专用捷径」的证据，正好支持论文选择只用 wearable 通道；如果没提升，就说明 wearable-only 没有损失。两种结果都能写。跑不了的话，删掉 `main.tex:374` 和 `main_zh.tex:320` 对应那句，比留一个空承诺好。

## 8. 任务 G（P2）：消融补到 3 seeds

现在正文诚实地标注了「单 seed attribution，属诊断证据」（`main.tex` 消融段 + 图 caption），这是可接受的。但如果想让消融进入主结论、报告标准差，需要五个模式各跑 seeds 7/13/23：

```bash
FAULTS_ALL="clean,insole_missing,encoder_dropout,imu_bias,packet_loss,stuck_imu,packet_loss_burst,packet_loss_partial,sensor_delay,sensor_delay_jitter"
PARALLEL=1 ABLATION_SEED=13 CORE_FAULTS="$FAULTS_ALL" \
  nohup bash scripts/reliability_paper_v2.sh ablations \
  > logs/abl_s13_$(date +%m%d_%H%M).log 2>&1 &
# 等上一批结束再跑
PARALLEL=1 ABLATION_SEED=23 CORE_FAULTS="$FAULTS_ALL" \
  nohup bash scripts/reliability_paper_v2.sh ablations \
  > logs/abl_s23_$(date +%m%d_%H%M).log 2>&1 &
```

注意 `run_ablations` 的输出目录是 `reports/v2_ablation_${mode}_seed${ABLATION_SEED}`，所以两批 seed 不会互相覆盖。但换 seed 之后第 1–4 阶段也是新训练，**这十个作业里没有一个可以复用 checkpoint**。

10 个新训练作业。优先级低于 E 和 F：单 seed 消融只要标注清楚就不算硬伤，而缺 CI 和缺 action 结果是硬伤。

## 9. 任务 H（P2）：deep-ensemble 评测

需要三个主 checkpoint 都在。用于说明 epistemic 通道（MC-dropout vs deep ensemble）的贡献，不是任何表格的必需项：

```bash
PARALLEL=0 GPUS="3" bash scripts/reliability_paper_v2.sh ensemble \
  2>&1 | tee logs/ensemble_$(date +%m%d_%H%M).log
```

产出 `reports/v2_ensemble_seed7/`。注意 `run_ensemble` 的 `--fault-scenarios` 用的是 `CORE_FAULTS`，想含 `stuck_imu` 就加 `CORE_FAULTS="clean,insole_missing,encoder_dropout,imu_bias,packet_loss,stuck_imu,packet_loss_burst,packet_loss_partial,sensor_delay,sensor_delay_jitter"`。

## 10. 任务 I（P3，可选）：逐 timestep 时序导出

`fig_method_overview` 的第三个面板（risk → gate → torque 时序）目前是示意图，caption 已注明 "schematic"。想换成真实数据，需要评测时导出逐 timestep 的 `prediction / risk / gate / commanded torque` 数组——当前 `run_reliability_experiment.py` 只写聚合指标，不保存时序。

这需要改代码（在 eval 路径里对一个选定试验 dump `.npz`），不是纯跑命令，所以放最后。如果时间不够，保留示意图 + caption 里的 "schematic" 标注是完全可接受的，不影响投稿。

## 11. 推荐执行顺序

```text
第 0 步  git pull + unittest（5 分钟，不过就停）
         ↓
第 1 批  A + B 并行（纯 eval，6 个作业，几十分钟到几小时）
         ↓
第 2 步  C（1 个 eval）+ 挪走旧 ablation json
         ↓
第 3 步  D 汇总建表 → 读 gap report → 把新数字回填 main.tex
         ↓
第 4 批  E（LOSO 15 折，整晚）+ F（action 4 个，可同批入队）
         ↓
第 5 步  重跑 D，补 LOSO 的 CI 统计
         ↓
可选     G / H / I
```

第 3 步做完，论文的所有现有数字就都是修正后口径了，可以先出一版可投的稿子。第 4 批做完，才能删掉 `main.tex:282` 和 `main.tex:286` 两条 TODO。

## 12. 拷回前的验收

按 `DATA_REQUIREMENTS.md` §7 执行，这里只列本次新增的检查项：

1. `tests/test_forecast_residual.py` 在生成这批报告的那个 commit 上通过（`reports/AAAI27_CODE_COMMIT.txt` 要记这个 commit）。
2. 所有 `prob_aug_recon_fc` 报告（main / stress / ablation 第 5 阶段 / action / LOSO / ensemble）的 mtime 都晚于 forecast 修正的 commit 时间；早于它的一律作废。
3. `test_{id,ood}/stuck_imu` 和两个 detection key 在三个主报告里都存在。
4. `reports/_stale_pre_forecast_fix/` 里的旧 json 没有被汇总进 `v2_paper_suite`（检查 `main_detection_aggregate.tsv` 的 `n` 列是否等于预期 seed 数）。
5. `paper_gap_report.md` 不再提示缺 action profile、缺 stress seeds。

打包（不含 `.pt`）：

```bash
git rev-parse HEAD > reports/AAAI27_CODE_COMMIT.txt
git status --short > reports/AAAI27_GIT_STATUS.txt
find reports -path '*/reliability_report.json' -size +0 -print | sort \
  > reports/AAAI27_REPORT_MANIFEST.txt

find reports -type f \( \
  -path 'reports/v2_*/reliability_report.json' -o \
  -path 'reports/v2_paper_suite/*' -o \
  -name 'AAAI27_CODE_COMMIT.txt' -o \
  -name 'AAAI27_GIT_STATUS.txt' -o \
  -name 'AAAI27_REPORT_MANIFEST.txt' \
\) -print0 | tar --null -T - -czf aaai27_metrics.tar.gz
sha256sum aaai27_metrics.tar.gz > aaai27_metrics.tar.gz.sha256
```

主 checkpoint 建议一并留着（不必拷回，但别删）：`stuck_imu` 这类「漏评测」问题只要有 checkpoint 就能补，没有就得重训。

## 13. 数字回填到论文时要同步的位置

拿到新结果后，`main.tex` 和 `main_zh.tex` 的数字**必须一起改**，中文版不会自动同步。另外图是脚本硬编码的常量，也要改：

| 论文位置 | 数据来源 |
|---|---|
| Abstract 三个数（clean RMSE change / wrong-direction reduction / clean retention） | `DATA_REQUIREMENTS.md` §5.1 |
| `tab:accuracy` | `fair_baseline_aggregate.tsv` + 三份主 JSON 的 nll/coverage_ece |
| `fig_detection_taxonomy`（`DETECTION_ROWS`、`MACRO_AUROC`） | `table1_detection.tsv` |
| `fig_safety_utility`（`SAFETY_ROWS`） | `table2_safety.tsv` |
| `fig_ablation_summary`（`ABLATION_ROWS`） | 五个 ablation JSON，按 §5.5 相邻差值 |
| `fig_stress_transfer`（`STRESS_ROWS`、`LOSO_ROWS`） | stress JSON + 15 个 LOSO JSON |
| 消融段的 retention 单调序列 $0.731\to0.798$ | 消融 JSON 的 gate retained ratio |
| 平均门控区间 $0.960$–$0.991$ | `main_aggregate.tsv` 的 `mean_gate_mean` |
| action 消融那句话（`main.tex:374`） | `core_aggregate_by_profile.tsv` + `detection_aggregate_by_profile.tsv` |
| LOSO 置信区间（新增，填 `main.tex:282` 的 TODO） | 第 6 节的 bootstrap 脚本输出 |

聚合表的列名规则是 `<field>_mean` / `<field>_std`（`summarize_reliability_suite.py:572`），字段名就是报告里的原始 key，例如 `gated_wrong_direction_ratio_mean`。

图的数据常量都在 `scripts/draw_aaai_figures.py` 顶部，改完跑 `cd paper/aaai27 && make figures` 重新出图，再目视检查一遍标注有没有重叠（数值变了标注位置可能撞在一起）。


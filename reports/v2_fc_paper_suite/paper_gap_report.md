# Paper-Readiness Gap Report

- Reports: 44
- Primary report: `reports/v2_fc_main_seed7/reliability_report.json`

## Evidence Gaps

- Core evidence coverage looks reasonable for this report set.

## Primary Performance Gaps

- Same-split deterministic baseline is better on test_id/clean: 0.1726 vs 0.1616 (delta +0.0110).
- Same-split deterministic baseline is better on test_ood/clean: 0.2506 vs 0.2282 (delta +0.0224).
- Secondary official-checkpoint comparison: OOD clean RMSE is worse by 0.0748; this checkpoint is not split-matched and must not be used as the primary baseline.
- Base predictor under-retains torque on test_ood/clean: ungated delta vs Nature is -0.3542; the gate itself keeps 96.0%.
- Base predictor under-retains torque on test_ood/sensor_delay: ungated delta vs Nature is -0.2987; the gate itself keeps 94.0%.
- Weak fault detectors: fault_detection/test_id/imu_bias=0.531, fault_detection/test_ood/imu_bias=0.530, fault_detection/train/imu_bias=0.528, fault_detection/val/imu_bias=0.533
- Detector fusion masks useful signals: fault_detection/test_id/packet_loss_burst: fused=0.757, best=0.924, fault_detection/test_ood/packet_loss: fused=0.796, best=0.958, fault_detection/test_ood/packet_loss_burst: fused=0.755, best=0.923, fault_detection/test_ood/packet_loss_partial: fused=0.815, best=1.000, fault_detection/test_ood/sensor_delay: fused=0.773, best=0.989, fault_detection/test_ood/sensor_delay_jitter: fused=0.768, best=0.964, fault_detection/test_ood/stuck_imu: fused=0.848, best=1.000, fault_detection/train/packet_loss_burst: fused=0.767, best=0.885 ...

## Recommended Next Runs

- Treat near-chance detectors as scoped empirical limitations unless a mechanism-based signal improves validation; do not tune detector candidates against test AUROC.
- Keep the fixed fault-by-signal matrix as primary evidence. The validation-selected ECDF candidate is rejected because it harms held-out burst detection; no further fusion tuning is prioritized without a new mechanism.

## Primary Best Validation

- No validation history found.

## Primary Gate Policy

- softness=2.0000, deadband=1.0520

## Primary Detector Policy

- Detector-all signals=residual,staleness,coherence; source=validation_fault_subset_search
- Detector-online signals=residual,staleness; candidate pool (K_gate)=logit,aleatoric,residual,forecast,epistemic,staleness
- validation AUROC: insole_missing=0.8586, encoder_dropout=0.9111, imu_bias=0.5021, packet_loss=0.7741, sensor_delay=0.8357

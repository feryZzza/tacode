# Reproducibility Notes for the Anonymous Submission

This document accompanies the separate AAAI-27 reproducibility checklist. It
contains no author identity, repository URL, or external download link.

## Scope and evidential limits

The paper reports offline replay on an existing, cited, publicly released
exoskeleton dataset. No physical exoskeleton was controlled in these
experiments. Three training seeds (7, 13, and 23) quantify optimization
stochasticity; they are not treated as independent participant replicates. The
primary uncertainty statement is instead the percentile-bootstrap 95% interval
over 15 leave-one-subject-out folds.

The historical experiment reports do not record the exact CPU/GPU models,
memory, operating system, or package versions used for every run. Accordingly,
the official checklist marks the computing-infrastructure item as `no`; this
file does not reconstruct or guess that metadata.

## Expected data layout

Place the parsed dataset under `data/Parsed`, or set `DATA_ROOT` to another
location. Preprocessing, participant/task splitting, normalization, fault
injection, estimator training, validation-only detector selection, gate tuning,
evaluation, and aggregation are implemented in this code package.

The participant-disjoint primary split contains 11 training, 2 validation, and
2 test participants. Jumping, cutting, weight lifting, and lunges are excluded
from estimator fitting. Clean trials in those categories from validation
participants are used for gate tuning, so the paper calls them
training-held-out tasks rather than fully unseen tasks.

## Software setup

Use Python 3 with the packages in the repository requirements file. A
CUDA-enabled PyTorch environment is recommended for the full suite. Commands
below accept `PY`, `DEVICE`, `DATA_ROOT`, `GPUS`, and worker-count overrides.
The final package should record an environment lock and the actual accelerator
model before the public release; these were not recoverable from the archived
reports used for the anonymous manuscript.

## End-to-end commands

From the repository root:

```bash
# Quick pipeline check.
PY=python3 DEVICE=cpu DATA_ROOT=data/Parsed \
  scripts/reproduce_aaai.sh smoke

# Main models, paired deterministic baselines, ordered ablations,
# action-input tests, stress tests, ensemble, and aggregation.
PY=python3 DEVICE=cuda DATA_ROOT=data/Parsed \
  EPOCHS=20 SEEDS="7 13 23" scripts/reproduce_aaai.sh core

# Add the 15 leave-one-subject-out folds.
PY=python3 DEVICE=cuda DATA_ROOT=data/Parsed \
  EPOCHS=20 SEEDS="7 13 23" scripts/reproduce_aaai.sh all

# Rebuild summary files from completed report JSON files.
PY=python3 scripts/reproduce_aaai.sh summary
```

For high-core-count machines, the server run used explicit numerical-library
thread limits to prevent each evaluation process from creating a full-machine
thread pool:

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
```

## Figure and manuscript regeneration

The figure build must report only `suite` or `export` data sources:

```bash
cd aaai27
make check-figures
make -B figures
```

The expected final status line is:

```text
data sources: detection=suite, macro=suite, safety=suite, ablation=suite, stress=suite, loso=suite, overview=export
```

Then compile the English manuscript, checklist, and anonymous supplement:

```bash
make submission
```

## Numerical verification

```bash
python3 scripts/check_paper_numbers.py \
  --suite-dir reports/v2_fc_paper_suite
python3 -m unittest discover -s tests -t . -v
```

The paper-number check must report zero missing keys. Test-participant data must
not enter normalization, score normalization, detector subset selection,
deadband selection, or softness selection. The three held-out corruption
operators (`packet_loss_burst`, `packet_loss_partial`, and
`sensor_delay_jitter`) must be absent from both estimator training and detector
selection. They remain within the packet-loss and delay families and are not
open-set hardware faults.

Recompute the seed- and split-matched clean-adjusted contrast with:

```bash
python3 scripts/analyze_gate_specificity.py
```

The command writes `gate_specificity.json` and `gate_specificity.md` under the
paper-suite directory. It operates on scenario aggregates and therefore does
not replace a window-level paired analysis.

## Metric interpretation

The opposition-weighted torque-product integral has units
`(Nm/kg)^2 s`. It is not mechanical energy because angular velocity is absent.
The reported torque-rate statistic is mean absolute command change divided by
the 200 Hz sampling interval, in `Nm kg^-1 s^-1`; it is not a kinematic jerk.
Legacy JSON/TSV keys containing `wrong_energy` and `jerk` are retained only for
backward compatibility with archived reports.

The archived local package does not contain the full per-timestep main-suite
outputs needed to recompute random-shutdown controls, exact closure rates,
shutdown durations, detection latency, or a rate-limited gate. Those analyses
require reevaluation from the server checkpoints and parsed 15-participant
dataset; they must not be inferred from the aggregate TSV files.

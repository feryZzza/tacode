# OpenExo: cross-dataset joint-moment estimation

This branch replaces the original single-dataset demonstration entry point with
a reproducible public-data benchmark for biological hip/knee moment estimation.
It is designed for the present constraint: no exoskeleton hardware and no new
human-subject recruitment.

The main scientific question is whether a causal wearable-sensor model can
transfer across people, devices and locomotion datasets using no target labels,
or only a few target participants.  The code unifies Nature 2024, Dryad and
Camargo into one schema, keeps missing sensors/labels as explicit masks, and
prevents subject leakage.

## What is implemented

- SI-unit, extension-positive canonical feature/target adapters;
- participant-disjoint pooled, source-only, unsupervised-domain-adaptation and
  few-participant protocols;
- a causal mask-aware TCN with optional domain-adversarial alignment;
- balanced domain sampling and physical sensor-group dropout;
- per-joint RMSE/MAE/R² overall, per dataset and per locomotion mode;
- saved configs, complete split manifests, checkpoints and JSON metrics.

The older `reliability/` pipeline and `run_reliability_experiment.py` are kept as
an archived, reproducible baseline for the previous paper.  They are not part
of the new experiment entry point.

## Environment

```bash
conda activate pytorch
pip install -r requirements.txt
```

`mat-io>=1.0.0` is required because Camargo stores MATLAB MCOS table objects;
older releases do not decode these files correctly.

The default paths already match this workstation:

```text
/media/fery/新加卷/Dataset/Task-Agnostic
/media/fery/新加卷/Dataset/Dryad
/media/fery/新加卷/Dataset/Camargo
```

Override them with `--nature-root`, `--dryad-root`, `--camargo-root`, or the
environment variables `NATURE_ROOT`, `DRYAD_ROOT`, `CAMARGO_ROOT`.

## Verify the installation

```bash
python train_open_data.py --mode smoke --protocol pooled --device cpu
```

Then run a real pooled baseline:

```bash
python train_open_data.py \
  --mode train \
  --protocol pooled \
  --feature-profile minimal \
  --alignment none \
  --epochs 20 \
  --device cuda \
  --run-name pooled_minimal_seed7
```

Run subject-disjoint unsupervised adaptation to Camargo. Four Camargo people
provide inputs but **no labels**; all evaluation people remain disjoint:

```bash
python train_open_data.py \
  --mode train \
  --protocol lodo \
  --target-dataset camargo \
  --adaptation-participants 4 \
  --target-supervision none \
  --alignment dann \
  --epochs 20 \
  --run-name uda_camargo_seed7
```

See [OPEN_DATA_EXPERIMENTS.md](OPEN_DATA_EXPERIMENTS.md) for the experiment
design and `scripts/run_open_data_matrix.sh` for the planned run matrix.

## Important scope boundary

This repository can support claims about offline estimation, cross-dataset
transfer, missing-sensor robustness and data efficiency.  Without hardware and
new participants it cannot support claims about closed-loop stability, user
safety, comfort, metabolic benefit or clinical efficacy.

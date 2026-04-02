# Physics-Informed TNO for Norne Field Simulation

Physics-informed neural operator surrogate for the Norne reservoir
simulation dataset using a 4D Temporal Neural Operator with derivative
regularization and mass conservation losses.

## Overview

This example trains a TNO on the Norne field dataset — a real-world
3D reservoir model based on the publicly available
[Norne Field](https://github.com/OPM/opm-data/tree/master/norne) dataset.
Norne requires volumetric 4D operators
(3D spatial + time) and handles complex geological features including
numerous faults with Non-Neighbor Connections (NNCs), pinch-outs, and
39% inactive cells.

A Design of Experiment study identified fault transmissibility and
KVKH multipliers as key uncertain parameters, which were varied using
Latin Hypercube Sampling (LHS) to generate 500 realizations.  The
primary LHS variable is PERMZ (vertical permeability), controlling the
Kv/Kh ratio.  All simulations were generated using the open-source
[OPM](https://opm-project.org/) reservoir simulator.

### Architecture

| Component | Configuration |
|-----------|--------------|
| Branch1 | MLP encoder, tanh |
| Branch2 (t-branch) | MLP encoder, tanh |
| Trunk | 8-layer tanh FNN, grid input (x,y,z,t), output activation |
| Decoder | MLP (2 layers, width 128, sigmoid output) |
| Width | 128 |
| Parameters | 196,161 |
| Dimensions | 4D (46 × 112 × 22 × 65 timesteps) |

### Physics-Informed Losses (Saturation Models)

| Loss | Weight | Description |
|------|--------|-------------|
| L1 | 1.0 | Data-fitting loss on active cells only |
| Derivative (dx, dy, dz) | 0.5 | Spatial gradient regularization in all 3 directions |
| Mass conservation | 0.5 | Weak mass balance with cell-volume weighting |

### Training Configuration

| Setting | Value |
|---------|-------|
| Regime | Autoregressive: 5 TF + 175 rollout epochs |
| Rollout mode | `detached` |
| L / K | 3 / 1 |
| Batch size | 2 per GPU × 8 GPUs |
| Optimizer | Adam, lr=1e-3, weight_decay=1e-4 |
| Scheduler | StepLR(step_size=10, gamma=0.85) |
| Masking | ACTNUM auto-detect (39.2% active cells) |

Note: Pressure model architecture is under active development and
uses a separate config (`pressure_model_config.yaml`).

## Dataset

Norne field reservoir simulation (500 LHS realizations, OPM simulator):

- **Grid**: 46 × 112 × 22 (113,344 total cells, 44,431 active — 39.2%)
- **Time steps**: 65 (0 to 3,260 days, ~9 years of operation)
- **Wells**: 36 (producers and injectors, multi-layer horizontal completions)
- **LHS variable**: PERMZ (vertical permeability / Kv/Kh ratio)
- **Input channels**: 11
  - Static: PERMX (log10), PERMZ (log10), PORO, PORV, NTG, ACTNUM
  - Coordinates: grid_x, grid_y, grid_z (normalized)
  - Dynamic: grid_t (normalized), WCID (+1 injector / -1 producer / 0 none)
- **Output variables** (separate models):
  - `pressure` — cell pressure (bar)
  - `swat` — water saturation (fraction)
  - `sgas` — gas saturation (fraction)
- **Samples**: 400 train / 50 val / 50 test

### Comparison with CO2 Dataset

| Aspect | CO2 | Norne |
|--------|-----|-------|
| Spatial dims | 2D (96 × 200) | 3D (46 × 112 × 22) |
| Total cells | 19,200 | 113,344 |
| Active cells | variable (~53%) | 44,431 (39.2%) |
| Timesteps | 24 | 65 |
| Samples | 5,500 | 500 |
| Input channels | 12 | 11 |
| Simulator | Custom | OPM |

## Usage

All commands from `neural_operator_factory/`:

### Training

```bash
# Pressure
sbatch examples/pi_norne/train.sbatch pressure_training_config

# Water saturation (SWAT)
sbatch examples/pi_norne/train.sbatch swat_training_config

# Gas saturation (SGAS)
sbatch examples/pi_norne/train.sbatch sgas_training_config
```

### Evaluation

```bash
sbatch examples/pi_norne/eval.sbatch pressure
sbatch examples/pi_norne/eval.sbatch swat
sbatch examples/pi_norne/eval.sbatch sgas

# With explicit checkpoint
CHECKPOINT=checkpoints/best_model_pressure_deeponet3d_tno_spatial.pth \
  sbatch examples/pi_norne/eval.sbatch pressure

# Pressure requires denormalization
NORMALIZE=1 sbatch examples/pi_norne/eval.sbatch pressure
```

## Results

### Water Saturation (SWAT)

| Metric | Value |
|--------|-------|
| MAE | 4.38e-3 |
| RMSE | 1.99e-2 |
| Relative L2 | 2.65% |
| R² | 0.9977 |
| Parameters | 196,161 |
| Training time | 1 hr 45 min (8× H100, 180 epochs) |

### Gas Saturation (SGAS)

| Metric | Value |
|--------|-------|
| MAE | 5.72e-3 |
| RMSE | 2.88e-2 |
| Relative L2 | 9.73% |
| R² | 0.9891 |
| Parameters | 196,161 |
| Training time | 1 hr 41 min (8× H100, 180 epochs) |

### Pressure

Under active development — results pending.

## Files

```text
pi_norne/
├── README.md
├── train.sbatch                        # 8 GPU, 4 hour time limit
├── eval.sbatch                         # 1 GPU evaluation
└── conf/
    ├── pressure_model_config.yaml      # Pressure architecture + loss (no mass conservation)
    ├── swat_model_config.yaml          # SWAT architecture + loss (mass conservation enabled)
    ├── sgas_model_config.yaml          # SGAS architecture + loss (mass conservation enabled)
    ├── pressure_training_config.yaml
    ├── swat_training_config.yaml
    └── sgas_training_config.yaml

Each variable has its own model config, allowing independent architecture tuning
for pressure vs. saturation variables.
```

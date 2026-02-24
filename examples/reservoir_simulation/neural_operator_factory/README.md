# Neural Operator Factory for Reservoir Simulation

A flexible framework for training and evaluating neural operator surrogate models for reservoir simulation, built on [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo). Supports FNO, DeepONet, and U-Net architectures on both 2D and 3D spatial datasets with full-mapping and autoregressive training regimes.

## Directory Structure

```
neural_operator_factory/
├── models/                         # Neural operator architectures
│   ├── __init__.py
│   ├── xfno.py                     # FNO variants: UFNO, UFNONet, FNO4D, FNO4DNet
│   ├── deeponet.py                 # DeepONet variants (2D/3D): 7 configurable variants
│   ├── unet.py                     # Custom UNet2D, UNet3D modules
│   └── physicsnemo_unet.py         # PhysicsNeMo UNet wrappers, StandaloneUNet
│
├── data/                           # Data loading and validation
│   ├── __init__.py
│   ├── dataloader.py               # ReservoirDataset (3D/4D), dataloaders, static masking
│   ├── validation.py               # Shape validation, dimension detection
│   └── scalar_utils.py             # MIONet scalar channel auto-detection
│
├── training/                       # Training utilities
│   ├── __init__.py
│   ├── losses.py                   # UnifiedLoss, SimpleRelativeL2Loss
│   ├── metrics.py                  # NumPy + PyTorch metrics, PhysicsNeMo imports
│   └── ar_utils.py                 # Autoregressive training (temporal bundling, rollout)
│
├── utils/                          # Utility functions
│   ├── __init__.py
│   ├── padding.py                  # Dimension-agnostic spatial padding
│   ├── co2_normalization.py        # CO2-specific denormalization helpers
│   └── co2_visualization.py        # CO2-specific plotting utilities
│
├── scripts/                        # Runnable entry points
│   ├── train.py                    # Main training script (DDP, AR, masking, MLflow)
│   ├── evaluate_norne.py           # Norne evaluation (full-mapping + AR rollout)
│   ├── evaluate_co2_pressure.py    # CO2-specific pressure evaluation
│   └── evaluate_co2_saturation.py  # CO2-specific saturation evaluation
│
├── conf/                           # Hydra configuration
│   ├── model_config.yaml           # Architecture and loss settings
│   └── training_config.yaml        # Training regime, data, masking, hyperparameters
│
├── tests/                          # Unit tests
│   ├── conftest.py
│   ├── test_xfno.py
│   ├── test_unet.py
│   ├── test_losses.py
│   ├── test_dataset.py
│   ├── test_data_validation.py
│   ├── test_padding.py
│   └── test_ar_utils.py
│
├── train.sbatch                    # Slurm submission script
├── README.md
└── requirements.txt
```

## Model Architectures

### FNO Family (`models/xfno.py`)

| Model | Description | Dimensions |
|-------|-------------|------------|
| **FNO** | Pure Fourier Neural Operator | 3D, 4D |
| **U-FNO** | FNO + U-Net skip connections | 3D only |
| **Conv-FNO** | FNO + 3D convolutions | 3D only |
| **FNO4D** | 4D FNO (3D spatial + time) | 4D only |

All FNO wrappers support flexible `target_times` for autoregressive temporal bundling.

### DeepONet Family (`models/deeponet.py`)

| Variant | Description |
|---------|-------------|
| `deeponet` | Basic DeepONet (MLP branch) |
| `u_deeponet` | U-Net enhanced spatial branch |
| `fourier_deeponet` | Fourier layers in spatial branch |
| `conv_deeponet` | Convolutional spatial branch |
| `hybrid_deeponet` | Fourier + U-Net + Conv combination |
| `mionet` | Multi-input operator network (2 branches) |
| `fourier_mionet` | MIONet with Fourier layers |

Both 2D spatial (`DeepONetWrapper`) and 3D spatial (`DeepONet3DWrapper`) versions are provided. All wrappers support `target_times` for autoregressive temporal bundling with flexible K (output window). MIONet variants auto-detect scalar input channels for branch2; if none are found, branch2 is disabled and the model runs as single-branch.

### U-Net Baselines (`models/unet.py`, `models/physicsnemo_unet.py`)

- Custom `UNet2D` / `UNet3D` with 3-level encoder-decoder
- PhysicsNeMo `StandaloneUNet` wrapper for baseline comparison

## Training Regimes

### Full Mapping

Predicts the entire trajectory in a single forward pass. The model receives all T timesteps and outputs all T timesteps at once.

```yaml
training:
  regime: full_mapping
  epochs: 200
```

### Autoregressive (with Temporal Bundling)

Predicts K timesteps from L context timesteps, with two-phase training:

1. **Teacher Forcing** — model sees ground-truth inputs (learns the physics)
2. **Rollout** — model sees its own predictions (learns self-correction)

Random starting points are sampled each iteration for memory efficiency and uniform trajectory coverage.

```yaml
training:
  regime: autoregressive
  autoregressive:
    input_window: 1       # L: context timesteps
    output_window: 3      # K: predicted timesteps per step
    teacher_forcing_epochs: 140
    rollout_epochs: 60
    max_rollout_steps: 4
    gradient_checkpointing: true
```

## Dataset Support

The `ReservoirDataset` class supports:

- **3D data**: `(N, H, W, T, C)` input, `(N, H, W, T)` output (e.g., CO2 sequestration)
- **4D data**: `(N, X, Y, Z, T, C)` input, `(N, X, Y, Z, T)` output (e.g., Norne field)
- Automatic dimension detection and validation
- Z-score normalization with distributed broadcast
- Flexible file naming patterns (`{mode}` placeholder for train/val/test)

### Spatial Masking

Inactive reservoir cells (outside the geological model) can be automatically excluded from loss and metrics. The ACTNUM channel is auto-detected by its unique binary/static/cross-sample fingerprint — no channel index configuration needed.

```yaml
data:
  mask:
    enabled: true    # Auto-detects ACTNUM, masks inactive cells
```

### MIONet Scalar Detection

For MIONet variants, input channels are automatically classified as spatial (varying across the grid) or scalar (constant per realization). Scalar channels are routed to branch2; spatial channels go to branch1. If no scalar channels are found, branch2 is disabled and the model runs as single-branch.

## Quick Start

### Training

1. Configure `conf/training_config.yaml` (data paths, regime, masking)
2. Configure `conf/model_config.yaml` (architecture, dimensions, loss)
3. Submit:
   ```bash
   sbatch train.sbatch
   ```

### Evaluation

```bash
# Norne — autoregressive rollout with ACTNUM masking
python scripts/evaluate_norne.py --mode autoregressive --L 1 --K 3 --mask

# Norne — full-mapping
python scripts/evaluate_norne.py --mode full_mapping

# Norne — pressure (with denormalization)
python scripts/evaluate_norne.py --mode autoregressive --L 1 --K 3 --mask --normalize \
    --output_file 'norne_{mode}_pressure.pt' --variable PRESSURE

# CO2-specific evaluation
python scripts/evaluate_co2_pressure.py --checkpoint checkpoints/best_model_pressure_*.pth
python scripts/evaluate_co2_saturation.py --checkpoint checkpoints/best_model_saturation_*.pth
```

### Testing

```bash
pytest tests/ -v
```

## Loss Functions

| Loss | Description |
|------|-------------|
| `mse` | Mean Squared Error |
| `l1` | Mean Absolute Error |
| `relative_l2` | Scale-invariant relative L2 |
| `huber` | Smooth L1 (Huber) |

Optional: spatial masking (ACTNUM), physics-informed derivative constraints.

## Evaluation Metrics

NumPy-based (post-processing): MAE, RMSE, MRE, MPE, R-squared, PSNR, Relative L1/L2, Normalized MSE

PyTorch-based (training): `mse_torch`, `rmse_torch`, `mae_torch`, `relative_l2_torch`, `r2_score_torch`, `psnr_torch`

PhysicsNeMo imports: `mse`, `rmse`, `Mean`, `Variance`, `WeightedMean`, `WeightedVariance`

## References

1. Wen, G. et al. (2022). "U-FNO—An enhanced Fourier neural operator-based deep-learning model for multiphase flow." *Advances in Water Resources*, 163, 104180.
2. Li, Z. et al. (2021). "Fourier Neural Operator for Parametric Partial Differential Equations." *ICLR 2021*.
3. Lu, L. et al. (2021). "Learning nonlinear operators via DeepONet." *Nature Machine Intelligence*, 3, 218-229.

## License

Apache License 2.0. See [LICENSE](../../../LICENSE.txt) for details.

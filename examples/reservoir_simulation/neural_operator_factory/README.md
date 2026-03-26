# Neural Operator Factory for Reservoir Simulation

A flexible, config-driven framework for training neural operator
surrogate models for reservoir simulation, built on
[PhysicsNeMo](https://github.com/NVIDIA/physicsnemo).
Train FNO, DeepONet, and U-Net architectures on both 2D and 3D
spatial datasets with a unified training pipeline, physics-informed
losses, and autoregressive temporal rollout.

## Key Features

- **Multiple architectures from one config**: switch between FNO,
  U-FNO, Conv-FNO, FNO4D, DeepONet (7 variants), TNO, and U-Net
  baselines by changing a YAML file.
- **2D and 4D support**: handles `(N, H, W, T, C)` and
  `(N, X, Y, Z, T, C)` datasets with automatic dimension detection.
- **Autoregressive training**: three-stage pipeline (teacher
  forcing, pushforward with curriculum, free-running rollout) for
  temporal problems.
- **Physics-informed losses**: spatial derivative regularization
  and weak mass conservation constraints alongside standard data
  losses.
- **Per-sample domain masking**: correctly excludes inactive
  cells from loss and metrics, with auto-detection or explicit
  configuration.
- **Multi-GPU DDP**: distributed training with SLURM batch scripts
  out of the box.
- **366 unit tests**: comprehensive test coverage for models,
  losses, metrics, padding, autoregressive utilities, and
  checkpointing.

## Directory Structure

```text
neural_operator_factory/
├── models/                        # Neural operator architectures
│   ├── xfno.py                    # FNO, U-FNO, Conv-FNO, FNO4D
│   ├── xdeeponet.py              # DeepONet variants (2D/3D), TNO
│   ├── unet.py                   # Custom UNet2D, UNet3D
│   └── physicsnemo_unet.py       # PhysicsNeMo UNet wrappers
│
├── data/                          # Data loading and validation
│   ├── dataloader.py              # ReservoirDataset, dataloaders
│   ├── validation.py              # Shape validation, dim detection
│   └── scalar_utils.py           # MIONet scalar channel detection
│
├── training/                      # Training utilities
│   ├── losses.py                  # UnifiedLoss (data + derivative)
│   ├── physics_losses.py          # Mass conservation loss
│   ├── ar_utils.py                # Autoregressive training helpers
│   └── metrics.py                 # NumPy + PyTorch metrics
│
├── utils/                         # Utility functions
│   ├── checkpoint.py              # Model save/load/reconstruct
│   ├── padding.py                 # Dimension-agnostic padding
│   ├── co2_normalization.py       # CO2-specific denormalization
│   └── co2_visualization.py       # CO2-specific plotting
│
├── scripts/                       # Entry points
│   ├── train.py                   # Training (DDP, AMP, Hydra)
│   └── evaluate_norne.py          # Norne test-set evaluation
│
├── conf/                          # Base Hydra configuration
│   ├── model_config.yaml          # Architecture and loss settings
│   └── training_config.yaml       # Training hyperparameters
│
├── examples/                      # Reproducible experiments
│   └── ufno_co2/                  # U-FNO paper reproduction
│
├── tests/                         # Unit tests (366 tests)
├── train.sbatch                   # SLURM training script
├── eval_norne.sbatch              # SLURM Norne evaluation
├── requirements.txt
└── README.md
```

## Dataset Requirements

The NOF expects input and output tensors in `.pt` format with a
specific layout.  The **last input channels** must follow a
fixed convention for derivative losses and DeepONet trunk queries
to work correctly.

### 2D Spatial Problems (`dimensions: 3d`)

**Input**: `(N, H, W, T, C)` — Output: `(N, H, W, T)`

| Position | Content | Required by |
|----------|---------|-------------|
| `0` to `C-4` | Feature channels | All models |
| `C-3` | **grid_x**: W-direction widths | Derivative (`dx`) |
| `C-2` | **grid_y**: H-direction widths | Derivative (`dy`) |
| `C-1` | **grid_t**: time coordinate | DeepONet trunk |

### 3D Spatial Problems (`dimensions: 4d`)

**Input**: `(N, X, Y, Z, T, C)` — Output: `(N, X, Y, Z, T)`

| Channel position | Content | Required by |
|-----------------|---------|-------------|
| `0` to `C-5` | Feature channels | All models |
| `C-4` | **grid_x**: cell widths in X direction | Derivative loss (`dx`) |
| `C-3` | **grid_y**: cell widths in Y direction | Derivative loss (`dy`) |
| `C-2` | **grid_z**: cell widths in Z direction | Derivative loss (`dz`) |
| `C-1` | **grid_t**: time coordinate | DeepONet trunk |

### Masking Convention

Inactive cells (outside the reservoir) must be **zero in all
input channels and in the output**.  The NOF auto-detects the
mask channel using this priority:

1. **Explicit config**: `mask_channel: 5` in the training config
2. **ACTNUM auto-detect**: binary {0,1} channel, static across
   time, whose zeros coincide with zero-output cells
3. **Non-zero channel**: any channel with a static zero pattern
   matching the output's inactive cells
4. **No mask**: all cells treated as active

When the mask varies across samples (e.g., CO2 dataset with
variable reservoir thickness), per-sample masks are constructed
at batch time.  Loss functions select only active cells per
sample for norm computation.

### File Naming

The dataset loader supports flexible file naming with
`{mode}` placeholders:

```yaml
data:
  data_path: /path/to/data
  input_file: norne_{mode}_a.pt     # {mode} = train/val/test
  output_file: norne_{mode}_swat.pt
```

Or CO2 convention (auto-detected from `variable`):

```yaml
data:
  data_path: /path/to/data
  variable: pressure   # resolves to dP_{mode}_a.pt / dP_{mode}_u.pt
```

## Model Architectures

### xFNO Family (`models/xfno.py`)

| Model | Layers | Dimensions | Parameters |
|-------|--------|------------|------------|
| **FNO** | Fourier only | 3D, 4D | ~31M |
| **U-FNO** | Fourier + U-Net skip | 3D only | ~33M |
| **Conv-FNO** | Fourier + Conv3d skip | 3D only | ~31M |
| **FNO4D** | SpectralConv4d | 4D only | configurable |

Architecture: lifting → [Fourier layers] →
[U-Fourier / Conv-Fourier layers] → decoder.
Configurable lifting (MLP/Conv), decoder depth,
Fourier modes per dimension, and activation function.

### xDeepONet Family (`models/xdeeponet.py`)

| Variant | Description |
|---------|-------------|
| `deeponet` | Basic DeepONet (MLP branch) |
| `u_deeponet` | U-Net enhanced spatial branch |
| `fourier_deeponet` | Fourier layers in spatial branch |
| `conv_deeponet` | Convolutional spatial branch |
| `hybrid_deeponet` | Fourier + U-Net + Conv combination |
| `mionet` | Multi-input operator network (2 branches) |
| `fourier_mionet` | MIONet with Fourier layers |
| `tno` | Temporal Neural Operator (branch2 = previous solution) |

Both 2D (`DeepONetWrapper`) and 3D (`DeepONet3DWrapper`)
versions are provided.  The **TNO** variant enables
autoregressive temporal predictions where branch2 receives
the model's own previous output as feedback.

Branch networks support configurable combinations of Fourier,
U-Net, and Conv layers.  The trunk network is a sinusoidal
MLP encoding time or full spatiotemporal coordinates.

### U-Net Baselines (`models/unet.py`, `models/physicsnemo_unet.py`)

- Custom `UNet2D` / `UNet3D` with 3-level encoder-decoder
- `PhysicsNemoUNet2D` / `PhysicsNemoUNet3D` wrappers around
  PhysicsNeMo's native 3D U-Net
- `StandaloneUNet` for standalone baseline comparison

## Training

### Configuration

All training is driven by two Hydra config files:

- `model_config.yaml`: architecture, loss function, derivative
  and physics loss settings
- `training_config.yaml`: data paths, batch size, epochs,
  optimizer, scheduler, logging, training regime

### Training Regimes

**Full-mapping** (`regime: full_mapping`): predict the entire
spatiotemporal trajectory in a single forward pass.

**Autoregressive** (`regime: autoregressive`): predict K
timesteps from L context timesteps using a three-stage pipeline:

| Stage | Description | Gradient |
|-------|-------------|----------|
| **Teacher forcing** | GT input at each step | Per-window |
| **Pushforward** | Live-gradient chains; curriculum | Through chain |
| **Rollout** | Free-running with detached feedback | Per-window |

Configurable parameters: `input_window` (L),
`output_window` (K), noise injection, feedback channel,
gradient checkpointing, LR reset at stage transitions.

### Running Training

```bash
# SLURM (8-GPU DDP)
sbatch train.sbatch

# Or with example configs
sbatch examples/ufno_co2/train.sbatch U-FNO saturation_training_config
```

## Loss Functions

The `UnifiedLoss` (`training/losses.py`) combines three
components:

### Data Losses

| Loss | Formula | Use case |
|------|---------|----------|
| `mse` | mean((pred - target)²) | Standard regression |
| `l1` | mean(\|pred - target\|) | Robust to outliers |
| `relative_l2` | \|\|pred - target\|\|₂ / \|\|target\|\|₂ | Scale-invariant |
| `huber` | Smooth L1 with delta threshold | MSE precision + L1 robustness |

Multiple losses can be combined with configurable weights.
When masking is active, only active cells are selected
per-sample before computing norms.

### Spatial Derivative Regularization

Penalizes differences in spatial gradients between prediction
and target using central finite differences on the grid-width
input channels:

```yaml
loss:
  derivative:
    enabled: true
    weight: 0.5
    dims: [dx]          # 2D: [dx, dy], 3D: [dx, dy, dz]
    metric: null         # inherits first data loss type
```

### Physics-Informed Losses

```yaml
loss:
  physics:
    mass_conservation:
      enabled: true
      weight: 0.5
      use_cell_volumes: true
```

Weak mass conservation penalizes discrepancies in
spatially-integrated quantities between prediction and
ground truth at each timestep.  Supports cell-volume
weighting from grid-width channels.

## Evaluation Metrics

**NumPy-based** (post-processing): MRE, MPE, MAE, R², PSNR,
Relative L1/L2, Normalized MSE

**PyTorch-based** (training): `mse_torch`, `rmse_torch`,
`mae_torch`, `relative_l2_torch`, `r2_score_torch`, `psnr_torch`

**PhysicsNeMo imports**: `mse`, `rmse`, `Mean`, `Variance`,
`WeightedMean`, `WeightedVariance`

## Examples

Self-contained experiments with their own configs and
SLURM scripts.  See [examples/README.md](examples/README.md).

| Example | Description | Dataset |
|---------|-------------|---------|
| [ufno_co2](examples/ufno_co2/) | U-FNO paper (Wen et al. 2022) | CO2 sequestration |

## Testing

```bash
cd examples/reservoir_simulation/neural_operator_factory
pytest tests/ -v
```

366 tests covering models, losses, metrics, padding,
autoregressive utilities, checkpointing, data validation,
and scalar detection.

## References

1. Wen, G. et al. (2022). "U-FNO — An enhanced Fourier neural
   operator-based deep-learning model for multiphase flow."
   *Advances in Water Resources*, 163, 104180.
2. Li, Z. et al. (2021). "Fourier Neural Operator for Parametric
   Partial Differential Equations." *ICLR 2021*.
3. Lu, L. et al. (2021). "Learning nonlinear operators via
   DeepONet." *Nature Machine Intelligence*, 3, 218-229.

## License

Apache License 2.0. See [LICENSE](../../../LICENSE.txt).

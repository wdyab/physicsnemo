# Neural Operator Factory for Reservoir Simulation

A flexible framework for training and evaluating neural operator surrogate models for reservoir simulation, built on [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo). Supports FNO, DeepONet, and U-Net architectures on both 2D and 3D spatial datasets.

## Directory Structure

```
neural_operator_factory/
├── models/                         # Neural operator architectures
│   ├── __init__.py                 # Exports all model classes
│   ├── xfno.py                     # FNO variants: UFNO, UFNONet, FNO4D, FNO4DNet
│   ├── deeponet.py                 # DeepONet variants (2D/3D): 7 configurable variants
│   ├── unet.py                     # Custom UNet2D, UNet3D modules
│   └── physicsnemo_unet.py         # PhysicsNeMo UNet wrappers, StandaloneUNet
│
├── data/                           # Data loading and validation
│   ├── __init__.py                 # Exports dataset and validation utilities
│   ├── dataloader.py               # ReservoirDataset (3D/4D), dataloaders
│   ├── validation.py               # Shape validation, dimension detection
│   └── scalar_utils.py             # MIONet scalar channel detection
│
├── training/                       # Loss functions and evaluation metrics
│   ├── __init__.py                 # Exports losses and metrics
│   ├── losses.py                   # UnifiedLoss, SimpleRelativeL2Loss
│   └── metrics.py                  # NumPy + PyTorch metrics, PhysicsNeMo imports
│
├── utils/                          # Utility functions
│   ├── __init__.py                 # Exports normalization and visualization
│   ├── normalization.py            # CO2 dataset denormalization helpers
│   └── visualization.py            # Plotting utilities
│
├── scripts/                        # Runnable entry points
│   ├── train.py                    # Main training script (DDP, AMP, MLflow)
│   ├── evaluate_pressure.py        # Pressure model evaluation
│   └── evaluate_saturation.py      # Saturation model evaluation
│
├── conf/                           # Hydra configuration
│   ├── model_config.yaml           # Architecture and loss settings
│   └── training_config.yaml        # Training hyperparameters and data paths
│
├── tests/                          # Unit tests
│   ├── conftest.py                 # Shared fixtures
│   ├── test_xfno.py               # FNO model tests
│   ├── test_unet.py               # UNet model tests
│   ├── test_losses.py             # Loss function tests
│   ├── test_dataset.py            # Dataset and dataloader tests
│   └── test_data_validation.py    # Data validation tests
│
├── docs/                           # Documentation
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

Both 2D spatial (`DeepONet`, `DeepONetWrapper`) and 3D spatial (`DeepONet3D`, `DeepONet3DWrapper`) versions are provided.

### U-Net Baselines (`models/unet.py`, `models/physicsnemo_unet.py`)

- Custom `UNet2D` / `UNet3D` with 3-level encoder-decoder
- PhysicsNeMo `StandaloneUNet` wrapper for baseline comparison

## Dataset Support

The `ReservoirDataset` class (`data/dataloader.py`) supports:

- **3D data**: `(N, H, W, T, C)` input, `(N, H, W, T)` output (e.g., CO2 sequestration)
- **4D data**: `(N, X, Y, Z, T, C)` input, `(N, X, Y, Z, T)` output (e.g., Norne field)
- Automatic dimension detection and validation
- Flexible file naming patterns
- Distributed data loading with normalization sharing

## Quick Start

### Training

1. Configure your data path in `conf/training_config.yaml`:
   ```yaml
   data:
     data_path: /path/to/your/data
     variable: pressure  # or 'saturation'
   ```

2. Select model and dimensions in `conf/model_config.yaml`:
   ```yaml
   arch:
     dimensions: 3d        # '3d' or '4d'
     model: xfno           # 'xfno' or 'xdeeponet'
   ```

3. Run training:
   ```bash
   # Single GPU
   python scripts/train.py

   # Multi-GPU (DDP)
   torchrun --nproc_per_node=4 scripts/train.py
   ```

### Evaluation

```bash
python scripts/evaluate_pressure.py --checkpoint checkpoints/best_model_pressure_*.pth
python scripts/evaluate_saturation.py --checkpoint checkpoints/best_model_saturation_*.pth
```

### Testing

```bash
cd examples/reservoir_simulation/neural_operator_factory
pytest tests/ -v
```

## Loss Functions

| Loss | Description |
|------|-------------|
| `mse` | Mean Squared Error |
| `l1` | Mean Absolute Error |
| `relative_l2` | Scale-invariant relative L2 |
| `huber` | Smooth L1 (Huber) |

Optional features: domain masking, physics-informed spatial derivative constraints.

## Evaluation Metrics

NumPy-based (post-processing): MRE, MPE, MAE, R², PSNR, Relative L1/L2, Normalized MSE

PyTorch-based (training): `mse_torch`, `rmse_torch`, `mae_torch`, `relative_l2_torch`, `r2_score_torch`, `psnr_torch`

PhysicsNeMo imports: `mse`, `rmse`, `Mean`, `Variance`, `WeightedMean`, `WeightedVariance`

## References

1. Wen, G. et al. (2022). "U-FNO—An enhanced Fourier neural operator-based deep-learning model for multiphase flow." *Advances in Water Resources*, 163, 104180.
2. Li, Z. et al. (2021). "Fourier Neural Operator for Parametric Partial Differential Equations." *ICLR 2021*.
3. Lu, L. et al. (2021). "Learning nonlinear operators via DeepONet." *Nature Machine Intelligence*, 3, 218-229.

## License

Apache License 2.0. See [LICENSE](../../../LICENSE.txt) for details.

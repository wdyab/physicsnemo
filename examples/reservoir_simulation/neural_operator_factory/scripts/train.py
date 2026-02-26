#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training script for neural operator reservoir simulation models."""

import sys
from pathlib import Path

# Add parent directory (neural_operator_factory/) to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig
from pathlib import Path
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import mlflow
import mlflow.pytorch

from models.xfno import UFNONet, FNO4DNet
from models.physicsnemo_unet import StandaloneUNet
from models.deeponet import DeepONetWrapper, DeepONet3DWrapper

def print_model_architecture(model, model_type: str, dimensions: str, cfg, logger):
    """Print detailed model architecture for any model type."""
    logger.info("=" * 80)
    logger.info("MODEL ARCHITECTURE")
    logger.info("=" * 80)
    
    # Get the actual model (unwrap DDP if needed)
    if hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model
    
    # Print model type and dimensions
    logger.info(f"Dimensions: {dimensions.upper()}")
    logger.info(f"Model Type: {model_type.upper()}")
    
    if model_type == "xdeeponet":
        variant = cfg.arch.xdeeponet.get("variant", "u_deeponet")
        logger.info(f"Variant: {variant}")
        logger.info("")
        
        # Branch configuration
        branch1_cfg = cfg.arch.xdeeponet.get("branch1", {})
        logger.info(f"Branch 1:")
        logger.info(f"  Type: {branch1_cfg.get('type', 'spatial')}")
        logger.info(f"  In Channels: auto (inferred from input tensor)")
        logger.info(f"  Fourier Layers: {branch1_cfg.get('num_fourier_layers', 0)}")
        logger.info(f"  UNet Layers: {branch1_cfg.get('num_unet_layers', 0)}")
        logger.info(f"  Conv Layers: {branch1_cfg.get('num_conv_layers', 0)}")
        logger.info(f"  Activation: {branch1_cfg.get('activation_fn', 'sin')}")
        
        if variant in ['mionet', 'fourier_mionet']:
            branch2_cfg = cfg.arch.xdeeponet.get("branch2", {})
            logger.info(f"Branch 2:")
            logger.info(f"  Type: {branch2_cfg.get('type', 'mlp')}")
            logger.info(f"  In Features: auto (inferred from input)")
            logger.info(f"  Activation: {branch2_cfg.get('activation_fn', 'relu')}")
        
        # Trunk configuration
        trunk_cfg = cfg.arch.xdeeponet.get("trunk", {})
        trunk_input = trunk_cfg.get('input_type', 'time')
        in_features = (4 if dimensions == '4d' else 3) if trunk_input == 'grid' else 1
        coord_desc = 'x,y,z,t' if dimensions == '4d' else 'x,y,t'
        logger.info(f"Trunk:")
        logger.info(f"  Input Type: {trunk_input} ({coord_desc if trunk_input == 'grid' else 'just t'})")
        logger.info(f"  In Features: {in_features}")
        logger.info(f"  Hidden Width: {trunk_cfg.get('hidden_width', 128)}")
        logger.info(f"  Num Layers: {trunk_cfg.get('num_layers', 6)}")
        logger.info(f"  Activation: {trunk_cfg.get('activation_fn', 'sin')}")
        
        # Decoder configuration
        logger.info(f"Decoder:")
        logger.info(f"  Type: {cfg.arch.xdeeponet.get('decoder_type', 'mlp')}")
        logger.info(f"  Width: {cfg.arch.xdeeponet.get('decoder_width', 128)}")
        logger.info(f"  Layers: {cfg.arch.xdeeponet.get('decoder_layers', 2)}")
        logger.info(f"  Activation: {cfg.arch.xdeeponet.get('decoder_activation_fn', 'relu')}")
        
        logger.info(f"Latent Width: {cfg.arch.xdeeponet.get('width', 64)}")
        logger.info(f"Padding: {cfg.arch.xdeeponet.get('padding', 8)}")
        
    elif model_type == "xfno":
        xfno_cfg = cfg.arch.xfno
        logger.info(f"Out Channels: {xfno_cfg.out_channels}")
        logger.info(f"Width: {xfno_cfg.width}")
        if dimensions == '4d':
            logger.info(f"Modes: ({xfno_cfg.modes1}, {xfno_cfg.modes2}, {xfno_cfg.modes3}, {xfno_cfg.modes4})")
        else:
            logger.info(f"Modes: ({xfno_cfg.modes1}, {xfno_cfg.modes2}, {xfno_cfg.modes3})")
        logger.info(f"FNO Layers: {xfno_cfg.num_fno_layers}")
        if dimensions == '3d':
            logger.info(f"U-Net Layers: {xfno_cfg.num_unet_layers}")
            logger.info(f"Conv Layers: {xfno_cfg.num_conv_layers}")
            logger.info(f"Lifting: type={xfno_cfg.lifting_type}, layers={xfno_cfg.lifting_layers}")
        else:
            logger.info(f"Coord Features: {xfno_cfg.coord_features}")
        logger.info(f"Activation: {xfno_cfg.activation_fn}")
        logger.info(f"Decoder: layers={xfno_cfg.decoder_layers}, width={xfno_cfg.decoder_width}")
    
    # Print full model structure
    logger.info("")
    logger.info("Full Model Structure:")
    logger.info("-" * 80)
    for line in str(actual_model).split('\n'):
        logger.info(line)
    logger.info("-" * 80)
    
    # Count parameters per component
    logger.info("")
    logger.info("Parameter Counts:")
    total_params = 0
    for name, module in actual_model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += params
        logger.info(f"  {name}: {params:,} parameters")
    logger.info(f"  TOTAL: {total_params:,} parameters")
    logger.info("=" * 80)


from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, LaunchLogger

from data.dataloader import create_dataloaders
from training.losses import get_loss_function, UnifiedLoss
from training.metrics import (
    mean_relative_error,
    mean_plume_error,
    mean_absolute_error,
    compute_relative_l2_error,
)
from utils.co2_normalization import dnorm_dP
from data.validation import validate_batch_dimensions, print_validation_summary
from training.ar_utils import (
    teacher_forcing_step,
    rollout_step,
    ar_validate_full_rollout,
)

# Registry of denormalization functions that can be selected via config.
_DENORM_REGISTRY = {
    "dnorm_dP": dnorm_dP,
}

# Registry of validation metric functions (numpy-based, operate on flat arrays).
def _rmse_np(y_pred, y_true):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

_METRIC_REGISTRY = {
    "rmse": ("RMSE", "val_rmse", _rmse_np),
    "mae": ("MAE", "val_mae", mean_absolute_error),
    "mre": ("MRE", "val_mre", mean_relative_error),
    "mpe": ("MPE", "val_mpe", mean_plume_error),
    "relative_l2": ("RelL2", "val_relative_l2", compute_relative_l2_error),
}


@hydra.main(version_base="1.3", config_path="../conf", config_name="training_config")
def main(cfg: DictConfig) -> None:
    """Main training function for neural operator reservoir simulation models."""

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Helper variable for MLFlow logging (only on rank 0)
    use_mlflow = cfg.logging.use_mlflow and dist.rank == 0

    # Set random seeds for reproducibility
    if hasattr(cfg, "seed"):
        import random
        import numpy as np

        seed = cfg.seed + dist.rank  # Different seed per rank for data augmentation
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Set deterministic behavior if requested
        if cfg.compute.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        elif cfg.compute.benchmark:
            torch.backends.cudnn.benchmark = True

    # Initialize logger
    logger = PythonLogger(name="nof_train")
    logger.file_logging()
    LaunchLogger.initialize()

    # Print header (only on rank 0)
    if dist.rank == 0:
        dimensions = cfg.arch.dimensions.lower()
        model_type = cfg.arch.model.lower()
        if model_type == "xdeeponet":
            model_name = cfg.arch.xdeeponet.variant.replace("_", "-").upper()
        else:
            model_name = model_type.upper()
        logger.info("=" * 80)
        logger.info(
            f"{model_name} ({dimensions.upper()}) Training | Variable: {cfg.data.variable} | GPUs: {dist.world_size}"
        )
        logger.info("=" * 80)

    # Auto-set num_workers based on num_gpus if not specified
    num_workers = cfg.data.num_workers
    if num_workers is None and hasattr(cfg.compute, "num_gpus"):
        num_workers = cfg.compute.num_gpus * 2  # 2 workers per GPU
    elif num_workers is None:
        num_workers = 4  # Default fallback

    # Get dimensions from config (used for model selection and data validation)
    expected_dimensions = cfg.arch.dimensions.lower()
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=cfg.data.data_path,
        batch_size=cfg.training.batch_size,
        normalize=cfg.data.normalize,
        num_workers=num_workers,
        device=dist.device,
        input_file=cfg.data.get("input_file", None),
        output_file=cfg.data.get("output_file", None),
        variable=cfg.data.get("variable", None),
        expected_dimensions=expected_dimensions,
        use_mask=cfg.data.get("mask_enabled", False),
    )

    # Get static mask (move to GPU if available)
    static_mask = train_loader.dataset.get_static_mask()
    if static_mask is not None:
        static_mask = static_mask.to(dist.device)

    # Print data info (only on rank 0)
    if dist.rank == 0:
        effective_batch_size = cfg.training.batch_size * dist.world_size
        logger.info(
            f"Data: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)} | Batch size={cfg.training.batch_size} per GPU (Effective: {effective_batch_size})"
        )

    # Validate data dimensions against config
    if dist.rank == 0:
        logger.info("Validating data dimensions...")

        # Get a sample batch to check dimensions
        sample_inputs, sample_targets = next(iter(train_loader))

        # Validate using centralized validation function
        validation_info = validate_batch_dimensions(sample_inputs, sample_targets, cfg.data.get("variable", "unknown"))
        detected_dimensions = validation_info["dimensions"]
        
        # Check that detected dimensions match config
        if detected_dimensions != expected_dimensions:
            raise ValueError(
                f"❌ Dimension mismatch! Config specifies '{expected_dimensions}' but data is '{detected_dimensions}'.\n"
                f"   Config: arch.dimensions = {expected_dimensions}\n"
                f"   Data: Input shape {tuple(sample_inputs.shape)} → {detected_dimensions}\n"
                f"   Please update arch.dimensions in config to match your dataset."
            )

        # Print validation summary
        print_validation_summary(
            input_shape=tuple(sample_inputs.shape),
            target_shape=tuple(sample_targets.shape),
            variable=cfg.data.get("variable", "unknown"),
            is_batch=True,
            logger=logger,
        )

    # Create model based on dimensions and model type
    dimensions = cfg.arch.dimensions.lower()
    model_type = cfg.arch.model.lower()
    
    # Get in_channels from first batch (for auto-discovery)
    sample_inputs, _ = next(iter(train_loader))
    in_channels = sample_inputs.shape[-1]  # Last dimension is channels

    if model_type == "xfno":
        xfno_cfg = cfg.arch.xfno
        
        if dimensions == "4d":
            # 4D FNO (3D spatial + time) - Pure FNO only
            logger.info(
                f"Creating FNO4D model (FNO layers: {xfno_cfg.num_fno_layers}, "
                f"modes: [{xfno_cfg.modes1}, {xfno_cfg.modes2}, {xfno_cfg.modes3}, {xfno_cfg.modes4}])"
            )
            model = FNO4DNet(
                in_channels=in_channels,
                out_channels=xfno_cfg.out_channels,
                width=xfno_cfg.width,
                modes1=xfno_cfg.modes1,
                modes2=xfno_cfg.modes2,
                modes3=xfno_cfg.modes3,
                modes4=xfno_cfg.modes4,
                num_fno_layers=xfno_cfg.num_fno_layers,
                padding=xfno_cfg.padding,
                activation_fn=xfno_cfg.activation_fn,
                lifting_layers=xfno_cfg.lifting_layers,
                decoder_layers=xfno_cfg.decoder_layers,
                decoder_width=xfno_cfg.decoder_width,
                coord_features=xfno_cfg.coord_features,
            ).to(dist.device)
            model_arch_name = "fno4d"
        else:
            # 3D FNO (2D spatial + time) - With optional U-Net/Conv
            num_unet = xfno_cfg.num_unet_layers
            num_conv = xfno_cfg.num_conv_layers

            if num_unet > 0 and num_conv > 0:
                logger.warning("⚠️  Using both U-Net and Conv layers (Conv-U-FNO).")
                model_arch_name = f"convufno_{xfno_cfg.unet_type}"
            elif num_unet > 0:
                model_arch_name = f"ufno_{xfno_cfg.unet_type}"
            elif num_conv > 0:
                model_arch_name = "convfno"
            else:
                model_arch_name = "fno"

            logger.info(
                f"Creating {model_arch_name.upper()} model (FNO: {xfno_cfg.num_fno_layers}, "
                f"U-Net: {num_unet}, Conv: {num_conv})"
            )

            model = UFNONet(
                in_channels=in_channels,
                out_channels=xfno_cfg.out_channels,
                width=xfno_cfg.width,
                modes1=xfno_cfg.modes1,
                modes2=xfno_cfg.modes2,
                modes3=xfno_cfg.modes3,
                num_fno_layers=xfno_cfg.num_fno_layers,
                num_unet_layers=num_unet,
                num_conv_layers=num_conv,
                padding=xfno_cfg.padding,
                conv_kernel_size=xfno_cfg.conv_kernel_size,
                unet_kernel_size=xfno_cfg.unet_kernel_size,
                unet_dropout=xfno_cfg.unet_dropout,
                unet_type=xfno_cfg.unet_type,
                activation_fn=xfno_cfg.activation_fn,
                lifting_type=xfno_cfg.lifting_type,
                lifting_layers=xfno_cfg.lifting_layers,
                lifting_width=xfno_cfg.lifting_width,
                decoder_type=xfno_cfg.decoder_type,
                decoder_layers=xfno_cfg.decoder_layers,
                decoder_width=xfno_cfg.decoder_width,
                decoder_activation_fn=xfno_cfg.get("decoder_activation_fn", None),
            ).to(dist.device)

    elif model_type == "xdeeponet":
        xdeeponet_cfg = cfg.arch.xdeeponet
        variant = xdeeponet_cfg.variant
        
        # Build branch configs from yaml
        branch1_config = dict(xdeeponet_cfg.branch1)
        branch2_config = dict(xdeeponet_cfg.branch2) if variant in ['mionet', 'fourier_mionet'] else None
        trunk_config = dict(xdeeponet_cfg.trunk)
        
        if dimensions == "4d":
            # 4D DeepONet (3D spatial + time)
            logger.info(
                f"Creating DeepONet3D model (variant: {variant}, "
                f"branch1: {branch1_config.get('type', 'spatial')}, width: {xdeeponet_cfg.width})"
            )
            model = DeepONet3DWrapper(
                padding=xdeeponet_cfg.padding,
                variant=variant,
                width=xdeeponet_cfg.width,
                branch1_config=branch1_config,
                branch2_config=branch2_config,
                trunk_config=trunk_config,
                decoder_type=xdeeponet_cfg.get("decoder_type", "mlp"),
                decoder_width=xdeeponet_cfg.decoder_width,
                decoder_layers=xdeeponet_cfg.decoder_layers,
                decoder_activation_fn=xdeeponet_cfg.get("decoder_activation_fn", "relu"),
            ).to(dist.device)
            model_arch_name = f"deeponet3d_{variant}_{branch1_config.get('type', 'spatial')}"
        else:
            # 3D DeepONet (2D spatial + time)
            logger.info(
                f"Creating DeepONet model (variant: {variant}, "
                f"branch1: {branch1_config.get('type', 'spatial')}, width: {xdeeponet_cfg.width})"
            )
            model = DeepONetWrapper(
                padding=xdeeponet_cfg.padding,
                variant=variant,
                width=xdeeponet_cfg.width,
                branch1_config=branch1_config,
                branch2_config=branch2_config,
                trunk_config=trunk_config,
                decoder_type=xdeeponet_cfg.get("decoder_type", "mlp"),
                decoder_width=xdeeponet_cfg.decoder_width,
                decoder_layers=xdeeponet_cfg.decoder_layers,
                decoder_activation_fn=xdeeponet_cfg.get("decoder_activation_fn", "relu"),
            ).to(dist.device)
            model_arch_name = f"deeponet_{variant}_{branch1_config.get('type', 'spatial')}"

    else:
        raise ValueError(f"Unknown model: {model_type}. Use 'xfno' or 'xdeeponet'.")

    # Initialize lazy modules with a dummy forward pass (required for DDP)
    # This is needed because nn.LazyLinear doesn't know its input size until first forward
    if dist.rank == 0:
        logger.info("Initializing model with dummy forward pass...")
    with torch.no_grad():
        dummy_batch = next(iter(train_loader))
        dummy_input = dummy_batch[0].to(dist.device)
        _ = model(dummy_input)
    if dist.rank == 0:
        logger.info("Model initialization complete.")

    # Wrap model with DistributedDataParallel for multi-GPU training
    if dist.world_size > 1:
        model = DDP(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.local_rank,
            find_unused_parameters=False,
        )

    # Count trainable parameters
    model_for_counting = model.module if isinstance(model, DDP) else model
    if hasattr(model_for_counting, "count_params"):
        trainable_params = model_for_counting.count_params()
    else:
        trainable_params = sum(
            p.numel() for p in model_for_counting.parameters() if p.requires_grad
        )

    # Print model info (only on rank 0)
    if dist.rank == 0:
        logger.info(
            f"Model: {model.__class__.__name__} | Parameters: {trainable_params:,}"
        )
        # Print detailed model architecture
        print_model_architecture(model, model_type, dimensions, cfg, logger)

    # Create training loss function
    loss_fn = get_loss_function(cfg.loss)

    # Create validation loss function (same as training loss for fair comparison)
    from omegaconf import DictConfig

    if cfg.loss.base_loss_type == "simple_relative_l2":
        # Use same simple loss for validation
        val_loss_fn = loss_fn
    else:
        # Use UnifiedLoss with same masking but no derivatives for validation
        val_loss_cfg = DictConfig(
            {
                "base_loss_type": cfg.loss.base_loss_type,
                "use_mask": cfg.loss.use_mask,  # Use same mask as training
                "use_derivative": False,  # No derivatives in validation
                "reduction": cfg.loss.get("reduction", "sum"),
            }
        )
        val_loss_fn = get_loss_function(val_loss_cfg)

    # Print loss info (only on rank 0)
    if dist.rank == 0:
        if cfg.loss.base_loss_type == "simple_relative_l2":
            loss_info = f"Train Loss: SIMPLE_RELATIVE_L2 (no mask, no derivatives) | Val Loss: SIMPLE_RELATIVE_L2"
        else:
            loss_info = f"Train Loss: {cfg.loss.base_loss_type.upper()} | Val Loss: {cfg.loss.base_loss_type.upper()}"
            if cfg.loss.use_derivative:
                loss_info += f" (+Derivative w={cfg.loss.derivative_weight})"
            if cfg.loss.use_mask:
                loss_info += " (+Masking)"
        logger.info(loss_info)

    # Create optimizer and scheduler
    optimizer = Adam(
        model.parameters(),
        lr=cfg.training.initial_lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    # Create scheduler based on config type
    scheduler_type = cfg.scheduler.type.lower()
    if scheduler_type == "step":
        scheduler = StepLR(
            optimizer, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma
        )
    elif scheduler_type == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=cfg.scheduler.gamma)
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. Must be 'step' or 'exponential'"
        )

    # Initialize AMP GradScaler if enabled
    scaler = GradScaler() if cfg.training.use_amp else None

    # Print optimizer info (only on rank 0)
    if dist.rank == 0:
        logger.info(
            f"Optimizer: Adam (lr={cfg.training.initial_lr}) | AMP: {cfg.training.use_amp}"
        )

    # Setup MLFlow tracking if enabled
    if use_mlflow:
        mlflow.set_experiment(cfg.logging.experiment_name)

        # Enable automatic system metrics logging (CPU, GPU, memory, disk, network)
        mlflow.enable_system_metrics_logging()

        mlflow.start_run()

        # Log hyperparameters
        mlflow_params = {
            "dimensions": dimensions,
            "model_type": model_type,
            "model_arch_name": model_arch_name,
            "batch_size": cfg.training.batch_size,
            "epochs": cfg.training.epochs,
            "learning_rate": cfg.training.initial_lr,
            "optimizer": "Adam",
            "train_loss": cfg.loss.base_loss_type,
            "loss_masking": cfg.loss.use_mask,
            "loss_derivative": cfg.loss.use_derivative,
            "use_amp": cfg.training.use_amp,
            "use_graphs": cfg.training.use_graphs,
            "variable": cfg.data.variable,
            "trainable_parameters": trainable_params,
            "in_channels": in_channels,
        }
        
        if model_type == "xfno":
            xfno_cfg = cfg.arch.xfno
            mlflow_params.update({
                "width": xfno_cfg.width,
                "modes1": xfno_cfg.modes1,
                "modes2": xfno_cfg.modes2,
                "modes3": xfno_cfg.modes3,
                "num_fno_layers": xfno_cfg.num_fno_layers,
                "padding": xfno_cfg.padding,
                "activation_fn": xfno_cfg.activation_fn,
            })
            if dimensions == "4d":
                mlflow_params["modes4"] = xfno_cfg.modes4
            else:
                mlflow_params["num_unet_layers"] = xfno_cfg.num_unet_layers
                mlflow_params["num_conv_layers"] = xfno_cfg.num_conv_layers
        elif model_type == "xdeeponet":
            xdeeponet_cfg = cfg.arch.xdeeponet
            mlflow_params.update({
                "variant": xdeeponet_cfg.variant,
                "width": xdeeponet_cfg.width,
                "padding": xdeeponet_cfg.padding,
                "branch1_type": xdeeponet_cfg.branch1.type,
            })
        
        mlflow.log_params(mlflow_params)

    # Setup checkpointing (make absolute path since chdir=False)
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        # If relative, make it relative to the working directory (U-FNO folder)
        checkpoint_dir = Path.cwd() / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if resuming
    start_epoch = 1
    best_val_loss = float("inf")
    best_val_mre = float("inf")

    if (
        hasattr(cfg.training, "resume_from_checkpoint")
        and cfg.training.resume_from_checkpoint
    ):
        checkpoint_path = Path(cfg.training.resume_from_checkpoint)
        if checkpoint_path.exists():
            if dist.rank == 0:
                logger.info(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=dist.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("val_loss", float("inf"))
            best_val_mre = checkpoint.get("val_mre", float("inf"))
            if dist.rank == 0:
                logger.success(
                    f"Resumed from epoch {checkpoint['epoch']}, best val loss: {best_val_loss:.6f}, best val MRE: {best_val_mre:.6f}"
                )
                logger.info(
                    f"Continuing training from epoch {start_epoch} to {cfg.training.epochs}"
                )
        else:
            if dist.rank == 0:
                logger.warning(
                    f"Checkpoint not found at {checkpoint_path}, starting from scratch"
                )
    else:
        if dist.rank == 0:
            logger.success("Starting training from scratch...")

    # ---------------------------------------------------------------------------
    # Determine training regime
    # ---------------------------------------------------------------------------
    regime = cfg.training.get("regime", "full_mapping").lower()
    if regime == "autoregressive":
        ar_cfg = cfg.training.autoregressive
        ar_L = ar_cfg.input_window
        ar_K = ar_cfg.output_window
        tf_epochs = ar_cfg.teacher_forcing_epochs
        ro_epochs = ar_cfg.rollout_epochs
        total_epochs = tf_epochs + ro_epochs
        ar_max_steps = ar_cfg.max_rollout_steps
        ar_checkpointing = ar_cfg.gradient_checkpointing

        if dist.rank == 0:
            logger.info("=" * 80)
            logger.info(f"AUTOREGRESSIVE TRAINING | L={ar_L}, K={ar_K}")
            logger.info(f"  Phase 1 — Teacher Forcing: {tf_epochs} epochs")
            logger.info(f"  Phase 2 — Rollout (max {ar_max_steps} steps): {ro_epochs} epochs")
            logger.info(f"  Total: {total_epochs} epochs")
            logger.info("=" * 80)
    else:
        total_epochs = cfg.training.epochs
        if dist.rank == 0:
            logger.info("=" * 80)
            logger.info("FULL-MAPPING TRAINING")
            logger.info(f"  Epochs: {total_epochs}")
            logger.info("=" * 80)

    # Training loop
    for epoch in range(start_epoch, total_epochs + 1):
        # Set epoch for distributed sampler (ensures proper shuffling across epochs)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # Training phase
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=len(train_loader)
        ) as log:
            model.train()
            total_loss = 0.0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(dist.device)
                targets = targets.to(dist.device)
                optimizer.zero_grad()

                if regime == "autoregressive":
                    # Determine phase: teacher forcing or rollout
                    is_rollout_phase = epoch > tf_epochs
                    if is_rollout_phase:
                        loss = rollout_step(
                            model, inputs, targets, loss_fn,
                            L=ar_L, K=ar_K,
                            max_steps=ar_max_steps,
                            use_checkpointing=ar_checkpointing,
                            spatial_mask=static_mask,
                        )
                    else:
                        loss = teacher_forcing_step(
                            model, inputs, targets, loss_fn,
                            L=ar_L, K=ar_K,
                            spatial_mask=static_mask,
                        )
                else:
                    # Full-mapping: single forward pass over entire trajectory
                    if cfg.training.use_amp:
                        with autocast():
                            pred = model(inputs)
                            loss = loss_fn(pred, targets, inputs, spatial_mask=static_mask)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        # Aggregate and continue (skip the common backward below)
                        if dist.world_size > 1:
                            loss_tensor = loss.detach().clone()
                            torch.distributed.all_reduce(
                                loss_tensor, op=torch.distributed.ReduceOp.SUM
                            )
                            total_loss += loss_tensor / dist.world_size
                        else:
                            total_loss += loss.detach()
                        continue
                    else:
                        pred = model(inputs)
                        loss = loss_fn(pred, targets, inputs, spatial_mask=static_mask)

                # Common backward pass (AR mode, or full-mapping without AMP)
                loss.backward()
                optimizer.step()

                if dist.world_size > 1:
                    loss_tensor = loss.detach().clone()
                    torch.distributed.all_reduce(
                        loss_tensor, op=torch.distributed.ReduceOp.SUM
                    )
                    total_loss += loss_tensor / dist.world_size
                else:
                    total_loss += loss.detach()

            avg_train_loss = total_loss / len(train_loader)

            # Log phase info for AR
            if regime == "autoregressive" and dist.rank == 0:
                phase_name = "ROLLOUT" if epoch > tf_epochs else "TEACHER-FORCING"
                if epoch == tf_epochs + 1:
                    logger.info("=" * 40)
                    logger.info("Switching to ROLLOUT phase")
                    logger.info("=" * 40)

            log.log_epoch({"loss": avg_train_loss})

            if cfg.logging.use_mlflow and dist.rank == 0:
                mlflow.log_metric("train_loss", float(avg_train_loss), step=epoch)

        # Validation phase
        if epoch % cfg.training.validate_freq == 0:
            with LaunchLogger("valid", epoch=epoch) as log:
                model.eval()
                total_val_loss = 0.0
                mre_list = []

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(dist.device)
                        targets = targets.to(dist.device)

                        # Forward pass — same regime as training
                        if regime == "autoregressive":
                            pred = ar_validate_full_rollout(
                                model, inputs, targets, L=ar_L, K=ar_K,
                            )
                        else:
                            pred = model(inputs)

                        if cfg.training.use_amp:
                            with autocast():
                                val_loss = val_loss_fn(pred, targets, inputs, spatial_mask=static_mask)
                        else:
                            val_loss = val_loss_fn(pred, targets, inputs, spatial_mask=static_mask)

                        # Aggregate validation loss across GPUs
                        if dist.world_size > 1:
                            val_loss_tensor = val_loss.detach().clone()
                            torch.distributed.all_reduce(
                                val_loss_tensor, op=torch.distributed.ReduceOp.SUM
                            )
                            val_loss_tensor = val_loss_tensor / dist.world_size
                            total_val_loss += val_loss_tensor
                        else:
                            total_val_loss += val_loss.detach()

                        # Calculate validation metric on rank 0 only (for logging)
                        if dist.rank == 0:
                            pred_cpu = pred.cpu().numpy()
                            targets_cpu = targets.cpu().numpy()
                            inputs_cpu = inputs.cpu().numpy()

                            # Optional denormalization (config-driven)
                            denorm_name = cfg.data.get("denormalize_fn", None)
                            if denorm_name and denorm_name in _DENORM_REGISTRY:
                                denorm_fn = _DENORM_REGISTRY[denorm_name]
                                pred_denorm = denorm_fn(pred_cpu)
                                targets_denorm = denorm_fn(targets_cpu)
                            else:
                                pred_denorm = pred_cpu
                                targets_denorm = targets_cpu

                            # Resolve metric function from config
                            val_metric_choice = cfg.data.get("val_metric", "rmse")
                            if val_metric_choice not in _METRIC_REGISTRY:
                                raise ValueError(
                                    f"Unknown val_metric '{val_metric_choice}'. "
                                    f"Choices: {list(_METRIC_REGISTRY.keys())}"
                                )
                            _, _, metric_fn = _METRIC_REGISTRY[val_metric_choice]

                            mask_np = static_mask.cpu().numpy() if static_mask is not None else None

                            for i in range(pred_denorm.shape[0]):
                                if mask_np is not None:
                                    y_pred = pred_denorm[i][mask_np]
                                    y_true = targets_denorm[i][mask_np]
                                else:
                                    y_pred = pred_denorm[i].ravel()
                                    y_true = targets_denorm[i].ravel()

                                mre_list.append(metric_fn(y_pred, y_true))

                avg_val_loss = total_val_loss / len(val_loader)
                avg_metric = np.mean(mre_list) if len(mre_list) > 0 else 0.0

                # Metric display name and logging key from config
                val_metric_choice = cfg.data.get("val_metric", "rmse")
                metric_name, metric_key, _ = _METRIC_REGISTRY[val_metric_choice]

                is_ratio_metric = val_metric_choice in ("mre", "mpe", "relative_l2")

                # Print validation metrics (only on rank 0)
                if dist.rank == 0:
                    if is_ratio_metric:
                        logger.info(
                            f"Epoch {epoch}: Val Loss = {avg_val_loss:.6f} | Val {metric_name} = {avg_metric:.6f} ({avg_metric * 100:.2f}%)"
                        )
                    else:
                        logger.info(
                            f"Epoch {epoch}: Val Loss = {avg_val_loss:.6f} | Val {metric_name} = {avg_metric:.6f}"
                        )

                # Log to MLFlow (only on rank 0)
                if cfg.logging.use_mlflow and dist.rank == 0:
                    mlflow.log_metric("val_loss", float(avg_val_loss), step=epoch)
                    mlflow.log_metric(metric_key, float(avg_metric), step=epoch)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_mre = avg_metric

                    # Print and log best validation loss (only on rank 0)
                    if dist.rank == 0:
                        if is_ratio_metric:
                            logger.success(
                                f"New best validation: Loss = {best_val_loss:.6f} | {metric_name} = {best_val_mre:.6f} ({best_val_mre * 100:.2f}%)"
                            )
                        else:
                            logger.success(
                                f"New best validation: Loss = {best_val_loss:.6f} | {metric_name} = {best_val_mre:.6f}"
                            )

                    # Log best loss to MLFlow (only on rank 0)
                    if cfg.logging.use_mlflow and dist.rank == 0:
                        mlflow.log_metric(
                            "best_val_loss", float(best_val_loss), step=epoch
                        )
                        mlflow.log_metric(
                            f"best_{metric_key}", float(best_val_mre), step=epoch
                        )

                    # Save best model (only on rank 0 to avoid race condition)
                    if dist.rank == 0:
                        best_model_path = (
                            checkpoint_dir
                            / f"best_model_{cfg.data.variable}_{model_arch_name}.pth"
                        )
                        # For DDP models, save the underlying module's state_dict
                        model_to_save = (
                            model.module if isinstance(model, DDP) else model
                        )

                        # Prepare model config to save with checkpoint
                        model_config = {
                            "dimensions": dimensions,
                            "model_type": model_type,
                            "model_arch_name": model_arch_name,
                            "variable": cfg.data.variable,
                            "in_channels": in_channels,
                        }

                        if model_type == "xfno":
                            xfno_cfg = cfg.arch.xfno
                            model_config.update({
                                "out_channels": xfno_cfg.out_channels,
                                "width": xfno_cfg.width,
                                "modes1": xfno_cfg.modes1,
                                "modes2": xfno_cfg.modes2,
                                "modes3": xfno_cfg.modes3,
                                "num_fno_layers": xfno_cfg.num_fno_layers,
                                "padding": xfno_cfg.padding,
                                "activation_fn": xfno_cfg.activation_fn,
                                "decoder_layers": xfno_cfg.decoder_layers,
                                "decoder_width": xfno_cfg.decoder_width,
                            })
                            if dimensions == "4d":
                                model_config.update({
                                    "modes4": xfno_cfg.modes4,
                                    "coord_features": xfno_cfg.coord_features,
                                    "lifting_layers": xfno_cfg.lifting_layers,
                                })
                            else:
                                model_config.update({
                                    "num_unet_layers": xfno_cfg.num_unet_layers,
                                    "num_conv_layers": xfno_cfg.num_conv_layers,
                                    "unet_type": xfno_cfg.unet_type,
                                    "lifting_type": xfno_cfg.lifting_type,
                                    "lifting_layers": xfno_cfg.lifting_layers,
                                    "lifting_width": xfno_cfg.lifting_width,
                                    "decoder_type": xfno_cfg.decoder_type,
                                })

                        elif model_type == "xdeeponet":
                            xdeeponet_cfg = cfg.arch.xdeeponet
                            model_config.update({
                                "variant": xdeeponet_cfg.variant,
                                "width": xdeeponet_cfg.width,
                                "padding": xdeeponet_cfg.padding,
                                "branch1_config": dict(xdeeponet_cfg.branch1),
                                "trunk_config": dict(xdeeponet_cfg.trunk),
                                "decoder_type": xdeeponet_cfg.get("decoder_type", "mlp"),
                                "decoder_width": xdeeponet_cfg.decoder_width,
                                "decoder_layers": xdeeponet_cfg.decoder_layers,
                                "decoder_activation_fn": xdeeponet_cfg.get("decoder_activation_fn", "relu"),
                            })
                            if xdeeponet_cfg.variant in ['mionet', 'fourier_mionet']:
                                model_config["branch2_config"] = dict(xdeeponet_cfg.branch2)

                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model_to_save.state_dict(),
                                "val_loss": best_val_loss,
                                metric_key: best_val_mre,
                                "model_config": model_config,
                            },
                            best_model_path,
                        )

                        # Log model to MLFlow
                        if cfg.logging.use_mlflow:
                            mlflow.log_artifact(str(best_model_path))

        # Learning rate scheduling (StepLR steps automatically every step_size epochs)
        scheduler.step()

    # Resolve metric metadata once for final summary / MLflow
    val_metric_choice = cfg.data.get("val_metric", "rmse")
    metric_name, metric_key, _ = _METRIC_REGISTRY[val_metric_choice]
    is_ratio_metric = val_metric_choice in ("mre", "mpe", "relative_l2")

    # Print training completion (only on rank 0)
    if dist.rank == 0:
        logger.success("Training completed!")
        if is_ratio_metric:
            logger.info(
                f"Best validation: Loss = {best_val_loss:.6f} | {metric_name} = {best_val_mre:.6f} ({best_val_mre * 100:.2f}%)"
            )
        else:
            logger.info(
                f"Best validation: Loss = {best_val_loss:.6f} | {metric_name} = {best_val_mre:.6f}"
            )

    # End MLFlow run (only on rank 0)
    if cfg.logging.use_mlflow and dist.rank == 0:
        mlflow.log_metric("final_best_val_loss", float(best_val_loss))
        mlflow.log_metric(f"final_best_{metric_key}", float(best_val_mre))
        mlflow.end_run()
        logger.info("MLFlow run completed")


if __name__ == "__main__":
    main()

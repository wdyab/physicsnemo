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

"""
Unified dataset loaders for reservoir simulation neural operators.

Supports both 3D (2D spatial + time) and 4D (3D spatial + time) datasets:
- 3D: Input (N, H, W, T, C), Output (N, H, W, T) - e.g., CO2 sequestration
- 4D: Input (N, X, Y, Z, T, C), Output (N, X, Y, Z, T) - e.g., Norne field
"""

from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List
import torch
from torch.utils.data import Dataset


def _log_message(msg: str, rank_zero_only: bool = True):
    """Print message, optionally only on rank 0 in distributed mode."""
    try:
        from physicsnemo.distributed import DistributedManager

        dist = DistributedManager()
        if not rank_zero_only or dist.rank == 0:
            print(msg)
    except:
        print(msg)


class ReservoirDataset(Dataset):
    """
    Unified dataset for reservoir simulation modeling.

    Automatically detects and handles both 3D and 4D data:
    - 3D: (N, H, W, T, C) input, (N, H, W, T) output
    - 4D: (N, X, Y, Z, T, C) input, (N, X, Y, Z, T) output

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to the data directory or directly to input file
    mode : str
        Dataset split: 'train', 'val', or 'test'
    input_file : str, optional
        Input filename pattern. Supports {mode} placeholder.
        Default: auto-detect from data_path
    output_file : str, optional
        Output filename pattern. Supports {mode} placeholder.
        Default: auto-detect from data_path
    normalize : bool
        Whether to normalize data (default: True)

    File Naming Patterns
    --------------------
    The dataset supports flexible file naming:

    1. Explicit files:
       >>> ReservoirDataset(data_path, mode='train',
       ...     input_file='train_inputs.pt', output_file='train_outputs.pt')

    2. Pattern with {mode} placeholder:
       >>> ReservoirDataset(data_path, mode='train',
       ...     input_file='data_{mode}_input.pt', output_file='data_{mode}_output.pt')

    3. CO2 dataset convention (auto-detected):
       Files: dP_train_a.pt, dP_train_u.pt (or sg_*)
       >>> ReservoirDataset(data_path, mode='train', variable='pressure')

    4. Generic convention (auto-detected):
       Files: train_input.pt, train_output.pt
       >>> ReservoirDataset(data_path, mode='train')

    Examples
    --------
    >>> # 3D CO2 dataset
    >>> ds = ReservoirDataset('data/co2', mode='train', variable='pressure')
    >>> x, y = ds[0]  # x: (H, W, T, C), y: (H, W, T)

    >>> # 4D Norne dataset with explicit files
    >>> ds = ReservoirDataset('data/norne', mode='train',
    ...     input_file='norne_{mode}_input.pt', output_file='norne_{mode}_output.pt')
    >>> x, y = ds[0]  # x: (X, Y, Z, T, C), y: (X, Y, Z, T)

    >>> # With dimension validation (from config)
    >>> ds = ReservoirDataset('data/norne', mode='train',
    ...     input_file='norne_{mode}_input.pt', output_file='norne_{mode}_output.pt',
    ...     expected_dimensions='4d')  # Raises error if data is 3d
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        mode: str = "train",
        input_file: Optional[str] = None,
        output_file: Optional[str] = None,
        variable: Optional[str] = None,
        normalize: bool = True,
        expected_dimensions: Optional[str] = None,
        use_mask: bool = False,
        mask_channel: Optional[int] = None,
        num_timesteps: Optional[int] = None,
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.mode = mode.lower()
        self.normalize = normalize
        self.variable = variable
        self.expected_dimensions = (
            expected_dimensions.lower() if expected_dimensions else None
        )
        self.use_mask = use_mask
        self._config_mask_channel = mask_channel
        self._num_timesteps = num_timesteps

        if self.mode not in ["train", "val", "test"]:
            raise ValueError(f"Mode must be 'train', 'val', or 'test', got {mode}")

        # Resolve file paths
        self.input_file, self.output_file = self._resolve_file_paths(
            input_file, output_file, variable
        )

        # Load data
        self._load_data()

        # Truncate time axis if requested
        if self._num_timesteps is not None:
            T = self._num_timesteps
            self.input_data = self.input_data[..., :T, :]
            self.output_data = self.output_data[..., :T]
            _log_message(f"  Truncated to {T} timesteps")

        # Detect dimensions and set metadata
        self._detect_dimensions()

        # Resolve masking strategy
        self.mask_channel = None
        self.mask_per_sample = False
        self.static_mask = None
        if self.use_mask:
            self._resolve_mask()

        # Compute normalization
        if self.normalize:
            self._compute_normalization()

    def _resolve_file_paths(
        self,
        input_file: Optional[str],
        output_file: Optional[str],
        variable: Optional[str],
    ) -> Tuple[Path, Path]:
        """Resolve input and output file paths with flexible naming support."""

        # Case 1: Explicit files provided
        if input_file is not None and output_file is not None:
            # Replace {mode} placeholder
            input_name = input_file.format(mode=self.mode)
            output_name = output_file.format(mode=self.mode)
            return self.data_path / input_name, self.data_path / output_name

        # Case 2: Variable-based naming (CO2 convention)
        if variable is not None:
            var_map = {"pressure": "dP", "saturation": "sg", "dP": "dP", "sg": "sg"}
            if variable.lower() not in var_map:
                raise ValueError(
                    f"Variable must be 'pressure' or 'saturation', got {variable}"
                )
            var_prefix = var_map[variable.lower()]
            return (
                self.data_path / f"{var_prefix}_{self.mode}_a.pt",
                self.data_path / f"{var_prefix}_{self.mode}_u.pt",
            )

        # Case 3: Auto-detect from directory
        return self._auto_detect_files()

    def _auto_detect_files(self) -> Tuple[Path, Path]:
        """Auto-detect input/output files from directory."""

        # Common naming patterns to try (in order of preference)
        patterns = [
            # Generic pattern
            (f"{self.mode}_input.pt", f"{self.mode}_output.pt"),
            (f"input_{self.mode}.pt", f"output_{self.mode}.pt"),
            (f"{self.mode}_x.pt", f"{self.mode}_y.pt"),
            (f"x_{self.mode}.pt", f"y_{self.mode}.pt"),
            # CO2 patterns (try both variables)
            (f"dP_{self.mode}_a.pt", f"dP_{self.mode}_u.pt"),
            (f"sg_{self.mode}_a.pt", f"sg_{self.mode}_u.pt"),
        ]

        for input_name, output_name in patterns:
            input_path = self.data_path / input_name
            output_path = self.data_path / output_name
            if input_path.exists() and output_path.exists():
                return input_path, output_path

        # List available .pt files for helpful error message
        pt_files = list(self.data_path.glob("*.pt"))
        raise FileNotFoundError(
            f"Could not auto-detect data files in {self.data_path}\n"
            f"Available .pt files: {[f.name for f in pt_files]}\n"
            f"Please specify input_file and output_file explicitly."
        )

    def _load_data(self):
        """Load data from disk."""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        if not self.output_file.exists():
            raise FileNotFoundError(f"Output file not found: {self.output_file}")

        _log_message(
            f"Loading {self.mode} data: {self.input_file.name} -> {self.output_file.name}"
        )

        self.input_data = torch.load(self.input_file, map_location="cpu")
        self.output_data = torch.load(self.output_file, map_location="cpu")

        _log_message(
            f"  Loaded {len(self.input_data)} samples | "
            f"Input: {tuple(self.input_data.shape)} | Output: {tuple(self.output_data.shape)}"
        )

    def _detect_dimensions(self):
        """Detect spatial dimensions (3D or 4D) from data shape and validate against expected."""
        input_ndim = self.input_data.dim()
        output_ndim = self.output_data.dim()

        # 3D data: Input (N, H, W, T, C), Output (N, H, W, T)
        if input_ndim == 5 and output_ndim == 4:
            self.dimensions = "3d"
            self.spatial_dims = 2  # H, W
            self.dim_names = ("H", "W", "T")

        # 4D data: Input (N, X, Y, Z, T, C), Output (N, X, Y, Z, T)
        elif input_ndim == 6 and output_ndim == 5:
            self.dimensions = "4d"
            self.spatial_dims = 3  # X, Y, Z
            self.dim_names = ("X", "Y", "Z", "T")

        else:
            raise ValueError(
                f"Unsupported data dimensions!\n"
                f"  Input: {input_ndim}D {tuple(self.input_data.shape)}\n"
                f"  Output: {output_ndim}D {tuple(self.output_data.shape)}\n"
                f"Expected:\n"
                f"  3D: Input (N, H, W, T, C), Output (N, H, W, T)\n"
                f"  4D: Input (N, X, Y, Z, T, C), Output (N, X, Y, Z, T)"
            )

        # Validate against expected dimensions (from config)
        if (
            self.expected_dimensions is not None
            and self.dimensions != self.expected_dimensions
        ):
            raise ValueError(
                f"❌ Dimension mismatch!\n"
                f"   Config expects: {self.expected_dimensions}\n"
                f"   Data has: {self.dimensions}\n"
                f"   Input shape: {tuple(self.input_data.shape)}\n"
                f"   Please update arch.dimensions in config to '{self.dimensions}' "
                f"or use a dataset with {self.expected_dimensions} data."
            )

        # Store shape info
        self.num_samples = self.input_data.shape[0]
        self.spatial_shape = tuple(self.input_data.shape[1:-2])  # Spatial dims only
        self.time_steps = self.input_data.shape[-2]
        self.num_channels = self.input_data.shape[-1]

        _log_message(
            f"  Detected: {self.dimensions.upper()} | "
            f"Spatial: {self.spatial_shape} | T: {self.time_steps} | C: {self.num_channels}"
        )

    def _output_inactive_cells(self):
        """Cells where the output is zero across all timesteps for sample 0.

        This identifies truly inactive (padded) cells: they have zero
        output at every timestep.  Used to validate that a candidate
        mask channel's zeros correspond to real inactive cells, not to
        feature values that happen to be zero (e.g. perforation = 0 in
        cells that still have non-zero pressure/saturation output).
        """
        return self.output_data[0].abs().sum(dim=-1) == 0

    def _find_actnum_channel(self):
        """Find ACTNUM channel index: binary {0,1}, static across time.

        An additional validation requires that the candidate's zero
        cells are a subset of output-inactive cells (truly padded).
        This prevents false positives from binary features like
        perforation maps.

        Returns ``(channel_index, cross_sample_consistent)`` or ``None``.
        """
        s0 = self.input_data[0]
        out_inactive = self._output_inactive_cells()
        n_check = min(self.input_data.shape[0], 3)
        candidates = []
        for ch in range(s0.shape[-1]):
            col = s0[..., 0, ch]
            vals = col.unique()
            if not (vals.numel() <= 2 and all(v in (0.0, 1.0) for v in vals.tolist())):
                continue
            if not torch.equal(s0[..., 0, ch], s0[..., -1, ch]):
                continue
            zeros = col == 0
            if (zeros & ~out_inactive).any():
                continue
            consistent = all(
                torch.equal(col, self.input_data[si][..., 0, ch])
                for si in range(1, n_check)
            )
            candidates.append((ch, zeros.sum().item(), consistent))
        if not candidates:
            return None
        best = max(candidates, key=lambda x: x[1])
        return best[0], best[2]

    def _find_nonzero_mask_channel(self):
        """Find a channel whose zero pattern marks inactive cells.

        Looks for a channel where some cells are zero at t=0, the same
        cells are zero at the last timestep (static pattern), and those
        zero cells are a subset of output-inactive cells (truly padded,
        not just a feature that happens to be zero).

        Returns ``(channel_index, cross_sample_consistent)`` or ``None``.
        """
        s0 = self.input_data[0]
        out_inactive = self._output_inactive_cells()
        n_check = min(self.input_data.shape[0], 3)
        candidates = []
        for ch in range(s0.shape[-1]):
            zeros_t0 = s0[..., 0, ch] == 0
            n_zeros = zeros_t0.sum().item()
            if n_zeros == 0 or n_zeros == zeros_t0.numel():
                continue
            zeros_tlast = s0[..., -1, ch] == 0
            if not torch.equal(zeros_t0, zeros_tlast):
                continue
            if (zeros_t0 & ~out_inactive).any():
                continue
            consistent = all(
                torch.equal(zeros_t0, self.input_data[si][..., 0, ch] == 0)
                for si in range(1, n_check)
            )
            candidates.append((ch, n_zeros, consistent))
        if not candidates:
            return None
        best = max(candidates, key=lambda x: x[1])
        return best[0], best[2]

    def _resolve_mask(self):
        """Determine which input channel indicates active cells.

        Detection priority:
        1. Explicit ``mask_channel`` from config
        2. ACTNUM auto-detect (binary {0,1}, static across time)
        3. Non-zero channel fallback (static zero pattern across time)
        4. No mask (all cells active)

        Sets ``self.mask_channel``, ``self.mask_per_sample``, and
        ``self.static_mask``.
        """
        # Priority 1: Explicit config
        if self._config_mask_channel is not None:
            ch = self._config_mask_channel
            self.mask_channel = ch
            mask_s0 = self.input_data[0][..., 0, ch] != 0
            n_check = min(self.input_data.shape[0], 3)
            consistent = all(
                torch.equal(mask_s0, self.input_data[si][..., 0, ch] != 0)
                for si in range(1, n_check)
            )
            self.mask_per_sample = not consistent
            self.static_mask = mask_s0 if not self.mask_per_sample else None
            n_act = mask_s0.sum().item()
            n_tot = mask_s0.numel()
            per_sample_str = " (per-sample)" if self.mask_per_sample else ""
            _log_message(
                f"  Mask [config ch {ch}]: {n_act}/{n_tot} active "
                f"({100 * n_act / n_tot:.1f}%){per_sample_str}"
            )
            return

        # Priority 2: ACTNUM auto-detect (binary, static across time)
        result = self._find_actnum_channel()
        if result is not None:
            ch, consistent = result
            self.mask_channel = ch
            self.mask_per_sample = not consistent
            mask_s0 = self.input_data[0][..., 0, ch] != 0
            self.static_mask = mask_s0 if not self.mask_per_sample else None
            n_act = mask_s0.sum().item()
            n_tot = mask_s0.numel()
            per_sample_str = " (per-sample)" if self.mask_per_sample else ""
            _log_message(
                f"  Mask [ACTNUM ch {ch}]: {n_act}/{n_tot} active "
                f"({100 * n_act / n_tot:.1f}%){per_sample_str}"
            )
            return

        # Priority 3: Non-zero channel fallback (static zero pattern)
        result = self._find_nonzero_mask_channel()
        if result is not None:
            ch, consistent = result
            self.mask_channel = ch
            self.mask_per_sample = not consistent
            mask_s0 = self.input_data[0][..., 0, ch] != 0
            self.static_mask = mask_s0 if not self.mask_per_sample else None
            n_act = mask_s0.sum().item()
            n_tot = mask_s0.numel()
            per_sample_str = " (per-sample)" if self.mask_per_sample else ""
            _log_message(
                f"  Mask [nonzero ch {ch}]: {n_act}/{n_tot} active "
                f"({100 * n_act / n_tot:.1f}%){per_sample_str}"
            )
            return

        # Priority 4: No mask
        self.mask_channel = None
        self.mask_per_sample = False
        self.static_mask = None
        _log_message("  Mask: none (all cells active)")

    def get_static_mask(self):
        """Return static spatial mask or None."""
        return self.static_mask

    def _compute_normalization(self):
        """Compute normalization statistics (dimension-agnostic)."""
        if self.mode == "train":
            # Compute mean/std across all dims except channels (last dim)
            # Works for both 5D (N,H,W,T,C) and 6D (N,X,Y,Z,T,C)
            reduce_dims = tuple(range(self.input_data.dim() - 1))  # All except last

            self.input_mean = self.input_data.mean(dim=reduce_dims, keepdim=True)
            self.input_std = self.input_data.std(dim=reduce_dims, keepdim=True)
            self.output_mean = self.output_data.mean()
            self.output_std = self.output_data.std()

            # Avoid division by zero
            self.input_std = torch.where(
                self.input_std > 1e-6, self.input_std, torch.ones_like(self.input_std)
            )
            if self.output_std < 1e-6:
                self.output_std = torch.tensor(1.0)

            _log_message(
                f"  Normalization: Output mean={self.output_mean.item():.4f}, "
                f"std={self.output_std.item():.4f}"
            )
        else:
            # Identity normalization for val/test (set from training)
            shape = [1] * (self.input_data.dim() - 1) + [self.num_channels]
            self.input_mean = torch.zeros(shape)
            self.input_std = torch.ones(shape)
            self.output_mean = torch.tensor(0.0)
            self.output_std = torch.tensor(1.0)

    def set_normalization(
        self,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor,
    ):
        """Set normalization parameters from external source (e.g., training set)."""
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std

    def get_normalization_stats(self) -> Tuple[torch.Tensor, ...]:
        """Return normalization statistics."""
        return (self.input_mean, self.input_std, self.output_mean, self.output_std)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            3D: input (H, W, T, C), output (H, W, T)
            4D: input (X, Y, Z, T, C), output (X, Y, Z, T)
        """
        input_sample = self.input_data[idx]
        output_sample = self.output_data[idx]

        if self.normalize:
            # Squeeze the batch dimension from normalization stats for single sample
            input_mean = self.input_mean.squeeze(0).to(input_sample.device)
            input_std = self.input_std.squeeze(0).to(input_sample.device)
            output_mean = self.output_mean.to(output_sample.device)
            output_std = self.output_std.to(output_sample.device)

            input_sample = (input_sample - input_mean) / input_std
            output_sample = (output_sample - output_mean) / output_std

        return input_sample, output_sample


# =============================================================================
# Collate Functions
# =============================================================================


def collate_fn(batch):
    """
    Universal collate function for reservoir data.

    Works for both 3D and 4D data - simply stacks samples along batch dimension.
    """
    inputs = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    return inputs, targets


# =============================================================================
# Dataloader Factory
# =============================================================================


def create_dataloaders(
    data_path: Union[str, Path],
    batch_size: int = 4,
    normalize: bool = True,
    num_workers: int = 4,
    device: Union[str, torch.device] = "cuda",
    # Flexible file specification
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    variable: Optional[str] = None,
    expected_dimensions: Optional[str] = None,
    use_mask: bool = False,
    mask_channel: Optional[int] = None,
    num_timesteps: Optional[int] = None,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train, validation, and test dataloaders.

    Supports both 3D and 4D datasets with flexible file naming.

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to the data directory
    batch_size : int
        Batch size (default: 4)
    normalize : bool
        Whether to normalize data (default: True)
    num_workers : int
        Number of dataloader workers (default: 4)
    device : Union[str, torch.device]
        Device for pin_memory optimization (default: "cuda")
    input_file : str, optional
        Input filename pattern with {mode} placeholder
    output_file : str, optional
        Output filename pattern with {mode} placeholder
    variable : str, optional
        Variable name for CO2 convention ('pressure' or 'saturation')
    expected_dimensions : str, optional
        Expected dimensions ('3d' or '4d') from config. If provided, validates
        that loaded data matches. Raises error on mismatch.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        (train_loader, val_loader, test_loader)

    Examples
    --------
    >>> # CO2 dataset (3D)
    >>> train, val, test = create_dataloaders('data/co2', variable='pressure')

    >>> # Norne dataset (4D) with explicit files
    >>> train, val, test = create_dataloaders(
    ...     'data/norne',
    ...     input_file='norne_{mode}_input.pt',
    ...     output_file='norne_{mode}_output.pt'
    ... )

    >>> # With dimension validation from config
    >>> train, val, test = create_dataloaders(
    ...     'data/norne',
    ...     input_file='norne_{mode}_input.pt',
    ...     output_file='norne_{mode}_output.pt',
    ...     expected_dimensions='4d'  # From cfg.arch.dimensions
    ... )
    """
    from torch.utils.data import DataLoader

    # Check distributed mode
    try:
        from physicsnemo.distributed import DistributedManager

        dist = DistributedManager()
        is_distributed = dist.world_size > 1
    except:
        is_distributed = False

    # Common kwargs for dataset creation
    dataset_kwargs = {
        "data_path": data_path,
        "input_file": input_file,
        "output_file": output_file,
        "variable": variable,
        "normalize": normalize,
        "expected_dimensions": expected_dimensions,
        "use_mask": use_mask,
        "mask_channel": mask_channel,
    }

    # Create datasets (num_timesteps truncates train/val only; test keeps full trajectory)
    train_dataset = ReservoirDataset(
        mode="train", num_timesteps=num_timesteps, **dataset_kwargs
    )
    val_dataset = ReservoirDataset(
        mode="val", num_timesteps=num_timesteps, **dataset_kwargs
    )
    test_dataset = ReservoirDataset(mode="test", **dataset_kwargs)

    # Share normalization from training set
    if normalize:
        norm_stats = train_dataset.get_normalization_stats()

        if is_distributed:
            import torch.distributed as dist_torch

            gpu_stats = []
            for stat in norm_stats:
                s = stat.cuda()
                dist_torch.broadcast(s, src=0)
                gpu_stats.append(s.cpu())
            norm_stats = tuple(gpu_stats)

        val_dataset.set_normalization(*norm_stats)
        test_dataset.set_normalization(*norm_stats)

    # Determine pin_memory setting
    use_pin_memory = (isinstance(device, torch.device) and device.type == "cuda") or (
        isinstance(device, str) and device == "cuda"
    )

    # Create samplers for distributed training
    train_sampler = val_sampler = test_sampler = None
    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False)

    # Create dataloaders
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
        "persistent_workers": num_workers > 0,
        "collate_fn": collate_fn,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, sampler=val_sampler, **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, sampler=test_sampler, **loader_kwargs
    )

    # Log dimensions info
    _log_message(
        f"Created dataloaders: {train_dataset.dimensions.upper()} data | "
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# Utility Functions
# =============================================================================


def get_dataset_info(data_path: Union[str, Path], **kwargs) -> Dict:
    """
    Get information about a dataset without loading all data.

    Returns
    -------
    Dict with keys: dimensions, spatial_shape, time_steps, num_channels, num_samples
    """
    ds = ReservoirDataset(data_path, mode="train", normalize=False, **kwargs)
    return {
        "dimensions": ds.dimensions,
        "spatial_shape": ds.spatial_shape,
        "time_steps": ds.time_steps,
        "num_channels": ds.num_channels,
        "num_samples": {
            "train": len(ds),
            "val": len(
                ReservoirDataset(data_path, mode="val", normalize=False, **kwargs)
            ),
            "test": len(
                ReservoirDataset(data_path, mode="test", normalize=False, **kwargs)
            ),
        },
    }

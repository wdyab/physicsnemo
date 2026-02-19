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
Autoregressive training utilities with temporal bundling.

Provides dimension-agnostic helpers for slicing time windows, constructing
AR model inputs, and running multi-step rollouts.  Works with both 3D
(B, H, W, T, C) and 4D (B, X, Y, Z, T, C) datasets.

Key concepts
------------
- **L** (input_window):  Number of context timesteps fed to the model.
- **K** (output_window): Number of timesteps the model predicts per step.
- The time axis is always the **second-to-last** dimension of the input
  tensor and the **last** dimension of the target tensor.
- For DeepONet models, explicit **target_times** (trunk query coordinates)
  are extracted from the full input tensor and passed to the model so that
  K can differ from L (temporal bundling).

Two-phase training
------------------
**Phase 1 — Teacher Forcing** (``teacher_forcing_step``):
    Each iteration samples one random window [t0, t0+L) from the trajectory
    and predicts the next K timesteps [t0+L, t0+L+K).  The model always
    receives **ground-truth** input.  This teaches the correct single-step
    mapping (the physics) and converges quickly.

**Phase 2 — Rollout** (``rollout_step``):
    Each iteration samples a random starting point and chains up to
    ``max_rollout_steps`` AR steps.  From the second step onward the model
    receives **its own prediction** as input instead of ground truth.
    Errors compound across steps and the loss backpropagates through the
    full chain, teaching the model to produce predictions that are robust
    to its own approximation errors.

Teacher forcing is run first to learn the physics from clean inputs.
Rollout is run second to fine-tune for self-correcting behavior during
long-horizon inference.  This avoids the "exposure bias" problem where a
model trained only on perfect inputs fails when it encounters its own
imperfect predictions at inference time.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# ---------------------------------------------------------------------------
# Time-axis helpers
# ---------------------------------------------------------------------------

def _time_axis_input(x: Tensor) -> int:
    """Time axis index for input ``(..., T, C)``."""
    return x.dim() - 2


def _time_axis_target(y: Tensor) -> int:
    """Time axis index for target ``(..., T)``."""
    return y.dim() - 1


# ---------------------------------------------------------------------------
# Random starting-point sampling
# ---------------------------------------------------------------------------

def sample_start_index(
    total_T: int,
    L: int,
    K: int,
    num_steps: int = 1,
) -> int:
    """Sample a random starting timestep for an AR training window.

    The window ``[t0, t0 + L + K * num_steps)`` must fit inside ``[0, total_T)``.
    """
    required = L + K * num_steps
    if required > total_T:
        raise ValueError(
            f"Window (L={L} + K={K} * steps={num_steps} = {required}) "
            f"exceeds trajectory length T={total_T}"
        )
    max_start = total_T - required
    return int(torch.randint(0, max_start + 1, (1,)).item())


# ---------------------------------------------------------------------------
# Time-window slicing
# ---------------------------------------------------------------------------

def slice_input_window(inputs: Tensor, t0: int, width: int) -> Tensor:
    """Extract ``(B, *spatial, width, C)`` from full-trajectory input."""
    return inputs.narrow(_time_axis_input(inputs), t0, width)


def slice_target_window(targets: Tensor, t0: int, width: int) -> Tensor:
    """Extract ``(B, *spatial, width)`` from full-trajectory target."""
    return targets.narrow(_time_axis_target(targets), t0, width)


# ---------------------------------------------------------------------------
# Target-time coordinate extraction
# ---------------------------------------------------------------------------

def extract_target_times(inputs: Tensor, t_start: int, K: int) -> Tensor:
    """Extract K target time coordinates from the full input tensor.

    The time coordinate is assumed to be the **last channel** (index -1)
    of the input tensor at a fixed spatial location ``[0, 0, ..., 0]``.

    Parameters
    ----------
    inputs : Tensor
        Full-trajectory input ``(B, *spatial, T, C)``.
    t_start : int
        First target timestep index.
    K : int
        Number of target timesteps.

    Returns
    -------
    Tensor
        Shape ``(K,)`` — the time coordinate values for the K target steps.
    """
    t_ax = _time_axis_input(inputs)
    ndim = inputs.dim()
    # Index into [0, 0, ..., t_start:t_start+K, -1] regardless of spatial dims
    # For 3D input (B, H, W, T, C): inputs[0, 0, 0, t_start:t_start+K, -1]
    # For 4D input (B, X, Y, Z, T, C): inputs[0, 0, 0, 0, t_start:t_start+K, -1]
    spatial_zeros = (0,) * (ndim - 3)  # ndim-3 = num spatial dims
    idx = (0,) + spatial_zeros  # (0, 0, ..., 0) for batch + spatial
    return inputs[idx][t_start : t_start + K, -1]


# ---------------------------------------------------------------------------
# Model call helper
# ---------------------------------------------------------------------------

def _call_model(
    model,
    x_window: Tensor,
    target_times: Optional[Tensor],
    use_checkpointing: bool = False,
) -> Tensor:
    """Call the model, optionally passing target_times (for DeepONet) and
    optionally using gradient checkpointing."""
    has_target_times = target_times is not None and _model_accepts_target_times(model)

    if use_checkpointing and model.training:
        if has_target_times:
            return grad_checkpoint(
                _forward_with_times, model, x_window, target_times,
                use_reentrant=False,
            )
        return grad_checkpoint(model, x_window, use_reentrant=False)

    if has_target_times:
        return model(x_window, target_times=target_times)
    return model(x_window)


def _forward_with_times(model, x_window, target_times):
    """Thin wrapper so ``grad_checkpoint`` can pass target_times."""
    return model(x_window, target_times=target_times)


def _model_accepts_target_times(model) -> bool:
    """Check if the model's forward() accepts a ``target_times`` kwarg."""
    import inspect
    # Unwrap DDP
    m = model.module if hasattr(model, "module") else model
    sig = inspect.signature(m.forward)
    return "target_times" in sig.parameters


# ---------------------------------------------------------------------------
# Teacher-forcing training step (one batch)
# ---------------------------------------------------------------------------

def teacher_forcing_step(
    model,
    inputs: Tensor,
    targets: Tensor,
    loss_fn,
    L: int,
    K: int,
) -> Tensor:
    """One teacher-forcing training iteration over a batch.

    Samples a **single random window** from the trajectory, runs one forward
    pass, and returns the loss.  Using one window per call keeps only one
    forward graph in GPU memory at a time.

    For models that accept ``target_times`` (DeepONet), the K target time
    coordinates are extracted from the full input tensor and passed to the
    trunk, enabling K != L temporal bundling.
    """
    total_T = targets.shape[_time_axis_target(targets)]
    t0 = sample_start_index(total_T, L, K, num_steps=1)

    x_window = slice_input_window(inputs, t0, L)
    y_target = slice_target_window(targets, t0 + L, K)
    target_times = extract_target_times(inputs, t0 + L, K)

    pred = _call_model(model, x_window, target_times)
    t_ax = _time_axis_target(pred)

    if pred.shape[t_ax] > K:
        pred = pred.narrow(t_ax, 0, K)
    elif pred.shape[t_ax] < K:
        actual_K = pred.shape[t_ax]
        y_target = slice_target_window(targets, t0 + L, actual_K)

    return loss_fn(pred, y_target, x_window)


# ---------------------------------------------------------------------------
# Rollout training step (one batch)
# ---------------------------------------------------------------------------

def rollout_step(
    model,
    inputs: Tensor,
    targets: Tensor,
    loss_fn,
    L: int,
    K: int,
    max_steps: int,
    use_checkpointing: bool = True,
) -> Tensor:
    """One rollout (free-running) training iteration.

    Samples a random starting point, chains up to ``max_steps`` AR steps
    (feeding predictions back), and computes loss over the full predicted
    window vs ground truth.
    """
    total_T = targets.shape[_time_axis_target(targets)]

    effective_steps = min(max_steps, (total_T - L) // K)
    if effective_steps < 1:
        effective_steps = 1

    t0 = sample_start_index(total_T, L, K, num_steps=effective_steps)

    preds = []
    gt_slices = []
    current_t = t0

    for step in range(effective_steps):
        target_start = current_t + L
        remaining = total_T - target_start
        actual_K = min(K, remaining)
        if actual_K <= 0:
            break

        x_window = slice_input_window(inputs, current_t, L)
        target_times = extract_target_times(inputs, target_start, actual_K)

        pred = _call_model(model, x_window, target_times, use_checkpointing)

        t_ax = _time_axis_target(pred)
        if pred.shape[t_ax] > actual_K:
            pred = pred.narrow(t_ax, 0, actual_K)

        preds.append(pred)
        gt_slices.append(slice_target_window(targets, target_start, actual_K))

        current_t += K

    if not preds:
        return torch.tensor(0.0, device=inputs.device, requires_grad=True)

    t_ax = _time_axis_target(preds[0])
    pred_cat = torch.cat(preds, dim=t_ax)
    gt_cat = torch.cat(gt_slices, dim=t_ax)

    rollout_T = pred_cat.shape[t_ax]
    max_input_T = inputs.shape[_time_axis_input(inputs)] - t0
    input_for_loss = slice_input_window(inputs, t0, min(rollout_T, max_input_T))
    return loss_fn(pred_cat, gt_cat, input_for_loss)


# ---------------------------------------------------------------------------
# Full autoregressive validation (all timesteps)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ar_validate_full_rollout(
    model,
    inputs: Tensor,
    targets: Tensor,
    L: int,
    K: int,
) -> Tensor:
    """Run a complete AR rollout over the full trajectory for validation.

    Always starts at t=0 and rolls out until all timesteps are covered.
    Returns the full predicted trajectory (same shape as ``targets``).
    """
    total_T = targets.shape[_time_axis_target(targets)]
    t_ax = _time_axis_target(targets)

    pred_slices = []
    current_t = 0

    while current_t + L < total_T:
        target_start = current_t + L
        remaining = total_T - target_start
        actual_K = min(K, remaining)
        if actual_K <= 0:
            break

        x_window = slice_input_window(inputs, current_t, L)
        target_times = extract_target_times(inputs, target_start, actual_K)

        pred = _call_model(model, x_window, target_times)

        pred_t_ax = _time_axis_target(pred)
        if pred.shape[pred_t_ax] > actual_K:
            pred = pred.narrow(pred_t_ax, 0, actual_K)

        pred_slices.append(pred)
        current_t += K

    if not pred_slices:
        return torch.zeros_like(targets)

    pred_full = torch.cat(pred_slices, dim=t_ax)

    gt_prefix = slice_target_window(targets, 0, L)
    pred_full = torch.cat([gt_prefix, pred_full], dim=t_ax)

    if pred_full.shape[t_ax] > total_T:
        pred_full = pred_full.narrow(t_ax, 0, total_T)
    elif pred_full.shape[t_ax] < total_T:
        deficit = total_T - pred_full.shape[t_ax]
        pad_slice = slice_target_window(targets, pred_full.shape[t_ax], deficit)
        pred_full = torch.cat([pred_full, pad_slice], dim=t_ax)

    return pred_full

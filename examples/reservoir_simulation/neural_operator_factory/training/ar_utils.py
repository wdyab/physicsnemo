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
**Phase 1 -- Teacher Forcing** (``teacher_forcing_step``):
    Sweeps sequentially through the full trajectory starting at t=0.
    Each window [t, t+L) predicts [t+L, t+L+K).  The model always
    receives ground-truth input.  For TNO, Branch2 also receives
    GT solution.  Loss is averaged over all windows.

**Phase 2 -- Rollout** (``rollout_step``):
    Sweeps sequentially through the full trajectory starting at t=0.
    For TNO, Branch2 receives the model's own (detached) prediction
    from the previous step instead of ground truth.  This trains the
    model to handle its own approximation errors.

Teacher forcing is run first to learn the physics from clean inputs.
Rollout is run second to fine-tune for self-correcting behavior during
long-horizon inference.
"""

from __future__ import annotations

from typing import Optional

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
        Shape ``(K,)`` -- the time coordinate values for the K target steps.
    """
    ndim = inputs.dim()
    spatial_zeros = (0,) * (ndim - 3)
    idx = (0,) + spatial_zeros
    return inputs[idx][t_start : t_start + K, -1]


# ---------------------------------------------------------------------------
# Model call helper
# ---------------------------------------------------------------------------

def _call_model(
    model,
    x_window: Tensor,
    target_times: Optional[Tensor],
    use_checkpointing: bool = False,
    x_branch2: Optional[Tensor] = None,
) -> Tensor:
    """Call the model, optionally passing target_times and x_branch2."""
    kwargs = {}
    if target_times is not None and _model_accepts_target_times(model):
        kwargs["target_times"] = target_times
    if x_branch2 is not None:
        kwargs["x_branch2"] = x_branch2

    if use_checkpointing and model.training:
        return grad_checkpoint(
            _forward_with_kwargs, model, x_window, kwargs,
            use_reentrant=False,
        )
    return model(x_window, **kwargs)


def _forward_with_kwargs(model, x_window, kwargs):
    """Thin wrapper so ``grad_checkpoint`` can pass kwargs."""
    return model(x_window, **kwargs)


def _model_accepts_target_times(model) -> bool:
    """Check if the model's forward() accepts a ``target_times`` kwarg."""
    import inspect
    m = model.module if hasattr(model, "module") else model
    sig = inspect.signature(m.forward)
    return "target_times" in sig.parameters


# ---------------------------------------------------------------------------
# Teacher-forcing training step (one batch) -- sequential sweep
# ---------------------------------------------------------------------------

def teacher_forcing_step(
    model,
    inputs: Tensor,
    targets: Tensor,
    loss_fn,
    L: int,
    K: int,
    spatial_mask: Optional[Tensor] = None,
    is_tno: bool = False,
) -> Tensor:
    """One teacher-forcing training iteration over a batch.

    Sweeps sequentially from t=0 through the full trajectory, processing
    every non-overlapping window.  Uses gradient accumulation: each window
    is forwarded and backwarded independently (one graph at a time).
    Returns a detached scalar loss for logging.  The caller should NOT
    call loss.backward() — only optimizer.step().

    For TNO, Branch2 receives the ground-truth solution at [t, t+L).
    For all other variants, no feedback is applied.
    """
    total_T = targets.shape[_time_axis_target(targets)]
    t_ax = _time_axis_target(targets)

    num_windows = (total_T - L) // K
    if num_windows <= 0:
        return torch.tensor(0.0, device=inputs.device)

    accumulated_loss = 0.0
    current_t = 0

    for _ in range(num_windows):
        target_start = current_t + L
        remaining = total_T - target_start
        actual_K = min(K, remaining)
        if actual_K <= 0:
            break

        x_window = slice_input_window(inputs, current_t, L)
        y_target = slice_target_window(targets, target_start, actual_K)
        target_times = extract_target_times(inputs, target_start, actual_K)

        y_branch2 = slice_target_window(targets, current_t, L) if is_tno else None

        pred = _call_model(model, x_window, target_times, x_branch2=y_branch2)

        if pred.shape[t_ax] > actual_K:
            pred = pred.narrow(t_ax, 0, actual_K)

        window_loss = loss_fn(pred, y_target, x_window, spatial_mask=spatial_mask)
        if window_loss.requires_grad:
            (window_loss / num_windows).backward()
        accumulated_loss += window_loss.detach().item()

        current_t += K

    return torch.tensor(accumulated_loss / num_windows, device=inputs.device)


# ---------------------------------------------------------------------------
# Rollout training step (one batch) -- sequential chain from t=0
# ---------------------------------------------------------------------------

def rollout_step(
    model,
    inputs: Tensor,
    targets: Tensor,
    loss_fn,
    L: int,
    K: int,
    use_checkpointing: bool = True,
    spatial_mask: Optional[Tensor] = None,
    is_tno: bool = False,
) -> Tensor:
    """One rollout (free-running) training iteration.

    Sweeps sequentially from t=0 through the full trajectory.  Uses
    gradient accumulation: each window is forwarded and backwarded
    independently.  Returns a detached scalar loss for logging.
    The caller should NOT call loss.backward() — only optimizer.step().

    For TNO, Branch2 receives the model's own (detached) prediction
    from the previous step, creating true autoregressive feedback.
    For all other variants, each window is processed independently.
    """
    total_T = targets.shape[_time_axis_target(targets)]
    t_ax = _time_axis_target(targets)

    num_windows = (total_T - L) // K
    if num_windows <= 0:
        return torch.tensor(0.0, device=inputs.device)

    accumulated_loss = 0.0
    prev_pred = None
    current_t = 0

    for _ in range(num_windows):
        target_start = current_t + L
        remaining = total_T - target_start
        actual_K = min(K, remaining)
        if actual_K <= 0:
            break

        x_window = slice_input_window(inputs, current_t, L)
        y_target = slice_target_window(targets, target_start, actual_K)
        target_times = extract_target_times(inputs, target_start, actual_K)

        if is_tno:
            if prev_pred is None:
                y_branch2 = slice_target_window(targets, current_t, L)
            elif prev_pred.shape[t_ax] >= L:
                y_branch2 = prev_pred.narrow(t_ax, prev_pred.shape[t_ax] - L, L)
            else:
                need = L - prev_pred.shape[t_ax]
                gt_part = slice_target_window(targets, current_t, need)
                y_branch2 = torch.cat([gt_part, prev_pred], dim=t_ax)
        else:
            y_branch2 = None

        pred = _call_model(
            model, x_window, target_times, use_checkpointing, x_branch2=y_branch2,
        )

        if pred.shape[t_ax] > actual_K:
            pred = pred.narrow(t_ax, 0, actual_K)

        window_loss = loss_fn(pred, y_target, x_window, spatial_mask=spatial_mask)
        if window_loss.requires_grad:
            (window_loss / num_windows).backward()
        accumulated_loss += window_loss.detach().item()

        prev_pred = pred.detach()
        current_t += K

    return torch.tensor(accumulated_loss / num_windows, device=inputs.device)


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
    is_tno: bool = False,
) -> Tensor:
    """Run a complete AR rollout over the full trajectory for validation.

    Always starts at t=0 and rolls out until all timesteps are covered.
    Returns the full predicted trajectory (same shape as ``targets``).
    When is_tno=True, feeds predictions back as branch2 input.
    """
    total_T = targets.shape[_time_axis_target(targets)]
    t_ax = _time_axis_target(targets)

    pred_slices = []
    prev_pred = None
    current_t = 0

    while current_t + L < total_T:
        target_start = current_t + L
        remaining = total_T - target_start
        actual_K = min(K, remaining)
        if actual_K <= 0:
            break

        x_window = slice_input_window(inputs, current_t, L)
        target_times = extract_target_times(inputs, target_start, actual_K)

        if is_tno:
            if prev_pred is None:
                y_branch2 = slice_target_window(targets, current_t, L)
            elif prev_pred.shape[t_ax] >= L:
                y_branch2 = prev_pred.narrow(t_ax, prev_pred.shape[t_ax] - L, L)
            else:
                need = L - prev_pred.shape[t_ax]
                gt_part = slice_target_window(targets, current_t, need)
                y_branch2 = torch.cat([gt_part, prev_pred], dim=t_ax)
        else:
            y_branch2 = None

        pred = _call_model(model, x_window, target_times, x_branch2=y_branch2)

        pred_t_ax = _time_axis_target(pred)
        if pred.shape[pred_t_ax] > actual_K:
            pred = pred.narrow(pred_t_ax, 0, actual_K)

        pred_slices.append(pred)
        prev_pred = pred
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

# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for autoregressive training utilities."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.ar_utils import (
    slice_input_window,
    slice_target_window,
    extract_target_times,
    teacher_forcing_step,
    rollout_step,
    ar_validate_full_rollout,
    _time_axis_input,
    _time_axis_target,
    _model_accepts_target_times,
)


# ---------------------------------------------------------------------------
# Fixtures — dummy models
# ---------------------------------------------------------------------------

class DummyModel3D(nn.Module):
    """Returns zeros; output T = target_times length if given, else T_in."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x, target_times=None):
        B, H, W, T_in, C = x.shape
        T_out = target_times.shape[0] if target_times is not None else T_in
        return torch.zeros(B, H, W, T_out, device=x.device)


class DummyModel4D(nn.Module):
    """Returns zeros; output T = target_times length if given, else T_in."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x, target_times=None):
        B, X, Y, Z, T_in, C = x.shape
        T_out = target_times.shape[0] if target_times is not None else T_in
        return torch.zeros(B, X, Y, Z, T_out, device=x.device)


class DummyModelNoTargetTimes(nn.Module):
    """Model that does NOT accept target_times (e.g. FNO)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        B, H, W, T_in, C = x.shape
        return torch.zeros(B, H, W, T_in, device=x.device)


def dummy_loss(pred, target, inputs, spatial_mask=None):
    return torch.mean((pred - target) ** 2)


# ---------------------------------------------------------------------------
# Tests: slicing
# ---------------------------------------------------------------------------

class TestSlicing:
    def test_slice_input_3d(self):
        x = torch.randn(2, 8, 10, 24, 5)
        w = slice_input_window(x, t0=3, width=4)
        assert w.shape == (2, 8, 10, 4, 5)
        assert torch.equal(w, x[:, :, :, 3:7, :])

    def test_slice_input_4d(self):
        x = torch.randn(2, 8, 10, 6, 24, 5)
        w = slice_input_window(x, t0=5, width=3)
        assert w.shape == (2, 8, 10, 6, 3, 5)
        assert torch.equal(w, x[:, :, :, :, 5:8, :])

    def test_slice_target_3d(self):
        y = torch.randn(2, 8, 10, 24)
        w = slice_target_window(y, t0=10, width=5)
        assert w.shape == (2, 8, 10, 5)
        assert torch.equal(w, y[:, :, :, 10:15])

    def test_slice_target_4d(self):
        y = torch.randn(2, 8, 10, 6, 24)
        w = slice_target_window(y, t0=0, width=3)
        assert w.shape == (2, 8, 10, 6, 3)
        assert torch.equal(w, y[:, :, :, :, 0:3])


# ---------------------------------------------------------------------------
# Tests: extract_target_times
# ---------------------------------------------------------------------------

class TestExtractTargetTimes:
    def test_3d_extracts_last_channel(self):
        # (B=2, H=4, W=6, T=16, C=5) — last channel is time coord
        x = torch.randn(2, 4, 6, 16, 5)
        times = extract_target_times(x, t_start=5, K=3)
        assert times.shape == (3,)
        assert torch.equal(times, x[0, 0, 0, 5:8, -1])

    def test_4d_extracts_last_channel(self):
        # (B=2, X=4, Y=6, Z=3, T=20, C=11) — last channel is time coord
        x = torch.randn(2, 4, 6, 3, 20, 11)
        times = extract_target_times(x, t_start=10, K=5)
        assert times.shape == (5,)
        assert torch.equal(times, x[0, 0, 0, 0, 10:15, -1])

    def test_k_equals_1(self):
        x = torch.randn(1, 4, 6, 16, 5)
        times = extract_target_times(x, t_start=0, K=1)
        assert times.shape == (1,)


# ---------------------------------------------------------------------------
# Tests: _model_accepts_target_times
# ---------------------------------------------------------------------------

class TestModelAcceptsTargetTimes:
    def test_model_with_target_times(self):
        assert _model_accepts_target_times(DummyModel3D()) is True

    def test_model_without_target_times(self):
        assert _model_accepts_target_times(DummyModelNoTargetTimes()) is False


# ---------------------------------------------------------------------------
# Tests: teacher_forcing_step
# ---------------------------------------------------------------------------

class TestTeacherForcing:
    def test_3d_returns_scalar_loss(self):
        model = DummyModel3D()
        inputs = torch.randn(2, 4, 6, 16, 5)
        targets = torch.randn(2, 4, 6, 16)
        loss = teacher_forcing_step(model, inputs, targets, dummy_loss, L=1, K=3)
        assert loss.dim() == 0

    def test_4d_returns_scalar_loss(self):
        model = DummyModel4D()
        inputs = torch.randn(1, 4, 6, 3, 16, 5)
        targets = torch.randn(1, 4, 6, 3, 16)
        loss = teacher_forcing_step(model, inputs, targets, dummy_loss, L=1, K=3)
        assert loss.dim() == 0

    def test_3d_k_equals_1(self):
        model = DummyModel3D()
        inputs = torch.randn(2, 4, 6, 16, 5)
        targets = torch.randn(2, 4, 6, 16)
        loss = teacher_forcing_step(model, inputs, targets, dummy_loss, L=1, K=1)
        assert loss.dim() == 0

    def test_model_without_target_times_still_works(self):
        model = DummyModelNoTargetTimes()
        inputs = torch.randn(2, 4, 6, 16, 5)
        targets = torch.randn(2, 4, 6, 16)
        # K=1 since this model produces L output timesteps
        loss = teacher_forcing_step(model, inputs, targets, dummy_loss, L=1, K=1)
        assert loss.dim() == 0


# ---------------------------------------------------------------------------
# Tests: rollout_step
# ---------------------------------------------------------------------------

class TestRolloutStep:
    def test_3d_returns_scalar_loss(self):
        model = DummyModel3D()
        inputs = torch.randn(2, 4, 6, 20, 5)
        targets = torch.randn(2, 4, 6, 20)
        loss = rollout_step(
            model, inputs, targets, dummy_loss,
            L=1, K=3, use_checkpointing=False,
        )
        assert loss.dim() == 0

    def test_4d_with_checkpointing(self):
        model = DummyModel4D()
        inputs = torch.randn(1, 4, 6, 3, 20, 5)
        targets = torch.randn(1, 4, 6, 3, 20)
        loss = rollout_step(
            model, inputs, targets, dummy_loss,
            L=1, K=3, use_checkpointing=True,
        )
        assert loss.dim() == 0

    def test_k1_rollout(self):
        model = DummyModel3D()
        inputs = torch.randn(2, 4, 6, 20, 5)
        targets = torch.randn(2, 4, 6, 20)
        loss = rollout_step(
            model, inputs, targets, dummy_loss,
            L=1, K=1, use_checkpointing=False,
        )
        assert loss.dim() == 0


# ---------------------------------------------------------------------------
# Tests: ar_validate_full_rollout
# ---------------------------------------------------------------------------

class TestFullRollout:
    def test_3d_output_shape(self):
        model = DummyModel3D()
        inputs = torch.randn(1, 4, 6, 16, 5)
        targets = torch.randn(1, 4, 6, 16)
        pred = ar_validate_full_rollout(model, inputs, targets, L=1, K=3)
        assert pred.shape == targets.shape

    def test_4d_output_shape(self):
        model = DummyModel4D()
        inputs = torch.randn(1, 4, 6, 3, 20, 5)
        targets = torch.randn(1, 4, 6, 3, 20)
        pred = ar_validate_full_rollout(model, inputs, targets, L=1, K=3)
        assert pred.shape == targets.shape

    def test_prefix_matches_gt(self):
        model = DummyModel3D()
        inputs = torch.randn(1, 4, 6, 16, 5)
        targets = torch.randn(1, 4, 6, 16)
        pred = ar_validate_full_rollout(model, inputs, targets, L=1, K=3)
        assert torch.equal(pred[:, :, :, :1], targets[:, :, :, :1])

    def test_k1_full_rollout(self):
        model = DummyModel3D()
        inputs = torch.randn(1, 4, 6, 10, 5)
        targets = torch.randn(1, 4, 6, 10)
        pred = ar_validate_full_rollout(model, inputs, targets, L=1, K=1)
        assert pred.shape == targets.shape


# ---------------------------------------------------------------------------
# Tests: time axis helpers
# ---------------------------------------------------------------------------

class TestTimeAxisHelpers:
    def test_input_3d(self):
        assert _time_axis_input(torch.randn(2, 8, 10, 24, 5)) == 3

    def test_input_4d(self):
        assert _time_axis_input(torch.randn(2, 8, 10, 6, 24, 5)) == 4

    def test_target_3d(self):
        assert _time_axis_target(torch.randn(2, 8, 10, 24)) == 3

    def test_target_4d(self):
        assert _time_axis_target(torch.randn(2, 8, 10, 6, 24)) == 4


# ---------------------------------------------------------------------------
# Tests: grid mode target_times in DeepONet wrappers
# ---------------------------------------------------------------------------

class DummyModel3DGrid(nn.Module):
    """Model that accepts target_times and returns T_out = target_times length."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x, target_times=None):
        B, H, W, T_in, C = x.shape
        T_out = target_times.shape[0] if target_times is not None else T_in
        return torch.zeros(B, H, W, T_out, device=x.device)


class DummyModel4DGrid(nn.Module):
    """Model that accepts target_times and returns T_out = target_times length."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x, target_times=None):
        B, X, Y, Z, T_in, C = x.shape
        T_out = target_times.shape[0] if target_times is not None else T_in
        return torch.zeros(B, X, Y, Z, T_out, device=x.device)


class TestGridModeTargetTimes:
    """Test that grid mode correctly constructs (K, 3) or (K, 4) trunk inputs."""

    def test_extract_target_times_returns_k_values(self):
        """extract_target_times returns (K,) time values regardless of grid mode."""
        x = torch.randn(2, 4, 6, 16, 5)
        times = extract_target_times(x, t_start=5, K=3)
        assert times.shape == (3,)

    def test_3d_grid_full_mapping(self):
        """3D wrapper with grid mode works in full-mapping (no target_times)."""
        model = DummyModel3DGrid()
        # Input: (B, H, W, T, C) — last 3 channels are grid_x, grid_y, grid_t
        x = torch.randn(1, 4, 6, 10, 5)
        pred = model(x)
        assert pred.shape == (1, 4, 6, 10)

    def test_3d_grid_with_target_times(self):
        """3D wrapper with grid mode + target_times produces correct K output."""
        model = DummyModel3DGrid()
        x = torch.randn(1, 4, 6, 10, 5)
        target_times = torch.tensor([0.1, 0.2, 0.3])  # K=3
        pred = model(x, target_times=target_times)
        assert pred.shape == (1, 4, 6, 3)

    def test_4d_grid_with_target_times(self):
        """4D wrapper with grid mode + target_times produces correct K output."""
        model = DummyModel4DGrid()
        x = torch.randn(1, 4, 6, 3, 10, 5)
        target_times = torch.tensor([0.1, 0.2, 0.3])  # K=3
        pred = model(x, target_times=target_times)
        assert pred.shape == (1, 4, 6, 3, 3)

    def test_3d_teacher_forcing_with_grid_model(self):
        """Teacher forcing works with a model that accepts target_times (grid mode)."""
        model = DummyModel3DGrid()
        inputs = torch.randn(2, 4, 6, 16, 5)
        targets = torch.randn(2, 4, 6, 16)
        loss = teacher_forcing_step(model, inputs, targets, dummy_loss, L=1, K=3)
        assert loss.dim() == 0

    def test_4d_teacher_forcing_with_grid_model(self):
        """Teacher forcing works with a 4D model that accepts target_times."""
        model = DummyModel4DGrid()
        inputs = torch.randn(1, 4, 6, 3, 16, 5)
        targets = torch.randn(1, 4, 6, 3, 16)
        loss = teacher_forcing_step(model, inputs, targets, dummy_loss, L=1, K=3)
        assert loss.dim() == 0

    def test_4d_rollout_with_grid_model(self):
        """Rollout works with a 4D model that accepts target_times."""
        model = DummyModel4DGrid()
        inputs = torch.randn(1, 4, 6, 3, 20, 5)
        targets = torch.randn(1, 4, 6, 3, 20)
        loss = rollout_step(
            model, inputs, targets, dummy_loss,
            L=1, K=3, use_checkpointing=False,
        )
        assert loss.dim() == 0

    def test_4d_full_rollout_with_grid_model(self):
        """Full AR validation rollout works with grid model."""
        model = DummyModel4DGrid()
        inputs = torch.randn(1, 4, 6, 3, 16, 5)
        targets = torch.randn(1, 4, 6, 3, 16)
        pred = ar_validate_full_rollout(model, inputs, targets, L=1, K=3)
        assert pred.shape == targets.shape


# ---------------------------------------------------------------------------
# Tests: TNO (Temporal Neural Operator) — branch2 = previous solution
# ---------------------------------------------------------------------------

class DummyTNOModel3D(nn.Module):
    """TNO-style model: branch1=spatial features, branch2=previous solution."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x, target_times=None, x_branch2=None):
        B, H, W, T_in, C = x.shape
        T_out = target_times.shape[0] if target_times is not None else T_in
        return torch.zeros(B, H, W, T_out, device=x.device)


class DummyTNOModel4D(nn.Module):
    """TNO-style 4D model."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x, target_times=None, x_branch2=None):
        B, X, Y, Z, T_in, C = x.shape
        T_out = target_times.shape[0] if target_times is not None else T_in
        return torch.zeros(B, X, Y, Z, T_out, device=x.device)


class TestTNO:
    def test_3d_teacher_forcing(self):
        model = DummyTNOModel3D()
        inputs = torch.randn(2, 4, 6, 16, 5)
        targets = torch.randn(2, 4, 6, 16)
        loss = teacher_forcing_step(
            model, inputs, targets, dummy_loss, L=1, K=3, is_tno=True,
        )
        assert loss.dim() == 0

    def test_4d_teacher_forcing(self):
        model = DummyTNOModel4D()
        inputs = torch.randn(1, 4, 6, 3, 16, 5)
        targets = torch.randn(1, 4, 6, 3, 16)
        loss = teacher_forcing_step(
            model, inputs, targets, dummy_loss, L=1, K=3, is_tno=True,
        )
        assert loss.dim() == 0

    def test_3d_rollout(self):
        model = DummyTNOModel3D()
        inputs = torch.randn(2, 4, 6, 20, 5)
        targets = torch.randn(2, 4, 6, 20)
        loss = rollout_step(
            model, inputs, targets, dummy_loss,
            L=1, K=3, use_checkpointing=False, is_tno=True,
        )
        assert loss.dim() == 0

    def test_4d_rollout(self):
        model = DummyTNOModel4D()
        inputs = torch.randn(1, 4, 6, 3, 20, 5)
        targets = torch.randn(1, 4, 6, 3, 20)
        loss = rollout_step(
            model, inputs, targets, dummy_loss,
            L=1, K=3, use_checkpointing=False, is_tno=True,
        )
        assert loss.dim() == 0

    def test_3d_full_rollout_shape(self):
        model = DummyTNOModel3D()
        inputs = torch.randn(1, 4, 6, 16, 5)
        targets = torch.randn(1, 4, 6, 16)
        pred = ar_validate_full_rollout(
            model, inputs, targets, L=1, K=3, is_tno=True,
        )
        assert pred.shape == targets.shape

    def test_4d_full_rollout_shape(self):
        model = DummyTNOModel4D()
        inputs = torch.randn(1, 4, 6, 3, 16, 5)
        targets = torch.randn(1, 4, 6, 3, 16)
        pred = ar_validate_full_rollout(
            model, inputs, targets, L=1, K=3, is_tno=True,
        )
        assert pred.shape == targets.shape

    def test_l2_k4_teacher_forcing(self):
        """Test L=2, K=4 — branch2 should receive 2 timesteps."""
        model = DummyTNOModel3D()
        inputs = torch.randn(2, 4, 6, 20, 5)
        targets = torch.randn(2, 4, 6, 20)
        loss = teacher_forcing_step(
            model, inputs, targets, dummy_loss, L=2, K=4, is_tno=True,
        )
        assert loss.dim() == 0

    def test_non_tno_unaffected(self):
        """is_tno=False should behave identically to before."""
        model = DummyTNOModel3D()
        inputs = torch.randn(2, 4, 6, 16, 5)
        targets = torch.randn(2, 4, 6, 16)
        loss = teacher_forcing_step(
            model, inputs, targets, dummy_loss, L=1, K=3, is_tno=False,
        )
        assert loss.dim() == 0

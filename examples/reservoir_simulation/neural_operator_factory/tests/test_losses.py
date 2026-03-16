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

"""Comprehensive unit tests for loss functions."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.losses import SimpleRelativeL2Loss, UnifiedLoss, get_loss_function


# ---------------------------------------------------------------------------
# Helpers: build inputs with grid-width channels (NOF convention)
# ---------------------------------------------------------------------------


def _make_2d_inputs(B, H, W, T, C, dx=None, dy=None):
    """Create (B, H, W, T, C) inputs with grid widths in last 3 channels."""
    inputs = torch.randn(B, H, W, T, C)
    if dx is None:
        dx = torch.ones(W)
    if dy is None:
        dy = torch.ones(H)
    dt = torch.linspace(0, 30, T)
    inputs[..., -3] = dx.view(1, 1, W, 1).expand(B, H, W, T)
    inputs[..., -2] = dy.view(1, H, 1, 1).expand(B, H, W, T)
    inputs[..., -1] = dt.view(1, 1, 1, T).expand(B, H, W, T)
    return inputs


def _make_3d_inputs(B, X, Y, Z, T, C, dx=None, dy=None, dz=None):
    """Create (B, X, Y, Z, T, C) inputs with grid widths in last 4 channels."""
    inputs = torch.randn(B, X, Y, Z, T, C)
    if dx is None:
        dx = torch.ones(X)
    if dy is None:
        dy = torch.ones(Y)
    if dz is None:
        dz = torch.ones(Z)
    dt = torch.linspace(0, 30, T)
    inputs[..., -4] = dx.view(1, X, 1, 1, 1).expand(B, X, Y, Z, T)
    inputs[..., -3] = dy.view(1, 1, Y, 1, 1).expand(B, X, Y, Z, T)
    inputs[..., -2] = dz.view(1, 1, 1, Z, 1).expand(B, X, Y, Z, T)
    inputs[..., -1] = dt.view(1, 1, 1, 1, T).expand(B, X, Y, Z, T)
    return inputs


# ===================================================================
# SimpleRelativeL2Loss
# ===================================================================


class TestSimpleRelativeL2Loss:
    """Tests for SimpleRelativeL2Loss."""

    def test_zero_for_identical(self):
        target = torch.randn(2, 8, 16, 4)
        loss = SimpleRelativeL2Loss()(target.clone(), target)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_positive_for_different(self):
        target = torch.randn(2, 8, 16, 4)
        pred = target + 0.1 * torch.randn_like(target)
        assert SimpleRelativeL2Loss()(pred, target) > 0

    def test_scale_invariance(self):
        target = torch.randn(2, 8, 16, 4) + 2
        pred = target + 0.1 * torch.randn_like(target)
        fn = SimpleRelativeL2Loss()
        assert torch.isclose(fn(pred, target), fn(pred * 5, target * 5), rtol=1e-4)

    def test_epsilon_prevents_nan(self):
        target = torch.zeros(2, 4, 4, 2)
        pred = torch.ones(2, 4, 4, 2)
        loss = SimpleRelativeL2Loss(eps=1e-8)(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


# ===================================================================
# UnifiedLoss — Data Losses
# ===================================================================


class TestDataLosses:
    """Tests for DataLosses."""

    @pytest.mark.parametrize("loss_type", ["mse", "l1", "relative_l2", "huber"])
    def test_all_types_run(self, loss_type):
        pred = torch.randn(2, 8, 16, 4)
        target = torch.randn(2, 8, 16, 4)
        fn = UnifiedLoss(types=[loss_type], weights=[1.0])
        loss = fn(pred, target)
        assert loss > 0
        assert not torch.isnan(loss)

    def test_mse_value(self):
        target = torch.zeros(1, 4, 4, 2)
        pred = torch.ones(1, 4, 4, 2)
        loss = UnifiedLoss(types=["mse"])(pred, target)
        assert torch.isclose(loss, torch.tensor(1.0))

    def test_l1_value(self):
        target = torch.zeros(1, 4, 4, 2)
        pred = torch.full_like(target, 2.0)
        loss = UnifiedLoss(types=["l1"])(pred, target)
        assert torch.isclose(loss, torch.tensor(2.0))

    def test_huber_equals_mse_for_small_errors(self):
        target = torch.randn(2, 8, 8, 4)
        pred = target + 0.01 * torch.randn_like(target)
        mse_loss = UnifiedLoss(types=["mse"])(pred, target)
        huber_loss = UnifiedLoss(types=["huber"], huber_delta=1.0)(pred, target)
        assert torch.isclose(mse_loss, huber_loss * 2, rtol=0.1)

    def test_relative_l2_epsilon_zero_target(self):
        target = torch.zeros(2, 4, 4, 2)
        pred = torch.ones(2, 4, 4, 2)
        loss = UnifiedLoss(types=["relative_l2"], eps=1e-6)(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_multiple_losses_combined(self):
        pred = torch.randn(2, 8, 8, 4)
        target = torch.randn(2, 8, 8, 4)
        fn_single = UnifiedLoss(types=["mse"], weights=[1.0])
        fn_multi = UnifiedLoss(types=["mse", "l1"], weights=[1.0, 0.0])
        assert torch.isclose(fn_single(pred, target), fn_multi(pred, target))

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Loss type must be"):
            UnifiedLoss(types=["invalid"])

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError, match="reduction"):
            UnifiedLoss(reduction="invalid")

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            UnifiedLoss(types=["mse", "l1"], weights=[1.0])

    def test_gradient_flow(self):
        target = torch.randn(2, 8, 8, 4)
        pred = torch.randn(2, 8, 8, 4, requires_grad=True)
        loss = UnifiedLoss(types=["mse"])(pred, target)
        loss.backward()
        assert pred.grad is not None


# ===================================================================
# UnifiedLoss — Masking
# ===================================================================


class TestMasking:
    """Tests for Masking."""

    def test_mse_mask_only_active(self):
        """MSE should average only over active cells."""
        B, H, W, T = 1, 4, 4, 2
        target = torch.zeros(B, H, W, T)
        pred = torch.ones(B, H, W, T)
        mask = torch.zeros(H, W, dtype=torch.bool)
        mask[0, 0] = True  # only one cell active

        fn = UnifiedLoss(types=["mse"])
        loss = fn(pred, target, spatial_mask=mask)
        assert torch.isclose(loss, torch.tensor(1.0))

    def test_l1_mask_only_active(self):
        B, H, W, T = 1, 4, 4, 2
        target = torch.zeros(B, H, W, T)
        pred = torch.full((B, H, W, T), 3.0)
        mask = torch.zeros(H, W, dtype=torch.bool)
        mask[0, 0] = True

        loss = UnifiedLoss(types=["l1"])(pred, target, spatial_mask=mask)
        assert torch.isclose(loss, torch.tensor(3.0))

    def test_mask_zeros_dont_leak_2d(self):
        """Error only in masked-out region => loss = 0."""
        B, H, W, T = 1, 8, 8, 4
        target = torch.randn(B, H, W, T)
        pred = target.clone()
        pred[:, :4, :, :] += 10.0

        mask = torch.zeros(H, W, dtype=torch.bool)
        mask[4:, :] = True

        loss = UnifiedLoss(types=["mse"])(pred, target, spatial_mask=mask)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_mask_3d(self):
        B, X, Y, Z, T = 1, 4, 4, 2, 3
        target = torch.randn(B, X, Y, Z, T)
        pred = target.clone()
        pred[:, :2, :, :, :] += 10.0

        mask = torch.zeros(X, Y, Z, dtype=torch.bool)
        mask[2:, :, :] = True

        loss = UnifiedLoss(types=["mse"])(pred, target, spatial_mask=mask)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


# ===================================================================
# UnifiedLoss — Derivative Loss (2D)
# ===================================================================


class TestDerivativeLoss2D:
    """Tests for DerivativeLoss2D."""

    def test_derivative_dx_runs(self):
        B, H, W, T, C = 2, 8, 16, 4, 12
        inputs = _make_2d_inputs(B, H, W, T, C)
        pred = torch.randn(B, H, W, T)
        target = torch.randn(B, H, W, T)
        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dx"]},
        )
        loss = fn(pred, target, inputs)
        assert loss > 0
        assert not torch.isnan(loss)

    def test_derivative_dy_runs(self):
        B, H, W, T, C = 2, 8, 16, 4, 12
        inputs = _make_2d_inputs(B, H, W, T, C)
        pred = torch.randn(B, H, W, T)
        target = torch.randn(B, H, W, T)
        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dy"]},
        )
        loss = fn(pred, target, inputs)
        assert loss > 0

    def test_derivative_both_dims(self):
        B, H, W, T, C = 2, 8, 16, 4, 12
        inputs = _make_2d_inputs(B, H, W, T, C)
        pred = torch.randn(B, H, W, T)
        target = torch.randn(B, H, W, T)
        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dx", "dy"]},
        )
        loss = fn(pred, target, inputs)
        assert loss > 0

    def test_derivative_zero_for_identical(self):
        B, H, W, T, C = 1, 8, 16, 4, 12
        inputs = _make_2d_inputs(B, H, W, T, C)
        target = torch.randn(B, H, W, T)
        pred = target.clone()
        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 1.0, "dims": ["dx"]},
        )
        loss = fn(pred, target, inputs)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_derivative_gradient_flow(self):
        B, H, W, T, C = 2, 8, 16, 4, 12
        inputs = _make_2d_inputs(B, H, W, T, C)
        target = torch.randn(B, H, W, T)
        pred = torch.randn(B, H, W, T, requires_grad=True)
        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dx"]},
        )
        loss = fn(pred, target, inputs)
        loss.backward()
        assert pred.grad is not None

    def test_derivative_non_uniform_grid(self):
        """Verify derivative uses non-uniform spacing correctly."""
        B, H, W, T, C = 1, 4, 6, 2, 12
        dx = torch.tensor([10.0, 20.0, 30.0, 15.0, 25.0, 10.0])
        dy = torch.tensor([5.0, 10.0, 5.0, 10.0])
        inputs = _make_2d_inputs(B, H, W, T, C, dx=dx, dy=dy)

        target = torch.zeros(B, H, W, T)
        pred = torch.zeros(B, H, W, T)
        pred[0, :, :, 0] = torch.arange(W).float().unsqueeze(0).expand(H, W)

        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 1.0, "dims": ["dx"]},
        )
        loss = fn(pred, target, inputs)
        assert loss > 0
        assert not torch.isnan(loss)

    def test_requires_inputs(self):
        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dx"]},
        )
        with pytest.raises(ValueError, match="inputs required"):
            fn(torch.randn(2, 8, 8, 4), torch.randn(2, 8, 8, 4))


# ===================================================================
# UnifiedLoss — Derivative Loss (3D)
# ===================================================================


class TestDerivativeLoss3D:
    """Tests for DerivativeLoss3D."""

    def test_derivative_all_3d_dims(self):
        B, X, Y, Z, T, C = 2, 6, 8, 4, 3, 11
        inputs = _make_3d_inputs(B, X, Y, Z, T, C)
        pred = torch.randn(B, X, Y, Z, T)
        target = torch.randn(B, X, Y, Z, T)
        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={
                "enabled": True,
                "weight": 0.5,
                "dims": ["dx", "dy", "dz"],
            },
        )
        loss = fn(pred, target, inputs)
        assert loss > 0
        assert not torch.isnan(loss)

    def test_derivative_single_dim_3d(self):
        B, X, Y, Z, T, C = 1, 6, 8, 4, 2, 11
        inputs = _make_3d_inputs(B, X, Y, Z, T, C)
        pred = torch.randn(B, X, Y, Z, T)
        target = torch.randn(B, X, Y, Z, T)
        for dim in ["dx", "dy", "dz"]:
            fn = UnifiedLoss(
                types=["mse"],
                derivative_config={"enabled": True, "weight": 0.5, "dims": [dim]},
            )
            loss = fn(pred, target, inputs)
            assert loss > 0, f"Failed for {dim}"

    def test_invalid_dim_for_2d_raises(self):
        B, H, W, T, C = 1, 8, 16, 4, 12
        inputs = _make_2d_inputs(B, H, W, T, C)
        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dz"]},
        )
        with pytest.raises(ValueError, match="not valid for 2D"):
            fn(torch.randn(B, H, W, T), torch.randn(B, H, W, T), inputs)


# ===================================================================
# get_loss_function factory
# ===================================================================


class TestFactory:
    """Tests for Factory."""

    def test_defaults(self):
        fn = get_loss_function({})
        assert isinstance(fn, UnifiedLoss)
        assert fn.loss_types == ["relative_l2"]

    def test_with_physics(self):
        cfg = {
            "types": ["mse"],
            "weights": [1.0],
            "physics": {"mass_conservation": {"enabled": True, "weight": 0.5}},
        }
        fn = get_loss_function(cfg, variable="saturation")
        assert "mass_conservation" in fn._physics_losses

    def test_new_derivative_config(self):
        cfg = {
            "types": ["mse"],
            "weights": [1.0],
            "derivative": {"enabled": True, "weight": 0.3, "dims": ["dx", "dy"]},
        }
        fn = get_loss_function(cfg)
        assert fn._deriv_enabled is True
        assert fn._deriv_weight == 0.3
        assert fn._deriv_dims == ["dx", "dy"]

    def test_pressure_warning(self):
        cfg = {
            "types": ["mse"],
            "weights": [1.0],
            "physics": {"mass_conservation": {"enabled": True, "weight": 1.0}},
        }
        with pytest.warns(UserWarning, match="pressure"):
            get_loss_function(cfg, variable="pressure")


# ===================================================================
# AR window compatibility
# ===================================================================


class TestARCompatibility:
    """Tests for ARCompatibility."""

    def test_single_timestep_2d(self):
        B, H, W, C = 2, 8, 16, 12
        pred = torch.randn(B, H, W, 1)
        target = torch.randn(B, H, W, 1)
        inputs = _make_2d_inputs(B, H, W, 1, C)
        fn = UnifiedLoss(
            types=["relative_l2"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dx"]},
        )
        loss = fn(pred, target, inputs)
        assert not torch.isnan(loss)

    def test_small_window_3d(self):
        B, X, Y, Z, C = 2, 6, 8, 4, 11
        K = 3
        pred = torch.randn(B, X, Y, Z, K)
        target = torch.randn(B, X, Y, Z, K)
        inputs = _make_3d_inputs(B, X, Y, Z, K, C)
        fn = UnifiedLoss(types=["mse"])
        loss = fn(pred, target, inputs)
        assert not torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

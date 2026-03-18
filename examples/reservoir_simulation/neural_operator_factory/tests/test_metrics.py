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

"""Unit tests for evaluation metrics (numpy and torch)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.metrics import (
    mean_relative_error,
    mean_plume_error,
    mean_absolute_error,
    max_absolute_error,
    compute_r2_score,
    compute_relative_l2_error,
    compute_relative_l1_error,
    normalized_mse,
    peak_signal_to_noise_ratio,
    mse_torch,
    rmse_torch,
    mae_torch,
    relative_l2_torch,
    relative_l1_torch,
    r2_score_torch,
    max_error_torch,
    psnr_torch,
)


class TestNumpyMetrics:
    """Tests for numpy-based metrics."""

    def test_mae_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        assert abs(mean_absolute_error(y_pred, y_true) - 0.5) < 1e-8

    def test_mae_identical(self):
        y = np.random.randn(100)
        assert mean_absolute_error(y, y) == 0.0

    def test_max_absolute_error(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 3.0, 2.0])
        assert abs(max_absolute_error(y_pred, y_true) - 3.0) < 1e-8

    def test_mre_known_value(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 21.0, 31.0])
        data_range = 30.0 - 10.0  # 20
        expected = np.mean(np.abs(y_pred - y_true)) / data_range  # 1/20 = 0.05
        assert abs(mean_relative_error(y_pred, y_true) - expected) < 1e-8

    def test_mpe_only_plume_region(self):
        y_true = np.array([0.0, 0.0, 0.5, 0.8])
        y_pred = np.array([0.0, 0.0, 0.6, 0.9])
        mpe = mean_plume_error(y_pred, y_true)
        assert abs(mpe - 0.1) < 1e-8

    def test_mpe_no_plume(self):
        y_true = np.zeros(10)
        y_pred = np.zeros(10)
        assert mean_plume_error(y_pred, y_true) == 0.0

    def test_r2_perfect(self):
        y = np.random.randn(50)
        assert abs(compute_r2_score(y, y) - 1.0) < 1e-8

    def test_r2_constant_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full_like(y_true, y_true.mean())
        assert abs(compute_r2_score(y_pred, y_true)) < 1e-8

    def test_r2_negative_for_bad_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        assert compute_r2_score(y_pred, y_true) < 0

    def test_relative_l2_known(self):
        y_true = np.array([3.0, 4.0])  # norm = 5
        y_pred = np.array([0.0, 0.0])  # diff norm = 5
        assert abs(compute_relative_l2_error(y_pred, y_true) - 1.0) < 1e-6

    def test_relative_l1_known(self):
        y_true = np.array([1.0, 2.0, 3.0])  # L1 norm = 6
        y_pred = np.array([0.0, 0.0, 0.0])  # diff L1 = 6
        assert abs(compute_relative_l1_error(y_pred, y_true) - 1.0) < 1e-6

    def test_nmse_variance(self):
        y_true = np.array([1.0, 3.0])  # var = 1
        y_pred = np.array([2.0, 2.0])  # mse = 1
        assert abs(normalized_mse(y_pred, y_true, "variance") - 1.0) < 1e-6

    def test_psnr_perfect(self):
        y = np.random.randn(100)
        assert peak_signal_to_noise_ratio(y, y) == float("inf")

    def test_psnr_known(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.0, 0.9])  # mse = 0.005, range = 1
        psnr = peak_signal_to_noise_ratio(y_pred, y_true)
        expected = 20 * np.log10(1.0 / np.sqrt(0.005))
        assert abs(psnr - expected) < 1e-4


class TestTorchMetrics:
    """Tests for torch-based metrics."""

    def test_mse_torch_value(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])
        assert torch.isclose(mse_torch(pred, target), torch.tensor(0.25))

    def test_rmse_torch_value(self):
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([2.0, 3.0])
        assert torch.isclose(rmse_torch(pred, target), torch.tensor(1.0))

    def test_mae_torch_value(self):
        pred = torch.tensor([0.0, 0.0])
        target = torch.tensor([1.0, 3.0])
        assert torch.isclose(mae_torch(pred, target), torch.tensor(2.0))

    def test_relative_l2_torch_known(self):
        target = torch.tensor([3.0, 4.0])  # norm = 5
        pred = torch.zeros(2)
        assert torch.isclose(
            relative_l2_torch(pred, target), torch.tensor(1.0), atol=1e-6
        )

    def test_relative_l1_torch_known(self):
        target = torch.tensor([1.0, 2.0, 3.0])
        pred = torch.zeros(3)
        assert torch.isclose(
            relative_l1_torch(pred, target), torch.tensor(1.0), atol=1e-6
        )

    def test_r2_torch_perfect(self):
        y = torch.randn(50)
        assert torch.isclose(r2_score_torch(y, y), torch.tensor(1.0))

    def test_r2_torch_negative(self):
        target = torch.tensor([1.0, 2.0, 3.0])
        pred = torch.tensor([10.0, 20.0, 30.0])
        assert r2_score_torch(pred, target) < 0

    def test_max_error_torch(self):
        pred = torch.tensor([0.0, 0.0])
        target = torch.tensor([1.0, 5.0])
        assert torch.isclose(max_error_torch(pred, target), torch.tensor(5.0))

    def test_psnr_torch_perfect(self):
        y = torch.randn(50)
        assert psnr_torch(y, y) == float("inf")


class TestTorchNumpyConsistency:
    """Verify torch and numpy metrics agree on the same data."""

    def test_mae_consistency(self):
        pred_np = np.random.randn(100)
        target_np = np.random.randn(100)
        np_val = mean_absolute_error(pred_np, target_np)
        torch_val = mae_torch(torch.tensor(pred_np), torch.tensor(target_np)).item()
        assert abs(np_val - torch_val) < 1e-6

    def test_relative_l2_consistency(self):
        pred_np = np.random.randn(100)
        target_np = np.random.randn(100) + 2
        np_val = compute_relative_l2_error(pred_np, target_np)
        torch_val = relative_l2_torch(
            torch.tensor(pred_np), torch.tensor(target_np)
        ).item()
        assert abs(np_val - torch_val) < 1e-5

    def test_r2_consistency(self):
        pred_np = np.random.randn(100)
        target_np = np.random.randn(100)
        np_val = compute_r2_score(pred_np, target_np)
        torch_val = r2_score_torch(
            torch.tensor(pred_np), torch.tensor(target_np)
        ).item()
        assert abs(np_val - torch_val) < 1e-5


class TestEdgeCases:
    """Edge cases for metrics."""

    def test_single_element(self):
        y = np.array([5.0])
        assert mean_absolute_error(y, y) == 0.0
        assert compute_r2_score(y, y) == 1.0

    def test_zero_target_relative_l2(self):
        pred = np.array([1.0])
        target = np.array([0.0])
        val = compute_relative_l2_error(pred, target)
        assert np.isfinite(val)

    def test_constant_target_r2(self):
        target = np.ones(10)
        pred = np.ones(10) * 1.1
        r2 = compute_r2_score(pred, target)
        assert r2 == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

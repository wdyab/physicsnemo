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
Training utilities: loss functions and evaluation metrics.
"""

from training.losses import (
    SimpleRelativeL2Loss,
    UnifiedLoss,
    get_loss_function,
)
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

__all__ = [
    # Losses
    "SimpleRelativeL2Loss",
    "UnifiedLoss",
    "get_loss_function",
    # Metrics
    "mean_relative_error",
    "mean_plume_error",
    "mean_absolute_error",
    "max_absolute_error",
    "compute_r2_score",
    "compute_relative_l2_error",
    "compute_relative_l1_error",
    "normalized_mse",
    "peak_signal_to_noise_ratio",
    "mse_torch",
    "rmse_torch",
    "mae_torch",
    "relative_l2_torch",
    "relative_l1_torch",
    "r2_score_torch",
    "max_error_torch",
    "psnr_torch",
]

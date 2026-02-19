# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for padding utilities (dimension-agnostic)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.padding import (
    compute_right_pad_to_multiple,
    compute_right_pad_to_multiple_per_dim,
    pad_right_nd,
    pad_spatial_right,
)


class TestComputePads:
    def test_compute_right_pad_to_multiple(self):
        assert compute_right_pad_to_multiple((16, 24), multiple=8, min_right_pad=0) == (0, 0)
        assert compute_right_pad_to_multiple((17, 24), multiple=8, min_right_pad=0) == (7, 0)
        assert compute_right_pad_to_multiple((16, 24), multiple=8, min_right_pad=8) == (8, 8)
        # Ensure min_right_pad does not break alignment (d + pad must stay multiple-of-8)
        assert compute_right_pad_to_multiple((46,), multiple=8, min_right_pad=8) == (10,)

    def test_compute_right_pad_to_multiple_per_dim(self):
        assert compute_right_pad_to_multiple_per_dim((16, 17), multiple=8, min_right_pad=(0, 0)) == (0, 7)
        assert compute_right_pad_to_multiple_per_dim((16, 17), multiple=8, min_right_pad=(8, 0)) == (8, 7)


class TestPadRightNd:
    def test_replicate_right_pad_6d(self):
        # Shape: (B, X, Y, Z, T, C)
        x = torch.zeros(1, 1, 1, 1, 2, 1)
        x[..., 0, 0] = 10.0
        x[..., 1, 0] = 20.0

        y = pad_right_nd(x, dims=(4,), right_pad=(3,), mode="replicate")
        assert y.shape == (1, 1, 1, 1, 5, 1)
        # Last value should replicate the original last along T (20)
        assert y[0, 0, 0, 0, -1, 0].item() == 20.0


class TestPadSpatialRight:
    def test_2d_spatial_keeps_rest(self):
        x = torch.randn(2, 5, 7, 3, 4)  # (B,H,W,T,C)
        y = pad_spatial_right(x, spatial_ndim=2, right_pad=(1, 2), mode="replicate")
        assert y.shape == (2, 6, 9, 3, 4)

    def test_3d_spatial_includes_time_when_requested(self):
        x = torch.randn(2, 5, 7, 3, 4)  # (B,H,W,T,C)
        y = pad_spatial_right(x, spatial_ndim=3, right_pad=(1, 2, 3), mode="replicate")
        assert y.shape == (2, 6, 9, 6, 4)

    def test_4d_spatial_works_for_6d_inputs(self):
        # (B,X,Y,Z,T,C)
        x = torch.tensor([[[[[[10.0], [20.0]]]]]])  # (1,1,1,1,2,1)
        y = pad_spatial_right(x, spatial_ndim=4, right_pad=(1, 1, 1, 2), mode="replicate")
        assert y.shape == (1, 2, 2, 2, 4, 1)
        assert y[0, -1, -1, -1, -1, 0].item() == 20.0


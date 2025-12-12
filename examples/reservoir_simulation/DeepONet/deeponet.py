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
U-DeepONet: Deep Operator Network with U-Net enhanced branch network.

This module implements DeepONet variants for learning operators in reservoir simulation:
- Standard DeepONet: MLP branch + MLP trunk
- U-DeepONet: U-Net branch + MLP trunk (this implementation)
- Fourier-DeepONet: FNO branch + MLP trunk (future work)

The architecture separates spatial encoding (branch) from temporal/coordinate 
encoding (trunk), then combines them via inner product for operator learning.

Reference:
    Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021).
    Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.
    Nature Machine Intelligence, 3(3), 218-229.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from typing import Optional, List, Union

from physicsnemo.models.module import Module
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.models.layers import get_activation
from physicsnemo.models.unet import UNet as PhysicsNemoUNet

from unet3d import UNet2D, UNet3D


# ==============================================================================
# Trunk Network (Time/Coordinate Encoder)
# ==============================================================================


class TrunkNet(nn.Module):
    """Trunk network for encoding query coordinates (e.g., time).
    
    This MLP encodes the coordinates where we want to evaluate the operator.
    For time-dependent problems, it encodes temporal coordinates.
    
    Parameters
    ----------
    in_features : int
        Input dimension (e.g., 1 for time, 3 for (x,y,t))
    out_features : int
        Output dimension (must match branch network output width)
    hidden_width : int
        Hidden layer width
    num_layers : int
        Number of hidden layers
    activation : str
        Activation function: 'sin', 'relu', 'gelu', 'tanh', etc.
        Note: 'sin' activation is common in DeepONet for periodic/smooth functions
    
    Example
    -------
    >>> trunk = TrunkNet(in_features=1, out_features=64, hidden_width=128, num_layers=6)
    >>> t = torch.linspace(0, 1, 24).unsqueeze(-1)  # (24, 1)
    >>> features = trunk(t)  # (24, 64)
    """
    
    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 64,
        hidden_width: int = 128,
        num_layers: int = 6,
        activation: str = "sin",
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_width = hidden_width
        self.num_layers = num_layers
        self.activation_name = activation
        
        # Set activation function
        if activation.lower() == "sin":
            self.activation = torch.sin
        else:
            self.activation = get_activation(activation)
        
        # Build MLP layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(self._create_linear(in_features, hidden_width))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(self._create_linear(hidden_width, hidden_width))
        
        # Output layer
        self.output_layer = self._create_linear(hidden_width, out_features)
    
    def _create_linear(self, in_dim: int, out_dim: int) -> nn.Linear:
        """Create a linear layer with Xavier initialization."""
        layer = nn.Linear(in_dim, out_dim)
        init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            init.zeros_(layer.bias)
        return layer
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through trunk network.
        
        Parameters
        ----------
        x : Tensor
            Input coordinates of shape (N, in_features) or (B, N, in_features)
        
        Returns
        -------
        Tensor
            Encoded features of shape (N, out_features) or (B, N, out_features)
        """
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.activation(self.output_layer(x))
        return x


# ==============================================================================
# Branch Network Components
# ==============================================================================


class UNetBlock2D(nn.Module):
    """2D U-Net block for branch network (single spatial slice).
    
    This is a lightweight U-Net designed for use within DeepONet's branch network.
    It processes 2D spatial data (H, W) with skip connections.
    
    Parameters
    ----------
    in_channels : int
        Input channels (should equal output for residual connection)
    out_channels : int
        Output channels
    kernel_size : int
        Convolution kernel size
    dropout_rate : float
        Dropout rate (0.0 to disable)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.conv1 = self._conv_block(in_channels, out_channels, kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv2 = self._conv_block(out_channels, out_channels, kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv2_1 = self._conv_block(out_channels, out_channels, kernel_size, stride=1, dropout_rate=dropout_rate)
        self.conv3 = self._conv_block(out_channels, out_channels, kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv3_1 = self._conv_block(out_channels, out_channels, kernel_size, stride=1, dropout_rate=dropout_rate)
        
        # Decoder
        self.deconv2 = self._deconv_block(out_channels, out_channels)
        self.deconv1 = self._deconv_block(out_channels * 2, out_channels)
        self.deconv0 = self._deconv_block(out_channels * 2, out_channels)
        
        # Output
        self.output_layer = nn.Conv2d(
            out_channels + in_channels, out_channels,
            kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2
        )
        self._initialize_weights(self.output_layer)
    
    def _initialize_weights(self, module):
        """Xavier initialization for conv layers."""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            init.xavier_normal_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
    
    def _conv_block(self, in_channels: int, out_channels: int, kernel_size: int, 
                    stride: int, dropout_rate: float) -> nn.Sequential:
        """Create encoder conv block."""
        layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        layers.apply(self._initialize_weights)
        return layers
    
    def _deconv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create decoder deconv block."""
        layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        layers.apply(self._initialize_weights)
        return layers
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through U-Net block.
        
        Parameters
        ----------
        x : Tensor
            Input of shape (B, C, H, W)
        
        Returns
        -------
        Tensor
            Output of shape (B, C, H, W)
        """
        # Encoder
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        
        # Decoder with skip connections
        out_deconv2 = self.deconv2(out_conv3)
        concat2 = torch.cat((out_conv2, out_deconv2), dim=1)
        
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), dim=1)
        
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), dim=1)
        
        out = self.output_layer(concat0)
        return out


# ==============================================================================
# U-DeepONet Architecture
# ==============================================================================


class UDeepONet(Module):
    """U-DeepONet: Deep Operator Network with U-Net enhanced branch network.
    
    Architecture:
    - Branch Network: Lifting + N U-Net blocks (encodes spatial input function)
    - Trunk Network: MLP (encodes query coordinates, e.g., time)
    - Combination: Inner product between branch and trunk outputs
    - Projection: MLP to final output
    
    The branch network processes the input function (e.g., initial conditions,
    permeability field) while the trunk network encodes where to evaluate
    the output (e.g., at different time points).
    
    Parameters
    ----------
    in_channels : int
        Number of input channels for branch network
    out_channels : int
        Number of output channels
    width : int
        Latent feature dimension
    num_unet_blocks : int
        Number of U-Net blocks in branch network
    unet_kernel_size : int
        Kernel size for U-Net convolutions
    unet_dropout : float
        Dropout rate in U-Net blocks
    unet_type : str
        Type of U-Net: "custom" (UNetBlock2D) or "physicsnemo"
    trunk_in_features : int
        Input dimension for trunk network (1 for time only)
    trunk_hidden_width : int
        Hidden width in trunk network
    trunk_num_layers : int
        Number of layers in trunk network
    trunk_activation : str
        Activation for trunk: 'sin', 'relu', 'gelu', etc.
    branch_activation : str
        Activation after U-Net blocks: 'sin', 'relu', 'gelu', etc.
    projection_hidden_width : int
        Hidden width in projection MLP
    projection_num_layers : int
        Number of layers in projection MLP
    
    Example
    -------
    >>> model = UDeepONet(
    ...     in_channels=12, out_channels=1, width=64,
    ...     num_unet_blocks=3, trunk_num_layers=6
    ... )
    >>> # x_spatial: (B, H, W, C=12), x_time: (T,) or (T, 1)
    >>> output = model(x_spatial, x_time)  # (B, H, W, T)
    """
    
    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 1,
        width: int = 64,
        num_unet_blocks: int = 3,
        unet_kernel_size: int = 3,
        unet_dropout: float = 0.0,
        unet_type: str = "custom",
        trunk_in_features: int = 1,
        trunk_hidden_width: int = 128,
        trunk_num_layers: int = 6,
        trunk_activation: str = "sin",
        branch_activation: str = "sin",
        projection_hidden_width: int = 128,
        projection_num_layers: int = 2,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.num_unet_blocks = num_unet_blocks
        self.unet_type = unet_type.lower()
        
        # Branch activation
        if branch_activation.lower() == "sin":
            self.branch_activation = torch.sin
        else:
            self.branch_activation = get_activation(branch_activation)
        
        # ===== Branch Network =====
        # Lifting layer: (B, H, W, in_channels) -> (B, H, W, width)
        self.lifting = nn.Linear(in_channels, width)
        init.xavier_normal_(self.lifting.weight)
        init.zeros_(self.lifting.bias)
        
        # U-Net blocks
        self.unet_blocks = nn.ModuleList()
        for _ in range(num_unet_blocks):
            if self.unet_type == "custom":
                self.unet_blocks.append(
                    UNetBlock2D(
                        in_channels=width,
                        out_channels=width,
                        kernel_size=unet_kernel_size,
                        dropout_rate=unet_dropout,
                    )
                )
            elif self.unet_type == "physicsnemo":
                # Use PhysicsNeMo's 2D UNet (via 3D with T=1)
                self.unet_blocks.append(
                    UNet2D(
                        input_channels=width,
                        output_channels=width,
                        kernel_size=unet_kernel_size,
                        dropout_rate=unet_dropout,
                    )
                )
            else:
                raise ValueError(f"Unknown unet_type: {self.unet_type}")
        
        # ===== Trunk Network =====
        self.trunk = TrunkNet(
            in_features=trunk_in_features,
            out_features=width,
            hidden_width=trunk_hidden_width,
            num_layers=trunk_num_layers,
            activation=trunk_activation,
        )
        
        # ===== Projection Network =====
        # After inner product: (B, T, H, W, width) -> (B, T, H, W, out_channels)
        if projection_num_layers == 1:
            self.projection = nn.Linear(width, out_channels)
        else:
            layers = []
            layers.append(nn.Linear(width, projection_hidden_width))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(projection_num_layers - 2):
                layers.append(nn.Linear(projection_hidden_width, projection_hidden_width))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(projection_hidden_width, out_channels))
            self.projection = nn.Sequential(*layers)
    
    def forward(self, x_spatial: Tensor, x_time: Tensor) -> Tensor:
        """Forward pass through U-DeepONet.
        
        Parameters
        ----------
        x_spatial : Tensor
            Spatial input of shape (B, H, W, C) where C=in_channels
        x_time : Tensor
            Time coordinates of shape (T,) or (T, 1)
        
        Returns
        -------
        Tensor
            Output of shape (B, H, W, T) or (B, H, W, T, out_channels) if out_channels > 1
        """
        batch_size = x_spatial.shape[0]
        H, W = x_spatial.shape[1], x_spatial.shape[2]
        
        # Ensure x_time has correct shape
        if x_time.dim() == 1:
            x_time = x_time.unsqueeze(-1)  # (T,) -> (T, 1)
        T = x_time.shape[0]
        
        # ===== Branch Network =====
        # Lifting: (B, H, W, C) -> (B, H, W, width)
        x_branch = self.lifting(x_spatial)
        
        # Permute for conv: (B, H, W, width) -> (B, width, H, W)
        x_branch = x_branch.permute(0, 3, 1, 2)
        
        # U-Net blocks with activation
        for unet in self.unet_blocks:
            x_branch = self.branch_activation(unet(x_branch))
        
        # Permute back: (B, width, H, W) -> (B, H, W, width)
        x_branch = x_branch.permute(0, 2, 3, 1)
        
        # ===== Trunk Network =====
        # (T, 1) -> (T, width)
        x_trunk = self.trunk(x_time)
        
        # ===== Combination via Inner Product =====
        # Branch: (B, H, W, width) -> (B, 1, H, W, width)
        # Trunk: (T, width) -> (1, T, 1, 1, width)
        x_branch = x_branch.unsqueeze(1)  # (B, 1, H, W, width)
        x_trunk = x_trunk.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1, T, 1, 1, width)
        
        # Element-wise multiplication (broadcasting)
        combined = x_branch * x_trunk  # (B, T, H, W, width)
        
        # ===== Projection =====
        output = self.projection(combined)  # (B, T, H, W, out_channels)
        
        # Squeeze if single output channel
        if self.out_channels == 1:
            output = output.squeeze(-1)  # (B, T, H, W)
        
        # Permute to match expected output format: (B, H, W, T)
        output = output.permute(0, 2, 3, 1)  # (B, H, W, T)
        
        return output
    
    def count_params(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# Network Wrapper (handles padding and data preparation)
# ==============================================================================


class UDeepONetWrapper(nn.Module):
    """Wrapper for UDeepONet that handles padding and input preparation.
    
    This wrapper:
    1. Applies padding to spatial dimensions for U-Net compatibility
    2. Separates spatial and temporal inputs from the full input tensor
    3. Calls UDeepONet with separated inputs
    4. Removes padding from output
    
    Parameters
    ----------
    padding : int
        Padding size for spatial dimensions
    **kwargs
        Arguments passed to UDeepONet
    
    Example
    -------
    >>> model = UDeepONetWrapper(padding=8, in_channels=12, width=64)
    >>> x = torch.randn(4, 96, 200, 24, 12)  # (B, H, W, T, C)
    >>> output = model(x)  # (B, H, W, T)
    """
    
    def __init__(self, padding: int = 8, **kwargs):
        super().__init__()
        
        # Ensure padding is divisible by 8 for U-Net compatibility
        if padding % 8 != 0:
            self.padding = ((padding + 7) // 8) * 8
            print(f"Warning: Padding adjusted from {padding} to {self.padding} for U-Net compatibility")
        else:
            self.padding = padding
        
        self.deeponet = UDeepONet(**kwargs)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with automatic input separation and padding.
        
        Parameters
        ----------
        x : Tensor
            Full input of shape (B, H, W, T, C) where C includes spatial features
            and the last 3 channels are grid coordinates (grid_x, grid_y, grid_t)
        
        Returns
        -------
        Tensor
            Output of shape (B, H, W, T)
        """
        batch_size = x.shape[0]
        H, W, T = x.shape[1], x.shape[2], x.shape[3]
        
        # Apply padding to spatial dimensions
        # Input: (B, H, W, T, C)
        # Pad H and W dimensions
        x_padded = F.pad(x, (0, 0, 0, 0, 0, self.padding, 0, self.padding), mode='replicate')
        
        # Separate spatial and temporal inputs
        # Spatial: Take first timestep, all spatial features except grid coords
        # Shape: (B, H+pad, W+pad, C)
        x_spatial = x_padded[:, :, :, 0, :-1]  # Exclude grid_t (last channel)
        
        # Time: Extract normalized time values
        # Shape: (T,) - time coordinates from input
        x_time = x[0, 0, 0, :, -1]  # (T,) - grid_t values
        
        # Forward through DeepONet
        output = self.deeponet(x_spatial, x_time)  # (B, H+pad, W+pad, T)
        
        # Remove padding
        output = output[:, :H, :W, :]  # (B, H, W, T)
        
        return output
    
    def count_params(self) -> int:
        """Count total number of trainable parameters."""
        return self.deeponet.count_params()


# ==============================================================================
# Factory function for easy instantiation
# ==============================================================================


def create_deeponet(config) -> nn.Module:
    """Factory function to create DeepONet from config.
    
    Parameters
    ----------
    config : DictConfig
        Configuration with deeponet parameters
    
    Returns
    -------
    nn.Module
        Configured DeepONet model
    """
    return UDeepONetWrapper(
        padding=config.get("padding", 8),
        in_channels=config.get("in_channels", 12),
        out_channels=config.get("out_channels", 1),
        width=config.get("width", 64),
        num_unet_blocks=config.get("num_unet_blocks", 3),
        unet_kernel_size=config.get("unet_kernel_size", 3),
        unet_dropout=config.get("unet_dropout", 0.0),
        unet_type=config.get("unet_type", "custom"),
        trunk_in_features=config.get("trunk_in_features", 1),
        trunk_hidden_width=config.get("trunk_hidden_width", 128),
        trunk_num_layers=config.get("trunk_num_layers", 6),
        trunk_activation=config.get("trunk_activation", "sin"),
        branch_activation=config.get("branch_activation", "sin"),
        projection_hidden_width=config.get("projection_hidden_width", 128),
        projection_num_layers=config.get("projection_num_layers", 2),
    )


# Export public API
__all__ = [
    "TrunkNet",
    "UNetBlock2D", 
    "UDeepONet",
    "UDeepONetWrapper",
    "create_deeponet",
]






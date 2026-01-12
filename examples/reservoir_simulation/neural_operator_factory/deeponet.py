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
DeepONet Variants for 2D and 3D spatial problems.

Supported variants:
    - deeponet: Basic DeepONet
    - u_deeponet: DeepONet with U-Net branch
    - fourier_deeponet: DeepONet with Fourier layers
    - conv_deeponet: DeepONet with convolutional layers
    - hybrid_deeponet: Combination of Fourier + U-Net + Conv
    - mionet: Multiple-input operator network (2 branches)
    - fourier_mionet: MIONet with Fourier layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from typing import Dict, Any, Optional

from physicsnemo.models.module import Module
from physicsnemo.models.layers import (
    SpectralConv2d,
    SpectralConv3d,
    Conv2dFCLayer,
    Conv3dFCLayer,
    ConvNdKernel1Layer,
    get_activation,
)
from physicsnemo.models.mlp import FullyConnected

from unet import UNet2D, UNet3D
from physicsnemo_unet import PhysicsNemoUNet2D, PhysicsNemoUNet3D


# =============================================================================
# Shared Components
# =============================================================================

class TrunkNet(nn.Module):
    """
    MLP trunk network for encoding query coordinates.
    
    Input: (T, in_features) - T query points with in_features dimensions
    Output: (T, out_features) - encoded representations
    """
    
    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 64,
        hidden_width: int = 128,
        num_layers: int = 6,
        activation_fn: str = "sin"
    ):
        super().__init__()
        
        if activation_fn.lower() == "sin":
            self.activation_fn = torch.sin
        else:
            self.activation_fn = get_activation(activation_fn)
        
        self.layers = nn.ModuleList()
        self.layers.append(self._make_linear(in_features, hidden_width))
        for _ in range(num_layers - 1):
            self.layers.append(self._make_linear(hidden_width, hidden_width))
        
        self.output_layer = self._make_linear(hidden_width, out_features)
    
    def _make_linear(self, in_dim: int, out_dim: int) -> nn.Linear:
        layer = nn.Linear(in_dim, out_dim)
        init.xavier_normal_(layer.weight)
        init.zeros_(layer.bias)
        return layer
    
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = self.activation_fn(layer(x))
        return self.activation_fn(self.output_layer(x))


class MLPBranch(nn.Module):
    """
    MLP branch network for scalar/vector inputs.
    
    Input: (B, in_features) - batch of scalar inputs
    Output: (B, out_features) - encoded representations
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_width: int = 64,
        num_layers: int = 3,
        activation_fn: str = "relu"
    ):
        super().__init__()
        
        if activation_fn.lower() == "sin":
            self.activation_fn = torch.sin
        else:
            self.activation_fn = get_activation(activation_fn)
        
        self.layers = nn.ModuleList()
        self.layers.append(self._make_linear(in_features, hidden_width))
        for _ in range(num_layers - 2):
            self.layers.append(self._make_linear(hidden_width, hidden_width))
        
        self.output_layer = self._make_linear(hidden_width, out_features)
    
    def _make_linear(self, in_dim: int, out_dim: int) -> nn.Linear:
        layer = nn.Linear(in_dim, out_dim)
        init.xavier_normal_(layer.weight)
        init.zeros_(layer.bias)
        return layer
    
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = self.activation_fn(layer(x))
        return self.activation_fn(self.output_layer(x))


# =============================================================================
# 2D Components
# =============================================================================

class SpatialBranch(nn.Module):
    """
    2D spatial branch network with Fourier, U-Net, and/or Conv layers.
    
    Input: (B, H, W, C) - batch of 2D spatial fields
    Output: (B, H, W, width) - encoded spatial representations
    """
    
    def __init__(
        self,
        in_channels: int,
        width: int,
        num_fourier_layers: int = 0,
        num_unet_layers: int = 0,
        num_conv_layers: int = 0,
        modes1: int = 12,
        modes2: int = 12,
        kernel_size: int = 3,
        dropout: float = 0.0,
        unet_impl: str = "custom",
        activation_fn: str = "gelu"
    ):
        super().__init__()
        
        self.num_fourier_layers = num_fourier_layers
        self.num_unet_layers = num_unet_layers
        self.num_conv_layers = num_conv_layers
        self.use_fourier_base = num_fourier_layers > 0
        
        total_layers = num_fourier_layers + num_unet_layers + num_conv_layers
        if total_layers == 0:
            raise ValueError("SpatialBranch requires at least one layer type")
        
        if activation_fn.lower() == "sin":
            self.activation_fn = torch.sin
        else:
            self.activation_fn = get_activation(activation_fn)
        
        # Lifting layer
        self.lift = nn.LazyLinear(width)
        
        # Spectral convolutions (Fourier layers)
        num_fourier_components = total_layers if self.use_fourier_base else num_fourier_layers
        self.spectral_convs = nn.ModuleList()
        self.conv_1x1s = nn.ModuleList()
        for _ in range(num_fourier_components):
            self.spectral_convs.append(SpectralConv2d(width, width, modes1, modes2))
            self.conv_1x1s.append(nn.Conv2d(width, width, kernel_size=1))
        
        # U-Net modules
        self.unet_modules = nn.ModuleList()
        for _ in range(num_unet_layers):
            if unet_impl == "custom":
                self.unet_modules.append(UNet2D(width, width, kernel_size, dropout))
            else:
                self.unet_modules.append(PhysicsNemoUNet2D(width, width, kernel_size))
        
        # Convolutional modules
        self.conv_modules = nn.ModuleList()
        padding = (kernel_size - 1) // 2
        for _ in range(num_conv_layers):
            self.conv_modules.append(nn.Sequential(
                nn.Conv2d(width, width, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(width)
            ))
    
    def forward(self, x: Tensor) -> Tensor:
        # Lift to width dimension
        x = self.lift(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, width) -> (B, width, H, W)
        
        # Fourier layers
        for i in range(self.num_fourier_layers):
            x = self.activation_fn(self.spectral_convs[i](x) + self.conv_1x1s[i](x))
        
        # Hybrid or standalone layers
        if self.use_fourier_base:
            for i in range(self.num_unet_layers):
                j = self.num_fourier_layers + i
                x = self.activation_fn(
                    self.spectral_convs[j](x) + self.conv_1x1s[j](x) + self.unet_modules[i](x)
                )
            for i in range(self.num_conv_layers):
                j = self.num_fourier_layers + self.num_unet_layers + i
                x = self.activation_fn(
                    self.spectral_convs[j](x) + self.conv_1x1s[j](x) + self.conv_modules[i](x)
                )
        else:
            for unet in self.unet_modules:
                x = self.activation_fn(unet(x))
            for conv in self.conv_modules:
                x = self.activation_fn(conv(x))
        
        return x.permute(0, 2, 3, 1)  # (B, width, H, W) -> (B, H, W, width)


class DeepONet(Module):
    """
    2D DeepONet for operator learning.
    
    Input: 
        - x_branch1: (B, H, W, C) for spatial or (B, in_features) for MLP
        - x_time: (T,) or (T, in_features) query coordinates
        - x_branch2: optional second branch input for MIONet
    Output: (B, H, W, T) for spatial or (B, T) for MLP
    """
    
    VALID_VARIANTS = [
        'deeponet', 'u_deeponet', 'fourier_deeponet', 'conv_deeponet',
        'hybrid_deeponet', 'mionet', 'fourier_mionet'
    ]
    
    def __init__(
        self,
        variant: str = "u_deeponet",
        width: int = 64,
        branch1_config: Dict[str, Any] = None,
        branch2_config: Dict[str, Any] = None,
        trunk_config: Dict[str, Any] = None,
        decoder_type: str = "mlp",
        decoder_width: int = 128,
        decoder_layers: int = 2,
        decoder_activation_fn: str = "relu"
    ):
        super().__init__()
        
        self.variant = variant.lower()
        self.width = width
        self.decoder_type = decoder_type.lower()
        self.decoder_activation_fn = decoder_activation_fn
        
        if self.variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Valid: {self.VALID_VARIANTS}")
        
        branch1_config = branch1_config or {}
        trunk_config = trunk_config or {}
        
        # Build networks
        self.branch1 = self._build_branch(branch1_config, width)
        
        self.has_branch2 = branch2_config is not None
        if self.has_branch2:
            self.branch2 = self._build_branch(branch2_config, width)
        
        self.trunk = TrunkNet(
            in_features=trunk_config.get('in_features', 1),
            out_features=width,
            hidden_width=trunk_config.get('hidden_width', 128),
            num_layers=trunk_config.get('num_layers', 6),
            activation_fn=trunk_config.get('activation_fn', 'sin')
        )
        
        self.decoder = self._build_decoder(
            width, 1, decoder_layers, decoder_width, decoder_type, decoder_activation_fn
        )
    
    def _build_branch(self, config: dict, width: int) -> nn.Module:
        branch_type = config.get('type', 'spatial')
        activation = config.get('activation_fn', 'sin')
        
        if branch_type == 'mlp':
            return MLPBranch(
                in_features=config.get('in_features', 12),
                out_features=width,
                hidden_width=config.get('hidden_width', 64),
                num_layers=config.get('num_layers', 3),
                activation_fn=activation
            )
        elif branch_type == 'spatial':
            return SpatialBranch(
                in_channels=config.get('in_channels', 12),
                width=width,
                num_fourier_layers=config.get('num_fourier_layers', 0),
                num_unet_layers=config.get('num_unet_layers', 0),
                num_conv_layers=config.get('num_conv_layers', 0),
                modes1=config.get('modes1', 12),
                modes2=config.get('modes2', 12),
                kernel_size=config.get('kernel_size', 3),
                dropout=config.get('dropout', 0.0),
                unet_impl=config.get('unet_impl', 'custom'),
                activation_fn=activation
            )
        else:
            raise ValueError(f"Unknown branch type: {branch_type}")
    
    def _build_decoder(
        self, width: int, out_channels: int, num_layers: int,
        hidden_width: int, decoder_type: str, activation_fn: str
    ) -> nn.Module:
        
        if decoder_type == "mlp":
            if num_layers == 0:
                return nn.Linear(width, out_channels)
            return FullyConnected(width, hidden_width, out_channels, num_layers, activation_fn)
        
        elif decoder_type == "conv":
            act = get_activation(activation_fn)
            if num_layers == 0:
                return Conv2dFCLayer(width, out_channels)
            
            layers = []
            in_ch = width
            for _ in range(num_layers):
                layers.extend([Conv2dFCLayer(in_ch, hidden_width), act])
                in_ch = hidden_width
            layers.append(Conv2dFCLayer(hidden_width, out_channels))
            return nn.Sequential(*layers)
        
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")
    
    def forward(
        self, x_branch1: Tensor, x_time: Tensor, x_branch2: Tensor = None
    ) -> Tensor:
        
        if x_time.dim() == 1:
            x_time = x_time.unsqueeze(-1)
        
        b1_out = self.branch1(x_branch1)
        
        if self.has_branch2:
            if x_branch2 is None:
                raise ValueError("x_branch2 required for mionet variants")
            b2_out = self.branch2(x_branch2)
        
        trunk_out = self.trunk(x_time)
        
        # Combine branch and trunk
        if b1_out.dim() == 4:  # Spatial branch
            b1_out = b1_out.unsqueeze(1)
            trunk_out = trunk_out.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            
            if self.has_branch2:
                if b2_out.dim() == 4:
                    b2_out = b2_out.unsqueeze(1)
                else:
                    b2_out = b2_out.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                combined = (b1_out + b2_out) * trunk_out
            else:
                combined = b1_out * trunk_out
            
            if self.decoder_type == "mlp":
                return self.decoder(combined).squeeze(-1).permute(0, 2, 3, 1)
            
            B, T, H, W, C = combined.shape
            combined = combined.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
            return self.decoder(combined).reshape(B, T, H, W).permute(0, 2, 3, 1)
        
        else:  # MLP branch
            b1_out = b1_out.unsqueeze(1)
            trunk_out = trunk_out.unsqueeze(0)
            
            if self.has_branch2:
                combined = (b1_out + b2_out.unsqueeze(1)) * trunk_out
            else:
                combined = b1_out * trunk_out
            
            return self.decoder(combined).squeeze(-1)
    
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepONetWrapper(nn.Module):
    """
    2D DeepONet wrapper with automatic padding and input extraction.
    
    Input: (B, H, W, T, C) - batch of spatiotemporal fields
    Output: (B, H, W, T) - predicted output field
    """
    
    def __init__(
        self,
        padding: int = 8,
        variant: str = "u_deeponet",
        width: int = 64,
        branch1_config: Dict[str, Any] = None,
        branch2_config: Dict[str, Any] = None,
        trunk_config: Dict[str, Any] = None,
        decoder_type: str = "mlp",
        decoder_width: int = 128,
        decoder_layers: int = 2,
        decoder_activation_fn: str = "relu"
    ):
        super().__init__()
        
        self.padding = ((padding + 7) // 8) * 8 if padding % 8 != 0 else padding
        self.variant = variant
        
        trunk_config = trunk_config or {}
        self.trunk_input = trunk_config.get('input_type', 'time').lower()
        
        if self.trunk_input not in ['time', 'grid']:
            raise ValueError(f"trunk input_type must be 'time' or 'grid'")
        
        if self.trunk_input == 'grid':
            trunk_config['in_features'] = 3  # (x, y, t)
        else:
            trunk_config['in_features'] = trunk_config.get('in_features', 1)
        
        self.model = DeepONet(
            variant=variant,
            width=width,
            branch1_config=branch1_config,
            branch2_config=branch2_config,
            trunk_config=trunk_config,
            decoder_type=decoder_type,
            decoder_width=decoder_width,
            decoder_layers=decoder_layers,
            decoder_activation_fn=decoder_activation_fn
        )
    
    def forward(self, x: Tensor, x_branch2: Tensor = None) -> Tensor:
        H, W = x.shape[1], x.shape[2]
        
        x = F.pad(x, (0, 0, 0, 0, 0, self.padding), "replicate")
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, self.padding), 'constant', 0)
        
        x_spatial = x.permute(0, 4, 1, 2, 3)[..., 0].permute(0, 2, 3, 1)
        
        if self.trunk_input == 'grid':
            x_trunk = x[0, 0, 0, :, -3:]
        else:
            x_trunk = x[0, 0, 0, :, -1].unsqueeze(-1)
        
        return self.model(x_spatial, x_trunk, x_branch2)[:, :H, :W, :]
    
    def count_params(self) -> int:
        return self.model.count_params()


# =============================================================================
# 3D Components
# =============================================================================

class SpatialBranch3D(nn.Module):
    """
    3D spatial branch network with Fourier, U-Net, and/or Conv layers.
    
    Input: (B, X, Y, Z, C) - batch of 3D spatial fields
    Output: (B, X, Y, Z, width) - encoded spatial representations
    """
    
    def __init__(
        self,
        in_channels: int,
        width: int,
        num_fourier_layers: int = 0,
        num_unet_layers: int = 0,
        num_conv_layers: int = 0,
        modes1: int = 10,
        modes2: int = 10,
        modes3: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.0,
        unet_impl: str = "custom",
        activation_fn: str = "gelu"
    ):
        super().__init__()
        
        self.num_fourier_layers = num_fourier_layers
        self.num_unet_layers = num_unet_layers
        self.num_conv_layers = num_conv_layers
        self.use_fourier_base = num_fourier_layers > 0
        
        total_layers = num_fourier_layers + num_unet_layers + num_conv_layers
        if total_layers == 0:
            raise ValueError("SpatialBranch3D requires at least one layer type")
        
        if activation_fn.lower() == "sin":
            self.activation_fn = torch.sin
        else:
            self.activation_fn = get_activation(activation_fn)
        
        # Lifting layer
        self.lift = nn.LazyLinear(width)
        
        # Spectral convolutions (Fourier layers)
        num_fourier_components = total_layers if self.use_fourier_base else num_fourier_layers
        self.spectral_convs = nn.ModuleList()
        self.conv_1x1s = nn.ModuleList()
        for _ in range(num_fourier_components):
            self.spectral_convs.append(SpectralConv3d(width, width, modes1, modes2, modes3))
            self.conv_1x1s.append(nn.Conv3d(width, width, kernel_size=1))
        
        # U-Net modules
        self.unet_modules = nn.ModuleList()
        for _ in range(num_unet_layers):
            if unet_impl == "custom":
                self.unet_modules.append(UNet3D(width, width, kernel_size, dropout))
            else:
                self.unet_modules.append(PhysicsNemoUNet3D(width, width, kernel_size))
        
        # Convolutional modules
        self.conv_modules = nn.ModuleList()
        padding = (kernel_size - 1) // 2
        for _ in range(num_conv_layers):
            self.conv_modules.append(nn.Sequential(
                nn.Conv3d(width, width, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm3d(width)
            ))
    
    def forward(self, x: Tensor) -> Tensor:
        # Lift to width dimension
        x = self.lift(x)
        x = x.permute(0, 4, 1, 2, 3)  # (B, X, Y, Z, width) -> (B, width, X, Y, Z)
        
        # Fourier layers
        for i in range(self.num_fourier_layers):
            x = self.activation_fn(self.spectral_convs[i](x) + self.conv_1x1s[i](x))
        
        # Hybrid or standalone layers
        if self.use_fourier_base:
            for i in range(self.num_unet_layers):
                j = self.num_fourier_layers + i
                x = self.activation_fn(
                    self.spectral_convs[j](x) + self.conv_1x1s[j](x) + self.unet_modules[i](x)
                )
            for i in range(self.num_conv_layers):
                j = self.num_fourier_layers + self.num_unet_layers + i
                x = self.activation_fn(
                    self.spectral_convs[j](x) + self.conv_1x1s[j](x) + self.conv_modules[i](x)
                )
        else:
            for unet in self.unet_modules:
                x = self.activation_fn(unet(x))
            for conv in self.conv_modules:
                x = self.activation_fn(conv(x))
        
        return x.permute(0, 2, 3, 4, 1)  # (B, width, X, Y, Z) -> (B, X, Y, Z, width)


class DeepONet3D(Module):
    """
    3D DeepONet for operator learning on volumetric data.
    
    Input:
        - x_branch1: (B, X, Y, Z, C) for spatial or (B, in_features) for MLP
        - x_time: (T,) or (T, in_features) query coordinates
        - x_branch2: optional second branch input for MIONet
    Output: (B, X, Y, Z, T) for spatial or (B, T) for MLP
    """
    
    VALID_VARIANTS = [
        'deeponet', 'u_deeponet', 'fourier_deeponet', 'conv_deeponet',
        'hybrid_deeponet', 'mionet', 'fourier_mionet'
    ]
    
    def __init__(
        self,
        variant: str = "u_deeponet",
        width: int = 64,
        branch1_config: Dict[str, Any] = None,
        branch2_config: Dict[str, Any] = None,
        trunk_config: Dict[str, Any] = None,
        decoder_type: str = "mlp",
        decoder_width: int = 128,
        decoder_layers: int = 2,
        decoder_activation_fn: str = "relu"
    ):
        super().__init__()
        
        self.variant = variant.lower()
        self.width = width
        self.decoder_type = decoder_type.lower()
        self.decoder_activation_fn = decoder_activation_fn
        
        if self.variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Valid: {self.VALID_VARIANTS}")
        
        branch1_config = branch1_config or {}
        trunk_config = trunk_config or {}
        
        # Build networks
        self.branch1 = self._build_branch(branch1_config, width)
        
        self.has_branch2 = branch2_config is not None
        if self.has_branch2:
            self.branch2 = self._build_branch(branch2_config, width)
        
        self.trunk = TrunkNet(
            in_features=trunk_config.get('in_features', 1),
            out_features=width,
            hidden_width=trunk_config.get('hidden_width', 128),
            num_layers=trunk_config.get('num_layers', 6),
            activation_fn=trunk_config.get('activation_fn', 'sin')
        )
        
        self.decoder = self._build_decoder(
            width, 1, decoder_layers, decoder_width, decoder_type, decoder_activation_fn
        )
    
    def _build_branch(self, config: dict, width: int) -> nn.Module:
        branch_type = config.get('type', 'spatial')
        activation = config.get('activation_fn', 'sin')
        
        if branch_type == 'mlp':
            return MLPBranch(
                in_features=config.get('in_features', 12),
                out_features=width,
                hidden_width=config.get('hidden_width', 64),
                num_layers=config.get('num_layers', 3),
                activation_fn=activation
            )
        elif branch_type == 'spatial':
            return SpatialBranch3D(
                in_channels=config.get('in_channels', 11),
                width=width,
                num_fourier_layers=config.get('num_fourier_layers', 0),
                num_unet_layers=config.get('num_unet_layers', 0),
                num_conv_layers=config.get('num_conv_layers', 0),
                modes1=config.get('modes1', 10),
                modes2=config.get('modes2', 10),
                modes3=config.get('modes3', 8),
                kernel_size=config.get('kernel_size', 3),
                dropout=config.get('dropout', 0.0),
                unet_impl=config.get('unet_impl', 'custom'),
                activation_fn=activation
            )
        else:
            raise ValueError(f"Unknown branch type: {branch_type}")
    
    def _build_decoder(
        self, width: int, out_channels: int, num_layers: int,
        hidden_width: int, decoder_type: str, activation_fn: str
    ) -> nn.Module:
        
        if decoder_type == "mlp":
            if num_layers == 0:
                return nn.Linear(width, out_channels)
            return FullyConnected(width, hidden_width, out_channels, num_layers, activation_fn)
        
        elif decoder_type == "conv":
            act = get_activation(activation_fn)
            if num_layers == 0:
                return Conv3dFCLayer(width, out_channels)
            
            layers = []
            in_ch = width
            for _ in range(num_layers):
                layers.extend([Conv3dFCLayer(in_ch, hidden_width), act])
                in_ch = hidden_width
            layers.append(Conv3dFCLayer(hidden_width, out_channels))
            return nn.Sequential(*layers)
        
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")
    
    def forward(
        self, x_branch1: Tensor, x_time: Tensor, x_branch2: Tensor = None
    ) -> Tensor:
        
        if x_time.dim() == 1:
            x_time = x_time.unsqueeze(-1)
        
        b1_out = self.branch1(x_branch1)
        
        if self.has_branch2:
            if x_branch2 is None:
                raise ValueError("x_branch2 required for mionet variants")
            b2_out = self.branch2(x_branch2)
        
        trunk_out = self.trunk(x_time)
        
        # Combine branch and trunk
        if b1_out.dim() == 5:  # Spatial branch
            b1_out = b1_out.unsqueeze(1)
            trunk_out = trunk_out.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
            
            if self.has_branch2:
                if b2_out.dim() == 5:
                    b2_out = b2_out.unsqueeze(1)
                else:
                    b2_out = b2_out.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                combined = (b1_out + b2_out) * trunk_out
            else:
                combined = b1_out * trunk_out
            
            if self.decoder_type == "mlp":
                return self.decoder(combined).squeeze(-1).permute(0, 2, 3, 4, 1)
            
            B, T, X, Y, Z, C = combined.shape
            combined = combined.permute(0, 1, 5, 2, 3, 4).reshape(B * T, C, X, Y, Z)
            return self.decoder(combined).reshape(B, T, X, Y, Z).permute(0, 2, 3, 4, 1)
        
        else:  # MLP branch
            b1_out = b1_out.unsqueeze(1)
            trunk_out = trunk_out.unsqueeze(0)
            
            if self.has_branch2:
                combined = (b1_out + b2_out.unsqueeze(1)) * trunk_out
            else:
                combined = b1_out * trunk_out
            
            return self.decoder(combined).squeeze(-1)
    
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepONet3DWrapper(nn.Module):
    """
    3D DeepONet wrapper with automatic padding and input extraction.
    
    Input: (B, X, Y, Z, T, C) - batch of 4D spatiotemporal fields
    Output: (B, X, Y, Z, T) - predicted output field
    """
    
    def __init__(
        self,
        padding: int = 8,
        variant: str = "u_deeponet",
        width: int = 64,
        branch1_config: Dict[str, Any] = None,
        branch2_config: Dict[str, Any] = None,
        trunk_config: Dict[str, Any] = None,
        decoder_type: str = "mlp",
        decoder_width: int = 128,
        decoder_layers: int = 2,
        decoder_activation_fn: str = "relu"
    ):
        super().__init__()
        
        self.padding = ((padding + 7) // 8) * 8 if padding % 8 != 0 else padding
        self.variant = variant
        
        trunk_config = trunk_config or {}
        self.trunk_input = trunk_config.get('input_type', 'time').lower()
        
        if self.trunk_input not in ['time', 'grid']:
            raise ValueError(f"trunk input_type must be 'time' or 'grid'")
        
        if self.trunk_input == 'grid':
            trunk_config['in_features'] = 4  # (x, y, z, t)
        else:
            trunk_config['in_features'] = trunk_config.get('in_features', 1)
        
        self.model = DeepONet3D(
            variant=variant,
            width=width,
            branch1_config=branch1_config,
            branch2_config=branch2_config,
            trunk_config=trunk_config,
            decoder_type=decoder_type,
            decoder_width=decoder_width,
            decoder_layers=decoder_layers,
            decoder_activation_fn=decoder_activation_fn
        )
    
    def forward(self, x: Tensor, x_branch2: Tensor = None) -> Tensor:
        X, Y, Z = x.shape[1], x.shape[2], x.shape[3]
        
        # Calculate padding
        if self.padding > 0:
            pad_x = max((8 - (X % 8)) % 8, self.padding)
            pad_y = max((8 - (Y % 8)) % 8, self.padding)
            pad_z = max((8 - (Z % 8)) % 8, self.padding)
        else:
            pad_x = (8 - (X % 8)) % 8 if X % 8 != 0 else 0
            pad_y = (8 - (Y % 8)) % 8 if Y % 8 != 0 else 0
            pad_z = (8 - (Z % 8)) % 8 if Z % 8 != 0 else 0
        
        if pad_x > 0 or pad_y > 0 or pad_z > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_z, 0, pad_y, 0, pad_x), "replicate")
        
        x_spatial = x[:, :, :, :, 0, :]
        
        if self.trunk_input == 'grid':
            x_trunk = x[0, 0, 0, 0, :, -4:]
        else:
            x_trunk = x[0, 0, 0, 0, :, -1].unsqueeze(-1)
        
        return self.model(x_spatial, x_trunk, x_branch2)[:, :X, :Y, :Z, :]
    
    def count_params(self) -> int:
        return self.model.count_params()

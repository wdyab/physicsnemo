#!/bin/bash
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

# ============================================================================
# Fourier-MIONet CO2 Example — Training
# ============================================================================
# Usage (submit from the neural_operator_factory/ directory):
#
#   sbatch examples/fourier_mionet_co2/train.sbatch vanilla_mionet saturation_training_config
#   sbatch examples/fourier_mionet_co2/train.sbatch fourier_mionet pressure_training_config
#
# Arguments:
#   $1 = architecture: vanilla_mionet, mionet_fnn, fourier_mionet
#   $2 = config name (default: saturation_training_config)
# ============================================================================

#SBATCH --job-name=coreai_climate_earth2-nof.fmionet_co2
#SBATCH --partition=batch
#SBATCH --account=coreai_climate_earth2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs checkpoints

ARCH="${1:-fourier_mionet}"
CONFIG_NAME="${2:-saturation_training_config}"

echo "========================================"
echo "  Fourier-MIONet CO2 — Training"
echo "  Architecture: $ARCH"
echo "  Config:       $CONFIG_NAME"
echo "  Job ID:       $SLURM_JOB_ID"
echo "  Node:         $(hostname)"
echo "  GPUs:         8"
echo "========================================"

pip install --quiet -r requirements.txt 2>/dev/null || true

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    scripts/train.py \
    --config-path "${SLURM_SUBMIT_DIR}/examples/fourier_mionet_co2/conf/${ARCH}" \
    --config-name "$CONFIG_NAME"

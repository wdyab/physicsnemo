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
# U-FNO CO2 Example — Training
# ============================================================================
# Usage (submit from the neural_operator_factory/ directory):
#
#   # U-FNO on gas saturation
#   sbatch examples/ufno_co2/train.sbatch U-FNO saturation_training_config
#
#   # U-FNO on pressure buildup
#   sbatch examples/ufno_co2/train.sbatch U-FNO pressure_training_config
#
#   # FNO baseline on saturation
#   sbatch examples/ufno_co2/train.sbatch FNO saturation_training_config
#
#   # Conv-FNO on pressure
#   sbatch examples/ufno_co2/train.sbatch Conv-FNO pressure_training_config
#
# Arguments:
#   $1 = architecture folder: FNO, Conv-FNO, or U-FNO       (default: U-FNO)
#   $2 = config name: saturation_training_config or          (default: saturation_training_config)
#                     pressure_training_config
# ============================================================================

#SBATCH --job-name=coreai_climate_earth2-nof.ufno_co2
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

ARCH="${1:-U-FNO}"
CONFIG_NAME="${2:-saturation_training_config}"

echo "========================================"
echo "  U-FNO CO2 Example — Training"
echo "  Architecture: $ARCH"
echo "  Config:       $CONFIG_NAME"
echo "  Job ID:       $SLURM_JOB_ID"
echo "  Node:         $(hostname)"
echo "  GPUs:         8"
echo "  Dir:          $SLURM_SUBMIT_DIR"
echo "========================================"

pip install --quiet -r requirements.txt 2>/dev/null || true

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    scripts/train.py \
    --config-path "${SLURM_SUBMIT_DIR}/examples/ufno_co2/conf/${ARCH}" \
    --config-name "$CONFIG_NAME"

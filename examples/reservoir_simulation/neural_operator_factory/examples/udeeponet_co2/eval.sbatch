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
# U-DeepONet CO2 Example — Evaluation
# ============================================================================
# Usage (submit from the neural_operator_factory/ directory):
#
#   sbatch examples/udeeponet_co2/eval.sbatch saturation
#   sbatch examples/udeeponet_co2/eval.sbatch pressure
#
#   CHECKPOINT=checkpoints/best_model_pressure_deeponet_u_deeponet_spatial.pth \
#       sbatch examples/udeeponet_co2/eval.sbatch pressure
# ============================================================================

#SBATCH --job-name=coreai_climate_earth2-nof.udeeponet_co2_eval
#SBATCH --partition=batch
#SBATCH --account=coreai_climate_earth2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs visualizations

VARIABLE="${1:-saturation}"

if [ -z "${CHECKPOINT:-}" ]; then
    CKPT_PATTERN="checkpoints/best_model_${VARIABLE}_*.pth"
    CHECKPOINT=$(ls -t $CKPT_PATTERN 2>/dev/null | head -1)
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: No checkpoint found matching $CKPT_PATTERN"
        exit 1
    fi
fi

echo "========================================"
echo "  U-DeepONet CO2 Example — Evaluation"
echo "  Variable:   $VARIABLE"
echo "  Checkpoint: $CHECKPOINT"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Node:       $(hostname)"
echo "========================================"

if [ "$VARIABLE" = "pressure" ]; then
    python3 examples/udeeponet_co2/evaluate_pressure.py \
        --checkpoint "$CHECKPOINT"
elif [ "$VARIABLE" = "saturation" ]; then
    python3 examples/udeeponet_co2/evaluate_saturation.py \
        --checkpoint "$CHECKPOINT"
else
    echo "ERROR: Unknown variable '$VARIABLE'. Use 'pressure' or 'saturation'."
    exit 1
fi

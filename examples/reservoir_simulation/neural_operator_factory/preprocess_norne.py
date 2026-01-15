#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Norne Dataset Preprocessing Script for 4D Neural Operators

This script preprocesses the Norne reservoir simulation dataset into tensor format
suitable for 4D FNO/DeepONet training. The Norne dataset is 3D spatial + time,
requiring special handling compared to the 2D CO2 sequestration dataset.

Input channels (11):
    0: PERMX (log10 scaled) - Horizontal permeability
    1: PERMZ (log10 scaled) - Vertical permeability (varies between LHS samples)
    2: PORO - Porosity
    3: PORV (normalized) - Pore volume
    4: NTG - Net-to-gross
    5: ACTNUM - Active cell mask (binary)
    6: WCID - Well completion indicator (+1 injector, -1 producer, 0 none)
    7: grid_x (normalized) - X coordinate
    8: grid_y (normalized) - Y coordinate
    9: grid_z (normalized) - Z coordinate
    10: grid_t (normalized) - Time coordinate

Output channels (3):
    0: PRESSURE - Cell pressure
    1: SWAT - Water saturation
    2: SGAS - Gas saturation

Usage:
    python preprocess_norne.py --sim_dir /path/to/NORNE_LHS.sim --output_dir /path/to/output
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add sim_utils to path
REPO_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(REPO_ROOT))

from sim_utils import EclReader, Grid, Well

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_completion_info(grid, well_info) -> list:
    """
    Extract well completion information from simulation data.
    Adapted from XMGN graph_builder.py.
    
    Parameters:
        grid: Grid object containing grid information.
        well_info: Dictionary containing well data from restart files.
        
    Returns:
        list: List of Well dict objects for each timestep.
    """
    wells_lst = []
    
    for i in range(len(well_info["ZWEL"])):
        INTEHEAD = well_info["INTEHEAD"][i]
        ZWEL = well_info["ZWEL"][i]
        IWEL = well_info["IWEL"][i]
        ICON = well_info["ICON"][i]
        
        NWELLS = INTEHEAD[16]
        NCWMAX = INTEHEAD[17]
        NICONZ = INTEHEAD[32]
        
        if NWELLS == 0 or IWEL.size == 0:
            wells_lst.append({})
            continue
        
        IWEL = IWEL.reshape((-1, NWELLS), order='F')
        ICON = ICON.reshape((NICONZ, NCWMAX, NWELLS), order='F')
        
        well_names = ["".join(row).strip() for row in ZWEL if "".join(row).strip()]
        wells = {}
        
        for iwell, name in enumerate(well_names):
            well = Well(name=name, type_id=IWEL[6, iwell], stat=IWEL[10, iwell])
            
            for ic in range(NCWMAX):
                icon = ICON[:, ic, iwell]
                if icon[0] == 0:
                    break
                I, J, K = icon[1:4]
                well.add_completion(
                    I=I, J=J, K=K,
                    dir=icon[13],
                    stat=icon[5],
                    conx_factor=1.0
                )
                well.completions[-1].set_ijk(grid.ijk_from_I_J_K(I, J, K))
            
            wells[name] = well
        
        wells_lst.append(wells)
    
    return wells_lst


def extract_grid_info(case_path: str) -> dict:
    """
    Extract grid structure from a simulation case.
    
    Parameters:
        case_path: Path to .DATA file
        
    Returns:
        dict: Grid information including dimensions, active cell mapping, etc.
    """
    reader = EclReader(case_path)
    
    # Read grid and init data
    egrid_data = reader.read_egrid(["COORD", "ZCORN", "FILEHEAD", "NNC1", "NNC2"])
    init_data = reader.read_init(["INTEHEAD", "PORV"])
    
    grid = Grid(init_data, egrid_data)
    
    # Create inverse mapping: active_idx -> (i, j, k)
    active_to_ijk = np.zeros((grid.nact, 3), dtype=np.int32)
    for ijk_global, active_idx in grid.ijk_to_active.items():
        k = ijk_global // (grid.nx * grid.ny)
        remainder = ijk_global % (grid.nx * grid.ny)
        j = remainder // grid.nx
        i = remainder % grid.nx
        active_to_ijk[active_idx] = [i, j, k]
    
    # Create 3D actnum array
    actnum_3d = grid.actnum.reshape(grid.nx, grid.ny, grid.nz, order='F')
    
    return {
        "nx": grid.nx,
        "ny": grid.ny,
        "nz": grid.nz,
        "nact": grid.nact,
        "actnum_3d": actnum_3d,
        "ijk_to_active": grid.ijk_to_active,
        "active_to_ijk": active_to_ijk,
    }


def process_single_sample(case_path: str, grid_info: dict) -> tuple:
    """
    Process one simulation case and return input/output tensors.
    
    Parameters:
        case_path: Path to .DATA file
        grid_info: Grid information dict from extract_grid_info
        
    Returns:
        tuple: (input_tensor, output_tensor) with shapes:
               input: (X, Y, Z, T, 11)
               output: (X, Y, Z, T, 3)
    """
    reader = EclReader(case_path)
    
    nx, ny, nz = grid_info["nx"], grid_info["ny"], grid_info["nz"]
    nact = grid_info["nact"]
    active_to_ijk = grid_info["active_to_ijk"]
    actnum_3d = grid_info["actnum_3d"]
    
    # === Read all data ===
    egrid_data = reader.read_egrid(["COORD", "ZCORN", "FILEHEAD", "NNC1", "NNC2"])
    init_data = reader.read_init([
        "INTEHEAD", "PERMX", "PERMZ", "PORO", "PORV", "NTG"
    ])
    
    # Read restart data (outputs)
    # Note: read_restart returns {"DATE": [...], "TIME": [...], "PRESSURE": [...], ...}
    # where each list has one entry per timestep
    restart_data = reader.read_restart(["PRESSURE", "SWAT", "SGAS"])
    
    # Read well data
    well_data = reader.read_restart(["INTEHEAD", "ZWEL", "IWEL", "ICON", "SCON"])
    
    # Initialize grid for well processing
    grid = Grid(init_data, egrid_data)
    
    # Get timestep information
    num_timesteps = len(restart_data["TIME"])
    
    # Extract times
    times = np.array(restart_data["TIME"])
    
    # === Process Static Properties ===
    static = {}
    
    # PERMX, PERMZ with log10 scaling
    for prop in ["PERMX", "PERMZ"]:
        static[prop] = np.zeros((nx, ny, nz), dtype=np.float32)
        data = init_data[prop]
        data_scaled = np.log10(np.maximum(data, 1e-10))
        for idx in range(nact):
            i, j, k = active_to_ijk[idx]
            static[prop][i, j, k] = data_scaled[idx]
    
    # PORO, NTG without scaling
    for prop in ["PORO", "NTG"]:
        static[prop] = np.zeros((nx, ny, nz), dtype=np.float32)
        data = init_data[prop]
        for idx in range(nact):
            i, j, k = active_to_ijk[idx]
            static[prop][i, j, k] = data[idx]
    
    # PORV (from full grid, need to filter)
    static["PORV"] = np.zeros((nx, ny, nz), dtype=np.float32)
    porv_full = init_data["PORV"]
    actnum_flat = actnum_3d.flatten(order='F')
    porv_active = porv_full[actnum_flat > 0]
    for idx in range(nact):
        i, j, k = active_to_ijk[idx]
        static["PORV"][i, j, k] = porv_active[idx]
    
    # Coordinates from grid
    for coord_name, coord_arr in [("X", grid.X), ("Y", grid.Y), ("Z", grid.Z)]:
        static[coord_name] = np.zeros((nx, ny, nz), dtype=np.float32)
        for idx in range(nact):
            i, j, k = active_to_ijk[idx]
            static[coord_name][i, j, k] = coord_arr[idx]
    
    # === Process Well Data (WCID) ===
    wells_per_timestep = get_completion_info(grid, well_data)
    WCID = np.zeros((nx, ny, nz, num_timesteps), dtype=np.float32)
    
    for t_idx in range(num_timesteps):
        if t_idx < len(wells_per_timestep) and wells_per_timestep[t_idx]:
            wcid_inj, wcid_prd = grid.create_completion_array(wells_per_timestep[t_idx])
            # Combined: +1 for injector, -1 for producer
            wcid_combined = wcid_inj - wcid_prd
            for idx in range(nact):
                if wcid_combined[idx] != 0:
                    i, j, k = active_to_ijk[idx]
                    WCID[i, j, k, t_idx] = wcid_combined[idx]
    
    # === Process Outputs ===
    output = np.zeros((nx, ny, nz, num_timesteps, 3), dtype=np.float32)
    
    for t_idx in range(num_timesteps):
        for c, prop in enumerate(["PRESSURE", "SWAT", "SGAS"]):
            data = restart_data[prop][t_idx]
            for idx in range(nact):
                i, j, k = active_to_ijk[idx]
                output[i, j, k, t_idx, c] = data[idx]
    
    # === Assemble Input Tensor ===
    input_tensor = np.zeros((nx, ny, nz, num_timesteps, 11), dtype=np.float32)
    
    # Normalize coordinates (only over active cells)
    active_mask = actnum_3d > 0
    
    X_active = static["X"][active_mask]
    Y_active = static["Y"][active_mask]
    Z_active = static["Z"][active_mask]
    PORV_active = static["PORV"][active_mask]
    
    X_min, X_max = X_active.min(), X_active.max()
    Y_min, Y_max = Y_active.min(), Y_active.max()
    Z_min, Z_max = Z_active.min(), Z_active.max()
    PORV_max = PORV_active.max()
    time_max = times.max()
    
    # Create normalized versions
    X_norm = np.zeros_like(static["X"])
    Y_norm = np.zeros_like(static["Y"])
    Z_norm = np.zeros_like(static["Z"])
    PORV_norm = np.zeros_like(static["PORV"])
    
    X_norm[active_mask] = (static["X"][active_mask] - X_min) / (X_max - X_min + 1e-10)
    Y_norm[active_mask] = (static["Y"][active_mask] - Y_min) / (Y_max - Y_min + 1e-10)
    Z_norm[active_mask] = (static["Z"][active_mask] - Z_min) / (Z_max - Z_min + 1e-10)
    PORV_norm[active_mask] = static["PORV"][active_mask] / (PORV_max + 1e-10)
    
    time_norm = times / (time_max + 1e-10)
    
    # Fill input tensor
    # Channel order: PERMX, PERMZ, PORO, PORV, NTG, ACTNUM, WCID, grid_x, grid_y, grid_z, grid_t
    for t in range(num_timesteps):
        input_tensor[:, :, :, t, 0] = static["PERMX"]
        input_tensor[:, :, :, t, 1] = static["PERMZ"]
        input_tensor[:, :, :, t, 2] = static["PORO"]
        input_tensor[:, :, :, t, 3] = PORV_norm
        input_tensor[:, :, :, t, 4] = static["NTG"]
        input_tensor[:, :, :, t, 5] = actnum_3d.astype(np.float32)
        input_tensor[:, :, :, t, 6] = WCID[:, :, :, t]
        input_tensor[:, :, :, t, 7] = X_norm
        input_tensor[:, :, :, t, 8] = Y_norm
        input_tensor[:, :, :, t, 9] = Z_norm
        input_tensor[:, :, :, t, 10] = time_norm[t]
    
    return input_tensor, output


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Norne dataset for 4D neural operators"
    )
    parser.add_argument(
        "--sim_dir",
        type=str,
        default="/lustre/fsw/coreai_climate_earth2/tonishi/physicsnemo/examples/reservoir_simulation/dataset/Norne/NORNE_ATW2013_LHS.sim",
        help="Directory containing simulation .DATA files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/lustre/fsw/coreai_climate_earth2/wdyab/physicsnemo_data/norne",
        help="Output directory for preprocessed tensors"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (None for all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test split"
    )
    args = parser.parse_args()
    
    sim_dir = Path(args.sim_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all simulation cases
    case_files = sorted(sim_dir.glob("*.DATA"))
    if args.num_samples:
        case_files = case_files[:args.num_samples]
    
    num_cases = len(case_files)
    logger.info(f"Found {num_cases} simulation cases in {sim_dir}")
    
    if num_cases == 0:
        logger.error("No .DATA files found!")
        return
    
    # Extract grid info from first case
    logger.info("Extracting grid information from first case...")
    grid_info = extract_grid_info(str(case_files[0]))
    logger.info(f"  Grid dimensions: {grid_info['nx']} x {grid_info['ny']} x {grid_info['nz']}")
    logger.info(f"  Active cells: {grid_info['nact']} ({100*grid_info['nact']/(grid_info['nx']*grid_info['ny']*grid_info['nz']):.1f}%)")
    
    # Process all cases
    all_inputs = []
    all_outputs = []
    failed_cases = []
    
    for case_path in tqdm(case_files, desc="Processing cases"):
        try:
            inp, out = process_single_sample(str(case_path), grid_info)
            all_inputs.append(inp)
            all_outputs.append(out)
        except Exception as e:
            logger.warning(f"Failed to process {case_path.name}: {e}")
            failed_cases.append(case_path.name)
    
    if failed_cases:
        logger.warning(f"Failed cases ({len(failed_cases)}): {failed_cases[:10]}...")
    
    # Stack into arrays
    logger.info("Stacking tensors...")
    inputs = np.stack(all_inputs, axis=0)
    outputs = np.stack(all_outputs, axis=0)
    
    logger.info(f"Input tensor shape: {inputs.shape}")
    logger.info(f"Output tensor shape: {outputs.shape}")
    
    # Determine split sizes
    num_samples = len(inputs)
    np.random.seed(args.seed)
    indices = np.random.permutation(num_samples)
    
    n_train = int(num_samples * 0.8)
    n_val = int(num_samples * 0.1)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    logger.info(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    
    # Save tensors
    logger.info("Saving tensors...")
    
    torch.save(torch.from_numpy(inputs[train_idx]), output_dir / "norne_train_a.pt")
    torch.save(torch.from_numpy(outputs[train_idx]), output_dir / "norne_train_u.pt")
    logger.info(f"  Saved training data: {len(train_idx)} samples")
    
    torch.save(torch.from_numpy(inputs[val_idx]), output_dir / "norne_val_a.pt")
    torch.save(torch.from_numpy(outputs[val_idx]), output_dir / "norne_val_u.pt")
    logger.info(f"  Saved validation data: {len(val_idx)} samples")
    
    torch.save(torch.from_numpy(inputs[test_idx]), output_dir / "norne_test_a.pt")
    torch.save(torch.from_numpy(outputs[test_idx]), output_dir / "norne_test_u.pt")
    logger.info(f"  Saved test data: {len(test_idx)} samples")
    
    # Save metadata
    metadata = {
        "grid_dims": [grid_info["nx"], grid_info["ny"], grid_info["nz"]],
        "num_active_cells": int(grid_info["nact"]),
        "num_timesteps": int(inputs.shape[4]),
        "input_channels": 11,
        "output_channels": 3,
        "input_channel_names": [
            "PERMX_log10", "PERMZ_log10", "PORO", "PORV_norm", "NTG",
            "ACTNUM", "WCID", "grid_x", "grid_y", "grid_z", "grid_t"
        ],
        "output_channel_names": ["PRESSURE", "SWAT", "SGAS"],
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_test": int(len(test_idx)),
        "seed": args.seed,
        "source_dir": str(sim_dir),
        "failed_cases": failed_cases,
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Saved metadata to {output_dir / 'metadata.json'}")
    
    # Compute and save statistics
    logger.info("Computing statistics...")
    stats = {
        "input_mean": inputs.mean(axis=(0, 1, 2, 3, 4)).tolist(),
        "input_std": inputs.std(axis=(0, 1, 2, 3, 4)).tolist(),
        "output_mean": outputs.mean(axis=(0, 1, 2, 3, 4)).tolist(),
        "output_std": outputs.std(axis=(0, 1, 2, 3, 4)).tolist(),
        "input_min": inputs.min(axis=(0, 1, 2, 3, 4)).tolist(),
        "input_max": inputs.max(axis=(0, 1, 2, 3, 4)).tolist(),
        "output_min": outputs.min(axis=(0, 1, 2, 3, 4)).tolist(),
        "output_max": outputs.max(axis=(0, 1, 2, 3, 4)).tolist(),
    }
    
    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Saved statistics to {output_dir / 'statistics.json'}")
    
    logger.info(f"Done! Dataset saved to {output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Input shape:  {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()


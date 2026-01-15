# Norne Dataset Preprocessing Blueprint for 4D Neural Operators

> **Purpose**: This document serves as a comprehensive blueprint for preprocessing the Norne reservoir simulation dataset into tensor format suitable for 4D FNO/DeepONet training.
> 
> **Created**: January 2026
> **Status**: Planning Complete - Ready for Implementation

---

## Table of Contents
1. [Dataset Overview](#1-dataset-overview)
2. [Source Data Location](#2-source-data-location)
3. [Target Tensor Structure](#3-target-tensor-structure)
4. [Input Channels Specification](#4-input-channels-specification)
5. [Output Channels Specification](#5-output-channels-specification)
6. [Key Findings from Data Analysis](#6-key-findings-from-data-analysis)
7. [Utilities to Reuse](#7-utilities-to-reuse)
8. [Step-by-Step Implementation Plan](#8-step-by-step-implementation-plan)
9. [Memory Considerations](#9-memory-considerations)
10. [Validation Checklist](#10-validation-checklist)

---

## 1. Dataset Overview

### Comparison with CO2 Sequestration Dataset

| Aspect | CO2 Dataset | Norne Dataset |
|--------|-------------|---------------|
| **Spatial dimensions** | 2D (96×200) | **3D (46×112×22)** |
| **Total grid cells** | 19,200 | 113,344 |
| **Active cells** | 19,200 (100%) | 44,431 (39%) |
| **Temporal steps** | 24 | 65 |
| **Samples** | 5,500 | 500 |
| **Input channels** | 12 | 11 |
| **Output channels** | 1 | 3 |
| **Simulation time** | N/A | 3,260 days (~9 years) |
| **Simulator** | Custom | OPM (ECLIPSE-compatible) |

### Key Difference
- CO2 dataset is **2D spatial + time** (3D tensor)
- Norne dataset is **3D spatial + time** (4D tensor)
- This requires 4D FNO support (to be implemented later)

---

## 2. Source Data Location

```
Base Path: /lustre/fsw/coreai_climate_earth2/tonishi/physicsnemo/examples/reservoir_simulation/dataset/Norne/

Simulation Directory:
  NORNE_ATW2013_LHS.sim/
  ├── NORNE_ATW2013_DOE_TOP15_EXTENDED_LHS_0000.DATA
  ├── NORNE_ATW2013_DOE_TOP15_EXTENDED_LHS_0000.INIT
  ├── NORNE_ATW2013_DOE_TOP15_EXTENDED_LHS_0000.EGRID
  ├── NORNE_ATW2013_DOE_TOP15_EXTENDED_LHS_0000.UNRST
  ├── NORNE_ATW2013_DOE_TOP15_EXTENDED_LHS_0000.UNSMRY
  ├── ... (500 cases total: 0000-0499)

Utilities:
  sim_utils/           # EclReader, Grid, Well classes
  xmgn/src/data/       # graph_builder.py with tested utilities
```

---

## 3. Target Tensor Structure

### Final Output Files

```
Output Directory: /lustre/fsw/coreai_climate_earth2/wdyab/physicsnemo_data/norne/
(Same location as CO2 dataset, in separate 'norne' subfolder)

CO2 Dataset Location (for reference):
  /lustre/fsw/coreai_climate_earth2/wdyab/physicsnemo_data/
  ├── sg_train_a.pt, sg_train_u.pt  (saturation)
  ├── dP_train_a.pt, dP_train_u.pt  (pressure)
  └── norne/                         (NEW - Norne dataset)

Norne Output Files:
  norne_train_a.pt    # Input:  (400, 46, 112, 22, 65, 11)
  norne_train_u.pt    # Output: (400, 46, 112, 22, 65, 3)
  norne_val_a.pt      # Input:  (50, 46, 112, 22, 65, 11)
  norne_val_u.pt      # Output: (50, 46, 112, 22, 65, 3)
  norne_test_a.pt     # Input:  (50, 46, 112, 22, 65, 11)
  norne_test_u.pt     # Output: (50, 46, 112, 22, 65, 3)
  metadata.json       # Dataset metadata
  statistics.json     # Normalization statistics
```

### Tensor Dimensions

```
Input Tensor:  (N, X, Y, Z, T, C_in)  = (N, 46, 112, 22, 65, 11)
Output Tensor: (N, X, Y, Z, T, C_out) = (N, 46, 112, 22, 65, 3)

Where:
  N = number of samples (400 train, 50 val, 50 test)
  X = 46 (grid cells in X direction)
  Y = 112 (grid cells in Y direction)
  Z = 22 (grid cells in Z direction, vertical layers)
  T = 65 (timesteps)
  C_in = 11 (input channels)
  C_out = 3 (output channels)
```

### Data Split
- **Train**: 400 samples (80%)
- **Validation**: 50 samples (10%)
- **Test**: 50 samples (10%)

---

## 4. Input Channels Specification

### Channel Layout (11 total)

| Index | Name | Type | Source | Scaling | Notes |
|-------|------|------|--------|---------|-------|
| 0 | PERMX | Static | INIT | LOG10 | Horizontal permeability |
| 1 | PERMZ | Static | INIT | LOG10 | Vertical permeability (**varies between LHS samples**) |
| 2 | PORO | Static | INIT | None | Porosity |
| 3 | PORV | Static | INIT | Normalize [0,1] | Pore volume |
| 4 | NTG | Static | INIT | None | Net-to-gross |
| 5 | ACTNUM | Static | Computed | Binary 0/1 | Active cell mask |
| 6 | grid_x | Static | Grid.X | Normalize [0,1] | Cell center X coordinate |
| 7 | grid_y | Static | Grid.Y | Normalize [0,1] | Cell center Y coordinate |
| 8 | grid_z | Static | Grid.Z | Normalize [0,1] | Cell center Z coordinate |
| 9 | grid_t | Dynamic | TIME | Normalize [0,1] | Time coordinate |
| 10 | WCID | Dynamic | Wells | Signed: +1/-1/0 | Well completion indicator |

### Static Channels (0-8)
- Constant across all 65 timesteps
- Broadcast when assembling final tensor
- Inactive cells filled with zeros

### Dynamic Channels (9-10)
- Vary with timestep
- `grid_t`: Same value for all cells at timestep t
- `WCID`: Sparse, changes when wells open/close

### WCID Convention
```
+1.0 = Injector well completion (open)
-1.0 = Producer well completion (open)
 0.0 = No well completion / closed well
```

---

## 5. Output Channels Specification

### Channel Layout (3 total)

| Index | Name | Source | Units | Notes |
|-------|------|--------|-------|-------|
| 0 | PRESSURE | UNRST | bar | Cell pressure |
| 1 | SWAT | UNRST | fraction | Water saturation [0,1] |
| 2 | SGAS | UNRST | fraction | Gas saturation [0,1] |

### Notes
- All outputs are per-cell, per-timestep values
- Inactive cells should be zero (masked in loss function)
- SOIL (oil saturation) can be derived: `SOIL = 1 - SWAT - SGAS`

---

## 6. Key Findings from Data Analysis

### Grid Information
```
Grid dimensions: 46 × 112 × 22
Total cells: 113,344
Active cells: 44,431 (39.2%)
```

### Permeability Analysis
```
PERMX = PERMY (isotropic horizontally)
PERMZ ≠ PERMX (anisotropic vertically)
PERMZ/PERMX ratio: ~0.25 average (typical Kv/Kh)

Value ranges:
  PERMX/Y: 0.65 - 3,997 mD (mean: 390 mD)
  PERMZ:   0.01 - 2,437 mD (mean: 83 mD)
```

### LHS Variation (What Changes Between Samples)
```
Property    | Varies?
------------|--------
PERMX       | NO (identical)
PERMZ       | YES (key LHS variable!)
PORO        | NO (identical)
NTG         | NO (identical)
PORV        | NO (identical)
Well sched. | Partially (timesteps 56-64 differ)
```

**Key Insight**: The Latin Hypercube Sampling varies only PERMZ (Kv/Kh ratio).

### Well Information
```
Number of wells: 36
Well types: Producers and Injectors
Completions: Multi-layer horizontal wells
Schedule: Mostly identical, diverges at late timesteps (56-64)
```

### Time Information
```
Restart timesteps: 65
Time range: 0 to 3,260 days (~9 years)
First 5 times: [0, 8, 25, 41, 56] days
Summary timesteps: 347 (higher frequency, not used)
```

### Available Dynamic Properties
```
Property    | Timesteps | Shape per step | Use
------------|-----------|----------------|-----
PRESSURE    | 65        | (44431,)       | Output
SWAT        | 65        | (44431,)       | Output
SGAS        | 65        | (44431,)       | Output
RS          | 64        | (44431,)       | Not used
RV          | 63        | (44431,)       | Not used
```

---

## 7. Utilities to Reuse

### From sim_utils (`reservoir_simulation/sim_utils/`)

#### EclReader
```python
from sim_utils import EclReader

reader = EclReader("path/to/CASE.DATA")
init_data = reader.read_init(["PERMX", "PERMZ", "PORO", ...])
egrid_data = reader.read_egrid(["COORD", "ZCORN", "FILEHEAD", "NNC1", "NNC2"])
restart_data = reader.read_restart(["PRESSURE", "SWAT", "SGAS", ...])
```

#### Grid Class
```python
from sim_utils import Grid

grid = Grid(init_data, egrid_data)

# Key attributes:
grid.nx, grid.ny, grid.nz    # Dimensions
grid.nact                     # Number of active cells
grid.actnum                   # Active cell mask (1D, full grid)
grid.actnum_bool             # Boolean version
grid.X, grid.Y, grid.Z       # Cell center coordinates (per active cell)
grid.ijk_to_active           # Dict: global_ijk -> active_idx

# Key method:
wcid_inj, wcid_prd = grid.create_completion_array(wells_dict)
```

### From XMGN (`xmgn/src/data/graph_builder.py`)

#### get_completion_info
```python
def get_completion_info(grid, well_info) -> list:
    """
    Extract well completion information from simulation data.
    
    Parameters:
        grid: Grid object
        well_info: Dict with INTEHEAD, ZWEL, IWEL, ICON, SCON
    
    Returns:
        list: List of Well dicts per timestep
    """
```

#### Key patterns to follow:
```python
# Reading well data keys
rst_well_keys = ["INTEHEAD", "ZWEL", "IWEL", "ICON", "SCON"]
well_data = reader.read_restart(rst_well_keys)

# Creating completion arrays
wells_per_timestep = get_completion_info(grid, well_data)
wcid_inj, wcid_prd = grid.create_completion_array(wells_per_timestep[t])
```

---

## 8. Step-by-Step Implementation Plan

### Step 1: Setup and Imports

```python
import sys
import os
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Add paths
REPO_ROOT = Path("/home/wdyab/physicsnemo/examples/reservoir_simulation")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "xmgn/src"))

from sim_utils import EclReader, Grid, Well
```

### Step 2: Extract Grid Information (Once)

```python
def get_grid_info(first_case_path: str) -> dict:
    """Extract grid structure from first case (identical across all samples)."""
    
    reader = EclReader(first_case_path)
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
    
    return {
        "nx": grid.nx,
        "ny": grid.ny,
        "nz": grid.nz,
        "nact": grid.nact,
        "actnum_3d": grid.actnum.reshape(grid.nx, grid.ny, grid.nz, order='F'),
        "ijk_to_active": grid.ijk_to_active,
        "active_to_ijk": active_to_ijk,
    }
```

### Step 3: Adapt get_completion_info from XMGN

```python
def get_completion_info_adapted(grid, well_info) -> list:
    """
    Adapted from XMGN graph_builder.py
    Extract well completion information for each timestep.
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
```

### Step 4: Process Single Sample

```python
def process_single_sample(case_path: str, grid_info: dict) -> tuple:
    """Process one simulation case, return (input_tensor, output_tensor)."""
    
    reader = EclReader(case_path)
    nx, ny, nz = grid_info["nx"], grid_info["ny"], grid_info["nz"]
    nact = grid_info["nact"]
    active_to_ijk = grid_info["active_to_ijk"]
    actnum_3d = grid_info["actnum_3d"]
    
    # --- Read all data ---
    egrid_data = reader.read_egrid(["COORD", "ZCORN", "FILEHEAD", "NNC1", "NNC2"])
    init_data = reader.read_init(["INTEHEAD", "PERMX", "PERMZ", "PORO", "PORV", "NTG"])
    restart_data = reader.read_restart(["PRESSURE", "SWAT", "SGAS"])
    well_data = reader.read_restart(["INTEHEAD", "ZWEL", "IWEL", "ICON", "SCON"])
    
    grid = Grid(init_data, egrid_data)
    num_timesteps = len(restart_data["PRESSURE"])
    times = np.array(restart_data["TIME"])
    
    # --- Process Static Properties ---
    static = {}
    for prop in ["PERMX", "PERMZ", "PORO", "NTG"]:
        static[prop] = np.zeros((nx, ny, nz), dtype=np.float32)
        data = init_data[prop]
        if prop in ["PERMX", "PERMZ"]:
            data = np.log10(np.maximum(data, 1e-10))  # LOG10 scaling
        for idx in range(nact):
            i, j, k = active_to_ijk[idx]
            static[prop][i, j, k] = data[idx]
    
    # PORV (from full grid, filter to active)
    static["PORV"] = np.zeros((nx, ny, nz), dtype=np.float32)
    porv_active = init_data["PORV"][actnum_3d.flatten(order='F') > 0]
    for idx in range(nact):
        i, j, k = active_to_ijk[idx]
        static["PORV"][i, j, k] = porv_active[idx]
    
    # Coordinates
    for coord, arr in [("X", grid.X), ("Y", grid.Y), ("Z", grid.Z)]:
        static[coord] = np.zeros((nx, ny, nz), dtype=np.float32)
        for idx in range(nact):
            i, j, k = active_to_ijk[idx]
            static[coord][i, j, k] = arr[idx]
    
    # --- Process Well Data (WCID) ---
    wells_per_timestep = get_completion_info_adapted(grid, well_data)
    WCID = np.zeros((nx, ny, nz, num_timesteps), dtype=np.float32)
    
    for t in range(num_timesteps):
        if t < len(wells_per_timestep) and wells_per_timestep[t]:
            wcid_inj, wcid_prd = grid.create_completion_array(wells_per_timestep[t])
            wcid_combined = wcid_inj - wcid_prd  # +1 injector, -1 producer
            for idx in range(nact):
                if wcid_combined[idx] != 0:
                    i, j, k = active_to_ijk[idx]
                    WCID[i, j, k, t] = wcid_combined[idx]
    
    # --- Process Outputs ---
    output = np.zeros((nx, ny, nz, num_timesteps, 3), dtype=np.float32)
    for t in range(num_timesteps):
        for c, prop in enumerate(["PRESSURE", "SWAT", "SGAS"]):
            data = restart_data[prop][t]
            for idx in range(nact):
                i, j, k = active_to_ijk[idx]
                output[i, j, k, t, c] = data[idx]
    
    # --- Assemble Input Tensor ---
    input_tensor = np.zeros((nx, ny, nz, num_timesteps, 11), dtype=np.float32)
    
    # Normalize coordinates
    X_active = static["X"][actnum_3d > 0]
    Y_active = static["Y"][actnum_3d > 0]
    Z_active = static["Z"][actnum_3d > 0]
    
    X_norm = (static["X"] - X_active.min()) / (X_active.max() - X_active.min() + 1e-10)
    Y_norm = (static["Y"] - Y_active.min()) / (Y_active.max() - Y_active.min() + 1e-10)
    Z_norm = (static["Z"] - Z_active.min()) / (Z_active.max() - Z_active.min() + 1e-10)
    PORV_norm = static["PORV"] / (static["PORV"].max() + 1e-10)
    time_norm = times / (times.max() + 1e-10)
    
    for t in range(num_timesteps):
        input_tensor[:, :, :, t, 0] = static["PERMX"]
        input_tensor[:, :, :, t, 1] = static["PERMZ"]
        input_tensor[:, :, :, t, 2] = static["PORO"]
        input_tensor[:, :, :, t, 3] = PORV_norm
        input_tensor[:, :, :, t, 4] = static["NTG"]
        input_tensor[:, :, :, t, 5] = actnum_3d.astype(np.float32)
        input_tensor[:, :, :, t, 6] = X_norm
        input_tensor[:, :, :, t, 7] = Y_norm
        input_tensor[:, :, :, t, 8] = Z_norm
        input_tensor[:, :, :, t, 9] = time_norm[t]
        input_tensor[:, :, :, t, 10] = WCID[:, :, :, t]
    
    return input_tensor, output
```

### Step 5: Main Processing Loop

```python
def main(sim_dir: str, output_dir: str, num_workers: int = 8):
    """Main preprocessing function."""
    
    sim_dir = Path(sim_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all cases
    case_files = sorted(sim_dir.glob("*.DATA"))
    num_cases = len(case_files)
    print(f"Found {num_cases} simulation cases")
    
    # Get grid info from first case
    print("Extracting grid information...")
    grid_info = get_grid_info(str(case_files[0]))
    print(f"  Grid: {grid_info['nx']} x {grid_info['ny']} x {grid_info['nz']}")
    print(f"  Active cells: {grid_info['nact']}")
    
    # Process all cases
    all_inputs = []
    all_outputs = []
    
    for case_path in tqdm(case_files, desc="Processing"):
        try:
            inp, out = process_single_sample(str(case_path), grid_info)
            all_inputs.append(inp)
            all_outputs.append(out)
        except Exception as e:
            print(f"Failed {case_path.name}: {e}")
    
    # Stack
    inputs = np.stack(all_inputs, axis=0)
    outputs = np.stack(all_outputs, axis=0)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    
    # Split
    num_samples = len(inputs)
    indices = np.random.permutation(num_samples)
    n_train = int(num_samples * 0.8)
    n_val = int(num_samples * 0.1)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Save
    print("Saving tensors...")
    torch.save(torch.from_numpy(inputs[train_idx]), output_dir / "norne_train_a.pt")
    torch.save(torch.from_numpy(outputs[train_idx]), output_dir / "norne_train_u.pt")
    torch.save(torch.from_numpy(inputs[val_idx]), output_dir / "norne_val_a.pt")
    torch.save(torch.from_numpy(outputs[val_idx]), output_dir / "norne_val_u.pt")
    torch.save(torch.from_numpy(inputs[test_idx]), output_dir / "norne_test_a.pt")
    torch.save(torch.from_numpy(outputs[test_idx]), output_dir / "norne_test_u.pt")
    
    # Save metadata
    metadata = {
        "grid_dims": [grid_info["nx"], grid_info["ny"], grid_info["nz"]],
        "num_active_cells": grid_info["nact"],
        "num_timesteps": inputs.shape[4],
        "input_channels": 11,
        "output_channels": 3,
        "input_channel_names": [
            "PERMX_log10", "PERMZ_log10", "PORO", "PORV_norm", "NTG",
            "ACTNUM", "grid_x", "grid_y", "grid_z", "grid_t", "WCID"
        ],
        "output_channel_names": ["PRESSURE", "SWAT", "SGAS"],
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "num_test": len(test_idx),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Compute and save statistics
    stats = {
        "input_mean": inputs.mean(axis=(0,1,2,3,4)).tolist(),
        "input_std": inputs.std(axis=(0,1,2,3,4)).tolist(),
        "output_mean": outputs.mean(axis=(0,1,2,3,4)).tolist(),
        "output_std": outputs.std(axis=(0,1,2,3,4)).tolist(),
    }
    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Done! Saved to {output_dir}")
```

### Step 6: SLURM Batch Script

```bash
#!/bin/bash
#SBATCH --job-name=norne_preprocess
#SBATCH --account=coreai_climate_earth2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=logs/preprocess_norne_%j.out

python preprocess_norne.py \
    --sim_dir /lustre/fsw/coreai_climate_earth2/tonishi/physicsnemo/examples/reservoir_simulation/dataset/Norne/NORNE_ATW2013_LHS.sim \
    --output_dir /lustre/fsw/coreai_climate_earth2/wdyab/physicsnemo_data/norne \
    --num_workers 8
```

---

## 9. Memory Considerations

### Estimated Sizes

| Item | Size |
|------|------|
| Single input sample | 46×112×22×65×11×4 bytes = ~292 MB |
| Single output sample | 46×112×22×65×3×4 bytes = ~80 MB |
| All inputs (500) | ~146 GB |
| All outputs (500) | ~40 GB |
| **Total** | **~186 GB in memory** |

### Recommendations

1. **High-memory node**: Request 500GB+ memory for preprocessing
2. **Process in batches**: If memory limited, process 100 samples at a time
3. **Parallel I/O**: Use multiple workers for reading simulation files
4. **Save incrementally**: Write intermediate results to disk

---

## 10. Validation Checklist

### After Preprocessing

- [ ] Verify tensor shapes match expected:
  - Input: `(N, 46, 112, 22, 65, 11)`
  - Output: `(N, 46, 112, 22, 65, 3)`

- [ ] Check data ranges:
  - PERMX/PERMZ (log10): typically [-2, 4]
  - PORO: [0, 0.4]
  - SWAT/SGAS: [0, 1]
  - PRESSURE: [200, 400] bar
  - WCID: {-1, 0, +1}

- [ ] Verify inactive cells are zero:
  ```python
  actnum = inputs[0, :, :, :, 0, 5]  # ACTNUM channel
  inactive_mask = actnum == 0
  assert (inputs[0, inactive_mask, :, :] == 0).all()
  ```

- [ ] Check WCID sparsity:
  ```python
  wcid = inputs[:, :, :, :, :, 10]
  sparsity = (wcid != 0).sum() / wcid.numel()
  print(f"WCID sparsity: {sparsity:.4%}")  # Should be < 1%
  ```

- [ ] Verify train/val/test split sizes:
  - Train: 400 samples
  - Val: 50 samples
  - Test: 50 samples

- [ ] Load and inspect a sample visually (optional)

---

## Appendix: File Locations Summary

```
Source Data:
  /lustre/fsw/coreai_climate_earth2/tonishi/physicsnemo/examples/reservoir_simulation/dataset/Norne/NORNE_ATW2013_LHS.sim/

Utilities:
  /home/wdyab/physicsnemo/examples/reservoir_simulation/sim_utils/
  /home/wdyab/physicsnemo/examples/reservoir_simulation/xmgn/src/data/graph_builder.py

Output:
  /lustre/fsw/coreai_climate_earth2/wdyab/physicsnemo_data/norne/

Scripts (to create):
  /home/wdyab/physicsnemo/examples/reservoir_simulation/neural_operator_factory/preprocess_norne.py
  /home/wdyab/physicsnemo/examples/reservoir_simulation/neural_operator_factory/preprocess_norne.sbatch
```

---

**End of Blueprint**

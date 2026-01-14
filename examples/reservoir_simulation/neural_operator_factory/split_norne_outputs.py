#!/usr/bin/env python3
"""
Split Norne dataset outputs into single-channel files.

The original Norne output has 3 channels: [PRESSURE, SWAT, SGAS]
This script splits them into separate files for training individual models.

Usage:
    python split_norne_outputs.py --data_path /path/to/norne --output_path /path/to/output

Output files created:
    - norne_{mode}_pressure.pt  (channel 0: PRESSURE)
    - norne_{mode}_swat.pt      (channel 1: SWAT - water saturation)
    - norne_{mode}_sgas.pt      (channel 2: SGAS - gas saturation)
"""

import argparse
import json
from pathlib import Path

import torch


CHANNEL_NAMES = ["pressure", "swat", "sgas"]
CHANNEL_FULL_NAMES = ["PRESSURE", "SWAT", "SGAS"]


def split_output_file(
    input_path: Path,
    output_dir: Path,
    mode: str,
    verbose: bool = True,
    memory_efficient: bool = False
) -> dict:
    """Split a single output file into three single-channel files."""
    if verbose:
        print(f"\nProcessing {mode} set...")
        print(f"  Loading: {input_path}")
    
    shapes = {}
    
    if memory_efficient:
        for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
            if verbose:
                print(f"  Processing channel {channel_idx}: {channel_name}...")
            
            data = torch.load(input_path, map_location='cpu', weights_only=True)
            
            if verbose and channel_idx == 0:
                print(f"  Original shape: {data.shape}")
                print(f"  dtype: {data.dtype}")
            
            single_channel = data[..., channel_idx:channel_idx+1].clone()
            del data
            
            output_filename = f"norne_{mode}_{channel_name}.pt"
            output_path = output_dir / output_filename
            
            torch.save(single_channel, output_path)
            shapes[channel_name] = list(single_channel.shape)
            
            if verbose:
                size_mb = output_path.stat().st_size / (1024**2)
                print(f"  Created: {output_filename} | shape: {single_channel.shape} | {size_mb:.1f} MB")
            
            del single_channel
    else:
        data = torch.load(input_path, map_location='cpu', weights_only=True)
        
        if verbose:
            print(f"  Original shape: {data.shape}")
            print(f"  dtype: {data.dtype}")
        
        for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
            single_channel = data[..., channel_idx:channel_idx+1]
            
            output_filename = f"norne_{mode}_{channel_name}.pt"
            output_path = output_dir / output_filename
            
            torch.save(single_channel, output_path)
            shapes[channel_name] = list(single_channel.shape)
            
            if verbose:
                size_mb = output_path.stat().st_size / (1024**2)
                print(f"  Created: {output_filename} | shape: {single_channel.shape} | {size_mb:.1f} MB")
    
    return shapes


def copy_input_files(input_dir: Path, output_dir: Path, verbose: bool = True) -> None:
    """Create symlinks for input files (they remain unchanged)."""
    if verbose:
        print("\nLinking input files (unchanged)...")
    
    for mode in ["train", "val", "test"]:
        src = input_dir / f"norne_{mode}_a.pt"
        dst = output_dir / f"norne_{mode}_input.pt"
        
        if src.exists():
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
            if verbose:
                print(f"  Linked: {dst.name} -> {src}")


def update_metadata(input_dir: Path, output_dir: Path, shapes: dict, verbose: bool = True) -> None:
    """Create updated metadata for the split dataset."""
    meta_path = input_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            original_meta = json.load(f)
    else:
        original_meta = {}
    
    for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
        channel_meta = original_meta.copy()
        channel_meta["output_channels"] = 1
        channel_meta["output_channel_names"] = [CHANNEL_FULL_NAMES[channel_idx]]
        channel_meta["original_channel_index"] = channel_idx
        channel_meta["split_from"] = "norne_{mode}_u.pt"
        
        meta_output_path = output_dir / f"metadata_{channel_name}.json"
        with open(meta_output_path, 'w') as f:
            json.dump(channel_meta, f, indent=2)
        
        if verbose:
            print(f"  Created: metadata_{channel_name}.json")


def update_statistics(input_dir: Path, output_dir: Path, verbose: bool = True) -> None:
    """Create updated statistics for each split channel."""
    stats_path = input_dir / "statistics.json"
    if not stats_path.exists():
        if verbose:
            print("  No statistics.json found, skipping...")
        return
    
    with open(stats_path) as f:
        original_stats = json.load(f)
    
    if verbose:
        print("\nCreating per-channel statistics...")
    
    for channel_idx, channel_name in enumerate(CHANNEL_NAMES):
        channel_stats = {
            "input_mean": original_stats.get("input_mean", []),
            "input_std": original_stats.get("input_std", []),
            "input_min": original_stats.get("input_min", []),
            "input_max": original_stats.get("input_max", []),
            "output_mean": [original_stats["output_mean"][channel_idx]],
            "output_std": [original_stats["output_std"][channel_idx]],
            "output_min": [original_stats["output_min"][channel_idx]],
            "output_max": [original_stats["output_max"][channel_idx]],
        }
        
        stats_output_path = output_dir / f"statistics_{channel_name}.json"
        with open(stats_output_path, 'w') as f:
            json.dump(channel_stats, f, indent=2)
        
        if verbose:
            print(f"  Created: statistics_{channel_name}.json")
            print(f"    {CHANNEL_FULL_NAMES[channel_idx]}: "
                  f"mean={channel_stats['output_mean'][0]:.4f}, "
                  f"std={channel_stats['output_std'][0]:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Split Norne dataset outputs into single-channel files"
    )
    parser.add_argument(
        "--data_path", type=str,
        default="/lustre/fsw/coreai_climate_earth2/wdyab/physicsnemo_data/norne",
        help="Path to original Norne dataset directory"
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Output directory (default: same as data_path)"
    )
    parser.add_argument(
        "--channels", type=str, nargs="+",
        default=["pressure", "swat", "sgas"],
        choices=["pressure", "swat", "sgas"],
        help="Which channels to extract (default: all)"
    )
    parser.add_argument(
        "--link_inputs", action="store_true",
        help="Create symlinks for input files with consistent naming"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--memory_efficient", action="store_true",
        help="Process channels one at a time to reduce memory usage"
    )
    parser.add_argument(
        "--modes", type=str, nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which dataset splits to process (default: all)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.data_path)
    output_dir = Path(args.output_path) if args.output_path else input_dir
    verbose = not args.quiet
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("Norne Dataset Output Splitter")
        print("=" * 60)
        print(f"Input directory:  {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Channels to extract: {args.channels}")
        print(f"Modes to process: {args.modes}")
        print(f"Memory efficient: {args.memory_efficient}")
    
    all_shapes = {}
    for mode in args.modes:
        input_path = input_dir / f"norne_{mode}_u.pt"
        
        if not input_path.exists():
            if verbose:
                print(f"\nSkipping {mode} (file not found: {input_path})")
            continue
        
        shapes = split_output_file(
            input_path, output_dir, mode, verbose,
            memory_efficient=args.memory_efficient
        )
        all_shapes[mode] = shapes
    
    if args.link_inputs:
        copy_input_files(input_dir, output_dir, verbose)
    
    if verbose:
        print("\nCreating metadata files...")
    update_metadata(input_dir, output_dir, all_shapes, verbose)
    update_statistics(input_dir, output_dir, verbose)
    
    if verbose:
        print("\n" + "=" * 60)
        print("Done! Created files:")
        print("=" * 60)
        for channel in args.channels:
            print(f"\n  {channel.upper()} model files:")
            print(f"    - norne_train_{channel}.pt")
            print(f"    - norne_val_{channel}.pt")
            print(f"    - norne_test_{channel}.pt")
            print(f"    - metadata_{channel}.json")
            print(f"    - statistics_{channel}.json")
        
        print("\n  To train a model for a specific variable, use:")
        print(f"    output_file='norne_{{mode}}_pressure.pt'  # for PRESSURE")
        print(f"    output_file='norne_{{mode}}_swat.pt'      # for SWAT")
        print(f"    output_file='norne_{{mode}}_sgas.pt'      # for SGAS")


if __name__ == "__main__":
    main()

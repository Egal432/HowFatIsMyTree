#!/usr/bin/env python3
"""
Step 1: Tiling
Splits the large EcoSense LAZ file into smaller spatial tiles.
Output: out_dir/tiles/*.laz
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import laspy

try:
    import pdal
except ImportError as e:
    raise SystemExit("Need PDAL python bindings: `pip install pdal`") from e


def run_pdal(pipeline_obj: dict) -> int:
    pipe = pdal.Pipeline(json.dumps(pipeline_obj))
    return pipe.execute()


def tile_laz(in_laz: Path, tiles_dir: Path, tile_size: float) -> None:
    """Split huge LAZ into XY tiles, preserving all extra dimensions."""
    tiles_dir.mkdir(parents=True, exist_ok=True)

    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(in_laz)},
            {"type": "filters.splitter", "length": float(tile_size)},
            {
                "type": "writers.las",
                "filename": str(tiles_dir / "tile_#.laz"),
                "compression": "laszip",
                "extra_dims": "all",
                "minor_version": 4,
                "dataformat_id": 7,
            },
        ]
    }
    print("[STEP 1] Tiling big LAZ (PDAL running)...")
    count = run_pdal(pipeline)
    print(f"[STEP 1] Done. PDAL processed ~{count:,} points.")


def sanity_check_tile_has_predinstance(tiles_dir: Path) -> None:
    """Verify that the generated tiles contain the PredInstance dimension."""
    tiles = sorted(list(tiles_dir.glob("*.laz")) + list(tiles_dir.glob("*.las")))
    if not tiles:
        raise SystemExit("[ERROR] No tiles found to sanity-check.")

    sample = tiles[0]
    las = laspy.read(str(sample))
    dims = set(las.point_format.dimension_names)
    PRED_DIM = "PredInstance"
    
    if PRED_DIM not in dims:
        raise SystemExit(
            f"[ERROR] Tiles do NOT contain '{PRED_DIM}'. Example tile: {sample.name}\n"
            "Fix: Ensure writers.las has extra_dims='all'."
        )
    print(f"[STEP 1] Sanity check passed – '{PRED_DIM}' present in {sample.name}.")


def main():
    ap = argparse.ArgumentParser(description="Step 1: Tile the point cloud.")
    ap.add_argument("ecosense_laz", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--tile-size", type=float, default=50.0, help="Tile side length (metres)")
    args = ap.parse_args()

    tiles_dir = args.out_dir / "tiles"
    
    # Check if already tiled
    if tiles_dir.is_dir():
        print(f"[STEP 1] Tiles already exist at {tiles_dir}. Skipping tiling.")
    else:
        tile_laz(args.ecosense_laz, tiles_dir, args.tile_size)

    sanity_check_tile_has_predinstance(tiles_dir)
    print(f"[STEP 1] Complete. Tiles ready at {tiles_dir}")


if __name__ == "__main__":
    main()
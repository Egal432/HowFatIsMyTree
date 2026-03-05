#!/usr/bin/env python3
"""
preprocess_ecosense_match_and_export.py

EcoSense preprocessing (corrected to your needs):

You have:
- A huge LAZ file with per-point instance segmentation stored in: PredInstance
- An inventory CSV with true DBH (diameter_m) and coordinates (x_32632, y_32632)

Goal:
1) Match each inventory row (tree) to a PredInstance using nearest-neighbor in XY
2) Export each matched PredInstance as a single LAZ file (one file per inventory tree)
3) Write labels.csv for ML (file path + dbh + metadata)

Why tiling?
- ecosense.laz is ~450M points; cannot load into RAM.
- We split into spatial tiles first, then work per tile.

Workflow overview:

PASS 0: Load inventory.csv, validate required columns, assign sequential tree_id if needed.

PASS 1: Tile the big LAZ (PDAL splitter) into manageable pieces.
        (No ground normalization here because you said you want to keep raw and normalize later.)

PASS 2: Matching inventory -> PredInstance (per tile)
        - For each tile, select inventory points inside tile bbox (plus buffer)
        - Build a nearest-neighbor search structure on tile points (KDTree if scipy available)
          and query each inventory point to get the PredInstance of the closest point.
        - Record match distance for QA.

PASS 3: Export (per tile)
        - For each tile, write "parts" for every matched PredInstance contained in this tile.

PASS 4: Merge parts per inventory-tree into a single LAZ using PDAL, write to trees/
        - Then write labels.csv and match_report.csv

Notes:
- PredInstance must exist in the LAZ/tiles and be non-constant.
- Inventory coordinates must be in same CRS/units as LAZ XY (looks like EPSG:32632 meters).
- If your inventory contains multiple rows per tree or duplicates, we keep them as separate labels
  unless you decide to deduplicate by full_id or qr_code_id.

"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import laspy

try:
    import pdal
except ImportError as e:
    raise SystemExit(
        "PDAL python bindings not found. Install with `pip install pdal` and ensure PDAL is installed."
    ) from e

# Optional accelerator
try:
    from scipy.spatial import cKDTree  # type: ignore
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ----------------------------
# Config / required columns
# ----------------------------

REQUIRED_INV_COLS = [
    "x_32632",
    "y_32632",
    "diameter_m",
    # plus whatever else you want carried to labels.csv (we'll keep all columns)
]

PRED_DIM = "PredInstance"


@dataclass
class MatchResult:
    inv_index: int
    x: float
    y: float
    predinstance: Optional[int]
    nn_dist_m: Optional[float]
    tile_name: Optional[str]
    note: str


# ----------------------------
# PDAL helpers
# ----------------------------

def run_pdal(pipeline_obj: dict) -> int:
    """Run a PDAL JSON pipeline and return point count processed."""
    pipe = pdal.Pipeline(json.dumps(pipeline_obj))
    return pipe.execute()


def tile_laz(in_laz: Path, tiles_dir: Path, tile_size: float) -> None:
    """
    Split the huge LAZ into spatial tiles.

    We keep all dimensions (including extra dims like PredInstance).
    """
    tiles_dir.mkdir(parents=True, exist_ok=True)

    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(in_laz)},
            {"type": "filters.splitter", "length": float(tile_size)},
            {
                "type": "writers.las",
                "filename": str(tiles_dir / "tile_#.laz"),
                "compression": "laszip",
            },
        ]
    }

    print("[PASS1] Tiling big LAZ with PDAL splitter...")
    count = run_pdal(pipeline)
    print(f"[PASS1] Done. PDAL processed ~{count:,} points (reported).")
    print(f"[PASS1] Tiles written to: {tiles_dir.resolve()}")


def merge_parts_to_tree(parts: List[Path], out_path: Path) -> None:
    """
    Merge multiple LAZ parts into one LAZ using PDAL.
    This avoids loading all parts into Python RAM.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": [str(p) for p in parts]},
            {"type": "filters.merge"},
            {"type": "writers.las", "filename": str(out_path), "compression": "laszip"},
        ]
    }
    run_pdal(pipeline)


# ----------------------------
# Matching helpers
# ----------------------------

def read_tile_header_bounds(tile_path: Path) -> Tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) from tile header."""
    las = laspy.read(str(tile_path))
    xmin, ymin, _ = las.header.mins
    xmax, ymax, _ = las.header.maxs
    return float(xmin), float(ymin), float(xmax), float(ymax)


def build_nn_index_xy(x: np.ndarray, y: np.ndarray):
    """
    Build a nearest-neighbor index for tile points.
    - If scipy exists: cKDTree (fast).
    - Else: a simple fallback that does approximate grid lookup (slower but works).
    """
    if HAVE_SCIPY:
        pts = np.column_stack([x, y]).astype(np.float64, copy=False)
        return ("kdtree", cKDTree(pts))

    # Fallback: grid hash (approx). We'll still compute exact distance among candidates.
    # Cell size controls speed/accuracy; keep modest.
    cell = 0.25  # meters
    xmin = float(x.min())
    ymin = float(y.min())
    ix = np.floor((x - xmin) / cell).astype(np.int32)
    iy = np.floor((y - ymin) / cell).astype(np.int32)

    grid: Dict[Tuple[int, int], np.ndarray] = {}
    # Store indices per cell (cap per cell to limit memory)
    cap = 500
    for i in range(x.size):
        key = (int(ix[i]), int(iy[i]))
        if key not in grid:
            grid[key] = np.array([i], dtype=np.int32)
        else:
            if grid[key].size < cap:
                grid[key] = np.append(grid[key], i)

    return ("grid", (grid, xmin, ymin, cell))


def query_nn(index_obj, qx: float, qy: float, x: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """
    Return (nn_index, nn_distance_m) for query point (qx,qy).
    Works with both KDTree and grid fallback.
    """
    mode, data = index_obj
    if mode == "kdtree":
        tree = data
        dist, idx = tree.query([qx, qy], k=1)
        return int(idx), float(dist)

    grid, xmin, ymin, cell = data
    # start at query cell and expand ring until we find candidates
    cx = int(math.floor((qx - xmin) / cell))
    cy = int(math.floor((qy - ymin) / cell))

    best_i = -1
    best_d2 = float("inf")

    # Expand search up to a reasonable radius in cells
    # (inventory should fall near a tree, so we usually find candidates quickly)
    for r in range(0, 20):  # 20 cells * 0.25m = 5m max
        found_any = False
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                key = (cx + dx, cy + dy)
                if key not in grid:
                    continue
                idxs = grid[key]
                found_any = True
                # Exact distance among candidates
                dxv = x[idxs] - qx
                dyv = y[idxs] - qy
                d2 = dxv * dxv + dyv * dyv
                j = int(np.argmin(d2))
                if float(d2[j]) < best_d2:
                    best_d2 = float(d2[j])
                    best_i = int(idxs[j])
        if found_any and best_i >= 0:
            break

    if best_i < 0:
        # As last resort, brute force (slow, but only if grid failed badly)
        dxv = x - qx
        dyv = y - qy
        d2 = dxv * dxv + dyv * dyv
        best_i = int(np.argmin(d2))
        best_d2 = float(d2[best_i])

    return best_i, float(math.sqrt(best_d2))


# ----------------------------
# Export helpers
# ----------------------------

def write_part(tile_las: laspy.LasData, mask: np.ndarray, out_path: Path) -> None:
    """
    Write a subset of points (mask) from tile_las into a LAZ part file.

    We preserve:
    - point format / extra dimensions (PredInstance, Reflectance, etc.)
    - scales/offsets (via header copy)
    """
    if not np.any(mask):
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    sub = laspy.create(point_format=tile_las.header.point_format, file_version=tile_las.header.version)
    # Copy header properties (scales/offsets etc.)
    sub.header = tile_las.header

    # laspy supports slicing points directly
    sub.points = tile_las.points[mask]
    sub.write(str(out_path))


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ecosense_laz", type=Path)
    ap.add_argument("inventory_csv", type=Path)
    ap.add_argument("out_dir", type=Path)

    ap.add_argument("--tile-size", type=float, default=50.0, help="Tile size (meters) for PDAL splitter")
    ap.add_argument("--buffer", type=float, default=2.0, help="Extra bbox buffer (m) when selecting inventory pts for a tile")
    ap.add_argument("--search-radius", type=float, default=1.0, help="Max allowed XY match distance (m). Beyond => flagged")
    ap.add_argument("--reservoir", type=int, default=0, help="Unused here (kept for compatibility).")

    ap.add_argument("--skip-tiling", action="store_true", help="If tiles already exist, skip PASS1 tiling")
    ap.add_argument("--keep-parts", action="store_true", help="Keep temporary parts files (debugging)")

    args = ap.parse_args()

    out_dir: Path = args.out_dir
    tiles_dir = out_dir / "tiles"
    parts_dir = out_dir / "parts"
    trees_dir = out_dir / "trees"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # PASS 0: Load inventory
    # ----------------------------
    inv = pd.read_csv(args.inventory_csv)

    missing = [c for c in REQUIRED_INV_COLS if c not in inv.columns]
    if missing:
        raise SystemExit(f"Inventory missing required columns: {missing}\nFound columns: {list(inv.columns)}")

    # Ensure sequential tree_id exists (you said: choose sequential tree id)
    if "tree_id" not in inv.columns:
        inv = inv.copy()
        inv["tree_id"] = np.arange(1, len(inv) + 1, dtype=int)

    # Make sure XY is numeric
    inv["x_32632"] = pd.to_numeric(inv["x_32632"], errors="coerce")
    inv["y_32632"] = pd.to_numeric(inv["y_32632"], errors="coerce")
    inv["diameter_m"] = pd.to_numeric(inv["diameter_m"], errors="coerce")

    # Drop rows with missing coords/dbh (or keep and flag)
    bad_xy = inv["x_32632"].isna() | inv["y_32632"].isna()
    if bad_xy.any():
        print(f"[WARN] {bad_xy.sum()} inventory rows have missing XY. They will be unmatched.")
    inv_valid = inv[~bad_xy].copy()

    # ----------------------------
    # PASS 1: Tile the big LAZ
    # ----------------------------
    if not args.skip_tiling:
        tile_laz(args.ecosense_laz, tiles_dir, tile_size=args.tile_size)
    else:
        print("[PASS1] Skipping tiling (using existing tiles).")

    tiles = sorted(list(tiles_dir.glob("*.laz")) + list(tiles_dir.glob("*.las")))
    if not tiles:
        raise SystemExit(f"No tiles found in {tiles_dir}. Did tiling fail?")

    # ----------------------------
    # PASS 2: Match inventory -> PredInstance
    # ----------------------------
    # We’ll store best match per inventory row (if it appears in multiple tiles due to buffer overlap).
    best_match: Dict[int, MatchResult] = {
        int(i): MatchResult(inv_index=int(i), x=float(inv_valid.loc[i, "x_32632"]), y=float(inv_valid.loc[i, "y_32632"]),
                            predinstance=None, nn_dist_m=None, tile_name=None, note="unmatched")
        for i in inv_valid.index
    }

    print("[PASS2] Matching inventory XY to PredInstance (tile-by-tile)...")
    for ti, tile_path in enumerate(tiles, start=1):
        # Read bounds from header quickly
        xmin, ymin, xmax, ymax = read_tile_header_bounds(tile_path)

        # Select inventory points within bbox (+buffer)
        buf = float(args.buffer)
        sel = inv_valid[
            (inv_valid["x_32632"] >= xmin - buf) &
            (inv_valid["x_32632"] <= xmax + buf) &
            (inv_valid["y_32632"] >= ymin - buf) &
            (inv_valid["y_32632"] <= ymax + buf)
        ]
        if sel.empty:
            continue

        las = laspy.read(str(tile_path))
        dims = set(las.point_format.dimension_names)
        if PRED_DIM not in dims:
            raise RuntimeError(f"Tile {tile_path.name} does not contain {PRED_DIM}.")
        # Coordinates
        x = las.x.astype(np.float64, copy=False)
        y = las.y.astype(np.float64, copy=False)
        pred = np.asarray(las[PRED_DIM]).astype(np.int64, copy=False)

        # Build NN index
        nn_index = build_nn_index_xy(x, y)

        # Match each selected inventory point
        for inv_i, row in sel.iterrows():
            qx = float(row["x_32632"])
            qy = float(row["y_32632"])

            idx_nn, dist = query_nn(nn_index, qx, qy, x, y)
            inst = int(pred[idx_nn])

            # Keep best (smallest distance) across possible tiles
            cur = best_match[int(inv_i)]
            if cur.nn_dist_m is None or dist < cur.nn_dist_m:
                note = "ok"
                if dist > float(args.search_radius):
                    note = f"far_match>{args.search_radius}m"
                best_match[int(inv_i)] = MatchResult(
                    inv_index=int(inv_i),
                    x=qx,
                    y=qy,
                    predinstance=inst,
                    nn_dist_m=float(dist),
                    tile_name=tile_path.name,
                    note=note
                )

        if ti % 50 == 0:
            print(f"[PASS2] processed {ti}/{len(tiles)} tiles... matched so far: "
                  f"{sum(1 for m in best_match.values() if m.predinstance is not None)}")

    # Convert match results to DataFrame
    match_df = pd.DataFrame([m.__dict__ for m in best_match.values()])
    match_df.to_csv(out_dir / "match_report.csv", index=False)
    print(f"[PASS2] Match report written: {(out_dir / 'match_report.csv').resolve()}")

    # Join matches back to full inventory
    inv_out = inv.copy()
    inv_out["predinstance"] = np.nan
    inv_out["match_dist_m"] = np.nan
    inv_out["match_tile"] = ""
    inv_out["match_note"] = "unmatched"

    for m in best_match.values():
        inv_out.loc[m.inv_index, "predinstance"] = m.predinstance if m.predinstance is not None else np.nan
        inv_out.loc[m.inv_index, "match_dist_m"] = m.nn_dist_m if m.nn_dist_m is not None else np.nan
        inv_out.loc[m.inv_index, "match_tile"] = m.tile_name or ""
        inv_out.loc[m.inv_index, "match_note"] = m.note

    # Filter to matched rows for export
    matched = inv_out[inv_out["predinstance"].notna()].copy()
    matched["predinstance"] = matched["predinstance"].astype(int)

    if matched.empty:
        raise SystemExit("[FAIL] No inventory rows were matched to PredInstance. Check CRS / coordinates / search-radius.")

    # Many inventory rows may map to same PredInstance (duplicates / multiple stems etc.)
    # We export ONE tree file PER inventory row (because you want labels per true DBH record).
    # That means if two inventory rows map to same PredInstance, they will share the same point cloud file
    # unless you want to duplicate the LAZ (we avoid duplicating to save storage).
    #
    # We’ll export point clouds per unique PredInstance, then labels.csv references the shared file.
    unique_inst = sorted(matched["predinstance"].unique().tolist())
    inst_to_file = {inst: f"pred_{inst}.laz" for inst in unique_inst}

    print(f"[PASS3] Unique matched instances: {len(unique_inst)}")
    print("[PASS3] Writing per-instance parts from tiles...")

    # ----------------------------
    # PASS 3: Extract per-instance parts per tile
    # ----------------------------
    # We create partial LAZ files per tile per instance (only when present),
    # then merge them per instance into one final LAZ in PASS 4.
    parts_index: Dict[int, List[Path]] = {inst: [] for inst in unique_inst}
    unique_set = set(unique_inst)

    for ti, tile_path in enumerate(tiles, start=1):
        las = laspy.read(str(tile_path))
        dims = set(las.point_format.dimension_names)
        if PRED_DIM not in dims:
            raise RuntimeError(f"Tile {tile_path.name} does not contain {PRED_DIM}.")

        pred = np.asarray(las[PRED_DIM]).astype(np.int64, copy=False)

        # Fast pre-check: does this tile contain any instance we need?
        tile_insts = np.unique(pred)
        needed_here = [int(i) for i in tile_insts if int(i) in unique_set]
        if not needed_here:
            continue

        for inst in needed_here:
            mask = (pred == inst)
            if not np.any(mask):
                continue
            part_path = parts_dir / f"pred_{inst}" / f"{tile_path.stem}__part.laz"
            write_part(las, mask, part_path)
            parts_index[inst].append(part_path)

        if ti % 50 == 0:
            print(f"[PASS3] processed {ti}/{len(tiles)} tiles...")

    # ----------------------------
    # PASS 4: Merge parts -> final per-instance LAZ
    # ----------------------------
    print("[PASS4] Merging parts into final per-instance LAZ files...")
    trees_dir.mkdir(parents=True, exist_ok=True)

    for k, inst in enumerate(unique_inst, start=1):
        parts = parts_index.get(inst, [])
        if not parts:
            print(f"[WARN] No parts found for PredInstance {inst} (matched but not extracted).")
            continue

        out_tree_path = trees_dir / inst_to_file[inst]
        merge_parts_to_tree(parts, out_tree_path)

        if k % 100 == 0:
            print(f"[PASS4] merged {k}/{len(unique_inst)} instances...")

    # ----------------------------
    # labels.csv (one row per INVENTORY record)
    # ----------------------------
    # This is what you’ll feed your ML:
    # - tree_id (sequential)
    # - predinstance
    # - tree_pointcloud_file (relative path)
    # - diameter_m (true DBH label)
    # - plus your other inventory metadata columns (species, plot_id, ...)
    labels = inv_out.copy()
    labels = labels[labels["predinstance"].notna()].copy()
    labels["predinstance"] = labels["predinstance"].astype(int)
    labels["tree_pointcloud_file"] = labels["predinstance"].map(lambda i: str(Path("trees") / inst_to_file[int(i)]))

    # Keep your requested label fields + keep everything else
    # (you can drop columns later in the dataloader)
    labels_path = out_dir / "labels.csv"
    labels.to_csv(labels_path, index=False)
    print(f"[DONE] labels.csv written: {labels_path.resolve()}")

    # Optional cleanup
    if not args.keep_parts:
        # Remove parts directory to save space
        # (commented out for safety; enable if you want auto cleanup)
        # import shutil
        # shutil.rmtree(parts_dir, ignore_errors=True)
        print("[NOTE] Temporary parts kept (cleanup disabled by default). "
              "If you want auto cleanup, enable it in the script or run with --keep-parts off + uncomment cleanup.")

    print(f"[DONE] Final tree clouds in: {trees_dir.resolve()}")
    print(f"[DONE] Match QA report in: {(out_dir / 'match_report.csv').resolve()}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
EcoSense preprocessing:

You have:
- ecosense.laz with instance segmentation in PredInstance (extra dimension)
- inventory.csv with true DBH + tree coordinates x_32632/y_32632

Goal:
- Match each inventory XY to a PredInstance (nearest neighbor in XY)
- Export each matched PredInstance as a single-tree LAZ
- Write labels.csv for ML

Key scalability trick:
- Tile the big LAZ into manageable chunks (PDAL splitter), then:
  - match inventory -> PredInstance tile-by-tile
  - extract per-instance parts per tile
  - merge parts per instance into final single-tree LAZ

Important:
- PredInstance is usually an ExtraBytes dimension. PDAL will drop it on write unless
  writers.las is told to keep extra dims: extra_dims="all".
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import laspy

try:
    import pdal
except ImportError as e:
    raise SystemExit("Need PDAL python bindings: `pip install pdal` + PDAL installed.") from e

# Progress bars
try:
    from tqdm import tqdm
except ImportError as e:
    raise SystemExit("Install tqdm: `pip install tqdm`") from e

# Optional accelerator
try:
    from scipy.spatial import cKDTree  # type: ignore
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


REQUIRED_INV_COLS = ["x_32632", "y_32632", "diameter_m"]
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


def run_pdal(pipeline_obj: dict) -> int:
    pipe = pdal.Pipeline(json.dumps(pipeline_obj))
    return pipe.execute()


def tile_laz(in_laz: Path, tiles_dir: Path, tile_size: float) -> None:
    """
    Split huge LAZ into XY tiles.

    CRITICAL: preserve ExtraBytes dims (like PredInstance).
    - extra_dims="all" tells PDAL to forward all extra dimensions to output
    - minor_version=4 + dataformat_id=7 forces LAS 1.4 point format 7 (matches your input)
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
                "extra_dims": "all",
                "minor_version": 4,
                "dataformat_id": 7,
            },
        ]
    }

    print("[PASS1] Tiling big LAZ (this can take a while, PDAL runs as one big job)...")
    count = run_pdal(pipeline)
    print(f"[PASS1] Done. PDAL processed ~{count:,} points (reported).")


def list_tiles(tiles_dir: Path) -> List[Path]:
    return sorted(list(tiles_dir.glob("*.laz")) + list(tiles_dir.glob("*.las")))


def sanity_check_tile_has_predinstance(tiles: List[Path]) -> None:
    """
    After tiling, verify PredInstance is actually present in tiles.
    If missing, stop early with an actionable error message.
    """
    if not tiles:
        raise SystemExit("[FAIL] No tiles found to sanity-check.")

    sample = tiles[0]
    las = laspy.read(str(sample))
    dims = set(las.point_format.dimension_names)
    if PRED_DIM not in dims:
        raise SystemExit(
            f"[FAIL] Tiles do NOT contain '{PRED_DIM}'. Example tile: {sample.name}\n"
            f"Tile dimensions: {sorted(dims)}\n\n"
            "This usually means PDAL dropped ExtraBytes/custom dims when writing.\n"
            "Fix: ensure writers.las has extra_dims='all' (already in this script).\n"
            "If it still fails, your PDAL build may not support forwarding ExtraBytes correctly "
            "or PredInstance is not encoded as a standard ExtraBytes VLR.\n"
            "Next step: check whether PDAL sees PredInstance in the ORIGINAL file:\n"
            "  pdal info --dimensions dataset/Ecosense/ecosense.laz\n"
        )
    print(f"[OK] Sanity check: '{PRED_DIM}' exists in tiles (sample {sample.name}).")


def read_tile_header_bounds(tile_path: Path) -> Tuple[float, float, float, float]:
    las = laspy.read(str(tile_path))
    xmin, ymin, _ = las.header.mins
    xmax, ymax, _ = las.header.maxs
    return float(xmin), float(ymin), float(xmax), float(ymax)


def build_nn_index_xy(x: np.ndarray, y: np.ndarray):
    if HAVE_SCIPY:
        pts = np.column_stack([x, y]).astype(np.float64, copy=False)
        return ("kdtree", cKDTree(pts))

    # Fallback: simple brute force (slow). Prefer installing scipy.
    return ("bruteforce", None)


def query_nn(index_obj, qx: float, qy: float, x: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    mode, tree = index_obj
    if mode == "kdtree":
        dist, idx = tree.query([qx, qy], k=1)
        return int(idx), float(dist)

    # brute force fallback
    dx = x - qx
    dy = y - qy
    d2 = dx * dx + dy * dy
    idx = int(np.argmin(d2))
    return idx, float(math.sqrt(float(d2[idx])))


def write_part(tile_las: laspy.LasData, mask: np.ndarray, out_path: Path) -> None:
    if not np.any(mask):
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sub = laspy.create(point_format=tile_las.header.point_format, file_version=tile_las.header.version)
    sub.header = tile_las.header
    sub.points = tile_las.points[mask]
    sub.write(str(out_path))


def merge_parts_to_tree(parts: List[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": [str(p) for p in parts]},
            {"type": "filters.merge"},
            {"type": "writers.las", "filename": str(out_path), "compression": "laszip", "extra_dims": "all",
             "minor_version": 4, "dataformat_id": 7},
        ]
    }
    run_pdal(pipeline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ecosense_laz", type=Path)
    ap.add_argument("inventory_csv", type=Path)
    ap.add_argument("out_dir", type=Path)

    ap.add_argument("--tile-size", type=float, default=50.0)
    ap.add_argument("--buffer", type=float, default=2.0, help="BBox buffer for selecting inv points per tile (m)")
    ap.add_argument("--search-radius", type=float, default=1.0, help="Flag matches beyond this NN distance (m)")
    ap.add_argument("--skip-tiling", action="store_true")
    ap.add_argument("--keep-parts", action="store_true")

    args = ap.parse_args()

    out_dir = args.out_dir
    tiles_dir = out_dir / "tiles"
    parts_dir = out_dir / "parts"
    trees_dir = out_dir / "trees"
    out_dir.mkdir(parents=True, exist_ok=True)

    # PASS 0: load inventory
    inv = pd.read_csv(args.inventory_csv)
    missing = [c for c in REQUIRED_INV_COLS if c not in inv.columns]
    if missing:
        raise SystemExit(f"Inventory missing required columns: {missing}\nFound: {list(inv.columns)}")

    if "tree_id" not in inv.columns:
        inv = inv.copy()
        inv["tree_id"] = np.arange(1, len(inv) + 1, dtype=int)

    inv["x_32632"] = pd.to_numeric(inv["x_32632"], errors="coerce")
    inv["y_32632"] = pd.to_numeric(inv["y_32632"], errors="coerce")
    inv["diameter_m"] = pd.to_numeric(inv["diameter_m"], errors="coerce")

    bad_xy = inv["x_32632"].isna() | inv["y_32632"].isna()
    if bad_xy.any():
        print(f"[WARN] {bad_xy.sum()} rows have missing XY → will remain unmatched.")
    inv_valid = inv[~bad_xy].copy()

    # PASS 1: tiling
    if not args.skip_tiling:
        tile_laz(args.ecosense_laz, tiles_dir, args.tile_size)
    else:
        print("[PASS1] Skipping tiling (using existing tiles).")

    tiles = list_tiles(tiles_dir)
    if not tiles:
        raise SystemExit(f"[FAIL] No tiles found in {tiles_dir}")

    # Verify tiles preserve PredInstance before we waste time
    sanity_check_tile_has_predinstance(tiles)

    # PASS 2: match inventory -> PredInstance
    best_match: Dict[int, MatchResult] = {
        int(i): MatchResult(
            inv_index=int(i),
            x=float(inv_valid.loc[i, "x_32632"]),
            y=float(inv_valid.loc[i, "y_32632"]),
            predinstance=None,
            nn_dist_m=None,
            tile_name=None,
            note="unmatched",
        )
        for i in inv_valid.index
    }

    print("[PASS2] Matching inventory XY to PredInstance (tile-by-tile)...")
    for tile_path in tqdm(tiles, desc="Matching tiles", unit="tile"):
        xmin, ymin, xmax, ymax = read_tile_header_bounds(tile_path)
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
            raise RuntimeError(f"Tile {tile_path.name} missing {PRED_DIM} despite sanity check.")

        x = las.x.astype(np.float64, copy=False)
        y = las.y.astype(np.float64, copy=False)
        pred = np.asarray(las[PRED_DIM]).astype(np.int64, copy=False)

        nn_index = build_nn_index_xy(x, y)

        for inv_i, row in sel.iterrows():
            qx = float(row["x_32632"])
            qy = float(row["y_32632"])
            idx_nn, dist = query_nn(nn_index, qx, qy, x, y)
            inst = int(pred[idx_nn])

            cur = best_match[int(inv_i)]
            if cur.nn_dist_m is None or dist < cur.nn_dist_m:
                note = "ok" if dist <= float(args.search_radius) else f"far_match>{args.search_radius}m"
                best_match[int(inv_i)] = MatchResult(
                    inv_index=int(inv_i),
                    x=qx,
                    y=qy,
                    predinstance=inst,
                    nn_dist_m=float(dist),
                    tile_name=tile_path.name,
                    note=note,
                )

    match_df = pd.DataFrame([m.__dict__ for m in best_match.values()])
    match_df.to_csv(out_dir / "match_report.csv", index=False)
    print(f"[PASS2] Wrote match_report.csv → {(out_dir / 'match_report.csv').resolve()}")

    # Join match to inventory
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

    matched = inv_out[inv_out["predinstance"].notna()].copy()
    if matched.empty:
        raise SystemExit("[FAIL] No rows matched. Check CRS, coordinates, or increase --buffer/--search-radius.")

    matched["predinstance"] = matched["predinstance"].astype(int)
    unique_inst = sorted(matched["predinstance"].unique().tolist())
    unique_set = set(unique_inst)

    inst_to_file = {inst: f"pred_{inst}.laz" for inst in unique_inst}
    parts_index: Dict[int, List[Path]] = {inst: [] for inst in unique_inst}

    # PASS 3: extract per-instance parts from tiles
    print("[PASS3] Extracting per-instance parts from tiles...")
    for tile_path in tqdm(tiles, desc="Extracting parts", unit="tile"):
        las = laspy.read(str(tile_path))
        pred = np.asarray(las[PRED_DIM]).astype(np.int64, copy=False)

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

    # PASS 4: merge parts per instance
    print("[PASS4] Merging parts into final per-instance LAZ files...")
    trees_dir.mkdir(parents=True, exist_ok=True)

    for inst in tqdm(unique_inst, desc="Merging instances", unit="tree"):
        parts = parts_index.get(inst, [])
        if not parts:
            continue
        merge_parts_to_tree(parts, trees_dir / inst_to_file[inst])

    # labels.csv (one row per inventory record)
    labels = inv_out[inv_out["predinstance"].notna()].copy()
    labels["predinstance"] = labels["predinstance"].astype(int)
    labels["tree_pointcloud_file"] = labels["predinstance"].map(
        lambda i: str(Path("trees") / inst_to_file[int(i)])
    )
    labels.to_csv(out_dir / "labels.csv", index=False)
    print(f"[DONE] Wrote labels.csv → {(out_dir / 'labels.csv').resolve()}")

    if not args.keep_parts:
        print("[NOTE] Parts cleanup is not auto-enabled in this version. "
              "If you want, I’ll add a safe cleanup step at the end.")

    print(f"[DONE] Final trees in → {trees_dir.resolve()}")


if __name__ == "__main__":
    main()
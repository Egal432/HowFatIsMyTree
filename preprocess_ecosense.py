#!/usr/bin/env python3
"""
EcoSense preprocessing (v2)

Features added compared with the original script:
* If the output folder (especially the *tiles* sub‑folder) already exists,
  tiling is skipped and the script proceeds directly to the matching step.
* Two log files are created:
    - tiles_success.log  → one line per tile that was processed without error
    - tiles_failed.log   → one line per tile that raised an exception
  These logs are consulted on subsequent runs so that already‑processed
  tiles are not re‑processed.
* The per‑tile loop now catches exceptions; a broken tile no longer aborts
  the whole pipeline.
* Compatibility fix for recent laspy versions (ScaledArrayView → np.asarray).

Usage (same as before):
    python preprocess_ecosense.py <ecosense.laz> <inventory.csv> <out_dir>
    [--tile-size 50] [--buffer 2] [--search-radius 1]
    [--skip-tiling] [--keep-parts]

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import laspy
import numpy as np
import pandas as pd

# -------------------------- optional deps --------------------------
try:
    import pdal
except ImportError as e:
    raise SystemExit(
        "Need PDAL python bindings: `pip install pdal` + PDAL installed."
    ) from e

try:
    from tqdm import tqdm
except ImportError as e:
    raise SystemExit("Install tqdm: `pip install tqdm`") from e

try:
    from scipy.spatial import cKDTree  # type: ignore
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
# ---------------------------------------------------------------

# ---------------------------- constants -------------------------
REQUIRED_INV_COLS = ["x_32632", "y_32632", "diameter_m"]
PRED_DIM = "PredInstance"
# ---------------------------------------------------------------

# --------------------------- dataclasses ------------------------


@dataclass
class MatchResult:
    inv_index: int
    x: float
    y: float
    predinstance: Optional[int]
    nn_dist_m: Optional[float]
    tile_name: Optional[str]
    note: str
# ---------------------------------------------------------------

# -------------------------- helper funcs ------------------------


def run_pdal(pipeline_obj: dict) -> int:
    """Execute a PDAL pipeline (JSON dict) and return the number of points processed."""
    pipe = pdal.Pipeline(json.dumps(pipeline_obj))
    return pipe.execute()


def tile_laz(in_laz: Path, tiles_dir: Path, tile_size: float) -> None:
    """Split a huge LAZ into square XY tiles, preserving all extra dimensions."""
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
    print("[PASS1] Tiling big LAZ (this can take a while)…")
    count = run_pdal(pipeline)
    print(f"[PASS1] Done. PDAL processed ~{count:,} points.")


def list_tiles(tiles_dir: Path) -> List[Path]:
    """Return a sorted list of all .laz/.las tiles."""
    return sorted(list(tiles_dir.glob("*.laz")) + list(tiles_dir.glob("*.las")))


def sanity_check_tile_has_predinstance(tiles: List[Path]) -> None:
    """Verify that a sample tile contains the PredInstance dimension."""
    if not tiles:
        raise SystemExit("[FAIL] No tiles found to sanity‑check.")
    sample = tiles[0]
    las = laspy.read(str(sample))
    dims = set(las.point_format.dimension_names)
    if PRED_DIM not in dims:
        raise SystemExit(
            f"[FAIL] Tiles do NOT contain '{PRED_DIM}'. Example tile: {sample.name}\n"
            f"Tile dimensions: {sorted(dims)}\n"
            "Fix: ensure writers.las uses extra_dims='all' (already in script).\n"
            "If it still fails, your PDAL build may not forward ExtraBytes correctly."
        )
    print(
        f"[OK] Sanity check passed – '{PRED_DIM}' present in sample tile {sample.name}.")


def read_tile_header_bounds(tile_path: Path) -> Tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) from a tile’s header."""
    las = laspy.read(str(tile_path))
    xmin, ymin, _ = las.header.mins
    xmax, ymax, _ = las.header.maxs
    return float(xmin), float(ymin), float(xmax), float(ymax)


def build_nn_index_xy(x: np.ndarray, y: np.ndarray):
    """Build a nearest‑neighbour index (KD‑tree if SciPy present)."""
    if HAVE_SCIPY:
        pts = np.column_stack([x, y]).astype(np.float64, copy=False)
        return ("kdtree", cKDTree(pts))
    # fallback – brute force (slow)
    return ("bruteforce", None)


def query_nn(index_obj, qx: float, qy: float, x: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return (index_of_nn, distance) for query point (qx, qy)."""
    mode, tree = index_obj
    if mode == "kdtree":
        dist, idx = tree.query([qx, qy], k=1)
        return int(idx), float(dist)
    # brute‑force fallback
    dx = x - qx
    dy = y - qy
    d2 = dx * dx + dy * dy
    idx = int(np.argmin(d2))
    return idx, float(math.sqrt(float(d2[idx])))


def write_part(tile_las: laspy.LasData, mask: np.ndarray, out_path: Path) -> None:
    """Write a masked subset of a tile to a temporary part file."""
    if not np.any(mask):
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sub = laspy.create(
        point_format=tile_las.header.point_format,
        file_version=tile_las.header.version,
    )
    sub.header = tile_las.header
    sub.points = tile_las.points[mask]
    sub.write(str(out_path))


def merge_parts_to_tree(parts: List[Path], out_path: Path) -> None:
    """Merge per‑tile parts of a single tree into one LAZ."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": [str(p) for p in parts]},
            {"type": "filters.merge"},
            {
                "type": "writers.las",
                "filename": str(out_path),
                "compression": "laszip",
                "extra_dims": "all",
                "minor_version": 4,
                "dataformat_id": 7,
            },
        ]
    }
    run_pdal(pipeline)
# ---------------------------------------------------------------

# -------------------------- logging helpers --------------------


def load_processed_log(log_path: Path) -> set[str]:
    """Read a log file (one tile name per line) and return a set."""
    if not log_path.is_file():
        return set()
    return {line.strip() for line in log_path.read_text().splitlines() if line.strip()}


def append_to_log(log_path: Path, line: str) -> None:
    """Append a single line to a log file (creates the file if needed)."""
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
# ---------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ecosense_laz", type=Path,
                    help="Path to the original ecosense LAZ")
    ap.add_argument("inventory_csv", type=Path,
                    help="CSV with x_32632, y_32632, diameter_m")
    ap.add_argument("out_dir", type=Path,
                    help="Root directory for all outputs")
    ap.add_argument("--tile-size", type=float, default=50.0,
                    help="Tile side length (metres)")
    ap.add_argument("--buffer", type=float, default=2.0,
                    help="BBox buffer when selecting inventory points per tile (m)")
    ap.add_argument("--search-radius", type=float, default=1.0,
                    help="Flag matches beyond this distance (m)")
    ap.add_argument("--skip-tiling", action="store_true",
                    help="Force skipping of tiling even if tiles folder is missing")
    ap.add_argument("--keep-parts", action="store_true",
                    help="Do NOT delete temporary per‑tile part files")
    args = ap.parse_args()

    # -----------------------------------------------------------------
    # Prepare directories
    out_dir = args.out_dir
    tiles_dir = out_dir / "tiles"
    parts_dir = out_dir / "parts"
    trees_dir = out_dir / "trees"

    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Load inventory (PASS 0)
    inv = pd.read_csv(args.inventory_csv)
    missing = [c for c in REQUIRED_INV_COLS if c not in inv.columns]
    if missing:
        raise SystemExit(
            f"Inventory missing required columns: {missing}\nFound: {list(inv.columns)}"
        )
    if "tree_id" not in inv.columns:
        inv = inv.copy()
        inv["tree_id"] = np.arange(1, len(inv) + 1, dtype=int)

    # Cast to numeric, coerce bad values to NaN
    inv["x_32632"] = pd.to_numeric(inv["x_32632"], errors="coerce")
    inv["y_32632"] = pd.to_numeric(inv["y_32632"], errors="coerce")
    inv["diameter_m"] = pd.to_numeric(inv["diameter_m"], errors="coerce")

    bad_xy = inv["x_32632"].isna() | inv["y_32632"].isna()
    if bad_xy.any():
        print(
            f"[WARN] {bad_xy.sum()} rows have missing XY → will remain unmatched.")
    inv_valid = inv[~bad_xy].copy()

    # -----------------------------------------------------------------
    # ----- PASS 1: tiling (skip if the tiles folder already exists) -----
    if args.skip_tiling:
        print("[PASS1] Skipping tiling because '--skip-tiling' was supplied.")
    elif tiles_dir.is_dir():
        print(
            f"[PASS1] Tiles folder already exists at {tiles_dir}. "
            "Skipping tiling and re‑using existing tiles."
        )
    else:
        tile_laz(args.ecosense_laz, tiles_dir, args.tile_size)

    tiles = list_tiles(tiles_dir)
    if not tiles:
        raise SystemExit(f"[FAIL] No tiles found in {tiles_dir}")

    # Verify that the required extra dimension survived tiling
    sanity_check_tile_has_predinstance(tiles)

    # -----------------------------------------------------------------
    # Logging files for processed/failed tiles
    success_log = out_dir / "tiles_success.log"
    fail_log = out_dir / "tiles_failed.log"
    processed_success = load_processed_log(success_log)
    processed_fail = load_processed_log(fail_log)

    # -----------------------------------------------------------------
    # ----- PASS 2: match inventory → PredInstance (tile‑by‑tile) -----
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

    print("[PASS2] Matching inventory XY to PredInstance (tile‑by‑tile)…")
    for tile_path in tqdm(tiles, desc="Matching tiles", unit="tile"):
        tile_name = tile_path.name

        # -----------------------------------------------------------------
        # Skip tiles that have already been logged as successful or failed
        if tile_name in processed_success:
            # already done – nothing to do
            continue
        if tile_name in processed_fail:
            # previously failed – we try again (optional: you could skip permanently)
            pass

        try:
            xmin, ymin, xmax, ymax = read_tile_header_bounds(tile_path)
            buf = float(args.buffer)

            # Select inventory points that fall inside the (buffered) tile extent
            sel = inv_valid[
                (inv_valid["x_32632"] >= xmin - buf)
                & (inv_valid["x_32632"] <= xmax + buf)
                & (inv_valid["y_32632"] >= ymin - buf)
                & (inv_valid["y_32632"] <= ymax + buf)
            ]
            if sel.empty:
                # nothing to do for this tile, but it counts as “processed successfully”
                append_to_log(success_log, tile_name)
                continue

            las = laspy.read(str(tile_path))

            # -------------------------------------------------------------
            # Extract XY as ordinary float64 Numpy arrays (las.x/las.y are
            # ScaledArrayView objects in recent laspy versions)
            x = np.asarray(las.x, dtype=np.float64)
            y = np.asarray(las.y, dtype=np.float64)

            # PredInstance values
            pred = np.asarray(las[PRED_DIM]).astype(np.int64, copy=False)

            # Build KD‑tree (or brute‑force fallback)
            nn_index = build_nn_index_xy(x, y)

            # -------------------------------------------------------------
            # For each inventory point in the buffered tile, find the nearest
            # point and record its instance ID (if this is the best match seen
            # so far for that inventory record)
            for inv_i, row in sel.iterrows():
                qx = float(row["x_32632"])
                qy = float(row["y_32632"])
                idx_nn, dist = query_nn(nn_index, qx, qy, x, y)
                inst = int(pred[idx_nn])

                cur = best_match[int(inv_i)]
                if cur.nn_dist_m is None or dist < cur.nn_dist_m:
                    note = (
                        "ok"
                        if dist <= float(args.search_radius)
                        else f"far_match>{args.search_radius}m"
                    )
                    best_match[int(inv_i)] = MatchResult(
                        inv_index=int(inv_i),
                        x=qx,
                        y=qy,
                        predinstance=inst,
                        nn_dist_m=float(dist),
                        tile_name=tile_name,
                        note=note,
                    )
            # -----------------------------------------------------------------
            # Tile processed without exception → log success
            append_to_log(success_log, tile_name)

        except Exception as exc:  # pylint: disable=broad-except
            # Log the failure but keep going
            err_msg = f"{tile_name}: {type(exc).__name__}: {exc}"
            print(f"[ERROR] {err_msg}", file=sys.stderr)
            append_to_log(fail_log, err_msg)
            # continue to next tile
            continue

    # -----------------------------------------------------------------
    # Write the match report (one line per inventory record)
    match_df = pd.DataFrame([m.__dict__ for m in best_match.values()])
    match_report_path = out_dir / "match_report.csv"
    match_df.to_csv(match_report_path, index=False)
    print(f"[PASS2] Match report written → {match_report_path.resolve()}")

    # -----------------------------------------------------------------
    # Merge the match information back into the inventory table
    inv_out = inv.copy()
    inv_out["predinstance"] = np.nan
    inv_out["match_dist_m"] = np.nan
    inv_out["match_tile"] = ""
    inv_out["match_note"] = "unmatched"

    for m in best_match.values():
        inv_out.loc[m.inv_index, "predinstance"] = (
            m.predinstance if m.predinstance is not None else np.nan
        )
        inv_out.loc[m.inv_index, "match_dist_m"] = (
            m.nn_dist_m if m.nn_dist_m is not None else np.nan
        )
        inv_out.loc[m.inv_index, "match_tile"] = m.tile_name or ""
        inv_out.loc[m.inv_index, "match_note"] = m.note

    # -----------------------------------------------------------------
    # Filter only rows that got a successful match
    matched = inv_out[inv_out["predinstance"].notna()].copy()
    if matched.empty:
        raise SystemExit(
            "[FAIL] No inventory rows matched any PredInstance. "
            "Check CRS, buffer size, or increase --search-radius."
        )
    matched["predinstance"] = matched["predinstance"].astype(int)

    unique_inst = sorted(matched["predinstance"].unique().tolist())
    unique_set = set(unique_inst)

    # Mapping from instance → output filename
    inst_to_file = {inst: f"pred_{inst}.laz" for inst in unique_inst}
    parts_index: Dict[int, List[Path]] = {inst: [] for inst in unique_inst}

    # -----------------------------------------------------------------
    # ----- PASS 3: extract per‑instance parts from tiles -----
    print("[PASS3] Extracting per‑instance parts from tiles…")
    for tile_path in tqdm(tiles, desc="Extracting parts", unit="tile"):
        tile_name = tile_path.name

        # (optional) skip tiles we already know failed – they probably won’t
        # contain useful data, but you may comment this out if you want to try again
        if tile_name in processed_fail:
            continue

        try:
            las = laspy.read(str(tile_path))
            pred = np.asarray(las[PRED_DIM]).astype(np.int64, copy=False)

            tile_insts = np.unique(pred)
            needed_here = [int(i) for i in tile_insts if int(i) in unique_set]
            if not needed_here:
                continue

            for inst in needed_here:
                mask = pred == inst
                if not np.any(mask):
                    continue
                part_path = parts_dir / \
                    f"pred_{inst}" / f"{tile_path.stem}__part.laz"
                write_part(las, mask, part_path)
                parts_index[inst].append(part_path)

        except Exception as exc:  # pylint: disable=broad-except
            err_msg = f"{tile_name} (PART EXTRACT): {type(exc).__name__}: {exc}"
            print(f"[ERROR] {err_msg}", file=sys.stderr)
            append_to_log(fail_log, err_msg)
            # do not abort – continue with the next tile

    # -----------------------------------------------------------------
    # ----- PASS 4: merge parts → final per‑tree LAZ files -----
    print("[PASS4] Merging per‑tree parts into final LAZ files…")
    trees_dir.mkdir(parents=True, exist_ok=True)

    for inst in tqdm(unique_inst, desc="Merging trees", unit="tree"):
        parts = parts_index.get(inst, [])
        if not parts:
            # No points for this instance – this can happen if the instance was
            # only present in a tile that failed earlier.
            continue
        merge_parts_to_tree(parts, trees_dir / inst_to_file[inst])

    # -----------------------------------------------------------------
    # Write the final labels.csv (one row per matched inventory record)
    labels = inv_out[inv_out["predinstance"].notna()].copy()
    labels["predinstance"] = labels["predinstance"].astype(int)
    labels["tree_pointcloud_file"] = labels["predinstance"].map(
        lambda i: str(Path("trees") / inst_to_file[int(i)])
    )
    labels_path = out_dir / "labels.csv"
    labels.to_csv(labels_path, index=False)
    print(f"[DONE] labels.csv written → {labels_path.resolve()}")

    # -----------------------------------------------------------------
    # Optional cleanup of temporary parts
    if not args.keep_parts:
        print("[NOTE] Parts cleanup is not automatic in this version. "
              "If you want to delete the temporary per‑tile part files, "
              "you can safely remove the directory:\n"
              f"  rm -r {parts_dir}")
    else:
        print("[NOTE] Keeping per‑tile part files because '--keep-parts' was supplied.")

    print(f"[DONE] Final per‑tree LAZ files are in → {trees_dir.resolve()}")
    print(f"[INFO] Processed tiles logged in → {success_log}")
    print(f"[INFO] Failed tiles logged in → {fail_log}")


if __name__ == "__main__":
    main()

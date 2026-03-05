#!/usr/bin/env python3
"""
preprocess_ecosense_copilot.py

Improved variant of preprocess_ecosense.py:
- header-only reads use laspy.open to avoid loading points when not needed
- grid fallback uses lists (no repeated np.append)
- handles sentinel/non-positive PredInstance values (treated as "no instance")
- filters out non-positive instances when exporting
- small robustness and clarity improvements
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
    raise SystemExit(
        "PDAL python bindings not found. Install with `pip install pdal` and ensure PDAL is installed."
    ) from e

try:
    from scipy.spatial import cKDTree  # type: ignore
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


REQUIRED_INV_COLS = [
    "x_32632",
    "y_32632",
    "diameter_m",
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


def run_pdal(pipeline_obj: dict) -> int:
    pipe = pdal.Pipeline(json.dumps(pipeline_obj))
    return pipe.execute()


def tile_laz(in_laz: Path, tiles_dir: Path, tile_size: float) -> None:
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": [str(p) for p in parts]},
            {"type": "filters.merge"},
            {"type": "writers.las", "filename": str(out_path), "compression": "laszip"},
        ]
    }
    run_pdal(pipeline)


def read_tile_header_bounds(tile_path: Path) -> Tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) from tile header (header-only read)."""
    with laspy.open(str(tile_path)) as fh:
        hdr = fh.header
        xmin, ymin, _ = hdr.mins
        xmax, ymax, _ = hdr.maxs
    return float(xmin), float(ymin), float(xmax), float(ymax)


def build_nn_index_xy(x: np.ndarray, y: np.ndarray):
    """
    Build nearest-neighbor index.
    - If scipy available: return ("kdtree", cKDTree)
    - Else: return ("grid", (grid_dict, xmin, ymin, cell))
    The grid stores numpy arrays of indices per cell for efficient access.
    """
    if HAVE_SCIPY:
        pts = np.column_stack([x, y]).astype(np.float64, copy=False)
        return ("kdtree", cKDTree(pts))

    cell = 0.25  # meters
    xmin = float(x.min())
    ymin = float(y.min())
    ix = np.floor((x - xmin) / cell).astype(np.int32)
    iy = np.floor((y - ymin) / cell).astype(np.int32)

    grid_lists: Dict[Tuple[int, int], List[int]] = {}
    cap = 500
    for i in range(x.size):
        key = (int(ix[i]), int(iy[i]))
        lst = grid_lists.get(key)
        if lst is None:
            grid_lists[key] = [i]
        else:
            if len(lst) < cap:
                lst.append(i)

    # convert lists to numpy arrays for faster indexing/distance ops later
    grid: Dict[Tuple[int, int], np.ndarray] = {
        k: np.array(v, dtype=np.int32) for k, v in grid_lists.items()
    }

    return ("grid", (grid, xmin, ymin, cell))


def query_nn(index_obj, qx: float, qy: float, x: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    mode, data = index_obj
    if mode == "kdtree":
        tree = data
        dist, idx = tree.query([qx, qy], k=1)
        return int(idx), float(dist)

    grid, xmin, ymin, cell = data
    cx = int(math.floor((qx - xmin) / cell))
    cy = int(math.floor((qy - ymin) / cell))

    best_i = -1
    best_d2 = float("inf")

    # Expand search radius in grid cells (up to ~5m)
    for r in range(0, 20):
        found_any = False
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                key = (cx + dx, cy + dy)
                if key not in grid:
                    continue
                idxs = grid[key]
                found_any = True
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
        # Fallback brute force (rare)
        dxv = x - qx
        dyv = y - qy
        d2 = dxv * dxv + dyv * dyv
        best_i = int(np.argmin(d2))
        best_d2 = float(d2[best_i])

    return best_i, float(math.sqrt(best_d2))


def write_part(tile_las: laspy.LasData, mask: np.ndarray, out_path: Path) -> None:
    if not np.any(mask):
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Create a LasData with same header then assign sliced points
    sub = laspy.create(point_format=tile_las.header.point_format, file_version=tile_las.header.version)
    sub.header = tile_las.header
    sub.points = tile_las.points[mask]
    sub.write(str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ecosense_laz", type=Path)
    ap.add_argument("inventory_csv", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--tile-size", type=float, default=50.0)
    ap.add_argument("--buffer", type=float, default=2.0)
    ap.add_argument("--search-radius", type=float, default=1.0)
    ap.add_argument("--skip-tiling", action="store_true")
    ap.add_argument("--keep-parts", action="store_true")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    tiles_dir = out_dir / "tiles"
    parts_dir = out_dir / "parts"
    trees_dir = out_dir / "trees"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    inv = pd.read_csv(args.inventory_csv)
    missing = [c for c in REQUIRED_INV_COLS if c not in inv.columns]
    if missing:
        raise SystemExit(f"Inventory missing required columns: {missing}\nFound columns: {list(inv.columns)}")

    if "tree_id" not in inv.columns:
        inv = inv.copy()
        inv["tree_id"] = np.arange(1, len(inv) + 1, dtype=int)

    inv["x_32632"] = pd.to_numeric(inv["x_32632"], errors="coerce")
    inv["y_32632"] = pd.to_numeric(inv["y_32632"], errors="coerce")
    inv["diameter_m"] = pd.to_numeric(inv["diameter_m"], errors="coerce")

    bad_xy = inv["x_32632"].isna() | inv["y_32632"].isna()
    if bad_xy.any():
        print(f"[WARN] {bad_xy.sum()} inventory rows have missing XY. They will be unmatched.")
    inv_valid = inv[~bad_xy].copy()

    if not args.skip_tiling:
        tile_laz(args.ecosense_laz, tiles_dir, tile_size=args.tile_size)
    else:
        print("[PASS1] Skipping tiling (using existing tiles).")

    tiles = sorted(list(tiles_dir.glob("*.laz")) + list(tiles_dir.glob("*.las")))
    if not tiles:
        raise SystemExit(f"No tiles found in {tiles_dir}. Did tiling fail?")

    # Best match per inventory row (use original inv_valid indices)
    best_match: Dict[int, MatchResult] = {
        int(i): MatchResult(inv_index=int(i),
                            x=float(inv_valid.loc[i, "x_32632"]),
                            y=float(inv_valid.loc[i, "y_32632"]),
                            predinstance=None, nn_dist_m=None, tile_name=None, note="unmatched")
        for i in inv_valid.index
    }

    print("[PASS2] Matching inventory XY to PredInstance (tile-by-tile)...")
    for ti, tile_path in enumerate(tiles, start=1):
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

        # Read full tile only when needed
        las = laspy.read(str(tile_path))
        dims = set(las.point_format.dimension_names)
        if PRED_DIM not in dims:
            raise RuntimeError(f"Tile {tile_path.name} does not contain {PRED_DIM}.")

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
                # note far/out-of-range vs ok
                note = "ok" if dist <= float(args.search_radius) else f"far_match>{args.search_radius}m"
                if inst <= 0:
                    # treat non-positive instance ids as "no instance"
                    best_match[int(inv_i)] = MatchResult(
                        inv_index=int(inv_i),
                        x=qx,
                        y=qy,
                        predinstance=None,
                        nn_dist_m=float(dist),
                        tile_name=tile_path.name,
                        note="no_instance"
                    )
                else:
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
            matched_count = sum(1 for m in best_match.values() if m.predinstance is not None)
            print(f"[PASS2] processed {ti}/{len(tiles)} tiles... matched so far: {matched_count}")

    match_df = pd.DataFrame([m.__dict__ for m in best_match.values()])
    match_df.to_csv(out_dir / "match_report.csv", index=False)
    print(f"[PASS2] Match report written: {(out_dir / 'match_report.csv').resolve()}")

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

    # keep only rows with assigned positive instance ids
    matched = inv_out[inv_out["predinstance"].notna()].copy()
    if matched.empty:
        raise SystemExit("[FAIL] No inventory rows were matched to PredInstance. Check CRS / coordinates / search-radius.")

    matched["predinstance"] = matched["predinstance"].astype(int)
    # exclude non-positive sentinel instance ids (defensive)
    unique_inst = sorted([int(i) for i in matched["predinstance"].unique().tolist() if int(i) > 0])
    inst_to_file = {inst: f"pred_{inst}.laz" for inst in unique_inst}

    print(f"[PASS3] Unique matched instances: {len(unique_inst)}")
    print("[PASS3] Writing per-instance parts from tiles...")

    parts_index: Dict[int, List[Path]] = {inst: [] for inst in unique_inst}
    unique_set = set(unique_inst)

    for ti, tile_path in enumerate(tiles, start=1):
        las = laspy.read(str(tile_path))
        dims = set(las.point_format.dimension_names)
        if PRED_DIM not in dims:
            raise RuntimeError(f"Tile {tile_path.name} does not contain {PRED_DIM}.")

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

        if ti % 50 == 0:
            print(f"[PASS3] processed {ti}/{len(tiles)} tiles...")

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

    labels = inv_out.copy()
    labels = labels[labels["predinstance"].notna()].copy()
    labels["predinstance"] = labels["predinstance"].astype(int)
    labels["tree_pointcloud_file"] = labels["predinstance"].map(lambda i: str(Path("trees") / inst_to_file[int(i)]))

    labels_path = out_dir / "labels.csv"
    labels.to_csv(labels_path, index=False)
    print(f"[DONE] labels.csv written: {labels_path.resolve()}")

    if not args.keep_parts:
        print("[NOTE] Temporary parts kept (cleanup disabled by default). "
              "If you want auto cleanup, enable it in the script or run with --keep-parts off + uncomment cleanup.")

    print(f"[DONE] Final tree clouds in: {trees_dir.resolve()}")
    print(f"[DONE] Match QA report in: {(out_dir / 'match_report.csv').resolve()}")


if __name__ == "__main__":
    main()
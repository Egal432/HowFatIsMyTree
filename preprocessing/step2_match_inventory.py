#!/usr/bin/env python3
"""
Step 2: Matching
Matches inventory coordinates to PredInstance IDs in the tiles.
Output: out_dir/match_report.csv
"""

from __future__ import annotations
import argparse
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import laspy
from tqdm import tqdm

# Optional accelerator
try:
    from scipy.spatial import cKDTree  # type: ignore
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


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


def read_tile_header_bounds(tile_path: Path) -> Tuple[float, float, float, float]:
    las = laspy.read(str(tile_path))
    xmin, ymin, _ = las.header.mins
    xmax, ymax, _ = las.header.maxs
    return float(xmin), float(ymin), float(xmax), float(ymax)


def build_nn_index_xy(x: np.ndarray, y: np.ndarray):
    if HAVE_SCIPY:
        pts = np.column_stack([x, y]).astype(np.float64, copy=False)
        return ("kdtree", cKDTree(pts))
    return ("bruteforce", None)


def query_nn(index_obj, qx: float, qy: float, x: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    mode, tree = index_obj
    if mode == "kdtree":
        dist, idx = tree.query([qx, qy], k=1)
        return int(idx), float(dist)
    # brute force
    dx = x - qx
    dy = y - qy
    d2 = dx * dx + dy * dy
    idx = int(np.argmin(d2))
    return idx, float(math.sqrt(float(d2[idx])))


def load_processed_log(log_path: Path) -> set[str]:
    if not log_path.is_file():
        return set()
    return {line.strip() for line in log_path.read_text().splitlines() if line.strip()}


def append_to_log(log_path: Path, line: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser(description="Step 2: Match inventory to predictions.")
    ap.add_argument("inventory_csv", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--buffer", type=float, default=2.0, help="Buffer for selecting points (m)")
    ap.add_argument("--search-radius", type=float, default=1.0, help="Max match distance (m)")
    args = ap.parse_args()

    out_dir = args.out_dir
    tiles_dir = out_dir / "tiles"
    match_report_path = out_dir / "match_report.csv"
    
    # Logs for robustness
    success_log = out_dir / "step2_success.log"
    fail_log = out_dir / "step2_failed.log"
    processed_success = load_processed_log(success_log)
    processed_fail = load_processed_log(fail_log)

    # Load inventory
    inv = pd.read_csv(args.inventory_csv)
    inv["x_32632"] = pd.to_numeric(inv["x_32632"], errors="coerce")
    inv["y_32632"] = pd.to_numeric(inv["y_32632"], errors="coerce")
    bad_xy = inv["x_32632"].isna() | inv["y_32632"].isna()
    if bad_xy.any():
        print(f"[WARN] {bad_xy.sum()} rows have missing XY.")
    inv_valid = inv[~bad_xy].copy()

    # Setup match dictionary
    best_match: Dict[int, MatchResult] = {
        int(i): MatchResult(
            inv_index=int(i),
            x=float(inv_valid.loc[i, "x_32632"]),
            y=float(inv_valid.loc[i, "y_32632"]),
            predinstance=None, nn_dist_m=None, tile_name=None, note="unmatched"
        )
        for i in inv_valid.index
    }

    # Process Tiles
    tiles = sorted(list(tiles_dir.glob("*.laz")) + list(tiles_dir.glob("*.las")))
    print("[STEP 2] Matching inventory XY to PredInstance...")

    for tile_path in tqdm(tiles, desc="Matching tiles", unit="tile"):
        tile_name = tile_path.name

        # Skip logic
        if tile_name in processed_success:
            continue
        if tile_name in processed_fail:
            # Optionally retry by removing from fail log, or just skip
            pass 

        try:
            xmin, ymin, xmax, ymax = read_tile_header_bounds(tile_path)
            buf = float(args.buffer)
            
            sel = inv_valid[
                (inv_valid["x_32632"] >= xmin - buf) & 
                (inv_valid["x_32632"] <= xmax + buf) &
                (inv_valid["y_32632"] >= ymin - buf) &
                (inv_valid["y_32632"] <= ymax + buf)
            ]
            if sel.empty:
                append_to_log(success_log, tile_name)
                continue

            las = laspy.read(str(tile_path))
            x = np.asarray(las.x, dtype=np.float64)
            y = np.asarray(las.y, dtype=np.float64)
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
                        inv_index=int(inv_i), x=qx, y=qy, predinstance=inst,
                        nn_dist_m=float(dist), tile_name=tile_name, note=note
                    )
            
            append_to_log(success_log, tile_name)

        except Exception as exc:
            err_msg = f"{tile_name}: {type(exc).__name__}: {exc}"
            print(f"[ERROR] {err_msg}", file=sys.stderr)
            append_to_log(fail_log, err_msg)
            continue

    # Write Report
    match_df = pd.DataFrame([m.__dict__ for m in best_match.values()])
    match_df.to_csv(match_report_path, index=False)
    print(f"[STEP 2] Done. Match report saved to {match_report_path}")


if __name__ == "__main__":
    main()
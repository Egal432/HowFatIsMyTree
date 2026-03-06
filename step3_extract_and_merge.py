#!/usr/bin/env python3
"""
Step 3: Extraction & Merging
Extracts per-instance points from tiles based on match_report.csv,
merges parts into final single-tree LAZ files, and creates labels.csv.
"""

from __future__ import annotations
import argparse
import json
import sys
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import laspy
from tqdm import tqdm

try:
    import pdal
except ImportError as e:
    raise SystemExit("Need PDAL python bindings: `pip install pdal`") from e


def run_pdal(pipeline_obj: dict) -> int:
    pipe = pdal.Pipeline(json.dumps(pipeline_obj))
    return pipe.execute()


def write_part(tile_las: laspy.LasData, mask: np.ndarray, out_path: Path) -> None:
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
    """
    Fixed Merge Logic: PDAL readers.las requires one reader stage per file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stages = []
    for p in parts:
        stages.append({"type": "readers.las", "filename": str(p)})
    
    stages.append({
        "type": "writers.las",
        "filename": str(out_path),
        "compression": "laszip",
        "extra_dims": "all",
        "minor_version": 4,
        "dataformat_id": 7,
    })
    
    run_pdal({"pipeline": stages})


def load_processed_log(log_path: Path) -> set[str]:
    if not log_path.is_file():
        return set()
    return {line.strip() for line in log_path.read_text().splitlines() if line.strip()}


def append_to_log(log_path: Path, line: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser(description="Step 3: Extract and Merge per-instance trees.")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--inventory-csv", type=Path, help="Original inventory (optional, for labels generation)")
    ap.add_argument("--keep-parts", action="store_true", help="Do not delete temporary part files")
    args = ap.parse_args()

    out_dir = args.out_dir
    tiles_dir = out_dir / "tiles"
    parts_dir = out_dir / "parts"
    trees_dir = out_dir / "trees"
    match_report_path = out_dir / "match_report.csv"
    
    # Logs
    success_log = out_dir / "step3_success.log"
    fail_log = out_dir / "step3_failed.log"
    processed_success = load_processed_log(success_log)
    processed_fail = load_processed_log(fail_log)

    # Load matches to find unique instances we care about
    if not match_report_path.exists():
        raise SystemExit(f"[ERROR] {match_report_path} not found. Run Step 2 first.")
    
    match_df = pd.read_csv(match_report_path)
    # Filter to successful matches
    matched = match_df[match_df["predinstance"].notna()].copy()
    if matched.empty:
        raise SystemExit("[ERROR] No successful matches found in match_report.csv.")
    
    matched["predinstance"] = matched["predinstance"].astype(int)
    unique_inst = sorted(matched["predinstance"].unique().tolist())
    unique_set = set(unique_inst)

    inst_to_file = {inst: f"pred_{inst}.laz" for inst in unique_inst}
    parts_index: Dict[int, List[Path]] = {inst: [] for inst in unique_inst}

    # PASS 3: Extract parts
    print("[STEP 3] Extracting per-instance parts from tiles...")
    tiles = sorted(list(tiles_dir.glob("*.laz")) + list(tiles_dir.glob("*.las")))
    
    for tile_path in tqdm(tiles, desc="Extracting parts", unit="tile"):
        tile_name = tile_path.name
        
        # Optional: skip tiles that failed in step2 or step3 previously
        if tile_name in processed_fail:
            continue

        try:
            las = laspy.read(str(tile_path))
            pred = np.asarray(las["PredInstance"]).astype(np.int64, copy=False)

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
            
            append_to_log(success_log, tile_name)

        except Exception as exc:
            err_msg = f"{tile_name} (EXTRACT): {type(exc).__name__}: {exc}"
            print(f"[ERROR] {err_msg}", file=sys.stderr)
            append_to_log(fail_log, err_msg)
            continue

    # PASS 4: Merge parts
    print("[STEP 3] Merging parts into final LAZ files...")
    trees_dir.mkdir(parents=True, exist_ok=True)
    
    for inst in tqdm(unique_inst, desc="Merging trees", unit="tree"):
        parts = parts_index.get(inst, [])
        if not parts:
            continue
        merge_parts_to_tree(parts, trees_dir / inst_to_file[inst])

    # Generate labels.csv
    labels_path = out_dir / "labels.csv"
    print("[STEP 3] Generating labels.csv...")
    
    # We reconstruct labels based on matched dataframe
    labels = matched.copy()
    labels["tree_pointcloud_file"] = labels["predinstance"].map(
        lambda i: str(Path("trees") / inst_to_file[int(i)])
    )
    labels.to_csv(labels_path, index=False)

    if not args.keep_parts:
        print(f"[NOTE] To save space, you can delete the parts directory: {parts_dir}")

    print(f"[STEP 3] Complete. Final trees in: {trees_dir.resolve()}")
    print(f"[STEP 3] Labels written to: {labels_path.resolve()}")


if __name__ == "__main__":
    main()
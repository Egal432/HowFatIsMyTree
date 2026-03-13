#!/usr/bin/env python3
"""
Step 4: Final Cleanup and Re-segmentation

1. Identifies PredInstances that match only 1 tree -> Renames to {tree_id}_ec.laz -> moves to 'clean_trees'.
2. Identifies PredInstances that match >1 trees -> Creates a manifest CSV -> Calls R script to split.
3. Orphan clusters found by R are moved to the tree folder nearest to them.
"""

import argparse
import subprocess
import shutil
import sys
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Step 4: Finalize tree files and trigger R re-segmentation.")
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()

    base_dir = args.out_dir
    source_dir = base_dir / "trees"
    match_report = base_dir / "match_report.csv"

    # Output directories
    clean_trees_dir = base_dir / "final_single_trees"
    resegmented_dir = base_dir / "resegmented_trees"

    clean_trees_dir.mkdir(parents=True, exist_ok=True)
    resegmented_dir.mkdir(parents=True, exist_ok=True)

    if not match_report.exists():
        raise SystemExit(
            f"[ERROR] {match_report} not found. Run Step 2 first.")

    # Load the match report
    df = pd.read_csv(match_report)

    # Filter only successful matches and ensure tree_id is int
    df = df[df["predinstance"].notna()].copy()
    df["predinstance"] = df["predinstance"].astype(int)

    # If tree_id doesn't exist (from earlier steps), use the inventory index
    if "tree_id" not in df.columns:
        df["tree_id"] = df["inv_index"]

    # Group by PredInstance
    grouped = df.groupby("predinstance")

    print(f"[INFO] Processing {len(grouped)} PredInstances...")

    for pred_id, group in grouped:
        laz_path = source_dir / f"pred_{pred_id}.laz"
        if not laz_path.exists():
            print(f"[WARN] Missing file for PredInstance {pred_id}, skipping.")
            continue

        count = len(group)

        if count == 1:
            # CASE 1: SINGLE TREE MATCH
            tree_id = int(group.iloc[0]["tree_id"])
            target_name = f"{tree_id}_ec.laz"
            target_path = clean_trees_dir / target_name

            if not target_path.exists():
                shutil.copy(laz_path, target_path)
            print(f"  [OK] Single tree copied -> {target_name}")

        else:
            # CASE 2: MULTI-TREE MATCH (Needs R processing)
            print(
                f"  [INFO] Multi-tree match ({count} trees) for PredInstance {pred_id} -> Calling R...")

            # Create a temporary manifest for this specific file containing ONLY these trees
            manifest_path = base_dir / f"temp_manifest_{pred_id}.csv"
            group_subset = group[["tree_id", "x_32632", "y_32632"]].copy()
            group_subset.to_csv(manifest_path, index=False)

            # Call the R script
            # Arguments: <Input_LAZ> <Manifest_CSV> <Output_Dir>
            try:
                subprocess.run(
                    ["Rscript", "split_multis.R", str(laz_path), str(
                        manifest_path), str(resegmented_dir)],
                    check=True
                )
                # Cleanup temp manifest
                manifest_path.unlink()
            except subprocess.CalledProcessError as e:
                print(f"  [ERROR] R script failed for {pred_id}: {e}")
                print(
                    f"  [NOTE] Temp manifest kept at {manifest_path} for debugging.")

    print(f"\n[DONE] Process complete.")
    print(f"  Clean trees (no splitting needed): {clean_trees_dir}")
    print(f"  Re-segmented trees (from R):      {resegmented_dir}")


if __name__ == "__main__":
    main()

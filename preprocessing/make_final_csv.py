#!/usr/bin/env python3
"""
make_final_csv.py  —  Generate final_trees.csv from clean tree folder + inventory

Scans the clean_trees folder for .laz files, looks up each full_id in the
inventory CSV, and writes a clean CSV with: full_id, x, y, diameter_m

Usage:
    python make_final_csv.py --clean  PATH/TO/clean_trees/
                             --inv    PATH/TO/inventory.csv
                             --out    PATH/TO/final_trees.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

INV_CSV_DEFAULT = r"E:\01_UAV_Frey_Group_3\HowFatIsMyTree\datasets\Ecosense\inventory.csv"
CLEAN_DIR_DEFAULT = r"E:\01_UAV_Frey_Group_3\HowFatIsMyTree\out\ecosense\clean_trees"
OUT_DEFAULT = r"E:\01_UAV_Frey_Group_3\HowFatIsMyTree\out\ecosense\final_trees.csv"


def main():
    ap = argparse.ArgumentParser(description="Generate final_trees.csv")
    ap.add_argument("--clean", default=CLEAN_DIR_DEFAULT,
                    help="Folder of clean .laz files")
    ap.add_argument("--inv",   default=INV_CSV_DEFAULT,
                    help="Inventory CSV")
    ap.add_argument("--out",   default=OUT_DEFAULT,
                    help="Output CSV path")
    args = ap.parse_args()

    clean_dir = Path(args.clean)
    if not clean_dir.exists():
        sys.exit(f"Clean folder not found: {clean_dir}")

    # Load inventory
    inv = pd.read_csv(args.inv)
    inv["full_id"] = inv["full_id"].astype(str).str.strip()
    inv = inv.drop_duplicates(subset="full_id", keep="first")
    inv_lookup = inv.set_index("full_id")[["x_32632", "y_32632", "diameter_m"]]
    # Collect full_ids from laz files
    laz_files = sorted(clean_dir.glob("*.laz"))
    if not laz_files:
        sys.exit(f"No .laz files found in {clean_dir}")

    print(f"Found {len(laz_files)} clean trees.")

    rows = []
    missing = []
    for f in laz_files:
        full_id = f.stem.split("__")[0].strip()
        if full_id not in inv_lookup.index:
            missing.append(full_id)
            continue
        r = inv_lookup.loc[full_id]
        rows.append({
            "full_id":    full_id,
            "x":          r["x_32632"],
            "y":          r["y_32632"],
            "diameter_m": r["diameter_m"],
        })

    if missing:
        print(
            f"  WARNING: {len(missing)} trees not found in inventory: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    out_df = pd.DataFrame(rows, columns=["full_id", "x", "y", "diameter_m"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"  Written {len(out_df)} rows → {out_path}")


if __name__ == "__main__":
    main()

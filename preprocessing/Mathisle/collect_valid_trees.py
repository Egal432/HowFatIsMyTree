"""
Collect Valid Trees
====================
Reads dbh_estimates.csv, filters to sensible DBH estimates,
and copies the corresponding LAZ files into a clean output folder.

Usage:
    python collect_valid_trees.py           # normal run
    python collect_valid_trees.py --dry-run # preview only
"""

import os
import sys
import shutil
import argparse
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DBH_CSV   = r"/mnt/e/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Mathisleweiher/dbh_estimates.csv"
TREES_DIR = r"/mnt/e/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Mathisleweiher/trees"
OUT_DIR   = r"/mnt/e/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Mathisleweiher/trees_valid"

# ─────────────────────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true",
                    help="Preview what would be copied without doing anything")
args = parser.parse_args()
DRY_RUN = args.dry_run

if DRY_RUN:
    print("=" * 60)
    print("DRY RUN — no files will be copied")

print("=" * 60)
print(f"DBH CSV:    {DBH_CSV}")
print(f"Source dir: {TREES_DIR}")
print(f"Output dir: {OUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD AND FILTER
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Loading DBH estimates...")

df = pd.read_csv(DBH_CSV)
print(f"  Total rows: {len(df)}")
print(f"\n  Flag breakdown:")
print(df["flag"].value_counts().to_string())

valid = df[df["flag"] == "ok"].copy()
print(f"\n  Valid trees (flag=ok): {len(valid)}")

if len(valid) == 0:
    print("ERROR: No valid trees found — check dbh_estimates.csv")
    sys.exit(1)

print(f"\n  DBH range:  {valid['dbh_cm'].min():.1f} – {valid['dbh_cm'].max():.1f} cm")
print(f"  DBH mean:   {valid['dbh_cm'].mean():.1f} cm")
print(f"  DBH median: {valid['dbh_cm'].median():.1f} cm")

# ─────────────────────────────────────────────────────────────────────────────
# RESOLVE FILE PATHS
# ─────────────────────────────────────────────────────────────────────────────
valid["src_path"] = valid["laz_file"].apply(
    lambda fn: os.path.join(TREES_DIR, fn)
)
valid["dst_path"] = valid["laz_file"].apply(
    lambda fn: os.path.join(OUT_DIR, fn)
)

# Check source files exist
missing = valid[~valid["src_path"].apply(os.path.exists)]
if len(missing):
    print(f"\n  WARNING: {len(missing)} source files not found:")
    for _, row in missing.iterrows():
        print(f"    {row['src_path']}")

to_copy = valid[valid["src_path"].apply(os.path.exists)]
print(f"\n  Files to copy: {len(to_copy)}")

if DRY_RUN:
    print("\n  Preview (first 10):")
    for _, row in to_copy.head(10).iterrows():
        print(f"    tree_{row['PredInstance']:06d}.laz  DBH={row['dbh_cm']:.1f} cm  inlier_ratio={row['inlier_ratio']:.2f}")
    if len(to_copy) > 10:
        print(f"    ... and {len(to_copy)-10} more")
    est_mb = sum(os.path.getsize(p) for p in to_copy["src_path"]) / 1024 / 1024
    print(f"\n  Estimated copy size: {est_mb:.0f} MB")
    print("\nDry run complete — no files copied.")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# COPY FILES
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

already_done = to_copy[to_copy["dst_path"].apply(os.path.exists)]
to_do        = to_copy[~to_copy["dst_path"].apply(os.path.exists)]

if len(already_done):
    print(f"\n  Skipping {len(already_done)} already-copied files")

print(f"  Copying {len(to_do)} files...\n")

copied = 0
for i, (_, row) in enumerate(to_do.iterrows()):
    shutil.copy2(row["src_path"], row["dst_path"])
    copied += 1
    if copied % 100 == 0 or copied == len(to_do):
        print(f"  {copied}/{len(to_do)} copied...", end="\r")

print()

# ─────────────────────────────────────────────────────────────────────────────
# SAVE FILTERED CSV ALONGSIDE THE LAZ FILES
# ─────────────────────────────────────────────────────────────────────────────
csv_out = os.path.join(OUT_DIR, "dbh_valid.csv")
to_copy.drop(columns=["src_path", "dst_path"]).to_csv(csv_out, index=False)
print(f"\n  Saved filtered CSV → {csv_out}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
all_out   = [f for f in os.listdir(OUT_DIR) if f.endswith(".laz")]
total_mb  = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in all_out) / 1024 / 1024
print(f"Output summary:")
print(f"  LAZ files:  {len(all_out)}")
print(f"  Total size: {total_mb:.0f} MB  ({total_mb/1024:.2f} GB)")
print(f"  CSV:        dbh_valid.csv")
print(f"  Location:   {OUT_DIR}")
print("\nDone.")

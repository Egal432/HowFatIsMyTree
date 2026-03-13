#!/usr/bin/env python3
"""
sort_preds.py  —  Sort and rename pred_*.laz files by inventory match count

Matches each pred_*.laz to inventory trees via exact x/y coordinate lookup
between labels.csv and inventory.csv, then copies files into:

    OUTPUT_DIR/
        single/          ← exactly 1 inventory tree  → renamed to full_id.laz
        multi_2/         ← exactly 2 inventory trees → pred_XXXX.laz + .txt
        multi_3/         ← exactly 3 inventory trees → pred_XXXX.laz + .txt
        ...
        unmatched/       ← pred exists in trees/ but has no row in labels.csv

    OUTPUT_DIR/sort_report.csv   ← full log of every pred and its matched trees

Usage:
    python sort_preds.py --trees   PATH/TO/trees/
                         --labels  PATH/TO/labels.csv
                         --inv     PATH/TO/inventory.csv
                         --out     PATH/TO/OUTPUT_DIR
    # add --dry-run to preview without copying anything
"""

import argparse
import shutil
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
INV_CSV_DEFAULT    = r"E:/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Ecosense/inventory.csv"
LABELS_CSV_DEFAULT = r"E:/01_UAV_Frey_Group_3/HowFatIsMyTree/out/ecosense/labels.csv"
TREES_DIR_DEFAULT  = r"E:/01_UAV_Frey_Group_3/HowFatIsMyTree/out/ecosense/trees"

# Tolerance for floating-point coordinate comparison (metres)
# Set to 0 for exact string/float match; increase slightly if CSVs have rounding
XY_TOLERANCE = 1e-3
# ──────────────────────────────────────────────────────────────────────────────


def load_data(labels_path: Path, inv_path: Path):
    labels = pd.read_csv(labels_path)
    inv    = pd.read_csv(inv_path)

    # Normalise column names
    labels.columns = labels.columns.str.strip()
    inv.columns    = inv.columns.str.strip()

    # Ensure numeric coords
    labels["x"] = pd.to_numeric(labels["x"], errors="coerce")
    labels["y"] = pd.to_numeric(labels["y"], errors="coerce")
    inv["x_32632"] = pd.to_numeric(inv["x_32632"], errors="coerce")
    inv["y_32632"] = pd.to_numeric(inv["y_32632"], errors="coerce")

    # Drop rows with missing coords
    labels = labels.dropna(subset=["x", "y"])
    inv    = inv.dropna(subset=["x_32632", "y_32632"])

    print(f"  Labels rows : {len(labels)}")
    print(f"  Inventory   : {len(inv)} trees")
    return labels, inv


def build_pred_to_trees(labels: pd.DataFrame, inv: pd.DataFrame) -> dict:
    """
    Returns dict:  predinstance (int)  →  list of full_id strings

    Strategy: for each row in labels, find the inventory tree whose
    (x_32632, y_32632) matches (x, y) within XY_TOLERANCE.
    """
    # Build a fast lookup from (rounded x, rounded y) → full_id
    # We round to avoid floating-point noise while still being exact enough
    def key(x, y):
        return (round(float(x), 4), round(float(y), 4))

    inv_lookup = {}
    for _, row in inv.iterrows():
        k = key(row["x_32632"], row["y_32632"])
        inv_lookup[k] = str(row["full_id"]).strip()

    pred_to_trees = defaultdict(list)
    unmatched_coords = []

    for _, row in labels.iterrows():
        pred = int(row["predinstance"])
        k    = key(row["x"], row["y"])

        # Try exact rounded match first
        full_id = inv_lookup.get(k)

        # If not found and tolerance > 0, do a nearest search
        if full_id is None and XY_TOLERANCE > 0:
            best_dist = XY_TOLERANCE
            for (ix, iy), fid in inv_lookup.items():
                d = ((ix - k[0])**2 + (iy - k[1])**2) ** 0.5
                if d <= best_dist:
                    best_dist = d
                    full_id   = fid

        if full_id is None:
            unmatched_coords.append((pred, row["x"], row["y"]))
        else:
            # Avoid duplicate full_ids per pred (same tree assigned twice)
            if full_id not in pred_to_trees[pred]:
                pred_to_trees[pred].append(full_id)

    if unmatched_coords:
        print(f"  WARNING: {len(unmatched_coords)} label rows had no inventory "
              f"coordinate match (first 5 shown):")
        for pred, x, y in unmatched_coords[:5]:
            print(f"    pred={pred}  x={x}  y={y}")

    return dict(pred_to_trees)


def main():
    ap = argparse.ArgumentParser(
        description="Sort pred_*.laz files by number of matched inventory trees")
    ap.add_argument("--trees",   default=TREES_DIR_DEFAULT,
                    help="Folder containing pred_*.laz files")
    ap.add_argument("--labels",  default=LABELS_CSV_DEFAULT,
                    help="labels.csv path")
    ap.add_argument("--inv",     default=INV_CSV_DEFAULT,
                    help="inventory.csv path")
    ap.add_argument("--out",     required=True,
                    help="Output root folder")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview decisions without copying any files")
    args = ap.parse_args()

    trees_dir = Path(args.trees)
    out_dir   = Path(args.out)

    if not trees_dir.exists():
        sys.exit(f"Trees folder not found: {trees_dir}")

    # ── Load CSVs ──
    print("Loading CSVs...")
    labels, inv = load_data(Path(args.labels), Path(args.inv))

    # ── Build mapping ──
    print("Matching labels → inventory...")
    pred_to_trees = build_pred_to_trees(labels, inv)
    print(f"  Matched {len(pred_to_trees)} unique pred instances.")

    # ── Collect LAZ files ──
    laz_files = {f.stem: f for f in sorted(trees_dir.glob("pred_*.laz"))}
    print(f"  Found {len(laz_files)} pred_*.laz files in trees folder.\n")

    # ── Process each pred ──
    report_rows = []
    counts = defaultdict(int)   # n_trees → how many preds had that count

    for stem, laz_path in laz_files.items():
        # Extract pred number from filename  e.g. "pred_1123" → 1123
        try:
            pred_num = int(stem.split("_", 1)[1])
        except (IndexError, ValueError):
            print(f"  SKIP (can't parse pred number): {stem}")
            continue

        matched = pred_to_trees.get(pred_num, [])
        n = len(matched)
        counts[n] += 1

        if n == 0:
            dest_dir  = out_dir / "unmatched"
            dest_name = laz_path.name          # keep original name
            note      = "no inventory match"

        elif n == 1:
            dest_dir  = out_dir / "single"
            dest_name = f"{matched[0]}.laz"    # rename to full_id
            note      = matched[0]

        else:
            dest_dir  = out_dir / f"multi_{n}"
            dest_name = laz_path.name          # keep pred_XXXX.laz
            note      = " | ".join(matched)

        report_rows.append({
            "pred":       stem,
            "pred_num":   pred_num,
            "n_trees":    n,
            "full_ids":   " | ".join(matched) if matched else "",
            "dest_dir":   str(dest_dir),
            "dest_name":  dest_name,
        })

        if args.dry_run:
            symbol = {0: "~", 1: "✓"}.get(n, "⚠")
            print(f"  {symbol} {stem:20s}  n={n}  →  {dest_dir.name}/{dest_name}")
            continue

        # ── Copy LAZ ──
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(laz_path), str(dest_dir / dest_name))

        # ── Write sidecar .txt for multi-tree preds ──
        if n > 1:
            txt_path = dest_dir / (laz_path.stem + ".txt")
            with open(txt_path, "w") as f:
                f.write(f"pred file : {laz_path.name}\n")
                f.write(f"n_trees   : {n}\n\n")
                f.write("matched inventory trees:\n")
                for fid in matched:
                    # Look up species + DBH for convenience
                    row = inv[inv["full_id"].astype(str).str.strip() == fid]
                    if not row.empty:
                        r = row.iloc[0]
                        sp  = str(r.get("species", "?"))
                        dbh = round(float(r.get("diameter_m", 0)) * 100, 1)
                        ht  = r.get("tls_treeheight", "?")
                        f.write(f"  {fid:15s}  species={sp:8s}  "
                                f"dbh={dbh:5.1f}cm  height={ht}m\n")
                    else:
                        f.write(f"  {fid}\n")

    # ── Save report CSV ──
    report_df = pd.DataFrame(report_rows).sort_values(["n_trees", "pred_num"])
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "sort_report.csv"
        report_df.to_csv(report_path, index=False)
        print(f"\nReport saved: {report_path}")

    # ── Print summary ──
    print(f"\n{'='*50}")
    print(f"  {'DRY RUN — ' if args.dry_run else ''}Results")
    print(f"{'='*50}")
    total = sum(counts.values())
    for n in sorted(counts.keys()):
        label = {0: "unmatched", 1: "single tree  → renamed to full_id"}.get(
            n, f"{n} trees     → multi_{n}/ + .txt sidecar")
        print(f"  n={n}  {counts[n]:>5} files   {label}")
    print(f"{'='*50}")
    print(f"  Total: {total} pred files processed")

    if args.dry_run:
        print("\n(dry-run: no files were copied)")


if __name__ == "__main__":
    main()

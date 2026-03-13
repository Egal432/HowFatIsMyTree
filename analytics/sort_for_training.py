#!/usr/bin/env python3
"""
sort_for_training.py  —  Sort segmented tree LAZ files for PointNet++ training

Reads the metrics.csv produced by batch_metrics.py and copies / moves trees
into:
    OUTPUT_DIR/
        training/    ← trees that pass all quality filters
        rejected/    ← trees that failed, with reason sub-folders
        review/      ← borderline cases worth a manual look

A summary CSV is written to OUTPUT_DIR/sort_summary.csv.

Usage:
    python sort_for_training.py --metrics PATH/TO/metrics.csv
                                --laz     PATH/TO/LAZ_FOLDER
                                --out     PATH/TO/OUTPUT_DIR
    # add --move to move files instead of copy (saves disk space)
    # add --dry-run to preview without touching files
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

# ── QUALITY THRESHOLDS ────────────────────────────────────────────────────────
# Adjust these to match what you want in your training set.

# Hard rejections — any one of these disqualifies the tree outright
REJECT_IF = [
    "flag_ground",           # too much ground contamination
    "flag_height_high",      # impossibly tall (stray points / segmentation error)
    "flag_height_low",       # stub / incomplete tree
    "flag_too_few_points",   # not enough points for PointNet++ to learn from
    "flag_dbh_high",         # DBH outlier (likely wrong segmentation)
    "flag_dbh_low",          # DBH outlier (likely wrong segmentation)
]

# Soft flags — tree goes to "review/" for manual inspection
REVIEW_IF = [
    "flag_height_mismatch",  # PC height differs notably from inventory
    "flag_dbh_mismatch",     # circle-fit DBH differs from inventory
    "flag_slenderness",      # unusual H/D ratio
    "flag_crown",            # very large crown (multi-stem? wrong seg?)
    "flag_no_trunk_slice",   # could not fit DBH circle — trunk missing
]

# Minimum absolute thresholds (applied on top of flags)
MIN_POINTS_TRAINING = 500     # at minimum for PointNet++
MIN_HEIGHT_M        = 3.0     # trees shorter than this aren't useful
MAX_HEIGHT_M        = 45.0    # absolute ceiling
MIN_DBH_CM          = 5.0     # tiny saplings excluded
MAX_GROUND_PCT      = 0.25    # stricter than the warning threshold

# ──────────────────────────────────────────────────────────────────────────────

REJECT_REASONS = {
    "flag_ground":          "ground_contamination",
    "flag_height_high":     "height_too_high",
    "flag_height_low":      "height_too_low",
    "flag_too_few_points":  "too_few_points",
    "flag_dbh_high":        "dbh_outlier_high",
    "flag_dbh_low":         "dbh_outlier_low",
    "min_points":           "below_min_points",
    "min_height":           "below_min_height",
    "max_height":           "above_max_height",
    "min_dbh":              "below_min_dbh",
    "max_ground_pct":       "ground_pct_too_high",
}


def decide(row: pd.Series):
    """
    Returns ('training' | 'rejected' | 'review', reason_str)
    """

    # ── 1. Hard rejections via flag columns ──
    for flag in REJECT_IF:
        if bool(row.get(flag, False)):
            return "rejected", REJECT_REASONS.get(flag, flag)

    # ── 2. Hard rejections via absolute thresholds ──
    n_pts = row.get("n_points", 0)
    if pd.notna(n_pts) and int(n_pts) < MIN_POINTS_TRAINING:
        return "rejected", REJECT_REASONS["min_points"]

    pc_h = row.get("pc_height_m")
    if pd.notna(pc_h):
        if float(pc_h) < MIN_HEIGHT_M:
            return "rejected", REJECT_REASONS["min_height"]
        if float(pc_h) > MAX_HEIGHT_M:
            return "rejected", REJECT_REASONS["max_height"]

    inv_dbh = row.get("inv_dbh_cm")
    if pd.notna(inv_dbh) and float(inv_dbh) < MIN_DBH_CM:
        return "rejected", REJECT_REASONS["min_dbh"]

    gnd = row.get("ground_pct")
    if pd.notna(gnd) and float(gnd) > MAX_GROUND_PCT:
        return "rejected", REJECT_REASONS["max_ground_pct"]

    # ── 3. Soft flags → review ──
    triggered = [f for f in REVIEW_IF if bool(row.get(f, False))]
    if triggered:
        reason = " | ".join(REJECT_REASONS.get(f, f) for f in triggered)
        return "review", reason

    return "training", "ok"


def copy_or_move(src: Path, dst_dir: Path, move: bool = False):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def main():
    ap = argparse.ArgumentParser(description="Sort tree LAZ files for PointNet++ training")
    ap.add_argument("--metrics", required=True, help="Path to metrics.csv from batch_metrics.py")
    ap.add_argument("--laz",     required=True, help="Folder containing the .laz files")
    ap.add_argument("--out",     required=True, help="Output root folder")
    ap.add_argument("--move",    action="store_true", help="Move files instead of copy")
    ap.add_argument("--dry-run", action="store_true", help="Preview decisions without copying")
    args = ap.parse_args()

    metrics_path = Path(args.metrics)
    laz_folder   = Path(args.laz)
    out_dir      = Path(args.out)

    if not metrics_path.exists():
        sys.exit(f"metrics.csv not found: {metrics_path}")
    if not laz_folder.exists():
        sys.exit(f"LAZ folder not found: {laz_folder}")

    df = pd.read_csv(metrics_path)
    print(f"Loaded {len(df)} rows from metrics CSV.")

    # ── Convert boolean columns (CSV reads them as strings sometimes) ──
    bool_cols = [c for c in df.columns if c.startswith("flag_") or c == "any_flag"]
    for c in bool_cols:
        df[c] = df[c].map(lambda v: str(v).strip().lower() in ("true", "1", "yes"))

    results = []
    counts  = {"training": 0, "rejected": 0, "review": 0, "laz_missing": 0}

    for _, row in df.iterrows():
        name   = str(row["name"])
        decision, reason = decide(row)

        # Find the LAZ file
        laz_path = laz_folder / f"{name}.laz"
        if not laz_path.exists():
            laz_path = laz_folder / f"{name}.las"
        if not laz_path.exists():
            counts["laz_missing"] += 1
            results.append({"name": name, "decision": "laz_missing", "reason": "file_not_found"})
            continue

        counts[decision] += 1

        # Copy/move
        if not args.dry_run:
            if decision == "rejected":
                # Sub-folder by rejection reason for easy browsing
                dest = out_dir / "rejected" / reason
            elif decision == "review":
                dest = out_dir / "review"
            else:
                dest = out_dir / "training"

            copy_or_move(laz_path, dest, move=args.move)

        results.append({"name": name, "decision": decision, "reason": reason})

        if args.dry_run:
            symbol = {"training": "✓", "rejected": "✗", "review": "?"}.get(decision, " ")
            print(f"  {symbol} {name:30s}  →  {decision:10s}  ({reason})")

    # ── Summary CSV ──
    summary_df = pd.DataFrame(results)
    # Merge key metrics columns for context
    keep_cols = ["name", "inv_dbh_cm", "pc_dbh_cm", "inv_height_m", "pc_height_m",
                 "n_points", "ground_pct", "any_flag"]
    merge_cols = [c for c in keep_cols if c in df.columns]
    summary_df = summary_df.merge(df[merge_cols], on="name", how="left")
    summary_df = summary_df.sort_values(["decision", "name"])

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "sort_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSort summary saved: {summary_path}")

    # ── Print summary table ──
    action = "Would sort" if args.dry_run else "Sorted"
    print(f"\n{'='*50}")
    print(f"  {action} {len(df)} trees")
    print(f"{'='*50}")
    print(f"  ✓ training    : {counts['training']:>5}")
    print(f"  ? review      : {counts['review']:>5}   ← check manually")
    print(f"  ✗ rejected    : {counts['rejected']:>5}")
    if counts['laz_missing']:
        print(f"  ! laz missing : {counts['laz_missing']:>5}")
    print(f"{'='*50}")

    if counts["rejected"] > 0:
        print("\nRejection breakdown:")
        for reason, grp in summary_df[summary_df.decision == "rejected"].groupby("reason"):
            print(f"  {reason:35s}: {len(grp)}")

    if counts["review"] > 0:
        print("\nReview breakdown:")
        for reason, grp in summary_df[summary_df.decision == "review"].groupby("reason"):
            print(f"  {reason:35s}: {len(grp)}")

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
merge_tree_folders.py  —  Merge multiple per-tree LAZ folders, handle conflicts

Scans three source folders for full_id.laz files, then copies them into:

    OUTPUT_DIR/
        merged/              ← trees that appear in exactly one source (clean)
        conflicts/
            16_27/
                16_27__single.laz        ← copy from 'single' folder
                16_27__r_segmentation.laz
            16_28/
                ...
        merge_report.csv     ← every tree, its sources, and verdict

Then run batch_metrics.py on conflicts/ to compare quality and pick a winner.

Usage:
    python merge_tree_folders.py --out PATH/TO/OUTPUT_DIR
    python merge_tree_folders.py --out PATH/TO/OUTPUT_DIR --dry-run
"""

import argparse
import shutil
import traceback
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ── SOURCE FOLDERS ────────────────────────────────────────────────────────────
# Label is used as the suffix in conflict filenames so you know where each came from
SOURCES = [
    ("single",        r"E:\01_UAV_Frey_Group_3\HowFatIsMyTree\out\ecosense\trees\single"),
    ("trees_by_id",   r"E:\01_UAV_Frey_Group_3\HowFatIsMyTree\out\ecosense\trees_by_id"),
    ("r_segmentation",r"E:\01_UAV_Frey_Group_3\HowFatIsMyTree\out\ecosense\r_segmentation"),
]
# ──────────────────────────────────────────────────────────────────────────────


def collect_files(sources: list) -> dict:
    """
    Returns dict:  full_id (str)  →  list of (label, Path) tuples
    Errors per folder are printed but never stop execution.
    """
    found = defaultdict(list)

    for label, folder_str in sources:
        folder = Path(folder_str)
        if not folder.exists():
            print(f"  WARNING: source folder not found, skipping — {folder}")
            continue

        try:
            laz_files = list(folder.glob("*.laz")) + list(folder.glob("*.las"))
        except Exception as e:
            print(f"  WARNING: could not scan {folder}: {e}")
            continue

        n = 0
        for f in laz_files:
            try:
                full_id = f.stem.strip()
                if full_id:
                    found[full_id].append((label, f))
                    n += 1
            except Exception as e:
                print(f"  WARNING: skipping {f}: {e}")

        print(f"  [{label}]  {n} files found in {folder}")

    return dict(found)


def safe_copy(src: Path, dst: Path, dry_run: bool) -> bool:
    """Copy src → dst, return True on success. Never raises."""
    try:
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
        return True
    except Exception as e:
        print(f"  ERROR copying {src.name} → {dst}: {e}")
        traceback.print_exc()
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Merge per-tree LAZ folders, route conflicts to their own subfolder")
    ap.add_argument("--out",      required=True, help="Output root folder")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Preview without copying files")
    ap.add_argument("--sources",  nargs="*", default=None,
                    help="Override source folders as label:path pairs "
                         "(e.g. single:E:/... trees_by_id:E:/...)")
    args = ap.parse_args()

    out_dir = Path(args.out)

    # Allow CLI override of source folders
    sources = SOURCES
    if args.sources:
        sources = []
        for item in args.sources:
            label, path = item.split(":", 1)
            sources.append((label.strip(), path.strip()))

    # ── Collect ──
    print("Scanning source folders...")
    all_files = collect_files(sources)
    print(f"  {len(all_files)} unique tree IDs found across all sources.\n")

    # ── Categorise ──
    clean     = {fid: srcs[0]      for fid, srcs in all_files.items() if len(srcs) == 1}
    conflicts = {fid: srcs          for fid, srcs in all_files.items() if len(srcs) > 1}

    print(f"  ✓ Clean (1 source)  : {len(clean)}")
    print(f"  ⚠ Conflicts (2+ src): {len(conflicts)}\n")

    merged_dir   = out_dir / "merged"
    conflict_dir = out_dir / "conflicts"

    report_rows = []
    n_ok = 0
    n_conflict_files = 0
    n_errors = 0

    # ── Copy clean files ──
    print("Copying clean files → merged/...")
    for full_id, (label, src_path) in sorted(clean.items()):
        dst = merged_dir / f"{full_id}.laz"
        ok  = safe_copy(src_path, dst, args.dry_run)
        if ok:
            n_ok += 1
        else:
            n_errors += 1

        report_rows.append({
            "full_id":  full_id,
            "verdict":  "merged",
            "n_sources": 1,
            "sources":  label,
            "dest":     str(dst),
            "error":    "" if ok else "copy_failed",
        })

        if args.dry_run:
            print(f"  ✓ {full_id:25s}  ←  [{label}]")

    # ── Copy conflict files ──
    print(f"\nCopying conflict files → conflicts/{{tree_id}}/...")
    for full_id, srcs in sorted(conflicts.items()):
        tree_conflict_dir = conflict_dir / full_id
        source_labels = []

        for label, src_path in srcs:
            # e.g.  conflicts/16_27/16_27__single.laz
            dst_name = f"{full_id}__{label}.laz"
            dst      = tree_conflict_dir / dst_name
            ok       = safe_copy(src_path, dst, args.dry_run)
            if ok:
                n_conflict_files += 1
            else:
                n_errors += 1
            source_labels.append(label)

            if args.dry_run:
                print(f"  ⚠ {full_id:25s}  ←  [{label}]  →  conflicts/{full_id}/{dst_name}")

        report_rows.append({
            "full_id":   full_id,
            "verdict":   "conflict",
            "n_sources": len(srcs),
            "sources":   " | ".join(source_labels),
            "dest":      str(tree_conflict_dir),
            "error":     "",
        })

    # ── Write conflict index .txt files (one per conflict tree) ──
    if not args.dry_run:
        for full_id, srcs in sorted(conflicts.items()):
            try:
                txt_path = conflict_dir / full_id / f"{full_id}__sources.txt"
                txt_path.parent.mkdir(parents=True, exist_ok=True)
                with open(txt_path, "w") as f:
                    f.write(f"tree    : {full_id}\n")
                    f.write(f"sources : {len(srcs)}\n\n")
                    for label, src_path in srcs:
                        f.write(f"  [{label}]\n")
                        f.write(f"    {src_path}\n")
                    f.write("\nRun batch_metrics.py on this folder to compare quality.\n")
            except Exception as e:
                print(f"  WARNING: could not write sidecar for {full_id}: {e}")

    # ── Save report ──
    report_df = pd.DataFrame(report_rows).sort_values(["verdict", "full_id"],
                                                       ascending=[False, True])
    if not args.dry_run:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / "merge_report.csv"
            report_df.to_csv(report_path, index=False)
            print(f"\nReport saved: {report_path}")
        except Exception as e:
            print(f"  WARNING: could not save report CSV: {e}")

    # ── Summary ──
    print(f"\n{'='*55}")
    print(f"  {'DRY RUN — ' if args.dry_run else ''}Merge complete")
    print(f"{'='*55}")
    print(f"  ✓ Merged (clean)     : {n_ok:>5} trees  →  merged/")
    print(f"  ⚠ Conflicts          : {len(conflicts):>5} trees  →  conflicts/{{id}}/")
    print(f"    Conflict copies    : {n_conflict_files:>5} files total")
    if n_errors:
        print(f"  ✗ Copy errors        : {n_errors:>5}  (check output above)")
    print(f"{'='*55}")

    if conflicts:
        print(f"\nNext step — compare conflict quality:")
        print(f"  python batch_metrics.py --folder \"{out_dir / 'conflicts'}\" --inv YOUR_INV.csv")
        print(f"  (each conflict tree has its own subfolder with both versions)")

    if args.dry_run:
        print("\n(dry-run: no files were copied)")


if __name__ == "__main__":
    main()

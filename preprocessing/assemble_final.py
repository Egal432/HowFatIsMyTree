#!/usr/bin/env python3
"""
assemble_final.py  —  Assemble final tree folder with priority-based source selection

Priority: single > r_segmentation > trees_by_id

For each unique tree (full_id), picks the best available version and copies
it as full_id.laz into the output folder. Writes a single provenance .txt
listing where every file came from.

Usage:
    python assemble_final.py --out E:/path/to/final_trees
    python assemble_final.py --out E:/path/to/final_trees --dry-run
"""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Priority order: first match wins
SOURCES = [
    ("single",         r"E:\01_UAV_Frey_Group_3\HowFatIsMyTree\out\ecosense\trees\single"),
    ("r_segmentation", r"E:\01_UAV_Frey_Group_3\HowFatIsMyTree\out\ecosense\r_segmentation"),
    ("trees_by_id",    r"E:\01_UAV_Frey_Group_3\HowFatIsMyTree\out\ecosense\trees_by_id"),
]
# ──────────────────────────────────────────────────────────────────────────────


def collect(sources):
    """
    Returns dict: full_id -> list of (priority_rank, label, Path)
    sorted by priority (lowest rank = highest priority).
    """
    found = defaultdict(list)
    for rank, (label, folder_str) in enumerate(sources):
        folder = Path(folder_str)
        if not folder.exists():
            print(f"  WARNING: folder not found, skipping — {folder}")
            continue
        n = 0
        for f in folder.glob("*.laz"):
            full_id = f.stem.split("__")[0].strip()
            if full_id:
                found[full_id].append((rank, label, f))
                n += 1
        print(f"  [{label}]  {n} files")
    return dict(found)


def main():
    ap = argparse.ArgumentParser(
        description="Assemble final tree folder with priority-based source selection")
    ap.add_argument("--out",     required=True, help="Output folder for final trees")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview without copying")
    args = ap.parse_args()

    out_dir = Path(args.out)

    print("Scanning source folders...")
    all_trees = collect(SOURCES)
    print(f"  {len(all_trees)} unique tree IDs found.\n")

    provenance = []   # (full_id, source_label, original_path)
    source_counts = defaultdict(int)

    for full_id, versions in sorted(all_trees.items()):
        # Sort by priority rank, pick first (lowest rank = highest priority)
        versions.sort(key=lambda x: x[0])
        rank, label, src_path = versions[0]

        dst = out_dir / f"{full_id}.laz"

        if args.dry_run:
            symbol = ["①", "②", "③"][rank] if rank < 3 else str(rank)
            print(f"  {symbol} {full_id:20s}  ←  [{label}]  {src_path.name}")
        else:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src_path), str(dst))
                source_counts[label] += 1
            except Exception as e:
                print(f"  ERROR copying {full_id}: {e}")
                label = f"ERROR ({label})"

        provenance.append((full_id, label, str(src_path)))

    # ── Write provenance txt ──
    if not args.dry_run:
        prov_path = out_dir / "_provenance.txt"
        with open(prov_path, "w") as f:
            f.write(f"Final tree dataset — provenance log\n")
            f.write(f"{'='*60}\n")
            f.write(f"Total trees: {len(provenance)}\n\n")
            f.write("Source counts:\n")
            for label, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
                f.write(f"  {label:25s}: {cnt}\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"{'full_id':<20}  {'source':<20}  original_path\n")
            f.write(f"{'-'*60}\n")
            for full_id, label, orig in sorted(provenance):
                f.write(f"{full_id:<20}  {label:<20}  {orig}\n")
        print(f"\nProvenance log saved: {prov_path}")

    # ── Summary ──
    print(f"\n{'='*45}")
    print(f"  {'DRY RUN — ' if args.dry_run else ''}Done")
    print(f"{'='*45}")
    print(f"  Total trees assembled: {len(provenance)}")
    if not args.dry_run:
        for label, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
            print(f"  [{label:20s}]  {cnt}")
    if args.dry_run:
        print("\n  (dry-run: no files copied)")


if __name__ == "__main__":
    main()

"""
Split LAZ by PredInstance
=========================
Reads the full point cloud in chunks and streams directly to
one .laz file per PredInstance — no full-cloud buffering in RAM.

Usage:
    python split_by_predinstance.py           # normal run
    python split_by_predinstance.py --dry-run # preview only
"""

import os
import sys
import argparse
import numpy as np
import laspy
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
LAZ_PATH = r"/mnt/e/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Mathisleweiher/mathisleweiher.laz"
OUTPUT_DIR = r"/mnt/e/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Mathisleweiher/trees"

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE = 5_000_000  # points per read chunk — reduce if RAM is tight
MIN_POINTS = 50         # skip PredInstances with fewer points than this
# PredInstance IDs to skip (0 = unclassified/background)
IGNORE_IDS = {0}

# ─────────────────────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true",
                    help="Scan and report without writing any files")
args = parser.parse_args()
DRY_RUN = args.dry_run

if DRY_RUN:
    print("=" * 60)
    print("DRY RUN — no files will be written")

print("=" * 60)
print(f"Input:      {LAZ_PATH}")
print(f"Output dir: {OUTPUT_DIR}")
print(f"Min points: {MIN_POINTS}")
print(f"Ignore IDs: {IGNORE_IDS}")

# ─────────────────────────────────────────────────────────────────────────────
# PASS 1 — count points per PredInstance (no data stored, just counters)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Pass 1: counting points per PredInstance...")

counts = defaultdict(int)

with laspy.open(LAZ_PATH) as f:
    total_pts = f.header.point_count
    print(f"  Total points: {total_pts:,}")
    for i, chunk in enumerate(f.chunk_iterator(CHUNK_SIZE)):
        pred = np.array(chunk["PredInstance"])
        ids, cnts = np.unique(pred, return_counts=True)
        for pid, cnt in zip(ids, cnts):
            counts[int(pid)] += int(cnt)
        pts_done = min((i + 1) * CHUNK_SIZE, total_pts)
        print(
            f"  {pts_done:>12,} / {total_pts:,}  ({100*pts_done/total_pts:.1f}%)", end="\r")

print()

valid_ids = sorted([p for p, c in counts.items()
                   if p not in IGNORE_IDS and c >= MIN_POINTS])
skip_ids = sorted([p for p, c in counts.items()
                  if p in IGNORE_IDS or c < MIN_POINTS])

print(f"  Total unique PredInstances: {len(counts)}")
print(f"  Will write:  {len(valid_ids)} (>= {MIN_POINTS} pts, not ignored)")
print(f"  Will skip:   {len(skip_ids)}")
print(
    f"  Point range: {min(counts[p] for p in valid_ids):,} – {max(counts[p] for p in valid_ids):,}")

bins = [0, 100, 500, 1000, 5000, 10000, 50000, float("inf")]
labels = ["<100", "100–500", "500–1k", "1k–5k", "5k–10k", "10k–50k", ">50k"]
print("\n  Point count distribution:")
for lo, hi, label in zip(bins[:-1], bins[1:], labels):
    n = sum(1 for p in valid_ids if lo <= counts[p] < hi)
    bar = "█" * min(n, 50) + (f" (+{n-50})" if n > 50 else "")
    print(f"    {label:>10}  {bar} {n}")

if DRY_RUN:
    # Estimate disk usage: assume ~15 bytes/pt compressed
    est_mb = sum(counts[p] for p in valid_ids) * 15 / 1024 / 1024
    print(f"\n  Estimated output size: ~{est_mb:.0f} MB")
    print("\nDry run complete — no files written.")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# PASS 2 — streaming write: one LasWriter per tree, open all upfront
#           write directly on each chunk pass — zero RAM buffering
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Skip already-done files
output_paths = {pid: os.path.join(
    OUTPUT_DIR, f"tree_{pid:06d}.laz") for pid in valid_ids}
already_done = {pid for pid, path in output_paths.items()
                if os.path.exists(path)}
to_write = [pid for pid in valid_ids if pid not in already_done]

if already_done:
    print(f"\n  Skipping {len(already_done)} already-existing files")
print(f"  Streaming {len(to_write)} tree files...\n")

if not to_write:
    print("All files already exist. Done.")
    sys.exit(0)

to_write_set = set(to_write)

with laspy.open(LAZ_PATH) as f:
    src_header = f.header

    # Open all writers at once — each is a tiny open file handle, no data yet
    writers = {}
    for pid in to_write:
        out_header = laspy.LasHeader(
            point_format=src_header.point_format,
            version=src_header.version
        )
        out_header.offsets = src_header.offsets
        out_header.scales = src_header.scales
        fh = open(output_paths[pid], "wb")
        writers[pid] = (fh, laspy.LasWriter(fh, header=out_header))

    print(f"  Opened {len(writers)} output file handles")
    print("  Reading and streaming points...\n")

    pts_done = 0
    n_chunks = (total_pts + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i, chunk in enumerate(f.chunk_iterator(CHUNK_SIZE)):
        pred = np.array(chunk["PredInstance"])

        # For each valid tree in this chunk, slice and write directly
        chunk_pids = np.intersect1d(np.unique(pred), list(to_write_set))
        for pid in chunk_pids:
            mask = pred == pid
            if mask.any():
                writers[pid][1].write_points(chunk[mask])

        pts_done += len(pred)
        print(f"  Chunk {i+1:>4}/{n_chunks}  |  "
              f"{pts_done:>12,} / {total_pts:,} pts  "
              f"({100*pts_done/total_pts:.1f}%)",
              end="\r")

    print()
    print("\n  Closing writers...")

    # Close all writers — this finalises each LAZ file
    for pid, (fh, writer) in writers.items():
        writer.close()
        fh.close()

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
all_files = [fn for fn in os.listdir(OUTPUT_DIR) if fn.endswith(".laz")]
total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, fn))
                 for fn in all_files)
print(f"Output summary:")
print(f"  Files:      {len(all_files)}")
print(
    f"  Total size: {total_size/1024/1024:.1f} MB  ({total_size/1024/1024/1024:.2f} GB)")
print(f"  Location:   {OUTPUT_DIR}")
print("\nDone.")

"""
visualize_sampling.py

Point cloud sampling visualizer for presentations.
Samples N points from a LAZ file and saves them as a new LAZ for CloudCompare.

Usage:
    python visualize_sampling.py --input path/to/tree.laz --n_points 1024
    python visualize_sampling.py --input path/to/tree.laz --n_points 2048
    python visualize_sampling.py --input path/to/tree.laz --n_points 1024 --n_points 2048
"""

import argparse
from pathlib import Path

import laspy
import numpy as np


def sample_and_save(input_path: Path, n_points: int, output_dir: Path, seed: int = 42):
    rng = np.random.default_rng(seed)

    las = laspy.read(str(input_path))
    total = len(las.points)

    print(f"  Loaded {total:,} points from {input_path.name}")

    if n_points >= total:
        print(f"  Warning: requested {n_points} but cloud only has {total} — using all points.")
        chosen_idx = np.arange(total)
    else:
        chosen_idx = rng.choice(total, size=n_points, replace=False)
        chosen_idx.sort()  # keep spatial order for nicer visualisation

    # Build new LAS with sampled points
    header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
    header.offsets = las.header.offsets
    header.scales = las.header.scales

    out_las = laspy.LasData(header=header)
    out_las.points = las.points[chosen_idx]

    # Add scalar field showing this is a sampled cloud
    out_las.add_extra_dim(laspy.ExtraBytesParams(name="sample_idx", type=np.int32))
    out_las.sample_idx = chosen_idx.astype(np.int32)

    stem = input_path.stem
    out_path = output_dir / f"{stem}_sampled_{n_points}.laz"
    out_las.write(str(out_path))
    print(f"  Saved {n_points:,} points → {out_path.name}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Sample points from a LAZ file for visualisation")
    parser.add_argument("--input", required=True, help="Path to input .laz file")
    parser.add_argument("--n_points", type=int, nargs="+", default=[1024, 2048],
                        help="Number(s) of points to sample (default: 1024 2048)")
    parser.add_argument("--output_dir", default=None,
                        help="Output folder (default: same folder as input)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:      {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Sampling:   {args.n_points} points\n")

    for n in args.n_points:
        print(f"--- {n:,} points ---")
        sample_and_save(input_path, n, output_dir, seed=args.seed)

    print("\nDone. Open the output .laz files in CloudCompare to compare sampling densities.")


if __name__ == "__main__":
    main()

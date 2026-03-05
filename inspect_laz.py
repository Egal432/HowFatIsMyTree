#!/usr/bin/env python3
"""
inspect_laz.py

Purpose
-------
Inspect a LAS/LAZ file to answer:
A1) Where is the tree segmentation stored? (which point attribute looks like "tree id")
A2) Is Z already height-normalized (above ground) or absolute elevation?

What it prints
--------------
- Basic header info (point count, bounds, scales/offsets, LAS version, point format)
- CRS info if available
- All available point dimensions (standard + extra bytes)
- Stats for candidate segmentation fields (unique counts, ranges)
- Classification distribution (useful to see if ground is classified)
- Heuristics about whether Z looks normalized

Install
-------
pip install laspy lazrs numpy

Usage
-----
python inspect_laz.py /path/to/bigfile.laz
"""

import argparse
import sys
import numpy as np

CANDIDATE_SEG_FIELDS = [
    # common places segmentation ends up
    "tree_id", "treeid", "TreeID", "TreeId",
    "instance", "instance_id", "segment_id", "seg_id",
    "point_source_id", "user_data",
    "classification",  # sometimes abused, usually not
    "gps_time",        # rarely abused
    "scanner_channel", "scan_angle_rank",
]


def _safe_unique_count(arr: np.ndarray, max_n=200_000) -> int:
    """Unique count, with subsampling for huge arrays."""
    n = arr.shape[0]
    if n > max_n:
        idx = np.random.choice(n, size=max_n, replace=False)
        arr = arr[idx]
    return int(np.unique(arr).shape[0])


def _top_counts(arr: np.ndarray, k=15, max_n=1_000_000):
    """Return (value, count) pairs for most frequent values."""
    n = arr.shape[0]
    if n > max_n:
        idx = np.random.choice(n, size=max_n, replace=False)
        arr = arr[idx]
    vals, counts = np.unique(arr, return_counts=True)
    order = np.argsort(counts)[::-1]
    vals = vals[order][:k]
    counts = counts[order][:k]
    return list(zip(vals.tolist(), counts.tolist()))


def _print_header(las):
    hdr = las.header
    print("=== HEADER ===")
    print(f"File version: {hdr.version}")
    print(f"Point format: {hdr.point_format}")
    print(f"Point count : {hdr.point_count:,}")
    print(f"Scales      : {hdr.scales}")
    print(f"Offsets     : {hdr.offsets}")
    print(f"Bounds min  : {hdr.mins}")
    print(f"Bounds max  : {hdr.maxs}")

    # CRS info (may be None)
    crs = None
    try:
        crs = hdr.parse_crs()
    except Exception:
        crs = None
    print(f"CRS         : {crs if crs is not None else 'None/Unknown'}")
    print()


def _list_dimensions(las):
    print("=== AVAILABLE POINT DIMENSIONS ===")
    names = list(las.point_format.dimension_names)
    print(f"Dimensions ({len(names)}):")
    for n in names:
        print(f"  - {n}")
    print()


def _dim_stats(las, name: str, n_preview=5):
    """Print basic stats for a dimension if it exists."""
    try:
        arr = las[name]
    except Exception:
        return None

    # Convert to numpy
    arr = np.asarray(arr)
    info = {
        "name": name,
        "dtype": str(arr.dtype),
        "min": None,
        "max": None,
        "mean": None,
        "unique_est": None,
        "top": None,
        "preview": None,
    }

    # numeric stats
    if np.issubdtype(arr.dtype, np.number):
        info["min"] = float(np.nanmin(arr))
        info["max"] = float(np.nanmax(arr))
        info["mean"] = float(np.nanmean(arr))
        # unique count (subsampled)
        info["unique_est"] = _safe_unique_count(arr)
        # top values for integer-ish things
        if np.issubdtype(arr.dtype, np.integer) or (info["unique_est"] < 5000):
            info["top"] = _top_counts(arr, k=12)
    else:
        # non-numeric (rare in las extra bytes)
        try:
            info["unique_est"] = _safe_unique_count(arr.astype("U"))
        except Exception:
            info["unique_est"] = None

    info["preview"] = arr[:n_preview].tolist()
    return info


def _check_z_normalization(z: np.ndarray):
    """
    Heuristic: if z looks like height above ground, expect:
      - many values near 0
      - min close to 0 (maybe slightly negative due to noise)
      - typical forest heights (0..50m)
    If absolute elevation, expect:
      - min/max like hundreds of meters depending on location
      - min far from 0
    """
    z = np.asarray(z, dtype=np.float64)

    zmin = float(np.nanmin(z))
    zmax = float(np.nanmax(z))
    zq1 = float(np.nanquantile(z, 0.01))
    zq99 = float(np.nanquantile(z, 0.99))

    near0 = np.mean((z >= -0.25) & (z <= 0.25))  # fraction near 0
    span = zq99 - zq1

    print("=== Z NORMALIZATION HEURISTICS ===")
    print(f"Z min/max          : {zmin:.3f} / {zmax:.3f}")
    print(f"Z 1% / 99%         : {zq1:.3f} / {zq99:.3f}")
    print(f"Z span (99%-1%)    : {span:.3f}")
    print(f"Fraction in [-0.25,0.25]m: {near0*100:.2f}%")

    # simple judgement
    looks_normalized = False
    reasons = []

    if abs(zmin) < 5 and zmax < 120:
        reasons.append("min close-ish to 0 and max not huge (<120m)")
        looks_normalized = True

    if near0 > 0.02:  # 2% of points within ±25cm of 0 is a decent hint
        reasons.append("non-trivial fraction of points near 0m")
        looks_normalized = True

    if zmin > 50 and zmax > 100:
        # common for absolute elevation in meters
        reasons.append("min is far above 0 (likely absolute elevation)")
        looks_normalized = False

    if zmin < -50:
        reasons.append("min is very negative (unlikely for normalized heights)")
        looks_normalized = False

    print("Heuristic verdict  :", "LIKELY normalized heights" if looks_normalized else "LIKELY absolute elevation")
    if reasons:
        print("Reasons            :", "; ".join(reasons))
    print()


def _suggest_segmentation_fields(las):
    print("=== SEGMENTATION FIELD HUNT ===")
    names = set(las.point_format.dimension_names)

    # Show obvious candidates that exist
    existing_candidates = [n for n in CANDIDATE_SEG_FIELDS if n in names]
    if existing_candidates:
        print("Candidate fields found in file:")
        for n in existing_candidates:
            info = _dim_stats(las, n)
            if info is None:
                continue
            print(f"\n-- {n} ({info['dtype']}) --")
            if info["min"] is not None:
                print(f"min/max/mean    : {info['min']:.3f} / {info['max']:.3f} / {info['mean']:.3f}")
            if info["unique_est"] is not None:
                print(f"unique (approx) : {info['unique_est']}")
            if info["top"] is not None:
                print("top values      :", info["top"])
            print("preview         :", info["preview"])
    else:
        print("No obvious candidate field names found among common options.")
        print("We will scan for 'ID-like' integer dimensions next...")

    # Scan all dimensions for something that looks like an instance id:
    # heuristics: integer-ish, many uniques, but not almost all unique (like gps_time)
    print("\n=== SCAN ALL DIMENSIONS FOR 'ID-LIKE' FIELDS ===")
    dim_names = list(las.point_format.dimension_names)

    likely = []
    for n in dim_names:
        info = _dim_stats(las, n)
        if info is None:
            continue

        # We want discrete IDs:
        # - integer dtype OR small number of unique values relative to N
        # - not gps_time-like (almost all unique)
        # - not x/y/z (float with huge unique)
        arr_dtype = info["dtype"]
        if "float" in arr_dtype.lower():
            continue
        if info["unique_est"] is None:
            continue

        N = las.header.point_count
        u = info["unique_est"]

        # likely segmentation id has:
        # - u in [50 .. 500000] (depends on dataset), but not close to N
        # - and values not just 0..31 typical bitfields
        if u >= 50 and u < 0.95 * N:
            likely.append((n, u, info["min"], info["max"], info["top"]))

    if likely:
        likely.sort(key=lambda x: x[1], reverse=True)
        print("Possible ID-like dimensions (sorted by unique count):")
        for n, u, mn, mx, top in likely[:15]:
            print(f"- {n:25s} unique≈{u:8d}  min={mn}  max={mx}  top={top[:5] if top else None}")
        print("\nTip: segmentation often has ~#trees unique values (e.g. 1500).")
        print("Pick a field whose unique count is close to your expected number of trees.")
    else:
        print("No obvious integer ID-like field found.")
        print("In that case, segmentation might be stored in an extra bytes field with a non-standard name.")
        print("Check the 'AVAILABLE POINT DIMENSIONS' list above for something suspicious (e.g. 'id', 'tree', 'instance').")

    print()


def _classification_info(las):
    names = set(las.point_format.dimension_names)
    if "classification" not in names:
        print("=== CLASSIFICATION ===")
        print("No 'classification' dimension present in this file format.")
        print()
        return

    cls = np.asarray(las["classification"])
    print("=== CLASSIFICATION ===")
    print("Classification unique count:", int(np.unique(cls).shape[0]))
    print("Top classes:", _top_counts(cls, k=20))
    print("Common note: ground is often class 2 (ASP RS), but depends on workflow.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("laz_path", help="Path to .las or .laz file")
    args = ap.parse_args()

    path = args.laz_path
    try:
        import laspy
    except ImportError:
        print("Missing dependency. Install with: pip install laspy lazrs numpy")
        sys.exit(1)

    las = laspy.read(path)

    print(f"\nInspecting: {path}\n")

    _print_header(las)
    _list_dimensions(las)

    # Basic Z heuristics
    z = np.asarray(las.z)  # scaled
    _check_z_normalization(z)

    # Classification / ground hint
    _classification_info(las)

    # Segmentation hunt
    _suggest_segmentation_fields(las)

    print("Done.")


if __name__ == "__main__":
    main()
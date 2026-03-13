"""
Tree Metrics from Segmented TLS Point Cloud
============================================
Requires dbh_estimates.csv from dbh_estimation.py to already exist.

Computes per-segment:
  - Tree height (max Z_norm)
  - Crown base height (inflection in vertical density)
  - Crown radius (horizontal spread above crown base)
  - Stem lean angle + lean direction
  - H/D ratio (slenderness)
  - Total point count (scan coverage proxy)

Exports for CloudCompare:
  1. tree_positions.txt   — one point per tree at ground, all scalar fields
  2. dbh_circles.txt      — circle rings at BH height (colored by DBH)
  3. stem_lines.txt       — vertical lines base→apex (colored by height)
  4. crown_circles.txt    — crown spread rings at crown base height
"""

import os
import glob
import numpy as np
import pandas as pd
import laspy
import rasterio
from rasterio.merge import merge

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DTM_FOLDER = r"/mnt/e/01_UAV_Frey_Group_3/GeoKram/full_dgm"
LAZ_PATH = r"/mnt/e/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Mathisleweiher/mathisleweiher.laz"
DBH_CSV = r"/mnt/e/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Mathisleweiher/dbh_estimates.csv"

OUT_DIR = r"/mnt/e/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Mathisleweiher/cc_viz"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
MAX_TREE_HEIGHT = 60.0   # discard segments taller than this (m) — noise guard
MIN_TREE_HEIGHT = 2.0    # discard segments shorter than this
CROWN_BASE_PERCENTILE = 25  # vertical density percentile for crown base detection

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DTM MOSAIC
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading DTM mosaic...")
tif_files = glob.glob(os.path.join(DTM_FOLDER, "*.tif"))
src_files = [rasterio.open(f) for f in tif_files]
mosaic_data, mosaic_transform = merge(src_files)
mosaic_data = mosaic_data[0]
dtm_crs = src_files[0].crs
dtm_nodata = src_files[0].nodata if src_files[0].nodata is not None else -9999
for s in src_files:
    s.close()
print(f"  Loaded {len(tif_files)} tiles")


def sample_dtm(x_arr, y_arr):
    from rasterio.transform import rowcol as rc
    nrows, ncols = mosaic_data.shape
    row_idx, col_idx = rc(mosaic_transform, x_arr, y_arr)
    row_idx = np.array(row_idx)
    col_idx = np.array(col_idx)
    valid = (col_idx >= 0) & (col_idx < ncols) & (
        row_idx >= 0) & (row_idx < nrows)
    ground = np.full(len(x_arr), np.nan)
    ground[valid] = mosaic_data[row_idx[valid], col_idx[valid]]
    if dtm_nodata is not None:
        ground[ground == dtm_nodata] = np.nan
    return ground


# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPUTE VERTICAL OFFSET (same logic as dbh_estimation.py)
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing vertical datum offset...")
sample_pts = []
with laspy.open(LAZ_PATH) as f:
    for chunk in f.chunk_iterator(2_000_000):
        x_s = np.array(chunk.x)
        y_s = np.array(chunk.y)
        z_s = np.array(chunk.z)
        sample_pts.append(np.column_stack([x_s, y_s, z_s]))
        if sum(len(g) for g in sample_pts) > 200_000:
            break

spt = np.vstack(sample_pts)
dtm_at_spts = sample_dtm(spt[:, 0], spt[:, 1])
valid_s = np.isfinite(dtm_at_spts)
raw_diff = spt[valid_s, 2] - dtm_at_spts[valid_s]
VERTICAL_OFFSET = float(np.percentile(raw_diff, 2))
print(f"  Vertical offset: {VERTICAL_OFFSET:.3f} m")

# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD DBH RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading DBH estimates...")
dbh_df = pd.read_csv(DBH_CSV)
print(f"  Total segments: {len(dbh_df)}")
print(f"  Flags: {dbh_df['flag'].value_counts().to_dict()}")

# Work with all segments — metrics may still be valid even if DBH flagged
all_pids = set(dbh_df["PredInstance"].astype(int).tolist())

# ─────────────────────────────────────────────────────────────────────────────
# 4. METRIC COMPUTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def crown_base_height(z_norm_seg, n_bins=40, smooth_window=3):
    """
    Estimate crown base height from vertical point density profile.
    Crown base = lowest bin where density rises sharply (stem → crown transition).
    Returns height in metres.
    """
    if len(z_norm_seg) < 20:
        return np.nan

    z_max = z_norm_seg.max()
    if z_max < 2.0:
        return np.nan

    bins = np.linspace(0, z_max, n_bins + 1)
    counts, _ = np.histogram(z_norm_seg, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Simple moving average smooth
    kernel = np.ones(smooth_window) / smooth_window
    counts_sm = np.convolve(counts.astype(float), kernel, mode='same')

    # Crown base = first bin from top where density drops below crown_base_percentile
    # of the upper canopy density — scan upward from stem
    upper_mean = np.mean(counts_sm[n_bins // 2:])
    threshold = upper_mean * 0.15  # 15% of upper canopy density

    cbh = np.nan
    for i in range(1, n_bins - 1):
        if counts_sm[i] >= threshold and counts_sm[i - 1] < threshold:
            cbh = bin_centers[i]
            break

    return cbh if cbh and cbh < z_max * 0.9 else z_max * 0.3


def stem_lean(x_arr, y_arr, z_norm_arr, stem_top=5.0):
    """
    Compute stem lean angle (degrees from vertical) and azimuth.
    Uses linear regression on stem points (Z_norm 0.5 – stem_top m).
    """
    stem_mask = (z_norm_arr >= 0.5) & (z_norm_arr <= stem_top)
    if stem_mask.sum() < 10:
        return np.nan, np.nan

    x_s = x_arr[stem_mask]
    y_s = y_arr[stem_mask]
    z_s = z_norm_arr[stem_mask]

    # Linear regression X ~ Z and Y ~ Z
    A = np.column_stack([z_s, np.ones_like(z_s)])
    try:
        cx, _ = np.linalg.lstsq(A, x_s, rcond=None)[0]
        cy, _ = np.linalg.lstsq(A, y_s, rcond=None)[0]
    except Exception:
        return np.nan, np.nan

    # cx, cy = horizontal displacement per metre of height
    lean_angle = float(np.degrees(np.arctan(np.sqrt(cx**2 + cy**2))))
    lean_azimuth = float(np.degrees(np.arctan2(cy, cx)) % 360)
    return lean_angle, lean_azimuth


def crown_radius_estimate(x_arr, y_arr, z_norm_arr, crown_base):
    """Horizontal spread of crown points above crown base."""
    if np.isnan(crown_base):
        return np.nan
    crown_mask = z_norm_arr > crown_base
    if crown_mask.sum() < 5:
        return np.nan
    cx = x_arr[crown_mask].mean()
    cy = y_arr[crown_mask].mean()
    dists = np.sqrt((x_arr[crown_mask] - cx)**2 + (y_arr[crown_mask] - cy)**2)
    return float(np.percentile(dists, 90))  # 90th pct = robust crown radius


# ─────────────────────────────────────────────────────────────────────────────
# 5. SECOND PASS — load ALL points per tile, compute full metrics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Computing tree metrics (full-segment pass)...")

tif_files_sorted = sorted(tif_files)
metrics_store = {}   # pid → dict of metrics

for tif_path in tif_files_sorted:
    tile_name = os.path.basename(tif_path)
    print(f"\n  Tile: {tile_name}")

    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        tile_data = src.read(1)
        tile_nd = src.nodata if src.nodata is not None else -9999

    valid_tile = tile_data[tile_data != tile_nd]
    if len(valid_tile) == 0:
        continue

    # Wide Z window — we want all vegetation, not just BH
    z_lo = float(valid_tile.min()) + VERTICAL_OFFSET - 0.5
    z_hi = float(valid_tile.max()) + VERTICAL_OFFSET + MAX_TREE_HEIGHT + 1.0

    chunk_pts = []
    with laspy.open(LAZ_PATH) as f:
        for chunk in f.chunk_iterator(5_000_000):
            x = np.array(chunk.x)
            y = np.array(chunk.y)
            z = np.array(chunk.z)
            mask = (
                (x >= bounds.left) & (x <= bounds.right) &
                (y >= bounds.bottom) & (y <= bounds.top) &
                (z >= z_lo) & (z <= z_hi)
            )
            if not mask.any():
                continue
            pred = np.array(chunk["PredInstance"])[mask]
            chunk_pts.append(np.column_stack(
                [x[mask], y[mask], z[mask], pred]))

    if not chunk_pts:
        continue

    arr = np.vstack(chunk_pts)
    print(f"    Points loaded: {len(arr):,}")

    # Normalize Z
    ground_z = sample_dtm(arr[:, 0], arr[:, 1])
    z_norm = arr[:, 2] - ground_z - VERTICAL_OFFSET

    # Keep only points with valid ground and reasonable normalized height
    valid_mask = np.isfinite(z_norm) & (
        z_norm >= -0.5) & (z_norm <= MAX_TREE_HEIGHT)
    arr = arr[valid_mask]
    z_norm = z_norm[valid_mask]

    # Only process segments we know about
    pred_ids = arr[:, 3].astype(int)
    tile_pids = np.intersect1d(np.unique(pred_ids), list(all_pids))
    print(f"    Segments to process: {len(tile_pids)}")

    for pid in tile_pids:
        if pid in metrics_store:
            continue  # already computed from an earlier tile

        seg = arr[pred_ids == pid]
        zn = z_norm[pred_ids == pid]

        if len(seg) < 5:
            continue

        x_seg, y_seg = seg[:, 0], seg[:, 1]

        height = float(np.percentile(zn, 99))   # 99th pct = robust max
        cbh = crown_base_height(zn)
        cr = crown_radius_estimate(x_seg, y_seg, zn, cbh)
        lean_ang, lean_az = stem_lean(x_seg, y_seg, zn)

        # Base position = centroid of lowest 10% of points
        low_mask = zn <= np.percentile(zn, 10)
        base_x = float(x_seg[low_mask].mean())
        base_y = float(y_seg[low_mask].mean())

        metrics_store[pid] = {
            "PredInstance":    int(pid),
            "base_x":          base_x,
            "base_y":          base_y,
            "height_m":        round(height, 2),
            "crown_base_m":    round(cbh, 2) if not np.isnan(cbh) else np.nan,
            "crown_radius_m":  round(cr, 2) if not np.isnan(cr) else np.nan,
            "lean_angle_deg":  round(lean_ang, 2) if not np.isnan(lean_ang) else np.nan,
            "lean_azimuth_deg": round(lean_az, 2) if not np.isnan(lean_az) else np.nan,
            "n_pts_total":     len(seg),
        }

    del arr, z_norm, chunk_pts
    print(f"    Done — {len(tile_pids)} segments processed")

# ─────────────────────────────────────────────────────────────────────────────
# 6. MERGE DBH + METRICS, COMPUTE DERIVED FIELDS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Merging metrics...")

metrics_df = pd.DataFrame(list(metrics_store.values()))
full_df = dbh_df.merge(metrics_df, on="PredInstance", how="left")

# Derived metrics
full_df["hd_ratio"] = full_df["height_m"] / \
    (full_df["dbh_m"].replace(0, np.nan))
full_df["crown_area_m2"] = np.pi * full_df["crown_radius_m"] ** 2
full_df["live_crown_ratio"] = (
    (full_df["height_m"] - full_df["crown_base_m"]) / full_df["height_m"]
)

# Height sanity flag
full_df.loc[
    (full_df["height_m"] < MIN_TREE_HEIGHT) |
    (full_df["height_m"] > MAX_TREE_HEIGHT),
    "flag"
] = "implausible_height"

full_df.to_csv(os.path.join(OUT_DIR, "tree_metrics.csv"), index=False)
print(f"  Saved tree_metrics.csv — {len(full_df)} trees")
print(f"  Columns: {list(full_df.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. CLOUDCOMPARE EXPORTS  (all Z values in absolute ellipsoidal height)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Exporting CloudCompare visualisation files...")

ok = full_df[full_df["flag"] == "ok"].dropna(
    subset=["base_x", "base_y", "height_m"])
print(f"  'ok' segments for export: {len(ok)}")

# ── Compute absolute ground Z at each tree base ──────────────────────────────
# absolute_Z = DTM(base_x, base_y) + VERTICAL_OFFSET
# This converts normalized heights back to the same ellipsoidal datum as the LAZ.
print("  Sampling DTM at tree base positions...")
base_x_arr = ok["base_x"].values
base_y_arr = ok["base_y"].values
ground_at_base = sample_dtm(base_x_arr, base_y_arr)          # orthometric
abs_ground = ground_at_base + VERTICAL_OFFSET             # ellipsoidal

ok = ok.copy()
ok["abs_ground_z"] = abs_ground

n_valid = np.sum(np.isfinite(abs_ground))
print(f"  DTM valid at {n_valid}/{len(ok)} tree positions")
print(
    f"  Absolute ground Z range: {np.nanmin(abs_ground):.1f} – {np.nanmax(abs_ground):.1f} m")


def abs_z(row, norm_height):
    """Convert a normalised height to absolute ellipsoidal Z for a given tree row."""
    return row["abs_ground_z"] + norm_height


# ── 7a. TREE POSITIONS ───────────────────────────────────────────────────────
# One point per tree sitting at absolute ground level.
# Load in CC: File > Open > ASCII, Space separator, assign X Y Z then scalar fields.
pos = ok[["base_x", "base_y", "PredInstance",
          "dbh_cm", "height_m", "crown_base_m",
          "crown_radius_m", "hd_ratio", "lean_angle_deg",
          "live_crown_ratio", "n_pts_total", "inlier_ratio", "abs_ground_z"]].copy()
pos.insert(2, "Z", ok["abs_ground_z"])
pos.to_csv(os.path.join(OUT_DIR, "tree_positions.txt"), sep=" ", index=False)
print("  ✓ tree_positions.txt — one point per tree at absolute ground Z")

# ── 7b. DBH CIRCLES ─────────────────────────────────────────────────────────
# Rings at absolute Z = ground + 1.3 m  (breast height in ellipsoidal coords)
n_ring = 72
angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
ok_dbh = ok.dropna(subset=["cx", "cy", "r", "abs_ground_z"])

ring_rows = []
for _, row in ok_dbh.iterrows():
    z_bh = row["abs_ground_z"] + 1.3   # absolute breast height
    ring_rows.append(pd.DataFrame({
        "X":            row["cx"] + row["r"] * np.cos(angles),
        "Y":            row["cy"] + row["r"] * np.sin(angles),
        "Z":            z_bh,
        "PredInstance": int(row["PredInstance"]),
        "DBH_cm":       round(row["dbh_cm"], 1),
    }))

if ring_rows:
    pd.concat(ring_rows).to_csv(
        os.path.join(OUT_DIR, "dbh_circles.txt"), sep=" ", index=False)
    print("  ✓ dbh_circles.txt — DBH rings at absolute breast height")

# ── 7c. STEM LINES ───────────────────────────────────────────────────────────
# Vertical sticks from absolute ground to absolute tree top.
# Each tree = 20 points along a vertical line — renders as a stick in CC.
n_stem_pts = 20
stem_rows = []
for _, row in ok.iterrows():
    if np.isnan(row["abs_ground_z"]):
        continue
    z_base = row["abs_ground_z"]
    z_top = row["abs_ground_z"] + row["height_m"]
    zs = np.linspace(z_base, z_top, n_stem_pts)
    stem_rows.append(pd.DataFrame({
        "X":            row["base_x"],
        "Y":            row["base_y"],
        "Z":            zs,
        "PredInstance": int(row["PredInstance"]),
        "Height_m":     round(row["height_m"], 2),
        "DBH_cm":       round(row["dbh_cm"], 1) if not np.isnan(row["dbh_cm"]) else 0,
    }))

if stem_rows:
    pd.concat(stem_rows).to_csv(
        os.path.join(OUT_DIR, "stem_lines.txt"), sep=" ", index=False)
    print("  ✓ stem_lines.txt — vertical stem sticks at absolute Z")

# ── 7d. CROWN CIRCLES ────────────────────────────────────────────────────────
# Rings at absolute Z = ground + crown_base_m
ok_crown = ok.dropna(subset=["crown_radius_m", "crown_base_m", "abs_ground_z"])
crown_rows = []
for _, row in ok_crown.iterrows():
    z_crown = row["abs_ground_z"] + row["crown_base_m"]
    crown_rows.append(pd.DataFrame({
        "X":            row["base_x"] + row["crown_radius_m"] * np.cos(angles),
        "Y":            row["base_y"] + row["crown_radius_m"] * np.sin(angles),
        "Z":            z_crown,
        "PredInstance": int(row["PredInstance"]),
        "Crown_r_m":    round(row["crown_radius_m"], 2),
    }))

if crown_rows:
    pd.concat(crown_rows).to_csv(
        os.path.join(OUT_DIR, "crown_circles.txt"), sep=" ", index=False)
    print("  ✓ crown_circles.txt — crown rings at absolute crown base Z")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SUMMARY STATS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Summary statistics (ok segments only):")
stats_cols = ["dbh_cm", "height_m", "crown_base_m",
              "crown_radius_m", "hd_ratio", "lean_angle_deg"]
print(ok[stats_cols].describe().round(2).to_string())

print(f"\nAll files saved to: {OUT_DIR}")
print("Done.")

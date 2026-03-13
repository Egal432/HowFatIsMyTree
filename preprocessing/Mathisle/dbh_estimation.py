"""
DBH Estimation from Split Tree LAZ Files
=========================================
Runs on the output of split_by_predinstance.py — one LAZ per PredInstance.
Each file is small, so this is fast and simple: no tile loop, no deduplication.

Pipeline:
  trees/tree_000042.laz  →  normalize Z with DTM  →  RANSAC circle at BH  →  DBH
  trees/tree_000043.laz  →  ...
  →  dbh_estimates.csv
"""

import os
import re
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
TREES_DIR = r"/mnt/e/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Mathisleweiher/trees"
OUTPUT_CSV = r"/mnt/e/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Mathisleweiher/dbh_estimates.csv"

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
BH_LOW = 1.2   # breast height slice lower bound (m, normalized)
BH_HIGH = 1.4   # breast height slice upper bound (m, normalized)
RANSAC_ITER = 300   # RANSAC iterations per tree
RANSAC_THR = 0.02  # inlier threshold (m)
MIN_PTS = 10    # minimum BH points to attempt a fit
MIN_DBH_CM = 5     # flag below this
MAX_DBH_CM = 90   # flag above this

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD AND MOSAIC DTM
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading DTM mosaic...")

tif_files = glob.glob(os.path.join(DTM_FOLDER, "*.tif"))
if not tif_files:
    raise FileNotFoundError(f"No .tif files found in {DTM_FOLDER}")

src_files = [rasterio.open(f) for f in tif_files]
mosaic_data, mosaic_transform = merge(src_files)
mosaic_data = mosaic_data[0]
dtm_nodata = src_files[0].nodata if src_files[0].nodata is not None else -9999
for s in src_files:
    s.close()

print(f"  Loaded {len(tif_files)} tiles")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DTM SAMPLING HELPER
# ─────────────────────────────────────────────────────────────────────────────


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
# 3. COMPUTE VERTICAL OFFSET  (ellipsoidal LAZ vs orthometric DTM)
#    Sample from first few tree files — offset is constant across the site
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing vertical datum offset...")

tree_files = sorted(glob.glob(os.path.join(TREES_DIR, "tree_*.laz")))
if not tree_files:
    raise FileNotFoundError(f"No tree_*.laz files found in {TREES_DIR}")

print(f"  Found {len(tree_files)} tree LAZ files")

sample_pts = []
for tf in tree_files[:5]:
    las = laspy.read(tf)
    sample_pts.append(np.column_stack(
        [np.array(las.x), np.array(las.y), np.array(las.z)]))
    if sum(len(s) for s in sample_pts) > 50_000:
        break

spt = np.vstack(sample_pts)
dtm_at_spts = sample_dtm(spt[:, 0], spt[:, 1])
valid_s = np.isfinite(dtm_at_spts)
raw_diff = spt[valid_s, 2] - dtm_at_spts[valid_s]
VERTICAL_OFFSET = float(np.percentile(raw_diff, 2))

print(f"  raw_Z - DTM_Z range: {raw_diff.min():.2f} – {raw_diff.max():.2f} m")
print(f"  Vertical offset (2nd pct): {VERTICAL_OFFSET:.3f} m")

# ─────────────────────────────────────────────────────────────────────────────
# 4. CIRCLE FITTING
# ─────────────────────────────────────────────────────────────────────────────


def circle_from_3pts(p):
    ax, ay = p[0]
    bx, by = p[1]
    cx, cy = p[2]
    D = 2 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by))
    if abs(D) < 1e-10:
        return None
    ux = ((ax**2+ay**2)*(by-cy) + (bx**2+by**2)
          * (cy-ay) + (cx**2+cy**2)*(ay-by)) / D
    uy = ((ax**2+ay**2)*(cx-bx) + (bx**2+by**2)
          * (ax-cx) + (cx**2+cy**2)*(bx-ax)) / D
    return ux, uy, np.sqrt((ax-ux)**2 + (ay-uy)**2)


def taubin_fit(pts):
    x0, y0 = pts[:, 0].mean(), pts[:, 1].mean()
    x, y = pts[:, 0] - x0, pts[:, 1] - y0
    M = np.column_stack([2*x, 2*y, np.ones(len(x))])
    b = x**2 + y**2
    try:
        res, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
        return res[0] + x0, res[1] + y0, np.sqrt(res[2] + res[0]**2 + res[1]**2)
    except Exception:
        return None


def ransac_circle(x, y):
    n = len(x)
    result = dict(cx=np.nan, cy=np.nan, r=np.nan, inlier_ratio=np.nan, n_pts=n)
    if n < MIN_PTS:
        return result

    pts = np.column_stack([x, y])
    best_score = 0
    best = None
    rng = np.random.default_rng()

    for _ in range(RANSAC_ITER):
        fit = circle_from_3pts(pts[rng.choice(n, 3, replace=False)])
        if fit is None:
            continue
        cx, cy, r = fit
        if r > 2.0 or r < 0.01:
            continue
        inliers = np.sum(
            np.abs(np.sqrt((x-cx)**2 + (y-cy)**2) - r) < RANSAC_THR)
        if inliers > best_score:
            best_score = inliers
            best = (cx, cy, r)

    if best is None:
        return result

    cx, cy, r = best
    inlier_mask = np.abs(np.sqrt((x-cx)**2 + (y-cy)**2) - r) < RANSAC_THR
    if inlier_mask.sum() >= 6:
        refined = taubin_fit(pts[inlier_mask])
        if refined is not None:
            cx, cy, r = refined

    result.update(cx=cx, cy=cy, r=r, inlier_ratio=best_score/n, n_pts=n)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN LOOP — one file per PredInstance
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Processing {len(tree_files)} tree files...\n")

results = []
n_skipped = 0

for i, tf in enumerate(tree_files):

    # Parse PredInstance ID from filename: tree_000042.laz → 42
    m = re.search(r"tree_(\d+)\.laz$", os.path.basename(tf))
    if not m:
        continue
    pid = int(m.group(1))

    if i % 50 == 0:
        print(f"  [{i+1:>4}/{len(tree_files)}]  tree_{pid:06d}")

    try:
        las = laspy.read(tf)
    except Exception as e:
        print(f"  WARN: could not read {os.path.basename(tf)}: {e}")
        n_skipped += 1
        continue

    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)

    if len(x) == 0:
        n_skipped += 1
        continue

    # Normalize Z using DTM
    ground_z = sample_dtm(x, y)
    z_norm = z - ground_z - VERTICAL_OFFSET

    # Slice to breast height
    bh_mask = (z_norm >= BH_LOW) & (z_norm <= BH_HIGH) & np.isfinite(ground_z)
    n_bh = int(bh_mask.sum())

    fit = ransac_circle(x[bh_mask], y[bh_mask])

    results.append({
        "PredInstance": pid,
        "laz_file":     os.path.basename(tf),
        "cx":           fit["cx"],
        "cy":           fit["cy"],
        "r":            fit["r"],
        "dbh_m":        fit["r"] * 2 if not np.isnan(fit["r"]) else np.nan,
        "dbh_cm":       fit["r"] * 200 if not np.isnan(fit["r"]) else np.nan,
        "inlier_ratio": fit["inlier_ratio"],
        "n_pts_bh":     n_bh,
        "n_pts_total":  len(x),
    })

print(f"\n  Done — {len(results)} trees processed, {n_skipped} skipped")

# ─────────────────────────────────────────────────────────────────────────────
# 6. FLAG AND SAVE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)

df = pd.DataFrame(results)

conditions = [
    df["n_pts_bh"] < MIN_PTS,
    df["inlier_ratio"] < 0.3,
    (df["dbh_cm"] < MIN_DBH_CM) | (df["dbh_cm"] > MAX_DBH_CM),
]
choices = ["too_few_points", "poor_circle_fit", "implausible_dbh"]
df["flag"] = np.select(conditions, choices, default="ok")

print("Flag summary:")
print(df["flag"].value_counts().to_string())
print(f"\nTotal trees: {len(df)}")

ok = df[df["flag"] == "ok"]
if len(ok):
    print(f"\nDBH summary (ok only, n={len(ok)}):")
    print(f"  mean:   {ok['dbh_cm'].mean():.1f} cm")
    print(f"  median: {ok['dbh_cm'].median():.1f} cm")
    print(f"  range:  {ok['dbh_cm'].min():.1f} – {ok['dbh_cm'].max():.1f} cm")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved → {OUTPUT_CSV}")
print("Done.")

#!/usr/bin/env python3
"""
batch_metrics.py  —  Batch tree point cloud metrics + QC dashboard

Processes all .laz/.las files in a folder, matches to inventory CSV,
computes smarter metrics (quantile height, trunk-slice DBH estimate,
crown area, density flags) and outputs:
  • metrics.csv          — one row per tree, all metrics + flags
  • _dashboard.png       — overview scatter / histogram panel

Usage:
    pip install laspy[lazrs] numpy matplotlib scipy pandas
    python batch_metrics.py --folder PATH/TO/LAZ_FOLDER
    python batch_metrics.py --folder PATH/TO/LAZ_FOLDER --inv PATH/TO/inventory.csv
"""

import argparse
import sys
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

try:
    import laspy
except ImportError:
    sys.exit("Install laspy:  pip install laspy[lazrs]")
try:
    import pandas as pd
except ImportError:
    sys.exit("Install pandas:  pip install pandas")
from scipy.spatial import ConvexHull

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
INV_CSV_DEFAULT = r"E:/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Ecosense/inventory.csv"

# Height: use this quantile instead of raw max (p99 rejects stray high outliers)
HEIGHT_QUANTILE = 0.99

# DBH estimation from trunk slice (metres above normalised ground)
DBH_SLICE_LOW = 1.1   # bottom of breast-height slice
DBH_SLICE_HIGH = 1.5   # top of breast-height slice
MIN_SLICE_POINTS = 20    # need at least this many to attempt circle fit

# Crown area: upper fraction of tree used for convex hull
CROWN_FRAC = 0.40  # top 40% of height

# Ground contamination
GROUND_Z_THRESH = 0.50  # metres above normalised base
GROUND_PCT_WARN = 0.30  # flag if >30% of points below threshold

# Minimum usable points
MIN_POINTS = 100

# Hard-limit flags (values outside these are suspicious)
MAX_DBH_CM = 150
MIN_DBH_CM = 3
MAX_HEIGHT_M = 55
MIN_HEIGHT_M = 2
MAX_CROWN_M2 = 250
MAX_DBH_MISMATCH_CM = 20    # flag if pc-estimated DBH differs >20 cm from inventory
MAX_HEIGHT_MISMATCH = 5.0   # flag if pc height differs >5 m from inventory height

# ── COLOURS ───────────────────────────────────────────────────────────────────
DARK = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
BLUE = "#58a6ff"
GREEN = "#3fb950"
ORANGE = "#d29922"
RED = "#f85149"
PURPLE = "#bc8cff"


# ── HELPERS ───────────────────────────────────────────────────────────────────

def fit_circle_algebraic(xy: np.ndarray):
    """
    Algebraic (Kåsa) circle fit — fast, no iteration.
    Returns (cx, cy, radius) or None on failure.
    Works well for partial arcs as long as the arc spans >90°.
    """
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([2*x, 2*y, np.ones(len(x))])
    b = x**2 + y**2
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, c = res
        r = np.sqrt(c + cx**2 + cy**2)
        if r <= 0 or r > 5.0:   # >5 m diameter trunk → reject
            return None
        return float(cx), float(cy), float(r)
    except Exception:
        return None


def load_inventory(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    inv = {}
    for _, row in df.iterrows():
        key = str(row["full_id"]).strip()
        inv[key] = {
            "inv_dbh_cm":    round(float(row["diameter_m"]) * 100, 1)
            if pd.notna(row.get("diameter_m")) else None,
            "inv_height_m":  float(row["tls_treeheight"])
            if pd.notna(row.get("tls_treeheight")) else None,
            "species":       str(row["species"])
            if pd.notna(row.get("species", "")) else "",
            "plot_id":       str(row["plot_id"])
            if pd.notna(row.get("plot_id", "")) else "",
        }
    print(f"  Loaded {len(inv)} trees from inventory.")
    return inv


# ── CORE METRICS ──────────────────────────────────────────────────────────────

def compute_metrics(path: Path, inv_data: dict) -> dict | None:
    tree_id = path.stem.split("__")[0]   # "16_27__single" → "16_27"

    inv = inv_data.get(tree_id)
    if inv is None:
        return {"name": tree_id, "_missing_inv": True}

    try:
        las = laspy.read(str(path))
        pts = np.stack([las.x, las.y, las.z], axis=1).astype(np.float32)
    except Exception as e:
        print(f"  ERROR reading {path.name}: {e}")
        return None

    n_pts = len(pts)
    if n_pts < MIN_POINTS:
        return {
            "name": tree_id, **inv,
            "n_points": n_pts, "flag_too_few_points": True,
            "_skip": True,
        }

    # ── Normalise Z ──
    z_min = pts[:, 2].min()
    pts[:, 2] -= z_min

    z = pts[:, 2]

    # ── Height (quantile, not raw max) ──
    pc_height = float(np.quantile(z, HEIGHT_QUANTILE))
    z_p50 = float(np.median(z))
    z_p25 = float(np.quantile(z, 0.25))
    z_p75 = float(np.quantile(z, 0.75))
    z_p95 = float(np.quantile(z, 0.95))
    z_p99 = float(np.quantile(z, 0.99))

    # ── DBH from trunk slice (circle fit at breast height) ──
    slice_mask = (z >= DBH_SLICE_LOW) & (z <= DBH_SLICE_HIGH)
    slice_pts = pts[slice_mask, :2]
    pc_dbh_cm = None
    if len(slice_pts) >= MIN_SLICE_POINTS:
        # Centre the slice before fitting
        xy_c = slice_pts - slice_pts.mean(axis=0)
        result = fit_circle_algebraic(xy_c)
        if result is not None:
            pc_dbh_cm = round(result[2] * 200, 1)   # radius→diameter, m→cm

    # ── Crown area (convex hull of top CROWN_FRAC) ──
    crown_thresh = pc_height * (1.0 - CROWN_FRAC)
    crown_mask = z > crown_thresh
    crown_pts = pts[crown_mask, :2]
    crown_area = None
    if len(crown_pts) >= 4:
        try:
            hull = ConvexHull(crown_pts)
            crown_area = round(float(hull.volume), 2)   # volume=area in 2D
        except Exception:
            pass

    # ── Crown width (approximate diameter of hull) ──
    crown_diam = None
    if crown_area is not None:
        crown_diam = round(float(np.sqrt(4 * crown_area / np.pi)), 2)

    # ── Point density (pts / m of height) ──
    pt_density = round(n_pts / max(pc_height, 0.1), 1)

    # ── Ground contamination ──
    ground_pct = float((z < GROUND_Z_THRESH).sum()) / n_pts

    # ── Slenderness (H/DBH) — common allometric check ──
    inv_dbh_cm = inv.get("inv_dbh_cm")
    slenderness = None
    if inv_dbh_cm and inv_dbh_cm > 0:
        slenderness = round(pc_height / (inv_dbh_cm / 100), 1)  # H[m] / D[m]

    # ── FLAGS ──────────────────────────────────────────────────────────────────
    inv_height = inv.get("inv_height_m")

    flag_ground = ground_pct > GROUND_PCT_WARN
    flag_height_high = pc_height > MAX_HEIGHT_M
    flag_height_low = pc_height < MIN_HEIGHT_M
    # already caught above but included for completeness
    flag_too_few_points = n_pts < MIN_POINTS
    flag_crown = crown_area is not None and crown_area > MAX_CROWN_M2
    flag_no_trunk_slice = len(slice_pts) < MIN_SLICE_POINTS

    # DBH flags (use inventory DBH as ground truth)
    flag_dbh_high = inv_dbh_cm is not None and inv_dbh_cm > MAX_DBH_CM
    flag_dbh_low = inv_dbh_cm is not None and inv_dbh_cm < MIN_DBH_CM

    # Cross-check: PC circle-fit vs inventory DBH
    flag_dbh_mismatch = (
        pc_dbh_cm is not None and inv_dbh_cm is not None and
        abs(pc_dbh_cm - inv_dbh_cm) > MAX_DBH_MISMATCH_CM
    )

    # Height: PC vs inventory
    flag_height_mismatch = (
        inv_height is not None and
        abs(pc_height - inv_height) > MAX_HEIGHT_MISMATCH
    )

    # Slenderness: <20 (super stubby) or >200 (impossibly thin) are suspicious
    flag_slenderness = slenderness is not None and (
        slenderness < 20 or slenderness > 200)

    any_flag = any([
        flag_ground, flag_height_high, flag_height_low,
        flag_too_few_points, flag_crown, flag_no_trunk_slice,
        flag_dbh_high, flag_dbh_low, flag_dbh_mismatch,
        flag_height_mismatch, flag_slenderness,
    ])

    return {
        # Identity
        "name":                  tree_id,
        "species":               inv.get("species", ""),
        "plot_id":               inv.get("plot_id", ""),
        # Inventory reference
        "inv_dbh_cm":            inv_dbh_cm,
        "inv_height_m":          inv_height,
        # Point cloud metrics
        "pc_height_m":           round(pc_height, 2),
        "pc_dbh_cm":             pc_dbh_cm,
        "crown_area_m2":         crown_area,
        "crown_diam_m":          crown_diam,
        "n_points":              n_pts,
        "pt_density_per_m":      pt_density,
        "ground_pct":            round(ground_pct, 4),
        "slenderness_H_D":       slenderness,
        "trunk_slice_npts":      int(slice_pts.shape[0]),
        # Height percentiles
        "z_p25":                 round(z_p25, 2),
        "z_p50":                 round(z_p50, 2),
        "z_p75":                 round(z_p75, 2),
        "z_p95":                 round(z_p95, 2),
        "z_p99":                 round(z_p99, 2),
        # Flags
        "flag_ground":           flag_ground,
        "flag_height_high":      flag_height_high,
        "flag_height_low":       flag_height_low,
        "flag_height_mismatch":  flag_height_mismatch,
        "flag_dbh_high":         flag_dbh_high,
        "flag_dbh_low":          flag_dbh_low,
        "flag_dbh_mismatch":     flag_dbh_mismatch,
        "flag_crown":            flag_crown,
        "flag_no_trunk_slice":   flag_no_trunk_slice,
        "flag_slenderness":      flag_slenderness,
        "flag_too_few_points":   flag_too_few_points,
        "any_flag":              any_flag,
        "source":   path.stem.split("__")[1] if "__" in path.stem else "unknown",
        "filename": path.name,
    }


# ── DASHBOARD ─────────────────────────────────────────────────────────────────

def flag_color(m: dict) -> str:
    if m.get("flag_ground"):
        return RED
    if m.get("flag_height_mismatch"):
        return ORANGE
    if m.get("flag_dbh_mismatch"):
        return ORANGE
    if m.get("flag_slenderness"):
        return ORANGE
    if m.get("flag_dbh_high") or m.get("flag_dbh_low"):
        return ORANGE
    if m.get("flag_height_high") or m.get("flag_height_low"):
        return PURPLE
    if m.get("flag_crown"):
        return PURPLE
    if m.get("flag_no_trunk_slice"):
        return MUTED
    return BLUE


def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(title, color=TEXT, fontsize=9, pad=6)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.grid(color=BORDER, lw=0.5, alpha=0.5)


def make_dashboard(metrics: list, out_path: Path):
    full = [m for m in metrics if not m.get(
        "_skip") and not m.get("_missing_inv")]
    if not full:
        print("  No plottable metrics — skipping dashboard.")
        return

    def arr(key, fallback=np.nan):
        return np.array([m.get(key, fallback) or fallback for m in full], dtype=float)

    dbh = arr("inv_dbh_cm")
    pc_h = arr("pc_height_m")
    inv_h = arr("inv_height_m")
    crown = arr("crown_area_m2")
    npts = arr("n_points")
    gnd = arr("ground_pct")
    pc_dbh = arr("pc_dbh_cm")
    slen = arr("slenderness_H_D")
    colors = [flag_color(m) for m in full]

    fig = plt.figure(figsize=(24, 15), facecolor=DARK)
    fig.suptitle("Tree Point Cloud — Batch QC Dashboard",
                 color=TEXT, fontsize=18, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.60, wspace=0.38,
                           top=0.93, bottom=0.08, left=0.05, right=0.97)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[1, 0])
    ax6 = fig.add_subplot(gs[1, 1])
    ax7 = fig.add_subplot(gs[1, 2])
    ax8 = fig.add_subplot(gs[1, 3])
    ax9 = fig.add_subplot(gs[2, :])   # full-width flag table

    # 1. DBH (inv) vs PC Height
    style_ax(ax1, "Inv DBH vs PC Height (p99)",
             "DBH inv (cm)", "Height p99 (m)")
    ax1.scatter(dbh, pc_h, c=colors, s=25, alpha=0.85,
                edgecolors='none', picker=True)

    # 2. Inv Height vs PC Height
    style_ax(ax2, "Inventory Height vs PC Height",
             "Inv height (m)", "PC height p99 (m)")
    valid = np.isfinite(inv_h) & np.isfinite(pc_h)
    if valid.any():
        ax2.scatter(inv_h[valid], pc_h[valid],
                    c=[c for c, v in zip(colors, valid) if v],
                    s=25, alpha=0.85, edgecolors='none')
        lim = (0, max(inv_h[valid].max(), pc_h[valid].max()) * 1.05)
        ax2.plot(lim, lim, color=MUTED, lw=1, ls='--', alpha=0.6, label="1:1")
        ax2.set_xlim(lim)
        ax2.set_ylim(lim)
        ax2.legend(facecolor=PANEL, labelcolor=MUTED, fontsize=7)

    # 3. Inv DBH vs PC DBH (circle fit)
    style_ax(ax3, "DBH: Inventory vs PC Trunk Slice",
             "DBH inv (cm)", "DBH pc-fit (cm)")
    valid3 = np.isfinite(dbh) & np.isfinite(pc_dbh)
    if valid3.any():
        ax3.scatter(dbh[valid3], pc_dbh[valid3],
                    c=[c for c, v in zip(colors, valid3) if v],
                    s=25, alpha=0.85, edgecolors='none')
        lim3 = (0, max(dbh[valid3].max(), pc_dbh[valid3].max()) * 1.05)
        ax3.plot(lim3, lim3, color=MUTED, lw=1, ls='--', alpha=0.6)
        ax3.set_xlim(lim3)
        ax3.set_ylim(lim3)

    # 4. Slenderness distribution
    style_ax(ax4, "Slenderness (H/D) Distribution\n(20–200 = normal range)",
             "H/D ratio", "N trees")
    sl_ok = slen[np.isfinite(slen)]
    if len(sl_ok):
        ax4.hist(sl_ok, bins=40, color=BLUE, alpha=0.8)
        ax4.axvline(20,  color=ORANGE, lw=1.2, ls='--', label="min 20")
        ax4.axvline(200, color=ORANGE, lw=1.2, ls='--', label="max 200")
        ax4.legend(facecolor=PANEL, labelcolor=MUTED, fontsize=7)

    # 5. PC Height vs Crown Area
    style_ax(ax5, "PC Height vs Crown Area",
             "Height p99 (m)", "Crown area (m²)")
    valid5 = np.isfinite(crown)
    if valid5.any():
        ax5.scatter(pc_h[valid5], crown[valid5],
                    c=[c for c, v in zip(colors, valid5) if v],
                    s=25, alpha=0.85, edgecolors='none')

    # 6. Ground contamination histogram
    style_ax(ax6, f"Ground Point % (flag >{GROUND_PCT_WARN*100:.0f}%)",
             "% below 0.5 m", "N trees")
    bins = np.linspace(0, max(gnd.max(), GROUND_PCT_WARN + 0.05), 40)
    ok = gnd <= GROUND_PCT_WARN
    bad = gnd > GROUND_PCT_WARN
    if ok.any():
        ax6.hist(gnd[ok],  bins=bins, color=GREEN, alpha=0.8, label="OK")
    if bad.any():
        ax6.hist(gnd[bad], bins=bins, color=RED,   alpha=0.8, label="Flagged")
    ax6.axvline(GROUND_PCT_WARN, color=ORANGE, lw=1.5, ls='--')
    ax6.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax6.legend(facecolor=PANEL, labelcolor=MUTED, fontsize=7)

    # 7. Point count histogram
    style_ax(ax7, "Point Count Distribution", "N points", "N trees")
    ax7.hist(npts, bins=50, color=BLUE, alpha=0.8)
    ax7.axvline(MIN_POINTS, color=RED, lw=1.2,
                ls='--', label=f"min {MIN_POINTS}")
    ax7.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax7.legend(facecolor=PANEL, labelcolor=MUTED, fontsize=7)

    # 8. Flag summary bar
    style_ax(ax8, "Flag Counts", "", "N trees")
    flag_keys = [
        "flag_ground", "flag_height_mismatch", "flag_dbh_mismatch",
        "flag_slenderness", "flag_height_high", "flag_height_low",
        "flag_dbh_high", "flag_dbh_low", "flag_crown",
        "flag_no_trunk_slice", "flag_too_few_points",
    ]
    flag_counts = [sum(1 for m in full if m.get(k)) for k in flag_keys]
    short_labels = [k.replace("flag_", "").replace("_", "\n")
                    for k in flag_keys]
    bar_colors = [RED, ORANGE, ORANGE, ORANGE, PURPLE, PURPLE,
                  ORANGE, ORANGE, PURPLE, MUTED, RED]
    bars = ax8.barh(short_labels, flag_counts, color=bar_colors, alpha=0.85)
    ax8.tick_params(axis='y', labelsize=7)
    for bar, cnt in zip(bars, flag_counts):
        if cnt > 0:
            ax8.text(cnt + 0.2, bar.get_y() + bar.get_height()/2,
                     str(cnt), va='center', ha='left', color=TEXT, fontsize=7)

    # 9. Flagged tree table (bottom strip)
    ax9.set_facecolor(PANEL)
    ax9.axis('off')
    ax9.set_title(f"⚠  Flagged Trees — top 30 by flag severity  "
                  f"(total flagged: {sum(1 for m in full if m.get('any_flag'))} / {len(full)})",
                  color=ORANGE, fontsize=9, loc='left', pad=5)

    flagged = sorted(
        [m for m in full if m.get("any_flag")],
        key=lambda m: -(
            m.get("flag_ground", 0)*4 +
            m.get("flag_dbh_mismatch", 0)*3 +
            m.get("flag_height_mismatch", 0)*2 +
            m.get("flag_slenderness", 0)
        )
    )[:30]

    cols = ["Tree", "Sp.", "InvDBH", "PcDBH", "InvH",
            "PcH", "Crown", "Gnd%", "Pts", "Flags"]
    col_x = [0.00, 0.08, 0.14, 0.20, 0.27, 0.33, 0.40, 0.47, 0.54, 0.61]
    for cx, lbl in zip(col_x, cols):
        ax9.text(cx, 0.98, lbl, color=MUTED, fontsize=7, fontweight='bold',
                 transform=ax9.transAxes, va='top')

    if flagged:
        for ri, m in enumerate(flagged):
            y = 0.91 - ri * 0.028
            active = [k.replace("flag_", "").upper()
                      for k in flag_keys if m.get(k)]
            row = [
                m["name"], m.get("species", "")[:6],
                f"{m['inv_dbh_cm']:.1f}" if m.get("inv_dbh_cm") else "—",
                f"{m['pc_dbh_cm']:.1f}" if m.get("pc_dbh_cm") else "—",
                f"{m['inv_height_m']:.1f}" if m.get("inv_height_m") else "—",
                f"{m['pc_height_m']:.1f}" if m.get("pc_height_m") else "—",
                f"{m['crown_area_m2']:.1f}" if m.get("crown_area_m2") else "—",
                f"{m['ground_pct']*100:.1f}%" if m.get(
                    "ground_pct") is not None else "—",
                f"{m['n_points']:,}" if m.get("n_points") else "—",
                " | ".join(active[:4]),
            ]
            fc = RED if m.get("flag_ground") else ORANGE
            for cx, val in zip(col_x, row):
                c = fc if cx == col_x[-1] else TEXT
                ax9.text(cx, y, val, color=c, fontsize=6.5,
                         transform=ax9.transAxes, va='top')
    else:
        ax9.text(0.5, 0.5, "✓ No trees flagged", color=GREEN,
                 fontsize=13, ha='center', va='center',
                 transform=ax9.transAxes)

    # Legend
    legend_items = [
        mpatches.Patch(facecolor=BLUE,   label="Clean",
                       edgecolor='none'),
        mpatches.Patch(facecolor=RED,    label="Ground contam.",
                       edgecolor='none'),
        mpatches.Patch(facecolor=ORANGE,
                       label="Outlier / mismatch", edgecolor='none'),
        mpatches.Patch(facecolor=PURPLE,
                       label="Geometry outlier",   edgecolor='none'),
        mpatches.Patch(facecolor=MUTED,  label="No trunk slice",
                       edgecolor='none'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=5,
               facecolor=PANEL, labelcolor=TEXT, fontsize=8,
               edgecolor=BORDER, framealpha=1, bbox_to_anchor=(0.5, 0.005))

    fig.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor=DARK)
    print(f"  Dashboard saved: {out_path}")
    plt.close(fig)


def _make_comparison(df: pd.DataFrame, out_dir: Path):
    """
    For trees that appear more than once (different sources), produce:
      - comparison.csv  : one row per tree, metrics from each source side by side + winner
      - _comparison_dashboard.png : scatter of winner scores, flag counts per source
    """
    import itertools

    if "source" not in df.columns:
        print("  --compare: no 'source' column found, skipping.")
        return
    df = df.dropna(subset=["source"])

    # Only look at trees with 2+ versions
    grouped = df.groupby("name")
    multi = {name: grp for name, grp in grouped if len(grp) > 1}

    if not multi:
        print("  --compare: no duplicate trees found (all trees have exactly one version).")
        return

    print(f"\n  Comparing {len(multi)} trees with multiple versions...")

    # Score each version — lower is better
    # Each flag adds penalty, more points and lower ground% is better
    def score(row):
        penalty = 0
        flag_cols = [c for c in df.columns if c.startswith(
            "flag_") and c != "flag_no_trunk_slice"]
        for c in flag_cols:
            if bool(row.get(c, False)):
                penalty += 1
        # Reward more points (normalised) and lower ground contamination
        pts = float(row.get("n_points", 0) or 0)
        gnd = float(row.get("ground_pct", 1) or 1)
        bonus = pts / 100000   # up to ~1.0 for 100k pts
        return round(penalty - bonus + gnd * 2, 4)

    comp_rows = []
    winner_counts = {}

    for name, grp in sorted(multi.items()):
        grp = grp.copy()
        grp["_score"] = grp.apply(score, axis=1)
        valid_scores = grp["_score"].dropna()
        if valid_scores.empty:
            best_source = grp["source"].iloc[0]  # fallback: just take first
        else:
            best_idx = valid_scores.idxmin()
            best_source = grp.loc[best_idx, "source"]

        row = {"name": name, "n_versions": len(grp), "winner": best_source}

        # Add per-source columns
        for _, r in grp.iterrows():
            src = str(r.get("source", "?"))
            winner_counts[src] = winner_counts.get(src, 0)
            for col in ["pc_height_m", "inv_height_m", "pc_dbh_cm", "inv_dbh_cm",
                        "n_points", "ground_pct", "crown_area_m2", "any_flag", "_score"]:
                row[f"{col}__{src}"] = r.get(col)

        winner_counts[best_source] = winner_counts.get(best_source, 0) + 1
        comp_rows.append(row)

    comp_df = pd.DataFrame(comp_rows).sort_values("name")
    comp_path = out_dir / "comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"  Comparison CSV saved: {comp_path}")

    # ── Winner summary ──
    print(f"\n  Winner counts (best version per tree):")
    for src, cnt in sorted(winner_counts.items(), key=lambda x: -x[1]):
        print(f"    {str(src):25s}: {cnt}")

    # ── Comparison dashboard ──
    sources = df["source"].unique().tolist()
    if len(sources) < 2:
        return

    fig, axes = plt.subplots(2, len(sources), figsize=(7 * len(sources), 10),
                             facecolor=DARK)
    if len(sources) == 2:
        axes = axes.reshape(2, -1)

    fig.suptitle("Conflict Version Comparison", color=TEXT,
                 fontsize=15, fontweight="bold", y=0.99)

    for col_i, src in enumerate(sources):
        src_df = df[df["source"] == src]

        # Top row: height scatter (inv vs pc)
        ax_top = axes[0][col_i]
        ax_top.set_facecolor(PANEL)
        for s in ax_top.spines.values():
            s.set_color(BORDER)
        ax_top.tick_params(colors=MUTED, labelsize=8)
        ax_top.set_title(f"[{src}]\nInv H vs PC H", color=TEXT, fontsize=10)
        ax_top.set_xlabel("Inv height (m)", color=MUTED, fontsize=8)
        ax_top.set_ylabel("PC height (m)", color=MUTED, fontsize=8)

        ih = src_df["inv_height_m"].to_numpy(dtype=float, na_value=np.nan)
        ph = src_df["pc_height_m"].to_numpy(dtype=float, na_value=np.nan)
        clrs = [flag_color(r.to_dict()) for _, r in src_df.iterrows()]
        valid = np.isfinite(ih) & np.isfinite(ph)
        if valid.any():
            ax_top.scatter(ih[valid], ph[valid],
                           c=[c for c, v in zip(clrs, valid) if v],
                           s=30, alpha=0.85, edgecolors='none')
            lim = (0, max(ih[valid].max(), ph[valid].max()) * 1.05)
            ax_top.plot(lim, lim, color=MUTED, lw=1, ls='--', alpha=0.5)
            ax_top.set_xlim(lim)
            ax_top.set_ylim(lim)

        # Bottom row: flag count bar
        ax_bot = axes[1][col_i]
        ax_bot.set_facecolor(PANEL)
        for s in ax_bot.spines.values():
            s.set_color(BORDER)
        ax_bot.tick_params(colors=MUTED, labelsize=7)
        ax_bot.set_title(f"[{src}]  Flag counts\n"
                         f"(n={len(src_df)}, winner={winner_counts.get(src, 0)}x)",
                         color=TEXT, fontsize=9)

        flag_keys = [c for c in df.columns if c.startswith("flag_")]
        fcounts = [int(src_df[k].fillna(False).astype(bool).sum())
                   for k in flag_keys]
        flabels = [k.replace("flag_", "").replace("_", "\n")
                   for k in flag_keys]
        ax_bot.barh(flabels, fcounts, color=ORANGE, alpha=0.8)
        ax_bot.tick_params(axis='y', labelsize=6)
        for i, cnt in enumerate(fcounts):
            if cnt > 0:
                ax_bot.text(cnt + 0.1, i, str(cnt), va='center',
                            color=TEXT, fontsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    dash_path = out_dir / "_comparison_dashboard.png"
    fig.savefig(str(dash_path), dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print(f"  Comparison dashboard saved: {dash_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Batch tree point cloud metrics + QC dashboard")
    ap.add_argument("--folder", required=True,
                    help="Folder containing per-tree .laz / .las files")
    ap.add_argument("--inv",    default=INV_CSV_DEFAULT,
                    help="Inventory CSV path")
    ap.add_argument("--out",    default=None,
                    help="Output directory (default: same as --folder)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel workers for reading LAZ files (default 4)")
    ap.add_argument("--no-dashboard", action="store_true",
                    help="Skip dashboard PNG generation")
    ap.add_argument("--recursive", action="store_true",
                    help="Scan subfolders recursively for LAZ files")
    ap.add_argument("--compare", action="store_true",
                    help="Compare versions of the same tree (use with --recursive on conflicts/)")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        sys.exit(f"Folder not found: {folder}")

    out_dir = Path(args.out) if args.out else folder
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.recursive:
        laz_files = sorted(list(folder.rglob("*.laz")) +
                           list(folder.rglob("*.las")))
    else:
        laz_files = sorted(list(folder.glob("*.laz")) +
                           list(folder.glob("*.las")))
    # Skip the log file if it somehow ends up here
    laz_files = [f for f in laz_files if not f.stem.startswith("_")]
    if not laz_files:
        sys.exit(f"No .laz / .las files found in {folder}")

    print(f"Found {len(laz_files)} LAZ files.")
    print(f"Loading inventory from: {args.inv}")
    inv_data = load_inventory(args.inv)

    # Process (parallel)
    metrics = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(compute_metrics, f, inv_data)                   : f for f in laz_files}
        for i, fut in enumerate(as_completed(futures), 1):
            f = futures[fut]
            try:
                m = fut.result()
                if m:
                    metrics.append(m)
            except Exception as e:
                print(f"  ERROR {f.name}: {e}")
            if i % 50 == 0 or i == len(laz_files):
                print(f"  {i}/{len(laz_files)} processed...")

    if not metrics:
        sys.exit("No metrics computed.")

    # Save CSV
    df = pd.DataFrame(metrics)
    # Sort: flagged first, then by name
    if "any_flag" in df.columns:
        df = df.sort_values(["any_flag", "name"], ascending=[False, True])
    csv_path = out_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nMetrics CSV saved: {csv_path}")

    # Summary
    if "any_flag" in df.columns:
        any_flag = df["any_flag"].fillna(True).astype(
            bool)  # treat missing-inv as flagged
        n_clean = int((~any_flag).sum())
        n_flagged = int(any_flag.sum())
    else:
        n_clean, n_flagged = "?", "?"
    print(f"  Total:   {len(df)}")
    print(f"  Clean:   {n_clean}")
    print(f"  Flagged: {n_flagged}")

    if not args.no_dashboard:
        make_dashboard(metrics, out_dir / "_dashboard.png")

    if args.compare:
        _make_comparison(df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

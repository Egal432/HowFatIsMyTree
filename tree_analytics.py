#!/usr/bin/env python3
"""
Tree Analytics Dashboard (Single File Version)
Computes metrics for a single LAZ file and visualizes them.

Usage:
    pip install laspy[lazrs] numpy matplotlib scipy pandas
    python tree_analytics.py path/to/your_tree.laz
    python tree_analytics.py path/to/your_tree.laz --save --csv metrics.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from scipy.spatial import ConvexHull

try:
    import laspy
except ImportError:
    sys.exit("Install laspy: pip install laspy[lazrs]")

try:
    import pandas as pd
except ImportError:
    sys.exit("Install pandas: pip install pandas")

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Default inventory path (can be overridden with --inv)
INV_CSV = r"E:/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Ecosense/inventory.csv"

# Ground contamination: flag if >X% of points are below this height
GROUND_Z_THRESH = 0.5    # metres above base
GROUND_PCT_WARN = 0.30   # 30% threshold
MIN_POINTS = 50

# Outlier thresholds for flagging
MAX_DBH_CM = 150
MAX_HEIGHT_M = 50
MAX_CROWN_M2 = 200

# ──────────────────────────────────────────────────────────────────────────────

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


def load_inventory(csv_path: str) -> dict:
    """Returns dict keyed by full_id with inventory fields."""
    df = pd.read_csv(csv_path)
    inv = {}
    for _, row in df.iterrows():
        key = str(row["full_id"]).strip()
        inv[key] = {
            "dbh_cm":      round(float(row["diameter_m"]) * 100, 1),
            "inv_height_m": float(row["tls_treeheight"]) if pd.notna(row["tls_treeheight"]) else None,
            "species":     str(row["species"]) if pd.notna(row.get("species", "")) else "",
            "plot_id":     str(row["plot_id"]) if pd.notna(row.get("plot_id", "")) else "",
        }
    print(f"Loaded {len(inv)} trees from inventory CSV.")
    return inv


def compute_metrics(path: Path, inv_data: dict) -> dict | None:
    """Compute point cloud metrics and merge with inventory data."""
    tree_id = path.stem   # e.g. "16_27"

    # Get inventory fields (DBH comes from here, not computed)
    inv = inv_data.get(tree_id)
    if inv is None:
        print(f"  WARNING: {tree_id} not found in inventory CSV — skipping")
        return None

    try:
        las = laspy.read(str(path))
        pts = np.stack([las.x, las.y, las.z], axis=1)
    except Exception as e:
        print(f"  ERROR reading {path.name}: {e}")
        return None

    if len(pts) < MIN_POINTS:
        return None

    # Normalise Z to start at 0
    pts[:, 2] -= pts[:, 2].min()
    pc_height = float(pts[:, 2].max())   # height from point cloud

    # Crown area from convex hull of upper 2/3
    crown_mask = pts[:, 2] > pc_height * 0.33
    crown_pts = pts[crown_mask, :2]
    crown_area = None
    if len(crown_pts) >= 4:
        try:
            hull = ConvexHull(crown_pts)
            crown_area = float(hull.volume)   # in 2D hull.volume = area
        except Exception:
            pass

    # Ground contamination
    low_mask = pts[:, 2] < GROUND_Z_THRESH
    ground_pct = float(low_mask.sum()) / len(pts)

    dbh_cm = inv["dbh_cm"]

    return {
        "name":          tree_id,
        "species":       inv["species"],
        "plot_id":       inv["plot_id"],
        # From inventory CSV
        "dbh_cm":        dbh_cm,
        "inv_height_m":  inv["inv_height_m"],
        # From point cloud
        "pc_height_m":   round(pc_height, 2),
        "crown_area":    round(crown_area, 2) if crown_area else None,
        "n_points":      len(pts),
        "ground_pct":    round(ground_pct, 3),
        "z_p05":         float(np.percentile(pts[:, 2], 5)),
        "z_p50":         float(np.percentile(pts[:, 2], 50)),
        "z_p95":         float(np.percentile(pts[:, 2], 95)),
        # Flags
        "flag_ground":   ground_pct > GROUND_PCT_WARN,
        "flag_dbh":      dbh_cm > MAX_DBH_CM,
        "flag_height":   pc_height > MAX_HEIGHT_M,
        "flag_crown":    crown_area is not None and crown_area > MAX_CROWN_M2,
        # Height mismatch: TLS inventory height vs point cloud height
        "flag_height_mismatch": (
            inv["inv_height_m"] is not None and
            abs(pc_height - inv["inv_height_m"]) > 5.0   # >5m difference
        ),
    }


def flag_color(m: dict) -> str:
    if m["flag_ground"]:
        return RED
    if m["flag_height_mismatch"]:
        return ORANGE
    if m["flag_dbh"]:
        return ORANGE
    if m["flag_height"]:
        return PURPLE
    if m["flag_crown"]:
        return ORANGE
    return BLUE


def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(title, color=TEXT, fontsize=10, pad=8)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(color=BORDER, linewidth=0.5, alpha=0.5)


def make_dashboard(metrics: list, save: bool = False):
    if not metrics:
        sys.exit("No metrics to plot.")

    # Filter to trees that have all fields needed for scatter plots
    plottable = [m for m in metrics if m["crown_area"] is not None]
    all_m = metrics

    dbh = np.array([m["dbh_cm"] for m in plottable])
    pc_heights = np.array([m["pc_height_m"] for m in plottable])
    inv_heights = np.array([m["inv_height_m"] if m["inv_height_m"] else m["pc_height_m"]
                           for m in plottable])
    crown = np.array([m["crown_area"] for m in plottable])
    n_pts = np.array([m["n_points"] for m in plottable])
    ground_pct = np.array([m["ground_pct"] for m in plottable])
    colors = [flag_color(m) for m in plottable]

    all_gpct = np.array([m["ground_pct"] for m in all_m])
    all_npts = np.array([m["n_points"] for m in all_m])
    all_colors = [flag_color(m) for m in all_m]

    fig = plt.figure(figsize=(22, 14), facecolor=DARK)
    fig.suptitle("Tree Point Cloud Analytics Dashboard",
                 color=TEXT, fontsize=20, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38,
                           top=0.93, bottom=0.07, left=0.06, right=0.97)

    ax_dbh_h = fig.add_subplot(gs[0, 0])
    ax_dbh_c = fig.add_subplot(gs[0, 1])
    ax_h_comp = fig.add_subplot(gs[0, 2])
    ax_h_crown = fig.add_subplot(gs[1, 0])
    ax_gnd_h = fig.add_subplot(gs[1, 1])
    ax_gnd_s = fig.add_subplot(gs[1, 2])
    ax_table = fig.add_subplot(gs[2, :])

    ann = fig.text(0.5, 0.025, "Click any point to identify the tree",
                   ha='center', color=MUTED, fontsize=9)

    # ── 1. DBH (inventory) vs Point Cloud Height ──────────────────────────────
    style_ax(ax_dbh_h, "DBH (inventory) vs Point Cloud Height",
             "DBH (cm)", "Point Cloud Height (m)")
    sc1 = ax_dbh_h.scatter(dbh, pc_heights, c=colors, s=40, alpha=0.85,
                           edgecolors='none', picker=True, pickradius=5)
    if len(dbh) > 3:
        z = np.polyfit(dbh, pc_heights, 1)
        xr = np.linspace(dbh.min(), dbh.max(), 100)
        ax_dbh_h.plot(xr, np.polyval(z, xr), color=GREEN, lw=1,
                      linestyle='--', alpha=0.6, label="trend")
        ax_dbh_h.legend(facecolor=PANEL, labelcolor=MUTED, fontsize=7)

    # ── 2. DBH (inventory) vs Crown Area ──────────────────────────────────────
    style_ax(ax_dbh_c, "DBH (inventory) vs Crown Area",
             "DBH (cm)", "Crown Area (m²)")
    sc2 = ax_dbh_c.scatter(dbh, crown, c=colors, s=40, alpha=0.85,
                           edgecolors='none', picker=True, pickradius=5)
    if len(dbh) > 3:
        z = np.polyfit(dbh, crown, 1)
        xr = np.linspace(dbh.min(), dbh.max(), 100)
        ax_dbh_c.plot(xr, np.polyval(z, xr), color=GREEN, lw=1,
                      linestyle='--', alpha=0.6)

    # ── 3. Inventory height vs Point Cloud height (should be ~1:1) ────────────
    style_ax(ax_h_comp, "Inventory Height vs Point Cloud Height\n(should follow 1:1 line)",
             "Inventory TLS Height (m)", "Point Cloud Height (m)")
    sc3 = ax_h_comp.scatter(inv_heights, pc_heights, c=colors, s=40, alpha=0.85,
                            edgecolors='none', picker=True, pickradius=5)
    lims = [min(inv_heights.min(), pc_heights.min()) - 1,
            max(inv_heights.max(), pc_heights.max()) + 1]
    ax_h_comp.plot(lims, lims, color=GREEN, lw=1, linestyle='--',
                   alpha=0.6, label="1:1")
    ax_h_comp.set_xlim(lims)
    ax_h_comp.set_ylim(lims)
    ax_h_comp.legend(facecolor=PANEL, labelcolor=MUTED, fontsize=7)

    # ── 4. Height vs Crown Area coloured by ground % ──────────────────────────
    style_ax(ax_h_crown, "Point Cloud Height vs Crown Area\n(colour = ground point %)",
             "Height (m)", "Crown Area (m²)")
    norm = Normalize(vmin=0, vmax=min(ground_pct.max(), 0.5))
    sc4 = ax_h_crown.scatter(pc_heights, crown,
                             c=ground_pct, cmap="RdYlGn_r", norm=norm,
                             s=40, alpha=0.85, edgecolors='none')
    cb = plt.colorbar(sc4, ax=ax_h_crown, pad=0.02)
    cb.set_label("Ground %", color=MUTED, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
    cb.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))

    # ── 5. Ground % histogram ─────────────────────────────────────────────────
    style_ax(ax_gnd_h,
             f"Ground Point % Distribution\n(flagged if >{GROUND_PCT_WARN*100:.0f}% below {GROUND_Z_THRESH}m)",
             "% points below 0.5m", "N trees")
    bins = np.linspace(0, max(all_gpct.max(), GROUND_PCT_WARN + 0.05), 40)
    ok_mask = all_gpct <= GROUND_PCT_WARN
    bad_mask = all_gpct > GROUND_PCT_WARN
    if ok_mask.any():
        ax_gnd_h.hist(all_gpct[ok_mask],  bins=bins, color=GREEN,
                      alpha=0.8, label=f"OK (≤{GROUND_PCT_WARN*100:.0f}%)")
    if bad_mask.any():
        ax_gnd_h.hist(all_gpct[bad_mask], bins=bins, color=RED,
                      alpha=0.8, label=f"Flagged (>{GROUND_PCT_WARN*100:.0f}%)")
    ax_gnd_h.axvline(GROUND_PCT_WARN, color=ORANGE, lw=1.5, linestyle='--')
    ax_gnd_h.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax_gnd_h.legend(facecolor=PANEL, labelcolor=MUTED, fontsize=8)

    # ── 6. Ground % vs point count ────────────────────────────────────────────
    style_ax(ax_gnd_s,
             "Ground Contamination vs Point Count\n(top-right = most ground points)",
             "N Points", "% below 0.5m")
    ax_gnd_s.scatter(all_npts, all_gpct, c=all_colors, s=30,
                     alpha=0.8, edgecolors='none')
    ax_gnd_s.axhline(GROUND_PCT_WARN, color=ORANGE, lw=1, linestyle='--',
                     label=f"{GROUND_PCT_WARN*100:.0f}% threshold")
    ax_gnd_s.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_gnd_s.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax_gnd_s.legend(facecolor=PANEL, labelcolor=MUTED, fontsize=8)

    # ── 7. Flagged tree table ─────────────────────────────────────────────────
    ax_table.set_facecolor(PANEL)
    ax_table.axis('off')
    for spine in ax_table.spines.values():
        spine.set_color(BORDER)
    ax_table.set_title("⚠  Flagged Trees (top 20 by ground contamination)",
                       color=ORANGE, fontsize=10, loc='left', pad=6)

    flagged = sorted(
        [m for m in all_m if any([m["flag_ground"], m["flag_dbh"],
                                  m["flag_height"], m["flag_crown"],
                                  m["flag_height_mismatch"]])],
        key=lambda m: -m["ground_pct"]
    )[:20]

    col_labels = ["Tree", "Species", "DBH(cm)", "InvH(m)", "PCH(m)",
                  "Crown(m²)", "Points", "Ground%", "Flags"]
    col_x = [0.00, 0.10, 0.20, 0.29, 0.38, 0.47, 0.58, 0.68, 0.78]

    for cx, lbl in zip(col_x, col_labels):
        ax_table.text(cx, 0.97, lbl, color=MUTED, fontsize=8,
                      fontweight='bold', transform=ax_table.transAxes, va='top')

    if flagged:
        for row_i, m in enumerate(flagged):
            y = 0.90 - row_i * 0.043
            flags = []
            if m["flag_ground"]:
                flags.append("GROUND")
            if m["flag_dbh"]:
                flags.append("DBH")
            if m["flag_height"]:
                flags.append("HEIGHT")
            if m["flag_crown"]:
                flags.append("CROWN")
            if m["flag_height_mismatch"]:
                flags.append("H_MISMATCH")

            row_vals = [
                m["name"],
                m["species"],
                str(m["dbh_cm"]),
                str(m["inv_height_m"]) if m["inv_height_m"] else "N/A",
                str(m["pc_height_m"]),
                str(m["crown_area"]) if m["crown_area"] else "N/A",
                f"{m['n_points']:,}",
                f"{m['ground_pct']*100:.1f}%",
                " | ".join(flags),
            ]
            flag_col = RED if m["flag_ground"] else ORANGE
            for cx, val in zip(col_x, row_vals):
                c = flag_col if cx == col_x[-1] else TEXT
                ax_table.text(cx, y, val, color=c, fontsize=7.5,
                              transform=ax_table.transAxes, va='top')
    else:
        ax_table.text(0.5, 0.5, "✓ No trees flagged", color=GREEN,
                      fontsize=14, ha='center', va='center',
                      transform=ax_table.transAxes)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=BLUE,   label="Normal",
                       edgecolor='none'),
        mpatches.Patch(facecolor=RED,
                       label="Ground contamination",  edgecolor='none'),
        mpatches.Patch(facecolor=ORANGE,
                       label="Outlier / H-mismatch", edgecolor='none'),
        mpatches.Patch(facecolor=PURPLE, label="Height outlier",
                       edgecolor='none'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=4,
               facecolor=PANEL, labelcolor=TEXT, fontsize=9,
               edgecolor=BORDER, framealpha=1,
               bbox_to_anchor=(0.5, 0.005))

    # ── Click to identify ─────────────────────────────────────────────────────
    def on_pick(event):
        if not hasattr(event, 'ind') or len(event.ind) == 0:
            return
        i = event.ind[0]
        m = plottable[i]
        msg = (f"  {m['name']}  ({m['species']})  |  "
               f"DBH: {m['dbh_cm']} cm  |  "
               f"InvH: {m['inv_height_m']} m  PCH: {m['pc_height_m']} m  |  "
               f"Crown: {m['crown_area']} m²  |  "
               f"Ground: {m['ground_pct']*100:.1f}%")
        ann.set_text(msg)
        ann.set_color(flag_color(m))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)

    if save:
        # Save in the same directory as the input LAZ file
        out_dir = Path(args.laz_file).parent
        out = out_dir / "_analytics_dashboard.png"
        fig.savefig(str(out), dpi=150, bbox_inches='tight', facecolor=DARK)
        print(f"Dashboard saved: {out}")

    plt.show()


def save_csv(metrics: list, out_path: str):
    import csv
    fields = ["name", "species", "plot_id", "dbh_cm", "inv_height_m",
              "pc_height_m", "crown_area", "n_points", "ground_pct",
              "z_p05", "z_p50", "z_p95",
              "flag_ground", "flag_dbh", "flag_height",
              "flag_crown", "flag_height_mismatch"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(metrics)
    print(f"CSV saved: {out_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Now takes the LAZ file as a positional argument instead of --folder
    ap.add_argument("laz_file", type=str, help="Path to the input LAZ file")
    ap.add_argument("--inv",     default=INV_CSV,    help="Inventory CSV path")
    ap.add_argument("--save",    action="store_true",
                    help="Save dashboard PNG")
    ap.add_argument("--csv",     default=None,
                    help="Save metrics to CSV")
    args = ap.parse_args()

    # Verify input file exists
    input_path = Path(args.laz_file)
    if not input_path.exists():
        sys.exit(f"Error: File not found at {args.laz_file}")

    inv = load_inventory(args.inv)

    # Compute metrics for the single file
    m = compute_metrics(input_path, inv)

    metrics = []
    if m:
        metrics.append(m)
        n_flagged = sum(1 for flag in [m["flag_ground"], m["flag_dbh"], m["flag_height"],
                                       m["flag_crown"], m["flag_height_mismatch"]] if flag)
        print(f"\nLoaded 1 tree. Flagged: {n_flagged}\n")
    else:
        print("No metrics computed (file might not match inventory or had insufficient points).")

    if args.csv and metrics:
        save_csv(metrics, args.csv)

    if metrics:
        make_dashboard(metrics, save=args.save)

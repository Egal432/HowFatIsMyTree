#!/usr/bin/env python3
"""
hd_filter.py  —  Filter trees by H-D plausibility and copy clean ones

Fits a power-law model  H = a * DBH^b  on inventory data, then for each
segmented tree checks whether its point-cloud height falls within
±N_SIGMA of the model prediction. Trees outside the envelope are rejected.

Outputs:
    CLEAN_DIR/           ← .laz files that pass the filter
    _hd_filter.png       ← H-D scatter with fitted curve + rejection envelope
    hd_filter_log.csv    ← every tree with predicted H, residual, verdict

Usage:
    python hd_filter.py --metrics PATH/TO/metrics.csv
                        --laz     PATH/TO/final_trees/
                        --inv     PATH/TO/inventory.csv
                        --out     PATH/TO/clean_trees/
    # add --dry-run to preview without copying
    # add --sigma 3.0 to loosen the envelope (default 2.0)
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from scipy.optimize import curve_fit
    from scipy.stats import norm
except ImportError:
    sys.exit("Install scipy:  pip install scipy")

# ── CONFIG ────────────────────────────────────────────────────────────────────
INV_CSV_DEFAULT     = r"E:\01_UAV_Frey_Group_3\HowFatIsMyTree\datasets\Ecosense\inventory.csv"
METRICS_CSV_DEFAULT = r"E:\01_UAV_Frey_Group_3\HowFatIsMyTree\out\ecosense\final_trees\metrics.csv"
LAZ_DIR_DEFAULT     = r"E:\01_UAV_Frey_Group_3\HowFatIsMyTree\out\ecosense\final_trees"
OUT_DIR_DEFAULT     = r"E:\01_UAV_Frey_Group_3\HowFatIsMyTree\out\ecosense\clean_trees"

N_SIGMA_DEFAULT     = 2.0   # flag if residual > N sigma from model
MIN_DBH_CM          = 3.0   # ignore tiny saplings when fitting
MIN_HEIGHT_M        = 2.0   # ignore stumps when fitting

DARK   = "#0d1117"; PANEL  = "#161b22"; BORDER = "#30363d"
TEXT   = "#e6edf3"; MUTED  = "#8b949e"
BLUE   = "#58a6ff"; GREEN  = "#3fb950"; ORANGE = "#d29922"
RED    = "#f85149"
# ──────────────────────────────────────────────────────────────────────────────


def power_law(dbh, a, b):
    return a * np.power(dbh, b)


def fit_hd_model(inv: pd.DataFrame):
    """
    Fit H = a * DBH^b on inventory data.
    Returns (a, b, sigma) where sigma is the std of log-residuals.
    """
    df = inv.copy()
    df["dbh_cm"]  = pd.to_numeric(df["diameter_m"],    errors="coerce") * 100
    df["height_m"] = pd.to_numeric(df["tls_treeheight"], errors="coerce")
    df = df.dropna(subset=["dbh_cm", "height_m"])
    df = df[(df["dbh_cm"] >= MIN_DBH_CM) & (df["height_m"] >= MIN_HEIGHT_M)]

    if len(df) < 10:
        sys.exit("Not enough inventory data to fit H-D model.")

    dbh = df["dbh_cm"].values
    ht  = df["height_m"].values

    # Initial guess: a=2, b=0.7 (typical forestry values)
    try:
        popt, _ = curve_fit(power_law, dbh, ht, p0=[2.0, 0.7],
                            bounds=([0.1, 0.1], [50.0, 2.0]),
                            maxfev=10000)
    except Exception as e:
        sys.exit(f"Curve fit failed: {e}")

    a, b = popt
    h_pred = power_law(dbh, a, b)

    # Residuals in log space (more homoscedastic than raw)
    log_res = np.log(ht) - np.log(h_pred)
    sigma   = float(np.std(log_res))

    print(f"  H-D model:  H = {a:.4f} * DBH^{b:.4f}")
    print(f"  Residual sigma (log space): {sigma:.4f}")
    print(f"  Fit based on {len(df)} inventory trees")

    return a, b, sigma, df


def main():
    ap = argparse.ArgumentParser(
        description="Filter trees by H-D plausibility, copy clean LAZ files")
    ap.add_argument("--metrics", default=METRICS_CSV_DEFAULT,
                    help="metrics.csv from batch_metrics.py")
    ap.add_argument("--laz",     default=LAZ_DIR_DEFAULT,
                    help="Folder containing final_trees .laz files")
    ap.add_argument("--inv",     default=INV_CSV_DEFAULT,
                    help="Inventory CSV")
    ap.add_argument("--out",     default=OUT_DIR_DEFAULT,
                    help="Output folder for clean trees")
    ap.add_argument("--sigma",   type=float, default=N_SIGMA_DEFAULT,
                    help=f"Rejection threshold in sigma (default {N_SIGMA_DEFAULT})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview without copying files")
    args = ap.parse_args()

    metrics_path = Path(args.metrics)
    laz_dir      = Path(args.laz)
    out_dir      = Path(args.out)

    if not metrics_path.exists():
        sys.exit(f"metrics.csv not found: {metrics_path}")
    if not laz_dir.exists():
        sys.exit(f"LAZ folder not found: {laz_dir}")

    # ── Load data ──
    print("Loading inventory...")
    inv = pd.read_csv(args.inv)
    print("Loading metrics...")
    metrics = pd.read_csv(metrics_path)

    # ── Fit model ──
    print("\nFitting H-D power law on inventory data...")
    a, b, sigma, inv_fit = fit_hd_model(inv)

    # ── Apply to segmented trees ──
    # Need pc_height_m and inv_dbh_cm
    df = metrics.copy()
    df["pc_height_m"] = pd.to_numeric(df["pc_height_m"], errors="coerce")
    df["inv_dbh_cm"]  = pd.to_numeric(df["inv_dbh_cm"],  errors="coerce")

    # Predicted height from model
    df["hd_pred_m"] = df["inv_dbh_cm"].apply(
        lambda d: float(power_law(d, a, b)) if pd.notna(d) and d > 0 else np.nan
    )

    # Log residual: positive = taller than expected, negative = shorter
    df["hd_log_resid"] = np.log(df["pc_height_m"] / df["hd_pred_m"]).where(
        df["pc_height_m"].notna() & df["hd_pred_m"].notna()
    )

    # Z-score in log space
    df["hd_z"] = df["hd_log_resid"] / sigma

    # Verdict
    df["hd_flag"] = df["hd_z"].abs() > args.sigma
    df["hd_verdict"] = df.apply(
        lambda r: (
            "no_dbh"    if pd.isna(r["inv_dbh_cm"]) else
            "no_pc_h"   if pd.isna(r["pc_height_m"]) else
            "rejected"  if r["hd_flag"] else
            "clean"
        ), axis=1
    )

    # ── Stats ──
    n_clean    = int((df["hd_verdict"] == "clean").sum())
    n_rejected = int((df["hd_verdict"] == "rejected").sum())
    n_no_data  = int(df["hd_verdict"].isin(["no_dbh", "no_pc_h"]).sum())

    print(f"\n  Sigma threshold : ±{args.sigma}")
    print(f"  Clean           : {n_clean}")
    print(f"  Rejected        : {n_rejected}")
    print(f"  No data (skip)  : {n_no_data}")

    if args.dry_run:
        print("\n  Rejected trees:")
        rej = df[df["hd_verdict"] == "rejected"].sort_values("hd_z")
        for _, r in rej.iterrows():
            direction = "TOO TALL" if r["hd_z"] > 0 else "TOO SHORT"
            print(f"    {str(r['name']):20s}  "
                  f"pc_h={r['pc_height_m']:.1f}m  "
                  f"pred={r['hd_pred_m']:.1f}m  "
                  f"z={r['hd_z']:.2f}  {direction}")

    # ── Save log CSV ──
    log_cols = ["name", "inv_dbh_cm", "pc_height_m",
                "hd_pred_m", "hd_log_resid", "hd_z", "hd_flag", "hd_verdict"]
    log_cols = [c for c in log_cols if c in df.columns]
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "hd_filter_log.csv"
        df[log_cols].sort_values("hd_z").to_csv(log_path, index=False)
        print(f"\n  Log saved: {log_path}")

    # ── Copy clean files ──
    n_copied = 0
    n_missing = 0
    for _, row in df[df["hd_verdict"] == "clean"].iterrows():
        name = str(row["name"])
        src  = laz_dir / f"{name}.laz"
        if not src.exists():
            src = laz_dir / f"{name}.las"
        if not src.exists():
            n_missing += 1
            continue
        if not args.dry_run:
            dst = out_dir / src.name
            try:
                shutil.copy2(str(src), str(dst))
                n_copied += 1
            except Exception as e:
                print(f"  ERROR copying {name}: {e}")

    if n_missing:
        print(f"  WARNING: {n_missing} clean trees had no .laz file found")
    if not args.dry_run:
        print(f"  Copied {n_copied} clean trees → {out_dir}")

    # ── Plot ──
    _make_plot(inv_fit, df, a, b, sigma, args.sigma, out_dir, args.dry_run)

    print("\nDone.")


def _make_plot(inv_fit, df, a, b, sigma, n_sigma, out_dir, dry_run):
    fig = plt.figure(figsize=(16, 8), facecolor=DARK)
    fig.suptitle(f"H-D Filter  (H = {a:.3f} · DBH^{b:.3f},  ±{n_sigma}σ envelope)",
                 color=TEXT, fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32,
                           left=0.07, right=0.97, top=0.91, bottom=0.10)

    # ── Left: H-D scatter with envelope ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL)
    for s in ax1.spines.values(): s.set_color(BORDER)
    ax1.tick_params(colors=MUTED, labelsize=8)
    ax1.set_title("H-D scatter: inventory fit + PC height verdicts",
                  color=TEXT, fontsize=10, pad=6)
    ax1.set_xlabel("Inventory DBH (cm)", color=MUTED, fontsize=9)
    ax1.set_ylabel("Height (m)",          color=MUTED, fontsize=9)
    ax1.grid(color=BORDER, lw=0.5, alpha=0.5)

    # Inventory points (grey)
    ax1.scatter(inv_fit["dbh_cm"], inv_fit["height_m"],
                c=MUTED, s=12, alpha=0.4, edgecolors='none', label="Inventory")

    # Fitted curve + envelope
    dbh_range = np.linspace(df["inv_dbh_cm"].dropna().min(),
                            df["inv_dbh_cm"].dropna().max(), 200)
    h_fit  = power_law(dbh_range, a, b)
    h_up   = h_fit * np.exp( n_sigma * sigma)
    h_down = h_fit * np.exp(-n_sigma * sigma)

    ax1.plot(dbh_range, h_fit,  color=BLUE,   lw=2,   label="H = a·DBH^b")
    ax1.plot(dbh_range, h_up,   color=ORANGE, lw=1.2, ls='--',
             label=f"+{n_sigma}σ envelope")
    ax1.plot(dbh_range, h_down, color=ORANGE, lw=1.2, ls='--',
             label=f"−{n_sigma}σ envelope")
    ax1.fill_between(dbh_range, h_down, h_up, color=ORANGE, alpha=0.08)

    # Segmented trees coloured by verdict
    clean = df[df["hd_verdict"] == "clean"]
    rej   = df[df["hd_verdict"] == "rejected"]
    nodat = df[df["hd_verdict"].isin(["no_dbh", "no_pc_h"])]

    if len(clean):
        ax1.scatter(clean["inv_dbh_cm"], clean["pc_height_m"],
                    c=GREEN, s=20, alpha=0.7, edgecolors='none', label="Clean")
    if len(rej):
        ax1.scatter(rej["inv_dbh_cm"], rej["pc_height_m"],
                    c=RED, s=25, alpha=0.9, edgecolors='none', label="Rejected")
    if len(nodat):
        ax1.scatter(nodat["inv_dbh_cm"].fillna(5), nodat["pc_height_m"].fillna(0),
                    c=MUTED, s=15, alpha=0.4, marker='x', label="No data")

    ax1.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=7, loc="upper left")

    # ── Right: Z-score distribution ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL)
    for s in ax2.spines.values(): s.set_color(BORDER)
    ax2.tick_params(colors=MUTED, labelsize=8)
    ax2.set_title("Z-score distribution (log residuals)",
                  color=TEXT, fontsize=10, pad=6)
    ax2.set_xlabel("Z-score  (0 = perfectly on model)", color=MUTED, fontsize=9)
    ax2.set_ylabel("N trees", color=MUTED, fontsize=9)
    ax2.grid(color=BORDER, lw=0.5, alpha=0.5)

    z_vals = df["hd_z"].dropna().values
    if len(z_vals):
        bins = np.linspace(z_vals.min(), z_vals.max(), 60)
        ok_mask  = np.abs(z_vals) <= n_sigma
        bad_mask = np.abs(z_vals) >  n_sigma
        if ok_mask.any():
            ax2.hist(z_vals[ok_mask],  bins=bins, color=GREEN, alpha=0.8,
                     label=f"Clean (|z| ≤ {n_sigma})")
        if bad_mask.any():
            ax2.hist(z_vals[bad_mask], bins=bins, color=RED,   alpha=0.8,
                     label=f"Rejected (|z| > {n_sigma})")
        ax2.axvline( n_sigma, color=ORANGE, lw=1.5, ls='--')
        ax2.axvline(-n_sigma, color=ORANGE, lw=1.5, ls='--')
        ax2.axvline(0, color=MUTED, lw=1, ls=':')

        # Overlay normal curve
        xg = np.linspace(bins[0], bins[-1], 200)
        yg = norm.pdf(xg, 0, 1) * len(z_vals) * (bins[1] - bins[0])
        ax2.plot(xg, yg, color=BLUE, lw=1.5, ls='--', alpha=0.6, label="N(0,1)")
        ax2.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=7)

    png_path = out_dir / "_hd_filter.png"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(png_path), dpi=150, bbox_inches='tight', facecolor=DARK)
        print(f"  Plot saved: {png_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

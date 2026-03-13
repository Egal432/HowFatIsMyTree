#!/usr/bin/env python3
import pathlib
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# --------------------------------------------------------------
# Helper – convert Windows paths to WSL paths


def win_to_wsl(p: str) -> pathlib.Path:
    p = p.strip().replace("\\", "/").strip('"').strip("'")
    if len(p) >= 2 and p[1] == ":":
        return pathlib.Path(f"/mnt/{p[0].lower()}{p[2:]}")
    return pathlib.Path(p)


# --------------------------------------------------------------
# Input files
csv_win = r"E:\01_UAV_Frey_Group_3\HowFatIsMyTree\datasets\Mathisleweiher\trees_valid\dbh_valid_fixed.csv"
txt_win = r"E:\01_UAV_Frey_Group_3\HowFatIsMyTree\datasets\Mathisleweiher\cc_viz\dbh_circles.txt"

csv_path = win_to_wsl(csv_win)
txt_path = win_to_wsl(txt_win)

print(f"CSV → {csv_path}")
print(f"TXT → {txt_path}")

# --------------------------------------------------------------
# 1️⃣  Read the CSV (robust version)


def read_dbh_csv(p: pathlib.Path) -> pd.DataFrame:
    rows = []
    for raw_line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip().strip("|").strip()
        if not line:
            continue
        parts = [seg.strip() for seg in line.split("|")]
        parts = [seg for seg in parts if seg]
        if len(parts) >= 4 and parts[0].startswith("tree_"):
            rows.append(parts[:4])
    if not rows:
        raise RuntimeError(
            "No parsable data rows found in the CSV – check file content.")
    df = pd.DataFrame(rows, columns=["full_id", "x", "y", "diameter_m"])
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["diameter_m"] = pd.to_numeric(df["diameter_m"], errors="coerce")
    df = df.dropna(subset=["x", "y", "diameter_m"]).reset_index(drop=True)
    return df


csv_df = read_dbh_csv(csv_path)
print(f"CSV loaded: {len(csv_df)} rows, columns: {list(csv_df.columns)}")

# --------------------------------------------------------------
# 2️⃣  Read the TXT (same logic as before)


def read_txt(p: pathlib.Path) -> pd.DataFrame:
    rows = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                continue
    if not rows:
        raise RuntimeError("No valid 5‑column rows found in the TXT file.")
    return pd.DataFrame(rows, columns=["x_txt", "y_txt", "z_txt", "pred_instance", "dbh_cm"])


txt_df = read_txt(txt_path)
print(f"TXT loaded: {len(txt_df)} rows, columns: {list(txt_df.columns)}")

# --------------------------------------------------------------
# 3️⃣  Prepare for matching
txt_df["x_int"] = txt_df["x_txt"].round().astype(int)
txt_df["y_int"] = txt_df["y_txt"].round().astype(int)

# --------------------------------------------------------------
# 4️⃣  Exact join on integer X/Y
merged = pd.merge(
    csv_df,
    txt_df[["x_int", "y_int", "z_txt"]],
    left_on=["x", "y"],
    right_on=["x_int", "y_int"],
    how="left"
)

# --------------------------------------------------------------
# 5️⃣  KD‑Tree fallback for missing Z
missing = merged["z_txt"].isna()
if missing.any():
    print(f"{missing.sum()} rows missed the exact join – trying KD‑Tree (tolerance=0.5 m).")
    tree = cKDTree(txt_df[["x_txt", "y_txt"]].values)
    query_pts = merged.loc[missing, ["x", "y"]].astype(float).values
    dists, idx = tree.query(query_pts, k=1)
    tol = 0.5
    ok = dists <= tol
    merged.loc[missing, "z_txt"] = np.where(
        ok,
        txt_df.iloc[idx]["z_txt"].values,
        np.nan
    )
    still_missing = merged["z_txt"].isna().sum()
    print(f"After KD‑Tree: {still_missing} rows still have no Z value.")

# --------------------------------------------------------------
# 6️⃣  Clean up and write output
merged = merged.drop(columns=["x_int", "y_int"])
merged = merged.rename(columns={"z_txt": "z_center"})
out_path = csv_path.with_name(csv_path.stem + "_withZ.csv")
merged.to_csv(out_path, index=False, float_format="%.6f")
print(f"\n✅ Finished – merged file written to: {out_path}")
print(
    f"Rows in output: {len(merged)} (rows with a Z value have non‑NaN in `z_center`).")

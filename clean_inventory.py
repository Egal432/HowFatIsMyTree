#!/usr/bin/env python3
"""
clean_inventory.py
------------------
Read the original, badly‑formatted inventory.csv, repair broken rows,
extract the useful columns (X, Y, optional Z, TreeID, Species, URL, …)
and write a clean CSV ready for CloudCompare or GIS.

Requirements:
    pip install pandas python-dateutil
"""

import csv
import re
from pathlib import Path
import pandas as pd

# ----------------------------------------------------------------------
# USER SETTINGS --------------------------------------------------------
# ----------------------------------------------------------------------
INPUT_FILE = Path(
    "E:\\01_UAV_Frey_Group_3\\HowFatIsMyTree\\datasets\\Ecosense\\inventory.csv")
OUTPUT_FILE = Path("inventory_clean.csv")    # result
DELIMITERS = [",", "|"]                     # possible delimiters in the file
KEEP_COLUMNS = [
    "TreeID", "Species", "URL", "X", "Y", "Z", "uuid", "loop"
]   # add / remove column names you want to keep

# ----------------------------------------------------------------------
# Helper functions ------------------------------------------------------
# ----------------------------------------------------------------------


def guess_delimiter(line: str) -> str:
    """Return the most likely delimiter for a given line."""
    counts = {d: line.count(d) for d in DELIMITERS}
    # pick the delimiter with the highest count (ignoring zeros)
    return max(counts, key=counts.get)


def split_line(line: str) -> list:
    """Split a line using the guessed delimiter and strip whitespace."""
    delim = guess_delimiter(line)
    parts = [p.strip() for p in line.split(delim)]
    # Remove empty strings that appear because of consecutive delimiters
    return [p for p in parts if p]


def merge_broken_rows(lines: list) -> list:
    """
    The original file sometimes breaks a logical row over several
    physical lines (e.g. a pipe‑delimited table that wraps). This
    function attempts to concatenate lines until we have a plausible
    number of columns (at least 10) or we encounter a line that starts
    with a numeric ID.
    """
    merged = []
    buffer = ""

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Start a new buffer if we see a leading integer (tree index)
        if re.match(r"^\s*\d+\s*[|,]", line):
            if buffer:
                merged.append(buffer)
            buffer = line
        else:
            # continuation of previous row
            buffer += " " + line

    if buffer:
        merged.append(buffer)

    return merged


def parse_rows(lines: list) -> list[dict]:
    rows = []
    for line in lines:
        # First, split on both commas and pipes, then strip empties
        parts = [p.strip() for p in re.split(r"[|,]", line) if p.strip()]

        # Skip lines that are completely empty after splitting
        if not parts:
            continue

        # Identify the numeric TreeID – it is the first integer we encounter
        tree_id = ""
        for token in parts:
            if token.isdigit():
                tree_id = token
                break

        # Species is usually the token right after the ID
        species = ""
        if tree_id:
            try:
                idx = parts.index(tree_id)
                species = parts[idx + 1] if idx + 1 < len(parts) else ""
            except ValueError:
                pass

        # URL – look for the first token that starts with "http"
        url = next((p for p in parts if p.startswith("http")), "")

        # X and Y – the two large numbers (~416xxx and ~5346xxx)
        x_val, y_val = "", ""
        large_numbers = [p for p in parts if re.fullmatch(r"\d{5,7}\.\d+", p)]
        if len(large_numbers) >= 2:
            x_val, y_val = large_numbers[0], large_numbers[1]

        # Z – the small number (< 1) that appears after X/Y
        z_val = "0"
        for p in parts:
            try:
                f = float(p)
                if 0 <= f <= 5 and p not in (x_val, y_val):
                    z_val = p
                    break
            except ValueError:
                continue

        # UUID – token that starts with "uuid:"
        uuid = next((p for p in parts if p.startswith("uuid:")), "")

        # Loop – token that looks like "loop[XX]"
        loop = next((p for p in parts if re.search(r"loop\[\d+\]", p)), "")

        rows.append({
            "TreeID": tree_id,
            "Species": species,
            "URL": url,
            "X": x_val,
            "Y": y_val,
            "Z": z_val,
            "uuid": uuid,
            "loop": loop,
        })
    return rows
# ----------------------------------------------------------------------
# Main routine ---------------------------------------------------------
# ----------------------------------------------------------------------


def main():
    # Load raw lines
    raw_lines = INPUT_FILE.read_text(encoding="utf-8").splitlines()

    # 1️⃣ Merge broken rows
    merged_lines = merge_broken_rows(raw_lines)

    # 2️⃣ Parse each row into a dict
    parsed = parse_rows(merged_lines)

    # 3️⃣ Create DataFrame, keep only desired columns
    df = pd.DataFrame(parsed)

    # Ensure X, Y, Z are numeric (coerce errors to NaN)
    for col in ["X", "Y", "Z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where X or Y are missing (cannot be used in CloudCompare)
    df.dropna(subset=["X", "Y"], inplace=True)

    # Keep only the columns the user asked for (or the default set)
    cols_to_save = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[cols_to_save]

    # 4️⃣ Write clean CSV
    df.to_csv(OUTPUT_FILE, index=False, float_format="%.6f")
    print(f"✅ Clean CSV written to {OUTPUT_FILE}")
    print(
        f"   {len(df)} valid tree records kept out of {len(merged_lines)} raw lines.")


if __name__ == "__main__":
    main()

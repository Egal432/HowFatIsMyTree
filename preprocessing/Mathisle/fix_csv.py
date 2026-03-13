import pandas as pd
from pathlib import Path

# Input and Output paths
input_path = "/mnt/e/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Mathisleweiher/trees_valid/dbh_valid.csv"
output_path = "/mnt/e/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Mathisleweiher/trees_valid/dbh_valid_fixed.csv"


def main():
    # 1. Load the CSV
    # The source looks like it might use pipes '|' or is just standard.
    # Try reading normally first.
    try:
        df = pd.read_csv(input_path)
    except:
        # Try with pipe separator if standard doesn't work
        df = pd.read_csv(input_path, sep='|')

    # 2. Ensure required columns exist
    # We map: cx -> x, cy -> y, dbh_m -> diameter_m
    required_map = {
        "cx": "x",
        "cy": "y",
        "dbh_m": "diameter_m"
    }

    for old, new in required_map.items():
        if old in df.columns:
            if new in df.columns:
                # Avoid duplicate names, drop old if new somehow exists
                df = df.drop(columns=[new])
            df = df.rename(columns={old: new})

    # 3. Fix the IDs
    # The "PredInstance" column must perfectly match the filename stem.
    # Example: tree_000002.laz -> PredInstance should be "tree_000002"
    if "laz_file" in df.columns and "PredInstance" in df.columns:
        print("Updating PredInstance to match filenames...")
        # Extract stem (filename without extension)
        df["PredInstance"] = df["laz_file"].apply(lambda x: Path(str(x)).stem)

    # 4. Select and Order columns for the final CSV
    final_cols = ["PredInstance", "x", "y", "diameter_m"]

    # Check if all exist
    missing = [c for c in final_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns after cleaning: {missing}")
        print(f"Available columns: {df.columns}")
        return

    df_final = df[final_cols]

    # 5. Save
    df_final.to_csv(output_path, index=False)
    print(f"Successfully created fixed CSV: {output_path}")
    print(f"Total rows: {len(df_final)}")

    # Print a sample to verify
    print("\nSample of first 3 rows:")
    print(df_final.head(3))


if __name__ == "__main__":
    main()

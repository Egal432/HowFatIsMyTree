import laspy
import numpy as np

# Path to your original file
laz_path = "datasets/Ecosense/ecosense.laz"

unique_instances = set()
chunk_size = 1_000_000  # Read 1 million points at a time to save RAM

print(f"Reading {laz_path} in chunks...")

try:
    # Open the file (this creates a LasReader)
    with laspy.open(laz_path) as reader:

        # CORRECTION: Access point_format via reader.header
        dims = reader.header.point_format.dimension_names

        if "PredInstance" not in dims:
            print(f"[ERROR] 'PredInstance' dimension not found in file header.")
            print(f"Available dimensions: {dims}")
        else:
            # Iterate through the file in chunks
            for points in reader.chunk_iterator(chunk_size):
                # Extract the PredInstance array for this chunk
                # 'points' is a LasData object, so we can access dimensions directly
                pred_data = points["PredInstance"]

                # Keep unique values
                unique_instances.update(np.unique(pred_data))

            print(
                f"\n[RESULT] Found {len(unique_instances)} unique PredInstance IDs in the file.")

except Exception as e:
    print(f"[ERROR] An error occurred: {e}")

#!/usr/bin/env/ Rscript
# split_multis.R
# Arguments:
# 1. Path to multi-tree LAZ file
# 2. Path to manifest CSV (columns: tree_id, x_32632, y_32632)
# 3. Output directory

library(lidR)
library(CspStandSegmentation)

# Parse Arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript split_multis.R <input_laz> <manifest_csv> <output_dir>")
}

input_laz  <- args[1]
manifest   <- args[2]
output_dir <- args[3]

# Create output dir if missing
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Load Data
las <- readLAS(input_laz)
inv_df <- read.csv(manifest)

# Check if data loaded
if (is.null(las) | nrow(inv_df) == 0) {
  stop("Error loading LAZ or manifest.")
}

# Perform Segmentation
# Note: CspStandSegmentation (csf/stand) usually updates the class.
# Using standard lidR watershed or Csp functions depending on what is available.
# Here we assume standard 'tree_segmentation' from the package or lidR watershed.
# Adjust parameters (voxel_size, height) if necessary for your specific data.

# Attempt 1: Using lidR's standard watershed if CspStandSeg is complex to call blindly
# or using Csp's tree_segmentation wrapper if exposed.
# We verify coordinates inside.

# We will use the inventory coordinates to 'guide' or 'match' the segments.
# Even if the segmentation creates noise, we will match it to the nearest inventory tree.

# Assuming standard 3D clustering or watershed segmentation:
segmentation <- segment_trees(las, stm_watershed(), th = 0) 

# --- Matching Logic ---
# Get centroids of segments
segment_metrics <- metrics(segmentation, func = list(x_mean = mean(Z), y_mean = mean(Y), z_mean = mean(Z), isn_count = length(Z)))
# Note: lidR aliases X = x, Z = check units. In lidR, X and Y are stored in the coordinates.
# We need to calculate centroids manually for sure.

coords_xyz <- las@data[, .(x, y, z, class)]
coords_xyz[, segment_id := class]

# Calculate centroids per segment ID
centroids <- coords_xyz[, .(cx = mean(x), cy = mean(y), cz = mean(z)), by = segment_id]

# Match Centroids to Inventory Trees (Nearest Neighbor)
# Simple Euclidean distance in XY
inv_coords <- as.matrix(inv_df[, c("x_32632", "y_32632")])
seg_coords  <- as.matrix(centroids[, c("cx", "cy")])

# Find nearest tree for each segment
library(sp)
# Using brute force or proxy if N is small (it is small here)
dist_matrix <- distances(rbind(inv_coords, seg_coords))
inv_n <- nrow(inv_coords)
seg_n <- nrow(seg_coords)

# Subset only segment-inventory distances:
dist_matrix <- dist_matrix[(inv_n+1):(inv_n+seg_n), 1:inv_n]

# Find min index
nearest_tree_idx <- apply(dist_matrix, 1, which.min)

# Assign Segment to Tree ID
centroids$assigned_tree_id <- inv_coords$tree_id[nearest_tree_idx]

# Export Logic
# We will replace the 'segment_id' in the LAS object with the 'tree_id'.
# This effectively merges orphans (hits their nearest tree).
# If a segment is noise and matches nowhere close? We already assigned to nearest.

# We need to replace the ID in the LAS data.
# The 'segmentation' object has a 'tree' or 'class' attribute.

# Merging the cluster IDs back into point cloud
# In lidR, 'segment_trees' adds an 'treeID' column.
# We will overwrite 'treeID' with our assigned 'assigned_tree_id'.

# Note: The column name might be 'treeID' or 'segmentID'. 
# Check the structure.
if("treeID" %in% names(segmentation@data)) {
  # We must map segment IDs (from segmentation) to Tree IDs (from centroids)
  # Create a lookup: old_segment_id -> new_tree_id
  lookup <- setDT(as.data.frame(centroids[, c("segment_id", "assigned_tree_id")]))
  setcolslookup(segmentation@data, lookup)
  
  # Name the final column 'treeID' if not present, or ensure it matches expectation
  # Actually CspStandSeg typically adds a 'class' or similar.
  
} else {
  stop("Segmentation did not produce expected ID column.")
}

# Now we have a point cloud where points have the correct 'tree_id'.
# We will split this cloud into individual LAZ files based on this ID.

unique_tree_ids <- unique(segmentation@data$assigned_tree_id)

for (tid in unique_tree_ids) {
  if (is.na(tid)) next
  
  # Filter points
  # sub_las = filter_poi(segmentation, assigned_tree_id == tid)
  # Function might differ by version.
  
  # Manual subsetting for safety:
  idx <- which(segmentation@data$assigned_tree_id == tid)
  sub_cloud <- las@data[idx]
  
  # Create new LAS
  out_name <- paste0(tid, "_ec.laz")
  out_path <- file.path(output_dir, out_name)
  
  # Write
  sub_las <- LAS(sub_cloud, header = las@header)
  writeLAS(sub_las, out_path)
}
# ============================================================
#  END-TO-END SEGMENTATION IN R (Moving Window)
#  - Reads tiles from the catalog (simulating big cloud)
#  - Uses Buffered Grid to avoid tiling artifacts
#  - Outputs: {UID}.laz files + detailed CSV log
# ============================================================

library(lidR)
library(CspStandSegmentation)
library(data.table)
library(sf)
library(tidyverse)

# --- CONFIGURATION ---------------------------------------------------------
# Paths pointing to your existing folders
INV_CSV_PATH   <- "E:/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Ecosense/inventory.csv"
TILES_DIR      <- "E:/01_UAV_Frey_Group_3/HowFatIsMyTree/out/ecosense/tiles"
OUTPUT_DIR     <- "E:/01_UAV_Frey_Group_3/HowFatIsMyTree/out/ecosense/r_segmentation"

# Algorithm Settings
GRID_SIZE      <- 50.0   # Size of the processing grid cell (meters)
BUFFER_SIZE    <- 15.0   # Buffer around the cell (meters) - prevents edge cuts
Z_BUFFER       <- 0.5    # Search radius (m) for estimating trunk Z height
VOXEL_SIZE     <- 0.2    # CSP Voxel size (keep at 0.3 for stability)

# --- TEST MODE SETTINGS -----------------------------------------------------
TEST_MODE      <- FALSE   # TRUE = Processes only the top 1 busiest grid cell
MAX_CELLS_TEST <- 1      # How many cells to process in test mode

# -----------------------------------------------------------------------------

if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

# 1. Load Inventory & Clean Data
cat("Loading inventory...\n")
inv <- fread(INV_CSV_PATH)

# --- CLEANING START ---
# Ensure columns are numeric (converts text/empty strings to NA)
inv[, x_32632 := as.numeric(x_32632)]
inv[, y_32632 := as.numeric(y_32632)]

# Count how many rows are trash
n_total <- nrow(inv)
bad_rows <- sum(is.na(inv$x_32632) | is.na(inv$y_32632) | !is.finite(inv$x_32632) | !is.finite(inv$y_32632))

# Filter: Keep only rows with valid X and Y
inv <- inv[!is.na(x_32632) & !is.na(y_32632) & is.finite(x_32632) & is.finite(y_32632)]

cat(sprintf("  Cleaned inventory: Removed %d invalid rows (kept %d).\n", bad_rows, nrow(inv)))
# --- CLEANING END ---

# Ensure columns exist (if they were missing from csv, we catch them now)
if (!"full_id" %in% names(inv)) inv[, full_id := as.character(.I)] 
if (nrow(inv) == 0) stop("No valid data left after cleaning inventory!")

# Convert to SimpleFeatures for spatial gridding
# This will no longer crash because we removed the NAs
inv_sf <- st_as_sf(inv, coords = c("x_32632", "y_32632"), crs = 32632, remove = FALSE)
# 2. Create Spatial Grid Processing Map
cat("Generating processing grid...\n")
# Make grid covering the extent of inventory
grid_obj <- st_make_grid(inv_sf, cellsize = GRID_SIZE, what = "polygons")
grid_df  <- st_sf(grid_df_id = seq_along(grid_obj), geometry = grid_obj)

# Join inventory seeds to grid cells
# This tells us which cell is responsible for which seeds
inv_joined <- st_join(inv_sf, grid_df, join = st_is_within_distance, dist = 0)

# Count seeds per grid cell to help testing
seed_counts <- inv_joined %>% 
  st_set_geometry(NULL) %>% 
  count(grid_df_id, name = "n_seeds") %>% 
  arrange(desc(n_seeds))

cat(sprintf("  Generated %d grid cells total.\n", nrow(grid_df)))

# 3. Apply Test Mode Filter
# We select the cell(s) with the most seeds to test
if (TEST_MODE) {
  cat("\n========================================\n")
  cat(" TEST MODE ACTIVE\n")
  cat("========================================\n")
  selected_cells <- head(seed_counts$grid_df_id, MAX_CELLS_TEST)
  cat(sprintf("  Selected %d busiest grid cells for validation.\n", length(selected_cells)))
  cat("  Modifying grid_df to test subset only.\n")
  grid_df <- grid_df[selected_cells, ]
}

# 4. Setup LidR Catalog (The "Big Cloud")
cat("Building LidR catalog from tiles...\n")
if (!dir.exists(TILES_DIR)) stop("Tiles directory not found.")
ctg <- lidR::catalog(TILES_DIR)
opt_progress(ctg) <- FALSE
opt_filter(ctg)   <- "" 

# 5. Enable Retries for cloud read errors
#catalog_options <- lidR::LidROptions(n越大 = 10, dataset_chunk_size = 1, filter_chunk_size = 1)

# 6. Main Processing Loop
# We will iterate over the GRID COLUMNS (efficient processing), but for simplicity
# let's just loop over the cells in our filtered list.
results_log <- data.table()

cat("Starting segmentation loop...\n")
flush.console()

# Get full grid list we are processing
cell_ids <- grid_df$grid_df_id

for (k in seq_along(cell_ids)) {
  
  gid <- cell_ids[k]
  
  # --- A. Determine Bounding Box ---
  # Get the grid cell polygon
  cell_poly <- grid_df$geometry[[k]]
  bounds    <- st_bbox(cell_poly)
  
  # Define CORE area
  xmin_core <- bounds$xmin
  xmax_core <- bounds$xmax
  ymin_core <- bounds$ymin
  ymax_core <- bounds$ymax
  
  # Define BUFFERED extraction area
  xmin_ext  <- xmin_core - BUFFER_SIZE
  xmax_ext  <- xmax_core + BUFFER_SIZE
  ymin_ext  <- ymin_core - BUFFER_SIZE
  ymax_ext  <- ymax_core + BUFFER_SIZE
  
  # --- B. Identify Seeds ---
  # 1. Seeds strictly inside CORE (we are responsible for saving these)
  core_seeds <- inv_joined %>%
    filter(grid_df_id == gid) %>% 
    st_set_geometry(NULL)
  
  # 2. Seeds inside BUFFER (we need these for segmentation context, but won't save them if they are outside core)
  # (Actually, we can just grab seeds within the extended buffer for the algorithm)
  buf_seeds_all <- inv_joined %>% 
    filter(x_32632 >= xmin_ext & x_32632 <= xmax_ext & 
             y_32632 >= ymin_ext & y_32632 <= ymax_ext) %>% 
    st_set_geometry(NULL)
  
  if (nrow(buf_seeds_all) == 0) {
    cat(sprintf("  [Cell %d/%d] No seeds. Skipping.\n", k, length(cell_ids)))
    next
  }
  
  cat(sprintf("  [Cell %d/%d] Extracting window (Seeds: %d)...\n", k, length(cell_ids), nrow(buf_seeds_all)))
  flush.console()
  
  # --- C. Extract Points from Catalog ---
  # This reads from tiles automatically and merges overlapping areas
  las_chunk <- tryCatch({
    # clip_rectangle returns a LAS object
    lidR::clip_rectangle(ctg, xmin_ext, ymin_ext, xmax_ext, ymax_ext)
  }, error = function(e) {
    cat(sprintf("    [ERROR] Data extraction failed: %s\n", e$message))
    return(NULL)
  })
  
  if (is.null(las_chunk) || lidR::npoints(las_chunk) == 0) {
    cat("    [SKIP] Empty or failed extraction.\n")
    log_entry <- list(grid_id = gid, status = "EMPTY_POINTS", n_trees_expected = nrow(core_seeds), n_trees_saved = 0)
    results_log <- rbind(results_log, log_entry)
    next
  }
  
  # --- D. Prepare Seed Map for CSP ---
  # Build map for ALL seeds in buffer (so CSP can separate them)
  # We need to map the subset of buf_seeds that we actually provided
  
  # To ensure alignment, let's re-filter buf_seeds_all against the actual loaded points 
  # to be super safe, but filtering by box is usually enough.
  
  # Identify which seeds in our "inventory list" are in our "buf_seeds_all"
  # (Variable names can be tricky, let's be explicit)
  
  seed_x   <- buf_seeds_all$x_32632
  seed_y   <- buf_seeds_all$y_32632
  seed_uid <- buf_seeds_all$full_id
  
  # Estimate Z (Height) for seeds
  z_vec <- sapply(seq_along(seed_x), function(j) {
    dx <- las_chunk@data$X - seed_x[j]
    dy <- las_chunk@data$Y - seed_y[j]
    nearby <- sqrt(dx^2 + dy^2) < Z_BUFFER
    
    if (sum(nearby) == 0) {
      return(unname(quantile(las_chunk@data$Z, probs = 0.05)))
    }
    unname(quantile(las_chunk@data$Z[nearby], probs = 0.05))
  })
  
  # Create seed_map for CSP
  # TreeID is sequential 1..N.
  seed_map <- data.frame(
    X      = seed_x,
    Y      = seed_y,
    Z      = z_vec,
    TreeID = 1:length(seed_x)
  )
  
  # --- E. Run CSP ---
  cat("    Running CSP...\n")
  seg_result <- tryCatch({
    las_geom <- CspStandSegmentation::add_geometry(las_chunk, n_cores = 4)
    CspStandSegmentation::csp_cost_segmentation(las_geom, seed_map, Voxel_size = VOXEL_SIZE, V_w = 0.5)
  }, error = function(e) {
    cat(sprintf("    [ERROR] CSP failed: %s\n", e$message))
    return(NULL)
  })
  
  if (is.null(seg_result)) {
    log_entry <- list(grid_id = gid, status = "CSP_CRASH", n_trees_expected = nrow(core_seeds), n_trees_saved = 0)
    results_log <- rbind(results_log, log_entry)
    next
  }
  
  # --- F. Save Results ---
  found_ids <- seg_result@data$TreeID
  found_ids <- found_ids[!is.na(found_ids) & found_ids != 0]
  
  n_saved <- 0
  
  for (tid in sort(unique(found_ids))) {
    
    # Get the real UID corresponding to this TreeID
    if (tid > length(seed_uid)) next # Safety
    real_uid <- seed_uid[tid]
    
    # STICT CHECK: Only save if this seed is in the CORE cell!
    # This prevents saving the same tree 4 times.
    is_core <- core_seeds$full_id == real_uid
    
    if (any(is_core)) {
      tree_las <- lidR::filter_poi(seg_result, TreeID == tid)
      if (lidR::npoints(tree_las) > 0) {
        out_path <- file.path(OUTPUT_DIR, paste0(real_uid, ".laz"))
        lidR::writeLAS(tree_las, out_path)
        n_saved <- n_saved + 1
      }
    }
  }
  
  cat(sprintf("    [OK] Saved %d trees.\n", n_saved))
  log_entry <- list(grid_id = gid, status = "OK", n_trees_expected = nrow(core_seeds), n_trees_saved = n_saved)
  results_log <- rbind(results_log, log_entry)
}

# 7. Final Summary
log_dt <- results_log
cat("\n========== SUMMARY ==========\n")
print(log_dt)

# Write Log CSV
log_path <- file.path(OUTPUT_DIR, "segmentation_log.csv")
fwrite(log_dt, log_path)
cat(sprintf("Log saved to: %s\n", log_path))

cat("\nDone!\n")

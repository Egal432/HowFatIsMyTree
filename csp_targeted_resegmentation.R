# ============================================================
#  CSP Targeted Re-Segmentation — Full Script
#  Output: one LAZ file per tree, named by full_id
#  e.g. 16_27.laz, 16_28.laz, 16_29.laz ...
#
#  Key fix: csp_cost_segmentation() expects a data.frame with
#  columns X, Y, TreeID — NOT a raster object.
# ============================================================

library(lidR)
library(CspStandSegmentation)
library(data.table)

############################################
# 1️⃣  PATHS
############################################
inv_csv_path <- "E:/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Ecosense/inventory.csv"
seg_csv_path <- "E:/01_UAV_Frey_Group_3/HowFatIsMyTree/out/ecosense/labels.csv"

# Root folder containing the "trees/" subfolder
laz_root <- "E:/01_UAV_Frey_Group_3/HowFatIsMyTree/out/ecosense"

# All output goes here — one .laz per tree
out_folder <- "E:/01_UAV_Frey_Group_3/HowFatIsMyTree/out/ecosense/trees_by_id"

if (!dir.exists(out_folder)) dir.create(out_folder, recursive = TRUE)

############################################
# 2️⃣  READ & JOIN CSVs
############################################
cat("Reading CSVs...\n"); flush.console()

inv <- fread(inv_csv_path)
seg <- fread(seg_csv_path)

# inv_index is 0-based row index into inventory
inv[, inv_index := .I - 1]

seg_joined <- merge(seg,
                    inv[, .(inv_index, full_id, x_32632, y_32632)],
                    by    = "inv_index",
                    all.x = TRUE)

cat("Segmentation rows:", nrow(seg_joined), "\n")
cat("Unmatched rows:   ", sum(is.na(seg_joined$x_32632)), "\n\n")
flush.console()

############################################
# 3️⃣  SPLIT INTO SINGLE vs MULTI-TREE
############################################
seg_groups <- seg_joined[, .(
  n_trees  = .N,
  full_ids = list(full_id),
  xs       = list(x_32632),
  ys       = list(y_32632),
  laz_file = first(tree_pointcloud_file),
  note     = first(note)
), by = predinstance]

single_segs <- seg_groups[n_trees == 1]
multi_segs  <- seg_groups[n_trees >  1]

cat(sprintf("Total segments:       %d\n", nrow(seg_groups)))
cat(sprintf("Single-tree (rename): %d\n", nrow(single_segs)))
cat(sprintf("Multi-tree  (re-run): %d\n", nrow(multi_segs)))
flush.console()

############################################
# 4️⃣  SINGLE-TREE SEGMENTS — copy & rename only
############################################
cat("\n--- Processing single-tree segments (copy + rename) ---\n")
flush.console()

single_ok      <- 0
single_missing <- 0

for (i in seq_len(nrow(single_segs))) {
  
  row      <- single_segs[i]
  full_id  <- unlist(row$full_ids)[1]
  laz_path <- file.path(laz_root, row$laz_file)
  out_path <- file.path(out_folder, paste0(full_id, ".laz"))
  
  if (!file.exists(laz_path)) {
    cat(sprintf("  SKIP (not found): %s\n", row$laz_file))
    flush.console()
    single_missing <- single_missing + 1
    next
  }
  
  file.copy(laz_path, out_path, overwrite = TRUE)
  single_ok <- single_ok + 1
}

cat(sprintf("Single-tree done: %d copied, %d missing\n\n", single_ok, single_missing))
flush.console()

############################################
# 5️⃣  MULTI-TREE SEGMENTS — Force Re-Run
############################################
cat("--- Processing multi-tree segments (Force Re-Segmentation) ---\n")
flush.console()

results_log <- vector("list", nrow(multi_segs))

for (i in seq_len(nrow(multi_segs))) {
  row      <- multi_segs[i]
  pred_id  <- row$predinstance
  n_trees  <- row$n_trees
  
  # Unlist coordinates
  xs <- as.numeric(unlist(row$xs))
  ys <- as.numeric(unlist(row$ys))
  ids <- unlist(row$full_ids)
  
  # Basic safety
  valid_idx <- !is.na(xs) & !is.na(ys)
  xs <- xs[valid_idx]
  ys <- ys[valid_idx]
  ids <- ids[valid_idx]
  
  if (length(xs) == 0) {
    cat(sprintf("  [%d/%d] pred_%s — Skipped (Invalid Seeds)\n", i, nrow(multi_segs), pred_id))
    results_log[[i]] <- list(predinstance = pred_id, status = "invalid_seeds")
    next
  }
  
  laz_path <- file.path(laz_root, row$laz_file)
  
  # ── FILE CHECK ─────────────────────────────────────────────
  if (!file.exists(laz_path)) {
    cat(sprintf("  [%d/%d] pred_%s — Source file MISSING\n", i, nrow(multi_segs), pred_id))
    flush.console()
    results_log[[i]] <- list(predinstance = pred_id, status = "file_not_found")
    next
  }
  
  cat(sprintf("  [%d/%d] pred_%s — Processing %d trees\n", i, nrow(multi_segs), pred_id, length(xs)))
  flush.console()
  
  # ── LOAD POINTS ────────────────────────────────────────────
  las_seg <- tryCatch(
    lidR::readLAS(laz_path),
    error = function(e) { message("    Read error: ", e$message); NULL }
  )
  if (is.null(las_seg) || lidR::npoints(las_seg) == 0) {
    cat("    SKIP — empty or unreadable LAZ\n")
    flush.console()
    results_log[[i]] <- list(predinstance = pred_id, status = "empty")
    next
  }
  
  # ── BUILD SEED MAP ────────────────────────────────────────
  # OPTIMIZED QUANTILE FIX:
  # We use unname() to remove the "5%" tag that causes the dimension crash.
  z_vec <- sapply(seq_along(xs), function(j) {
    dx <- las_seg@data$X - xs[j]
    dy <- las_seg@data$Y - ys[j]
    nearby <- sqrt(dx^2 + dy^2) < 0.5
    if (sum(nearby) == 0) {
      # Robust: Lowest 5% of the whole cloud segment
      return(unname(quantile(las_seg@data$Z, probs = 0.05)))
    }
    # Robust: Lowest 5% of points near trunk (Ground)
    return(unname(quantile(las_seg@data$Z[nearby], probs = 0.05)))
  })
  
  # Force clean data.frame - CSP is strict about types
  seed_map <- data.frame(
    X      = as.numeric(xs),
    Y      = as.numeric(ys),
    Z      = as.numeric(z_vec),
    TreeID = seq_along(xs)
  )
  
  # ── BOUNDS CHECK ───────────────────────────────────────────
  x_min <- las_seg@header@PHB[["Min X"]]
  x_max <- las_seg@header@PHB[["Max X"]]
  y_min <- las_seg@header@PHB[["Min Y"]]
  y_max <- las_seg@header@PHB[["Max Y"]]
  
  in_bounds <- seed_map$X >= x_min & seed_map$X <= x_max &
    seed_map$Y >= y_min & seed_map$Y <= y_max
  
  if (sum(in_bounds) == 0 && nrow(seed_map) > 0) {
    cat("    ERROR: Seeds out of bounds. Saving unsplit original.\n")
    for (j in seq_along(ids)) {
      file.copy(laz_path, file.path(out_folder, paste0(ids[j], ".laz")), overwrite = TRUE)
    }
    results_log[[i]] <- list(predinstance = pred_id, status = "seeds_out_of_bounds")
    next
  }
  
  if (sum(in_bounds) < nrow(seed_map)) {
    cat(sprintf("    Warning: %d seeds out of bounds, filtering.\n", sum(!in_bounds)))
    seed_map <- seed_map[in_bounds, ]
    ids      <- ids[in_bounds]
  }
  cat(sprintf("    Seed map prepared: %d trees.\n", nrow(seed_map)))
  flush.console()
  
  # ── RUN CSP ────────────────────────────────────────────────
  cat("    Running add_geometry...\n"); flush.console()
  
  seg_result <- tryCatch({
    las_geom <- CspStandSegmentation::add_geometry(las_seg, n_cores = 8)
    cat("    Running csp_cost_segmentation...\n"); flush.console()
    
    las_segmented <- CspStandSegmentation::csp_cost_segmentation(
      las_geom,
      seed_map,
      Voxel_size = 0.05,   # 5cm Resolution
      V_w        = 0.5,     
      N_cores    = 8        
    )
    return(las_segmented)
    
  }, error = function(e) {
    message(sprintf("    CSP ERROR: %s", e$message))
    return(NULL)
  })
  
  # ── FALLBACK (IF CSP FAILS) ────────────────────────────────
  if (is.null(seg_result)) {
    cat("    CSP FAILED -> Saving unsplit original.\n")
    flush.console()
    for (j in seq_along(ids)) {
      file.copy(laz_path, file.path(out_folder, paste0(ids[j], ".laz")), overwrite = TRUE)
    }
    results_log[[i]] <- list(predinstance = pred_id, status = "csp_failed")
    next
  }
  
  # ── SPLIT BY TreeID ────────────────────────────────────────
  tree_ids_found <- sort(unique(seg_result@data$TreeID))
  tree_ids_found <- tree_ids_found[!is.na(tree_ids_found) & tree_ids_found != 0]
  
  n_saved <- 0
  for (t in tree_ids_found) {
    las_tree <- lidR::filter_poi(seg_result, TreeID == t)
    if (lidR::npoints(las_tree) == 0) next
    
    tree_label <- if (t <= length(ids)) ids[t] else paste0("pred", pred_id, "_tree", t)
    out_path <- file.path(out_folder, paste0(tree_label, ".laz"))
    
    lidR::writeLAS(las_tree, out_path)
    n_saved <- n_saved + 1
  }
  
  results_log[[i]] <- list(predinstance = pred_id,
                           status       = "ok",
                           n_trees_in   = length(xs),
                           n_trees_out  = n_saved)
}

############################################
# 6️⃣  SUMMARY
############################################
cat("\n========== Summary ==========\n")
flush.console()

log_dt <- rbindlist(results_log, fill = TRUE)

ok_count      <- sum(log_dt$status == "ok",                       na.rm = TRUE)
failed_count  <- sum(log_dt$status == "csp_failed_kept_original", na.rm = TRUE)
missing_count <- sum(log_dt$status == "file_not_found",           na.rm = TRUE)
empty_count   <- sum(log_dt$status == "empty",                    na.rm = TRUE)

cat(sprintf("Single-tree segments copied:    %d\n", single_ok))
cat(sprintf("Multi-tree re-segmented (ok):   %d\n", ok_count))
cat(sprintf("Multi-tree fallback (CSP fail): %d\n", failed_count))
cat(sprintf("Files not found:                %d\n", missing_count))
cat(sprintf("Empty segments skipped:         %d\n", empty_count))

total_laz <- length(list.files(out_folder, pattern = "\\.laz$"))
cat(sprintf("\nTotal LAZ files in output: %d\n", total_laz))
cat(sprintf("Output folder: %s\n",               out_folder))

cat("\n✅ Done! One LAZ per tree in output folder.\n")
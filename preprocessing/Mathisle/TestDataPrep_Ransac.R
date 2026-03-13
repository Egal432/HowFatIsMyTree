library(lidR)
library(terra)
library(data.table)
library(conicfit)
library(sf)
# ── Option A: All TIFs in a folder ──────────────────────────────────────────
path <- r"(E:\01_UAV_Frey_Group_3\GeoKram\full_dgm)"

tif_files <- list.files(path, 
                        pattern = "\\.tif$", 
                        full.names = TRUE)

# Mosaic into a single virtual raster (no file written, lives in memory)
dtm <- mosaic(sprc(lapply(tif_files, rast)))

# Check CRS matches your LAZ — important!
crs(dtm)
#1.1
laz_path <- r"(E:\01_UAV_Frey_Group_3\HowFatIsMyTree\datasets\Mathisleweiher\mathisleweiher.laz)"
#1.2
cat("DTM tiles loaded:", length(tif_files), "\n")
cat("DTM extent:", as.character(ext(dtm)), "\n")
cat("DTM CRS:", crs(dtm, proj=TRUE), "\n")
cat("Terrain range:", minmax(dtm)[1], "to", minmax(dtm)[2], "m\n")

# Make sure it covers your point cloud extent
las_ext <- ext(min(dt$X), max(dt$X), min(dt$Y), max(dt$Y))
cat("LAZ extent:", as.character(las_ext), "\n")
# They should overlap — if not, CRS mismatch is the likely culprit
# ── 2. Read a raw Z band wide enough to capture BH across terrain variation
# e.g. if terrain varies ±5m, load Z = (min_elev + 1.0) to (max_elev + 1.6)
# Safest: just load a 3m raw Z window — adjust to your terrain
# You can get the terrain range from the DTM first:
z_min <- minmax(dtm)[1] + 1.0   # ground_min + just below BH
z_max <- minmax(dtm)[2] + 1.6   # ground_max + just above BH

cat("Loading raw Z between:", z_min, "and", z_max, "\n")

las <- readLAS(laz_path,
               select = "xyz",
               filter = paste("-keep_z", z_min, z_max))

cat("Points loaded:", nrow(las@data), "\n")

# ── 3. Normalize Z using DTM ─────────────────────────────────────────────────
dt <- as.data.table(las@data)

# Extract DTM elevation at each XY (vectorized — fast)
ground_z <- terra::extract(dtm, cbind(dt$X, dt$Y))[, 1]

dt[, Z_norm := Z - ground_z]

# Now filter to breast height
dt_bh <- dt[Z_norm >= 1.2 & Z_norm <= 1.4]

cat("Points in BH slice after normalization:", nrow(dt_bh), "\n")

# Check what the PredInstance column is actually called
cat("Column names:", names(dt_bh), "\n")
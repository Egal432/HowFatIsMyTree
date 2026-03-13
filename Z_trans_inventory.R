library(readr)

# Load inventory
inv   <- "E:/01_UAV_Frey_Group_3/HowFatIsMyTree/datasets/Ecosense/inventory.csv"
inv <- fread(INV_CSV_PATH)

# Option A: Place points at ground level (flat, matching cloud elevation)
ground_z <- 517.770  # match the Zg from CloudCompare

# Option B: Place at ground + tree height (tips of trees)
# inv$z_out <- ground_z + inv$tls_treeheight

inv$z_out <- ground_z  # use Option A

# Export for CloudCompare (X Y Z, space-separated, no header)
write.table(
  inv[, c("x_32632", "y_32632", "z_out")],
  file = "E:/01_UAV_Frey_Group_3/HowFatIsMyTree/inventory_for_cc.txt",
  row.names = FALSE,
  col.names = FALSE,
  sep = " "
)

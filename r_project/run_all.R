# =============================================================================
# run_all.R
# Master runner — sources all analysis scripts in order.
# Set DATA_DIR and OUTPUT_DIR before running, or export them as environment
# variables. Defaults assume you are running from the repo root.
# =============================================================================

DATA_DIR   <- Sys.getenv("SVH_DATA_DIR",   unset = "data/raw")
OUTPUT_DIR <- Sys.getenv("SVH_OUTPUT_DIR", unset = "output")

scripts <- c(
  "r_project/01_data_loading.R",
  "r_project/02_feature_engineering.R",
  "r_project/03_plan_a_sdoh_funnel.R",
  "r_project/04_plan_b_off_hours.R",
  "r_project/05_plan_c_continuity.R",
  "r_project/06_plan_d_geography.R"
)

for (s in scripts) {
  message("\n==============================")
  message("Running: ", s)
  message("==============================")
  source(s, local = FALSE)
}

message("\nAll scripts complete. Plots saved to: ", OUTPUT_DIR, "/plots/")

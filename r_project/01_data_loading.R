# =============================================================================
# 01_data_loading.R
# Load the seven raw CSVs, apply basic type coercions, and export a merged
# working dataset to data/processed/.
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
})

DATA_DIR      <- get0("DATA_DIR",      envir = .GlobalEnv, ifnotfound = "data/raw")
PROCESSED_DIR <- "data/processed"
dir.create(PROCESSED_DIR, showWarnings = FALSE, recursive = TRUE)

message("Loading raw CSVs from: ", DATA_DIR)

read_svh <- function(filename) {
  path <- file.path(DATA_DIR, filename)
  if (!file.exists(path)) stop("Missing file: ", path)
  read_csv(path, show_col_types = FALSE) |> clean_names()
}

encounters          <- read_svh("encounters.csv")
patients            <- read_svh("patients.csv")
diagnosis           <- read_svh("diagnosis.csv")
departments         <- read_svh("departments.csv")
providers           <- read_svh("providers.csv")
social_determinants <- read_svh("social_determinants.csv")
tigercensus         <- read_svh("tigercensuscodes.csv")

# ------ Minimal type fixes --------------------------------------------------

# encounters: cast binary flag columns
flag_cols <- c(
  "is_ed_visit", "is_hospital_admission", "is_hospital_outpatient_visit",
  "is_inpatient_admission", "is_observation", "is_outpatient_face_to_face_visit"
)
for (col in flag_cols) {
  if (col %in% names(encounters)) {
    encounters[[col]] <- encounters[[col]] == "Yes"
  }
}

# encounters: numeric admission/discharge components
num_cols <- c(
  "admit_year", "admit_month", "admit_day", "admit_hour", "admit_minute",
  "discharge_year", "discharge_month", "discharge_day",
  "discharge_hour", "discharge_minute"
)
for (col in num_cols) {
  if (col %in% names(encounters)) {
    suppressWarnings(encounters[[col]] <- as.numeric(encounters[[col]]))
  }
}

# tigercensus: numeric lat/lon/population
tigercensus <- tigercensus |>
  mutate(
    population_value = as.numeric(population_value),
    centlat          = as.numeric(centlat),
    centlon          = as.numeric(centlon)
  )

# ------ Persist raw-loaded tables -------------------------------------------
message("Saving loaded tables to ", PROCESSED_DIR)
saveRDS(encounters,          file.path(PROCESSED_DIR, "encounters_raw.rds"))
saveRDS(patients,            file.path(PROCESSED_DIR, "patients_raw.rds"))
saveRDS(diagnosis,           file.path(PROCESSED_DIR, "diagnosis_raw.rds"))
saveRDS(departments,         file.path(PROCESSED_DIR, "departments_raw.rds"))
saveRDS(providers,           file.path(PROCESSED_DIR, "providers_raw.rds"))
saveRDS(social_determinants, file.path(PROCESSED_DIR, "sdoh_raw.rds"))
saveRDS(tigercensus,         file.path(PROCESSED_DIR, "tigercensus_raw.rds"))

message("01_data_loading.R complete.")

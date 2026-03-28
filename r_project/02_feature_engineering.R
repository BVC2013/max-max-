# =============================================================================
# 02_feature_engineering.R
# Build all engineered features defined in the README.
# Reads processed RDS files; writes encounters_engineered.rds + CSV.
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
})

PROCESSED_DIR <- "data/processed"

encounters          <- readRDS(file.path(PROCESSED_DIR, "encounters_raw.rds"))
patients            <- readRDS(file.path(PROCESSED_DIR, "patients_raw.rds"))
diagnosis           <- readRDS(file.path(PROCESSED_DIR, "diagnosis_raw.rds"))
social_determinants <- readRDS(file.path(PROCESSED_DIR, "sdoh_raw.rds"))
tigercensus         <- readRDS(file.path(PROCESSED_DIR, "tigercensus_raw.rds"))

# =============================================================================
# 1. Core Parsing / Flags
# =============================================================================

message("Engineering core flags...")

enc <- encounters |>
  mutate(
    # Encounter date
    encounter_date = as.Date(date),

    # Admission / discharge timestamps
    admission_timestamp = make_datetime(
      year   = admit_year,
      month  = admit_month,
      day    = admit_day,
      hour   = admit_hour,
      min    = admit_minute
    ),
    discharge_timestamp = make_datetime(
      year   = discharge_year,
      month  = discharge_month,
      day    = discharge_day,
      hour   = discharge_hour,
      min    = discharge_minute
    ),

    # Duration
    encounter_duration_hours = as.numeric(
      difftime(discharge_timestamp, admission_timestamp, units = "hours")
    ),
    encounter_duration_hours = if_else(
      encounter_duration_hours < 0 | encounter_duration_hours > 8760,
      NA_real_,
      encounter_duration_hours
    ),

    # ED / hospital flags
    is_ed_visit_flag          = is_ed_visit,
    is_hospital_admission_flag = is_hospital_admission,

    # Off-hours: weekend OR hour < 7 OR hour >= 17
    is_off_hours = (
      wday(admission_timestamp) %in% c(1, 7) |  # Sunday=1, Saturday=7
        admit_hour < 7 |
        admit_hour >= 17
    )
  )

# =============================================================================
# 2. MyChart Active (patient-level)
# =============================================================================

message("Joining patient-level features (MyChart, geography)...")

pat <- patients |>
  mutate(
    my_chart_active = str_to_lower(my_chart_status) == "activated",
    census_block_fips_clean = if_else(
      str_detect(census_block_group_fips_code, "Unspecified") | is.na(census_block_group_fips_code),
      NA_character_,
      census_block_group_fips_code
    ),
    geo_data_available = !is.na(census_block_fips_clean)
  ) |>
  select(durable_key, my_chart_active, vital_status,
         census_block_fips_clean, geo_data_available)

enc <- enc |>
  left_join(pat, by = c("patient_durable_key" = "durable_key"))

# =============================================================================
# 3. SDoH Aggregated (patient-level)
# =============================================================================

message("Engineering SDoH barrier flags...")

positive_answers <- c(
  "Yes", "Sometimes", "Often", "Always",
  "Hard", "Very Hard",
  "A little bit", "Somewhat", "Quite a bit", "Very much",
  "Less than once a week", "Never",
  "I choose not to answer"
)

sdoh_flags <- social_determinants |>
  mutate(
    domain_lc      = str_to_lower(domain),
    positive_screen = str_trim(answer_text) %in% positive_answers
  ) |>
  group_by(patient_durable_key) |>
  summarise(
    has_transport_need      = any(str_detect(domain_lc, "transport")   & positive_screen, na.rm = TRUE),
    has_financial_strain    = any(str_detect(domain_lc, "financial")   & positive_screen, na.rm = TRUE),
    has_housing_instability = any(str_detect(domain_lc, "housing")     & positive_screen, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(
    any_sdoh_barrier = has_transport_need | has_financial_strain | has_housing_instability
  )

enc <- enc |>
  left_join(sdoh_flags, by = "patient_durable_key") |>
  mutate(across(c(has_transport_need, has_financial_strain,
                  has_housing_instability, any_sdoh_barrier),
                ~ replace_na(.x, FALSE)))

# =============================================================================
# 4. Join Diagnosis (DiagnosisValue for journey tracking)
# =============================================================================

message("Joining diagnosis values...")

diag_slim <- diagnosis |>
  select(diagnosis_key, diagnosis_value, diagnosis_name, group_code, group_name)

enc <- enc |>
  left_join(diag_slim, by = c("primary_diagnosis_key" = "diagnosis_key"))

# =============================================================================
# 5. Journey Features (per PatientDurableKey × DiagnosisValue)
# =============================================================================

message("Computing journey features — this may take a minute on large data...")

enc <- enc |>
  # Only compute journeys where we have a real diagnosis
  group_by(patient_durable_key, diagnosis_value) |>
  arrange(encounter_date, .by_group = TRUE) |>
  mutate(
    visit_number              = row_number(),
    journey_start_date        = first(encounter_date),
    cumulative_days_in_journey = as.numeric(encounter_date - journey_start_date),
    days_since_last_visit     = as.numeric(encounter_date - lag(encounter_date)),
  ) |>
  ungroup() |>
  # Is_Incident_Case: >180 days after the patient's first-ever observed visit
  # (across ALL diagnoses, not just within the current journey)
  group_by(patient_durable_key) |>
  mutate(
    first_ever_visit = min(encounter_date, na.rm = TRUE),
    is_incident_case = as.numeric(encounter_date - first_ever_visit) > 180,
  ) |>
  ungroup() |>
  group_by(patient_durable_key, diagnosis_value) |>
  arrange(encounter_date, .by_group = TRUE) |>
  mutate(

    # Follow-up: any later visit in same journey?
    has_follow_up = visit_number < max(visit_number),

    # Abandonment: no follow-up
    raw_journey_abandoned     = !has_follow_up,
    is_true_abandonment_risk  = raw_journey_abandoned & (vital_status != "DECEASED"),

    # Care transfer: attending provider changes within journey
    is_care_transfer = !is.na(attending_provider_durable_key) &
      attending_provider_durable_key != lag(attending_provider_durable_key, default = first(attending_provider_durable_key)),
    cumulative_care_transfers = cumsum(coalesce(as.integer(is_care_transfer), 0L))
  ) |>
  ungroup() |>
  mutate(
    is_true_abandonment_risk = replace_na(is_true_abandonment_risk, FALSE)
  )

# =============================================================================
# 6. Geography — join tigercensus for population & coordinates
# =============================================================================

message("Joining tigercensus geography...")

geo_slim <- tigercensus |>
  select(geoid, population_value, centlat, centlon) |>
  mutate(
    is_rural_census_block = population_value < 1500
  )

enc <- enc |>
  left_join(geo_slim, by = c("census_block_fips_clean" = "geoid"))

# =============================================================================
# 7. Save engineered dataset
# =============================================================================

message("Saving engineered dataset...")
saveRDS(enc, file.path(PROCESSED_DIR, "encounters_engineered.rds"))
write_csv(enc, file.path(PROCESSED_DIR, "encounters_engineered.csv"))

message("02_feature_engineering.R complete. Rows: ", nrow(enc), " | Cols: ", ncol(enc))

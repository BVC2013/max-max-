# SVH Patient Journey Analysis

> **Stormont Vail Health — ASA DataFest 2026**  
> Longitudinal patient-journey analysis using encounter, social-determinants, geography, and clinical data.

---

## Project Overview

This repository contains two parallel analysis projects — one in **R** and one in **Python** — that ingest the seven SVH CSV files, engineer a comprehensive feature set, and test four analytically-grounded plans for understanding patient journeys:

| Plan | Name | Core Question |
|------|------|---------------|
| A | Access Friction Funnel | Do SDoH barriers (transport, finance, housing) predict follow-up failure? |
| B | Off-Hours Throughput Strain | Does time-of-arrival shift ED utilisation and encounter duration? |
| C | Continuity Breakdown | Does provider/care-transfer fragmentation escalate utilisation? |
| D | Local Care Mismatch Map | Where do high-ED-risk + low-MyChart + low-density clusters appear? |

---

## Repository Structure

```
max-max-/
├── README.md
├── data/
│   └── raw/                  ← Place the 7 CSV files here (not committed)
│       ├── encounters.csv
│       ├── patients.csv
│       ├── diagnosis.csv
│       ├── departments.csv
│       ├── providers.csv
│       ├── social_determinants.csv
│       └── tigercensuscodes.csv
├── r_project/
│   ├── SVH_Analysis.Rproj
│   ├── run_all.R
│   ├── 01_data_loading.R
│   ├── 02_feature_engineering.R
│   ├── 03_plan_a_sdoh_funnel.R
│   ├── 04_plan_b_off_hours.R
│   ├── 05_plan_c_continuity.R
│   └── 06_plan_d_geography.R
└── python_project/
    ├── requirements.txt
    ├── run_all.py
    ├── src/
    │   ├── data_loading.py
    │   ├── feature_engineering.py
    │   ├── plan_a_sdoh_funnel.py
    │   ├── plan_b_off_hours.py
    │   ├── plan_c_continuity.py
    │   └── plan_d_geography.py
    └── output/
        └── plots/            ← Generated matplotlib figures saved here
```

---

## Data Setup

1. Obtain the seven CSV files from the DataFest data package.
2. Place them in `data/raw/` (this directory is git-ignored).
3. Run either project (R or Python) — processed artefacts are written to `data/processed/` and plots to `output/plots/`.

---

## Engineered Features

### Core Parsing / Flags
| Feature | Description |
|---|---|
| `EncounterDate` | Parsed date from the `Date` field |
| `AdmissionTimestamp` | Combined `AdmitYear`/`Month`/`Day`/`Hour`/`Minute` |
| `DischargeTimestamp` | Combined `DischargeYear`/`Month`/`Day`/`Hour`/`Minute` |
| `Encounter_Duration_Hours` | `DischargeTimestamp − AdmissionTimestamp` in hours |
| `IsEdVisit_Flag` | Boolean from `IsEdVisit == "Yes"` |
| `IsHospitalAdmission_Flag` | Boolean from `IsHospitalAdmission == "Yes"` |
| `Is_Off_Hours` | `TRUE` if weekend OR `AdmitHour < 7` OR `AdmitHour >= 17` |
| `MyChart_Active` | `TRUE` when `MyChartStatus == "activated"` |

### SDoH Aggregated (patient-level)
| Feature | Description |
|---|---|
| `Has_Transport_Need` | Any positive screen in the Transportation domain |
| `Has_Financial_Strain` | Any positive screen in the Financial Resource Strain domain |
| `Has_Housing_Instability` | Any positive screen in the Housing Stability domain |
| `Any_SDoH_Barrier` | `TRUE` if any of the three above are `TRUE` |

### Journey (DiagnosisValue-based, per patient)
| Feature | Description |
|---|---|
| `Visit_Number` | Sequential rank within a patient×DiagnosisValue journey |
| `Journey_Start_Date` | Date of first encounter in the journey |
| `Cumulative_Days_In_Journey` | Days from `Journey_Start_Date` to current encounter |
| `Days_Since_Last_Visit` | Days elapsed since the previous encounter in the journey |
| `Is_Incident_Case` | `TRUE` if > 180 days after first-ever observed visit for this patient |
| `Has_Follow_Up` | `TRUE` if a later encounter exists in the same journey |
| `Raw_Journey_Abandoned` | `TRUE` if no follow-up and journey is not ongoing |
| `Is_True_Abandonment_Risk` | `Raw_Journey_Abandoned AND VitalStatus != "DECEASED"` |
| `Is_Care_Transfer` | Transfer detected (change in attending provider within journey) |
| `Cumulative_Care_Transfers` | Running count of transfers within the journey |

### Geography
| Feature | Description |
|---|---|
| `CensusBlockGroupFipsCode_Clean` | Cleaned FIPS code (strip `*Unspecified`) |
| `Is_Rural_Census_Block` | `TRUE` if `PopulationValue < 1500` |
| `Geo_Data_Available` | `TRUE` when a valid FIPS code is present |

---

## Four Plans — Analysis Detail

### Plan A — Access Friction Funnel (SDoH → Follow-up Failure)
**Hard trend:** Among ED/hospital encounters, compare `Is_True_Abandonment_Risk` rates stratified by `Has_Transport_Need`, `Has_Financial_Strain`, and `Has_Housing_Instability`.  
**Test:** Logistic regression + chi-square; odds-ratios with 95 % CIs.  
**Circumstantial support:** CMS HRSN/SDOH framework links these three barriers to access and utilisation patterns.

### Plan B — Off-Hours Throughput Strain (Time-of-Arrival Effect)
**Hard trend:** Compare ED rate and `Encounter_Duration_Hours` by `Is_Off_Hours` and `AdmitHour` bucket (0–6, 7–16, 17–23).  
**Tests:** Welch t-test on duration; proportion test on ED rate; hourly heat-map.  
**Circumstantial support:** AHRQ ED boarding literature and hospital operations research.

### Plan C — Continuity Breakdown (Provider Fragmentation)
**Hard trend:** For each patient×DiagnosisValue journey, track `Cumulative_Care_Transfers` and regress against ED escalation probability and encounter duration.  
**Test:** Poisson / negative-binomial regression; Kaplan-Meier time-to-escalation by transfer stratum.  
**Circumstantial support:** CMS care-coordination framework; fragmented care is linked to repeated tests and higher utilisation.

### Plan D — Local Care Mismatch Map (Geography + Digital Engagement)
**Hard trend:** Block-group level heat-map using `CENTLAT`, `CENTLON`, `PopulationValue`. Identify clusters where ED rate is high AND MyChart activation is low AND block is rural.  
**Test:** Spatial clustering (DBSCAN or choropleth); Fisher exact for rural vs urban odds.  
**Circumstantial support:** SVH service area is predominantly rural (avg town < 1 000 population).

---

## Tableau Visualisation Guide

> The four approaches below can be implemented in Tableau Desktop or Tableau Public using the engineered output CSVs from either project. Connect Tableau to `data/processed/encounters_engineered.csv` (and related files) as your data source.

---

### Tableau Viz 1 — SDoH Abandonment Heat-Map (Plan A)

**What it shows:** A matrix of abandonment-risk rates by SDoH barrier combination. Each cell represents one of the 8 possible {Transport, Financial, Housing} combinations (2³). Colour encodes the `Is_True_Abandonment_Risk` rate; size encodes encounter volume.

**Build steps:**
1. Connect to `encounters_engineered.csv`.
2. Create a calculated field: `[SDoH_Combo] = STR([Has_Transport_Need]) + "|" + STR([Has_Financial_Strain]) + "|" + STR([Has_Housing_Instability])`.
3. Drag `SDoH_Combo` to Columns; drag a dimension (e.g., `IsEdVisit_Flag`) to Rows.
4. Drag `Is_True_Abandonment_Risk` (aggregated as AVG) to **Color**; drag `Number of Records` to **Size**.
5. Filter to ED/hospital encounters only.
6. Use a **diverging** colour palette (e.g., orange→red) where red = high risk.
7. Add `Has_Transport_Need`, `Has_Financial_Strain`, `Has_Housing_Instability` as quick filters in a dashboard for interactive drill-down.

**Insight to communicate:** Which specific SDoH barrier combination is the strongest predictor of follow-up failure?

---

### Tableau Viz 2 — Off-Hours ED Burden Timeline (Plan B)

**What it shows:** A dual-axis chart combining (a) % of encounters that are ED visits by hour-of-day and (b) average `Encounter_Duration_Hours` by hour, overlaid on an annotated band for off-hours.

**Build steps:**
1. Connect to `encounters_engineered.csv`.
2. Bin `AdmitHour` into discrete buckets (0–6, 7–16, 17–23) using **Create Group** or a calculated field.
3. Create calculated field: `[IsED_Num] = IF [IsEdVisit_Flag] = TRUE THEN 1 ELSE 0 END`.
4. Build a bar chart: `AdmitHour` on Columns; `AVG([Encounter_Duration_Hours])` on the primary Y-axis (left).
5. Add a second measure `SUM([IsED_Num]) / COUNT([EncounterKey])` on a **dual axis** (right Y-axis); sync axes off.
6. Add a reference band from hour 0–6 and 17–23 coloured light red to indicate off-hours.
7. Add `Is_Off_Hours` as a color mark on the bar chart.

**Insight to communicate:** Off-hours encounters last longer and drive disproportionate ED use — the boarding bottleneck is visible.

---

### Tableau Viz 3 — Care Transfer Escalation Funnel (Plan C)

**What it shows:** A funnel / step chart where the X-axis is `Cumulative_Care_Transfers` (0, 1, 2, 3+) and the Y-axes show (a) % of encounters that escalated to ED/hospital and (b) average `Encounter_Duration_Hours`.

**Build steps:**
1. Connect to `encounters_engineered.csv`.
2. Create a calculated field: `[Transfer_Bin] = IF [Cumulative_Care_Transfers] >= 3 THEN "3+" ELSE STR([Cumulative_Care_Transfers]) END`.
3. Build a bar chart: `Transfer_Bin` on Columns; `AVG([Encounter_Duration_Hours])` on Rows.
4. Overlay a line chart on a dual axis: `SUM([IsEdVisit_Flag_Num]) / COUNT([EncounterKey])` for escalation rate.
5. Annotate each bar with the count of patients (tooltip or label).
6. Filter to incident cases (`Is_Incident_Case = TRUE`) for a cleaner cohort.
7. Add a trend line (linear) to the line chart to show the escalation gradient.

**Insight to communicate:** Each additional care transfer is associated with longer encounters and higher ED escalation — continuity gaps compound over the journey.

---

### Tableau Viz 4 — Rural Risk Cluster Map (Plan D)

**What it shows:** A geographic dot map of Kansas, where each dot represents a census block group. Dot **size** = ED rate, dot **colour** = MyChart inactivity rate, dot **opacity** = rural flag (`Is_Rural_Census_Block`). High-risk clusters appear as large, dark, semi-transparent dots.

**Build steps:**
1. Join `encounters_engineered.csv` with `tigercensuscodes.csv` on `CensusBlockGroupFipsCode_Clean = GEOID`.
2. Drag `CENTLON` to Columns (set geographic role: **Longitude**) and `CENTLAT` to Rows (**Latitude**).
3. Set the mark type to **Circle**.
4. Drag `ED_Rate_By_Block` (pre-aggregated calculated field) to **Size**.
5. Drag `MyChart_Inactive_Rate_By_Block` to **Color** (use a sequential palette: light = active, dark = inactive).
6. Drag `Is_Rural_Census_Block` to **Detail**; set opacity via a calculated field so rural blocks are more opaque.
7. Add a background map layer (Tableau default or a custom Mapbox tile).
8. Create a **dashboard action** so clicking a block filters Plans A, B, C charts on the same dashboard.

**Insight to communicate:** High-risk rural clusters (large dark dots) pinpoint communities where SVH should invest in outreach, transportation programs, and MyChart activation drives.

---

## R Project

See `r_project/` for full scripts. Run in order via:

```r
source("r_project/run_all.R")
```

**Dependencies:** `tidyverse`, `lubridate`, `janitor`, `scales`, `ggplot2`, `ggridges`, `sf`, `survival`, `survminer`, `broom`, `MASS`, `lme4`

---

## Python Project

See `python_project/` for full scripts. Run via:

```bash
pip install -r python_project/requirements.txt
python python_project/run_all.py
```

**Dependencies:** `pandas`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`, `statsmodels`, `pyarrow`, `lifelines`

---

## Ethics & Data Handling

- Data are fully de-identified per HIPAA (covered by SVH and ASA DataFest agreements).
- Raw CSV files are **git-ignored** and must never be committed.
- Do not upload data to any public service.
- Delete all data after DataFest participation.

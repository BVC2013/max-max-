"""
feature_engineering.py
Build all engineered features defined in the README.
Reads raw parquet files; writes encounters_engineered.parquet + CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


POSITIVE_ANSWERS = {
    "Yes", "Sometimes", "Often", "Always",
    "Hard", "Very Hard",
    "A little bit", "Somewhat", "Quite a bit", "Very much",
    "Less than once a week", "Never",
    "I choose not to answer",
}


def _make_timestamp(df: pd.DataFrame, year: str, month: str, day: str,
                    hour: str, minute: str) -> pd.Series:
    """Safely construct a datetime from component columns."""
    parts = df[[year, month, day, hour, minute]].copy()
    parts.columns = ["year", "month", "day", "hour", "minute"]
    parts = parts.apply(pd.to_numeric, errors="coerce")

    valid = (
        parts["year"].between(2000, 2030) &
        parts["month"].between(1, 12) &
        parts["day"].between(1, 31)
    )
    result = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if valid.any():
        result[valid] = pd.to_datetime(
            parts[valid].astype(int).astype(str).agg(
                lambda r: f"{r['year']}-{r['month']:0>2}-{r['day']:0>2} "
                          f"{r['hour']:0>2}:{r['minute']:0>2}:00",
                axis=1
            ),
            errors="coerce",
        )
    return result


def run(data_dir: Path, output_dir: Path, processed_dir: Path) -> None:

    print("  Loading raw parquet files …")
    enc   = pd.read_parquet(processed_dir / "encounters_raw.parquet")
    pat   = pd.read_parquet(processed_dir / "patients_raw.parquet")
    diag  = pd.read_parquet(processed_dir / "diagnosis_raw.parquet")
    sdoh  = pd.read_parquet(processed_dir / "social_determinants_raw.parquet")
    tiger = pd.read_parquet(processed_dir / "tigercensus_raw.parquet")

    # =========================================================================
    # 1. Core Parsing / Flags
    # =========================================================================
    print("  Engineering core flags …")

    enc["EncounterDate"] = pd.to_datetime(enc["Date"], errors="coerce").dt.date

    enc["AdmissionTimestamp"] = _make_timestamp(
        enc, "AdmitYear", "AdmitMonth", "AdmitDay", "AdmitHour", "AdmitMinute"
    )
    enc["DischargeTimestamp"] = _make_timestamp(
        enc, "DischargeYear", "DischargeMonth", "DischargeDay",
        "DischargeHour", "DischargeMinute"
    )

    dur = (enc["DischargeTimestamp"] - enc["AdmissionTimestamp"]).dt.total_seconds() / 3600
    enc["Encounter_Duration_Hours"] = np.where(dur.between(0, 8760), dur, np.nan)

    enc["IsEdVisit_Flag"]           = enc["IsEdVisit"].astype(bool)
    enc["IsHospitalAdmission_Flag"] = enc["IsHospitalAdmission"].astype(bool)

    admit_hour = pd.to_numeric(enc["AdmitHour"], errors="coerce")
    is_weekend = pd.to_datetime(enc["Date"], errors="coerce").dt.dayofweek >= 5
    enc["Is_Off_Hours"] = is_weekend | (admit_hour < 7) | (admit_hour >= 17)
    enc["AdmitHour_num"] = admit_hour

    # =========================================================================
    # 2. Patient-level features — MyChart, vital status, geography
    # =========================================================================
    print("  Joining patient features …")

    pat["MyChart_Active"] = pat["MyChartStatus"].str.strip().str.lower() == "activated"
    pat["CensusBlockFipsCode_Clean"] = pat["CensusBlockGroupFipsCode"].where(
        ~pat["CensusBlockGroupFipsCode"].astype(str).str.contains("Unspecified", na=True)
    )
    pat["Geo_Data_Available"] = pat["CensusBlockFipsCode_Clean"].notna()

    pat_slim = pat[["DurableKey", "MyChart_Active", "VitalStatus",
                    "CensusBlockFipsCode_Clean", "Geo_Data_Available"]].copy()

    enc = enc.merge(
        pat_slim,
        left_on="PatientDurableKey", right_on="DurableKey",
        how="left"
    )

    # =========================================================================
    # 3. SDoH Aggregated (patient-level)
    # =========================================================================
    print("  Engineering SDoH barrier flags …")

    sdoh["domain_lc"] = sdoh["Domain"].str.lower().fillna("")
    sdoh["positive_screen"] = sdoh["AnswerText"].isin(POSITIVE_ANSWERS)

    sdoh_agg = sdoh.groupby("PatientDurableKey").apply(
        lambda g: pd.Series({
            "Has_Transport_Need":      bool((g["domain_lc"].str.contains("transport") & g["positive_screen"]).any()),
            "Has_Financial_Strain":    bool((g["domain_lc"].str.contains("financial") & g["positive_screen"]).any()),
            "Has_Housing_Instability": bool((g["domain_lc"].str.contains("housing")   & g["positive_screen"]).any()),
        })
    ).reset_index()
    sdoh_agg["Any_SDoH_Barrier"] = (
        sdoh_agg["Has_Transport_Need"] |
        sdoh_agg["Has_Financial_Strain"] |
        sdoh_agg["Has_Housing_Instability"]
    )

    enc = enc.merge(sdoh_agg, on="PatientDurableKey", how="left")
    for col in ["Has_Transport_Need", "Has_Financial_Strain",
                "Has_Housing_Instability", "Any_SDoH_Barrier"]:
        enc[col] = enc[col].fillna(False)

    # =========================================================================
    # 4. Join Diagnosis (DiagnosisValue for journey tracking)
    # =========================================================================
    print("  Joining diagnosis …")

    diag_slim = diag[["DiagnosisKey", "DiagnosisValue", "DiagnosisName",
                       "GroupCode", "GroupName"]].copy()
    enc = enc.merge(diag_slim, left_on="PrimaryDiagnosisKey",
                    right_on="DiagnosisKey", how="left")

    # =========================================================================
    # 5. Journey Features (per PatientDurableKey × DiagnosisValue)
    # =========================================================================
    print("  Computing journey features (this may take a moment) …")

    enc["EncounterDate_dt"] = pd.to_datetime(enc["Date"], errors="coerce")
    enc = enc.sort_values(["PatientDurableKey", "DiagnosisValue", "EncounterDate_dt"])

    grp = enc.groupby(["PatientDurableKey", "DiagnosisValue"], sort=False)

    enc["Visit_Number"]               = grp.cumcount() + 1
    enc["Journey_Start_Date"]         = grp["EncounterDate_dt"].transform("min")
    enc["Cumulative_Days_In_Journey"] = (
        enc["EncounterDate_dt"] - enc["Journey_Start_Date"]
    ).dt.days.astype(float)
    enc["Days_Since_Last_Visit"]      = grp["EncounterDate_dt"].diff().dt.days.astype(float)

    patient_first = enc.groupby("PatientDurableKey")["EncounterDate_dt"].transform("min")
    # Is_Incident_Case: >180 days after the patient's first-ever observed visit
    # (across ALL diagnoses, not just within the current diagnosis journey)
    enc["Is_Incident_Case"] = (enc["EncounterDate_dt"] - patient_first).dt.days > 180

    journey_max_visit = grp["Visit_Number"].transform("max")
    enc["Has_Follow_Up"]          = enc["Visit_Number"] < journey_max_visit
    enc["Raw_Journey_Abandoned"]  = ~enc["Has_Follow_Up"]
    enc["Is_True_Abandonment_Risk"] = (
        enc["Raw_Journey_Abandoned"] &
        (enc["VitalStatus"].fillna("").str.upper() != "DECEASED")
    )

    # Care transfers: attending provider change within journey
    prev_provider = grp["AttendingProviderDurableKey"].shift(1)
    enc["Is_Care_Transfer"] = (
        enc["AttendingProviderDurableKey"].notna() &
        (enc["AttendingProviderDurableKey"] != prev_provider)
    )
    enc["Is_Care_Transfer"] = enc["Is_Care_Transfer"].fillna(False)
    enc["Cumulative_Care_Transfers"] = grp["Is_Care_Transfer"].cumsum()

    # =========================================================================
    # 6. Geography — join tigercensus
    # =========================================================================
    print("  Joining tigercensus geography …")

    tiger_slim = tiger[["GEOID", "PopulationValue", "CENTLAT", "CENTLON"]].copy()
    tiger_slim["Is_Rural_Census_Block"] = tiger_slim["PopulationValue"] < 1500
    tiger_slim["GEOID"] = tiger_slim["GEOID"].astype(str)
    enc["CensusBlockFipsCode_Clean"] = enc["CensusBlockFipsCode_Clean"].astype(str)

    enc = enc.merge(tiger_slim, left_on="CensusBlockFipsCode_Clean",
                    right_on="GEOID", how="left")

    # =========================================================================
    # 7. Persist
    # =========================================================================
    print("  Saving engineered dataset …")
    enc.to_parquet(processed_dir / "encounters_engineered.parquet", index=False)
    enc.to_csv(processed_dir / "encounters_engineered.csv", index=False)

    print(f"feature_engineering complete. {len(enc):,} rows | {enc.shape[1]} cols")

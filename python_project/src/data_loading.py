"""
data_loading.py
Load and minimally coerce all seven SVH CSV files.
Saves parquet files to data/processed/ for fast downstream loading.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


FILENAMES = {
    "encounters":          "encounters.csv",
    "patients":            "patients.csv",
    "diagnosis":           "diagnosis.csv",
    "departments":         "departments.csv",
    "providers":           "providers.csv",
    "social_determinants": "social_determinants.csv",
    "tigercensus":         "tigercensuscodes.csv",
}

FLAG_COLS = [
    "IsEdVisit", "IsHospitalAdmission", "IsHospitalOutpatientVisit",
    "IsInpatientAdmission", "IsObservation", "IsOutpatientFaceToFaceVisit",
]

NUMERIC_ADMIT_COLS = [
    "AdmitYear", "AdmitMonth", "AdmitDay", "AdmitHour", "AdmitMinute",
    "DischargeYear", "DischargeMonth", "DischargeDay", "DischargeHour", "DischargeMinute",
]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def run(data_dir: Path, output_dir: Path, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, pd.DataFrame] = {}
    for name, filename in FILENAMES.items():
        print(f"  Loading {filename} …", end=" ", flush=True)
        df = _read(data_dir / filename)
        print(f"{len(df):,} rows x {df.shape[1]} cols")
        tables[name] = df

    # ---- encounters: boolean flags -----------------------------------------
    enc = tables["encounters"]
    for col in FLAG_COLS:
        if col in enc.columns:
            enc[col] = enc[col].str.strip().str.lower() == "yes"

    # numeric admit/discharge components
    for col in NUMERIC_ADMIT_COLS:
        if col in enc.columns:
            enc[col] = pd.to_numeric(enc[col], errors="coerce")

    tables["encounters"] = enc

    # ---- tigercensus: numeric coords / population --------------------------
    tc = tables["tigercensus"]
    for col in ("PopulationValue", "CENTLAT", "CENTLON"):
        if col in tc.columns:
            tc[col] = pd.to_numeric(tc[col], errors="coerce")
    tables["tigercensus"] = tc

    # ---- persist -----------------------------------------------------------
    for name, df in tables.items():
        out_path = processed_dir / f"{name}_raw.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved {out_path.name}")

    print("data_loading complete.")

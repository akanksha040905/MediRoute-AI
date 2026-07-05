"""
data_manager.py — Load, validate, persist, and log hospital data.
Supports 231-hospital Karnataka dataset with contact_number field.
"""

import os
import csv
import logging
import pandas as pd
from datetime import datetime
from config import CSV_PATH, LOG_PATH, CSV_COLUMNS

logger = logging.getLogger(__name__)

REQUIRED_COLS = [
    "hospital", "latitude", "longitude", "total_beds",
    "available_beds", "avg_daily_patients", "critical_patients",
    "ventilators", "waiting_queue",
]


def validate_row(row: dict) -> list:
    errors = []
    for col in REQUIRED_COLS:
        if col not in row or row[col] is None or str(row[col]).strip() == "":
            errors.append(f"Missing required field: {col}")
    if not errors:
        if float(row["available_beds"]) > float(row["total_beds"]):
            errors.append("available_beds cannot exceed total_beds")
        if not (-90 <= float(row["latitude"]) <= 90):
            errors.append("Latitude must be between -90 and 90")
        if not (-180 <= float(row["longitude"]) <= 180):
            errors.append("Longitude must be between -180 and 180")
        oxy = float(row.get("oxygen_supply_pct", 80))
        if not (0 <= oxy <= 100):
            errors.append("oxygen_supply_pct must be 0–100")
    return errors


def load_data() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        logger.warning("CSV not found — generating sample data.")
        from sample_data_generator import generate_sample_data
        df = generate_sample_data()
        save_data(df)
        return df
    df = pd.read_csv(CSV_PATH)
    df = _fill_missing_columns(df)
    df["contact_number"] = df["contact_number"].fillna("N/A").astype(str).str.strip()
    # Convert lat/lon to numeric, drop rows without valid coordinates
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if len(df) < before:
        logger.warning("Dropped %d hospitals with missing coordinates", before - len(df))
    logger.info("Loaded %d hospitals from %s", len(df), CSV_PATH)
    return df


def _fill_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "icu_specialist_count": 5,
        "oxygen_supply_pct":    80.0,
        "contact_number":       "N/A",
        "last_updated":         datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def save_data(df: pd.DataFrame) -> None:
    ordered = [c for c in CSV_COLUMNS if c in df.columns]
    extras  = [c for c in df.columns if c not in CSV_COLUMNS]
    df[ordered + extras].to_csv(CSV_PATH, index=False)
    logger.info("Data saved → %s (%d rows)", CSV_PATH, len(df))


def add_hospital(df: pd.DataFrame, new_row: dict) -> tuple:
    errors = validate_row(new_row)
    if errors:
        return df, errors
    new_row.setdefault("contact_number", "N/A")
    new_row["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_data(new_df)
    return new_df, []


def update_hospital(df: pd.DataFrame, hospital_name: str, updates: dict) -> pd.DataFrame:
    mask = df["hospital"] == hospital_name
    if not mask.any():
        raise ValueError(f"Hospital '{hospital_name}' not found.")
    for col, val in updates.items():
        if col in df.columns:
            df.loc[mask, col] = val
    df.loc[mask, "last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_data(df)
    return df


def delete_hospital(df: pd.DataFrame, hospital_name: str) -> pd.DataFrame:
    df = df[df["hospital"] != hospital_name].reset_index(drop=True)
    save_data(df)
    return df


LOG_COLUMNS = [
    "timestamp", "patient_lat", "patient_lon", "severity",
    "recommended_hospital", "eta_min", "predicted_beds", "contact_number",
]


def log_allocation(
    patient_lat: float,
    patient_lon: float,
    severity: str,
    recommended_hospital: str,
    eta_min: float,
    predicted_beds: int,
    contact_number: str = "N/A",
) -> None:
    record = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        patient_lat, patient_lon, severity,
        recommended_hospital, round(eta_min, 1),
        predicted_beds, contact_number,
    ]
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writerow(LOG_COLUMNS)
        writer.writerow(record)


def _recover_log() -> pd.DataFrame:
    good_rows = []
    try:
        with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                if len(row) == len(LOG_COLUMNS):
                    good_rows.append(row)
    except Exception as exc:
        logger.error("Could not open log for recovery: %s", exc)
    recovered = pd.DataFrame(good_rows, columns=LOG_COLUMNS)
    try:
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(LOG_COLUMNS)
            writer.writerows(good_rows)
    except Exception as exc:
        logger.error("Could not rewrite recovered log: %s", exc)
    return recovered


def load_log() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=LOG_COLUMNS)
    try:
        df = pd.read_csv(LOG_PATH, quoting=1)
        for col in LOG_COLUMNS:
            if col not in df.columns:
                df[col] = "N/A"
        df = df[LOG_COLUMNS]
        for col in ("patient_lat", "patient_lon", "eta_min", "predicted_beds"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as exc:
        logger.warning("Log file corrupt (%s) — attempting recovery.", exc)
        return _recover_log()
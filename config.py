"""
config.py — MediRoute AI configuration
"""
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")

CSV_PATH   = os.path.join(DATA_DIR,   "icu_beds.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "icu_model.pkl")
LOG_PATH   = os.path.join(LOGS_DIR,   "allocation_log.csv")

for _dir in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)

APP_TITLE           = "MediRoute AI — Karnataka ICU Network"
TOP_N_HOSPITALS     = 3
SEVERITY_LEVELS     = ["Low", "Medium", "High", "Critical"]
BED_ALERT_THRESHOLD = 10

DEFAULT_LAT = 12.9716
DEFAULT_LON = 77.5946
MAP_ZOOM    = 8

OSRM_BASE_URL      = "http://router.project-osrm.org/route/v1/driving"
ROUTE_TIMEOUT_SEC  = 6
FALLBACK_SPEED_KMH = 40

# How many nearest hospitals (by haversine) to consider per severity
# These are realistic ambulance ranges across Karnataka
PROXIMITY_CANDIDATES = {
    "Low":      25,   # search nearest 25 → pick best nearby
    "Medium":   30,
    "High":     35,
    "Critical": 20,   # critical → strictly nearest, fewer candidates
}

# Scoring weights — must sum to 1.0 per severity
# Critical: distance dominates (get there FAST)
# Low: beds and resources matter more than speed
SCORING_WEIGHTS = {
    "Low":      {"beds": 0.35, "distance": 0.30, "queue": 0.20, "readiness": 0.15},
    "Medium":   {"beds": 0.30, "distance": 0.40, "queue": 0.15, "readiness": 0.15},
    "High":     {"beds": 0.25, "distance": 0.55, "queue": 0.10, "readiness": 0.10},
    "Critical": {"beds": 0.15, "distance": 0.70, "queue": 0.05, "readiness": 0.10},
}

FEATURE_COLS = [
    "total_beds",
    "avg_daily_patients",
    "critical_patients",
    "waiting_queue",
    "ventilators",
    "icu_specialist_count",
    "oxygen_supply_pct",
    "occupancy_rate",
    "critical_ratio",
    "bed_pressure",
    "resource_strain",
]

TARGET_COL = "safe_capacity"

CSV_COLUMNS = [
    "hospital", "contact_number", "latitude", "longitude",
    "total_beds", "available_beds", "avg_daily_patients",
    "critical_patients", "ventilators", "waiting_queue",
    "icu_specialist_count", "oxygen_supply_pct", "last_updated",
]

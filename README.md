# MediRoute AI — Karnataka ICU Bed Allocation System

AI-powered ICU bed finder and routing system covering 231 Karnataka hospitals.
Built with Streamlit, scikit-learn ensemble ML, Folium maps, and OSRM routing.

---

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501** by default.

---

## Project structure

```
mediroute/
├── app.py                    # Main Streamlit UI
├── config.py                 # All constants and paths
├── data_manager.py           # CSV CRUD + allocation log
├── model.py                  # GBR + RF ensemble ML model
├── utils.py                  # Geo / routing / scoring helpers
├── sample_data_generator.py  # Seed data (231 Karnataka hospitals)
├── requirements.txt
├── data/
│   └── icu_beds.csv          # Auto-generated on first run
├── models/
│   └── icu_model.pkl         # Cached trained model
└── logs/
    └── allocation_log.csv    # Audit trail of recommendations
```

---

## Using your own hospital data

Drop a CSV at `data/icu_beds.csv` with these columns (in any order):

| Column | Type | Notes |
|---|---|---|
| hospital | str | Hospital name |
| contact_number | str | e.g. `+91-080-XXXXXXXX` |
| latitude | float | WGS-84 |
| longitude | float | WGS-84 |
| total_beds | int | Total ICU beds |
| available_beds | int | Currently available |
| avg_daily_patients | int | Rolling average |
| critical_patients | int | Current critical count |
| ventilators | int | Available ventilators |
| waiting_queue | int | Patients waiting |
| icu_specialist_count | int | On-duty ICU specialists |
| oxygen_supply_pct | float | 0–100 |
| last_updated | str | `YYYY-MM-DD HH:MM` |

Then click **Refresh & Retrain** in the sidebar.

---

## Features

- **Find Hospital** — AI-ranked top-3 ICU recommendations with live OSRM routing, ETA, and Folium map
- **Hospital Registry** — Search, add, edit, and delete hospitals
- **Analytics** — Network-wide bed availability, occupancy, ventilators, O2 supply charts
- **Allocation Log** — Full audit CSV with download; severity and frequency charts
- **Model Info** — GBR + RF ensemble metrics, confidence per hospital, one-click retrain

---

## Routing

Routes use the public [OSRM](http://project-osrm.org/) API when available.
If OSRM is unreachable (network restrictions, timeouts), the system falls back
to a haversine straight-line distance at `FALLBACK_SPEED_KMH = 40 km/h`.

---

## Configuration

All tuneable constants live in `config.py`:

- `BED_ALERT_THRESHOLD` — beds below this trigger a red alert (default: 10)
- `SCORING_WEIGHTS` — per-severity weighting of beds / distance / queue / ventilators
- `FALLBACK_SPEED_KMH` — ETA fallback speed when OSRM is down
- `TOP_N_HOSPITALS` — number of recommendations returned (default: 3)

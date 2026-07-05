"""
utils.py — Geo, routing, scoring.

FIX: enrich_with_routes now sorts by DISTANCE first (proximity_n nearest),
so the ranking always starts from hospitals closest to the patient.
Critical severity = fewest candidates = strictly nearest hospitals.

FIX 2: rank_hospitals now ranks by DISTANCE first among viable hospitals.
Score is only used to disqualify hospitals with unusably low beds
(already heavily penalised in score_hospitals) and as a tiebreaker
for hospitals at ~equal distance. Previously score dominated the final
sort, which let a farther hospital with more beds outrank a closer one.
"""

import math, logging
import numpy as np
import pandas as pd
import requests
from config import OSRM_BASE_URL, ROUTE_TIMEOUT_SEC, FALLBACK_SPEED_KMH, \
                   SCORING_WEIGHTS, TOP_N_HOSPITALS, PROXIMITY_CANDIDATES

logger = logging.getLogger(__name__)


def haversine_km(lat1, lon1, lat2, lon2):
    R    = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_route(lat1, lon1, lat2, lon2):
    try:
        url  = f"{OSRM_BASE_URL}/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
        resp = requests.get(url, timeout=ROUTE_TIMEOUT_SEC)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok":
            raise ValueError(data.get("code"))
        r   = data["routes"][0]
        pts = [(c[1], c[0]) for c in r["geometry"]["coordinates"]]
        return r["distance"]/1000, r["duration"]/60, pts
    except Exception as exc:
        logger.warning("OSRM failed: %s — haversine fallback", exc)
        d = haversine_km(lat1, lon1, lat2, lon2)
        return d, d/FALLBACK_SPEED_KMH*60, []


def enrich_with_routes(df: pd.DataFrame, plat: float, plon: float,
                       severity: str = "Medium") -> pd.DataFrame:
    """
    1. Compute haversine distance for ALL hospitals.
    2. Keep only the N nearest (by severity).
    3. Get real OSRM distance+ETA for those N.
    This guarantees the nearest hospitals are always in the candidate pool.
    """
    df = df.copy()
    # Drop rows with missing coordinates — these haven't been geocoded yet
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]
    if df.empty:
        return df

    # Step 1: haversine pre-sort
    df["distance_km"] = df.apply(
        lambda r: haversine_km(plat, plon, r["latitude"], r["longitude"]), axis=1
    )
    df = df.sort_values("distance_km").reset_index(drop=True)

    # Step 2: keep nearest N for the given severity
    n = PROXIMITY_CANDIDATES.get(severity, 30)
    df = df.head(n).copy()

    # Step 3: OSRM enrichment on the short list
    distances, durations, routes = [], [], []
    for _, row in df.iterrows():
        d, t, r = get_route(plat, plon, row["latitude"], row["longitude"])
        distances.append(d)
        durations.append(t)
        routes.append(r)

    df["distance_km"] = distances   # overwrite with real road distance
    df["eta_min"]     = durations
    df["route"]       = routes
    return df


def _mm(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return pd.Series(0.5, index=s.index) if hi == lo else (s - lo) / (hi - lo)


def score_hospitals(df: pd.DataFrame, severity: str) -> pd.DataFrame:
    """
    Score with DISTANCE as the strongest signal for High/Critical.
    Uses actual available_beds (65%) + predicted safe capacity (35%).
    Hard penalties for hospitals with very few beds.
    """
    df = df.copy()
    w  = SCORING_WEIGHTS.get(severity, SCORING_WEIGHTS["Medium"])

    # Bed score: actual beds ground truth + ML signal
    bed_score = 0.65 * _mm(df["available_beds"]) + 0.35 * _mm(df["predicted_beds"])

    # Distance score: shorter road distance AND faster ETA both matter
    dist_score = 0.5 * (1 - _mm(df["distance_km"])) + 0.5 * (1 - _mm(df["eta_min"]))

    # Queue: shorter wait is better
    queue_score = 1 - _mm(df["waiting_queue"])

    # Readiness: ventilators + oxygen + specialists
    vent_n  = _mm(df["ventilators"])
    oxy_n   = _mm(df.get("oxygen_supply_pct",   pd.Series(80.0, index=df.index)))
    staff_n = _mm(df.get("icu_specialist_count", pd.Series(5.0,  index=df.index)))
    readiness = 0.40 * vent_n + 0.35 * oxy_n + 0.25 * staff_n

    df["score"] = (
        w["beds"]      * bed_score   +
        w["distance"]  * dist_score  +
        w["queue"]     * queue_score +
        w["readiness"] * readiness
    )

    # Hard penalties — no beds = near-zero score regardless of other factors
    df.loc[df["available_beds"] == 0, "score"] *= 0.05
    df.loc[df["available_beds"] == 1, "score"] *= 0.40
    df.loc[df["available_beds"] == 2, "score"] *= 0.60
    df.loc[df["available_beds"] <= 3, "score"] *= 0.75
    return df


def rank_hospitals(df: pd.DataFrame, severity: str) -> pd.DataFrame:
    """
    TRUE proximity-first ranking:
    1. Score is used only to disqualify hospitals that are unusably bad
       (near-zero beds, already penalised heavily in score_hospitals).
    2. Among the viable candidates, rank by DISTANCE first.
       Score is only a tiebreaker for hospitals at ~equal distance.
    This matches the UI label "sorted by distance from your location".
    """
    scored = score_hospitals(df, severity)
    if scored.empty:
        return scored

    # Disqualify hospitals whose score is far below the best option in the
    # candidate pool (i.e. effectively no usable beds) — everything else
    # is considered "viable" and ranked by distance, not score.
    threshold = scored["score"].max() * 0.15
    viable = scored[scored["score"] >= threshold]
    if viable.empty:
        viable = scored  # fallback: nothing cleared the bar, use full pool

    ranked = viable.sort_values(["distance_km", "score"], ascending=[True, False])
    return ranked.head(TOP_N_HOSPITALS).reset_index(drop=True)


def detect_location_from_ip():
    for url in ["https://ipinfo.io/json", "https://ip-api.com/json/"]:
        try:
            data = requests.get(url, timeout=3).json()
            if "loc" in data:
                lat, lon = data["loc"].split(","); return float(lat), float(lon)
            if data.get("status") == "success":
                return float(data["lat"]), float(data["lon"])
        except Exception as e:
            logger.warning("IP geo failed %s: %s", url, e)
    return None
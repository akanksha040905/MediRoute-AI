"""
geocode_hospitals.py — MediRoute AI
Auto-geocodes hospital coordinates using OpenStreetMap's Nominatim API
(via geopy), so lat/long is pulled from a live database instead of
manually typed in (source of the "wrong hospital location" bugs).

WHAT IT DOES
------------
1. Loads data/icu_beds.csv
2. For each hospital, checks if the existing lat/lon looks valid
   (inside a Karnataka bounding box). If yes, it's left alone.
3. If missing/invalid, tries several query variants against Nominatim,
   from most to least specific, until one returns a result inside
   Karnataka.
4. Respects Nominatim's usage policy (max 1 request/sec, custom
   User-Agent) via geopy's RateLimiter.
5. Caches every query -> result so re-running the script (e.g. after
   it's interrupted, or after adding new hospitals) doesn't re-hit
   the API for hospitals already resolved.
6. Writes a report (geocode_report.csv) so you can see exactly which
   hospitals were auto-geocoded, which were kept as-is, and which
   need manual attention.

USAGE
-----
    pip install geopy --break-system-packages
    python geocode_hospitals.py                 # normal run
    python geocode_hospitals.py --force          # re-geocode everything
    python geocode_hospitals.py --csv path.csv   # custom CSV path

IMPORTANT: Nominatim's usage policy requires a real, descriptive
User-Agent (ideally with contact info). Edit USER_AGENT below before
running at scale — a generic/fake one can get your IP blocked.

NOTE ON ENCODING: All file reads/writes in this script explicitly use
UTF-8. Windows defaults to your system codepage (often cp1252) for
plain read_text()/to_csv() calls, which crashes as soon as any hospital
name, address, or OSM result contains a non-ASCII character (accents,
special punctuation, non-Latin script, etc). Every I/O call below is
pinned to encoding="utf-8" to avoid that class of bug entirely.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.distance import geodesic

# ══════════════════════════════════════════════════════
#  CONFIG — edit these
# ══════════════════════════════════════════════════════
CSV_PATH     = Path("data/icu_beds.csv")
CACHE_PATH   = Path("data/geocode_cache.json")
REPORT_PATH  = Path("data/geocode_report.csv")

# Nominatim requires a real identifying User-Agent (their usage policy).
# Replace the email with yours if you plan to geocode many hospitals.
USER_AGENT   = "mediroute-ai-geocoder (contact: akankshaanusha1@gmail.com)"

# Karnataka's approximate bounding box — used to sanity-check every
# result (both existing CSV values and new geocoded ones). Anything
# outside this box is almost certainly wrong.
KA_LAT_MIN, KA_LAT_MAX = 11.4, 18.6
KA_LON_MIN, KA_LON_MAX = 74.0, 78.6

MIN_DELAY_SECONDS = 1.1   # Nominatim policy: max 1 req/sec — stay under it
MAX_RETRIES       = 3
CHECKPOINT_EVERY  = 10    # rows between incremental CSV saves

# In --compare mode: existing vs newly-geocoded points further apart than
# this (km) get flagged as "mismatch" for manual review.
MISMATCH_KM_THRESHOLD = 3.0


def in_karnataka(lat, lon) -> bool:
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    if pd.isna(lat) or pd.isna(lon):
        return False
    return KA_LAT_MIN <= lat <= KA_LAT_MAX and KA_LON_MIN <= lon <= KA_LON_MAX


def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_query_variants(hospital: str, district: str | None, address: str | None = None) -> list[str]:
    """Most-specific to least-specific query strings to try in order."""
    hospital = hospital.strip()
    variants = []

    # If a real address is available, lead with "<hospital name>, <address>" —
    # this is usually the most accurate query since it pins down the place
    # name even if OSM doesn't have the exact hospital name tagged.
    if address and str(address).strip().lower() not in ("nan", ""):
        addr = str(address).strip()
        variants.append(f"{hospital}, {addr}")
        variants.append(addr)  # fall back to just the address/place itself

    if district and str(district).strip().lower() not in ("nan", ""):
        district = str(district).strip()
        variants.append(f"{hospital}, {district}, Karnataka, India")
    variants.append(f"{hospital}, Karnataka, India")
    variants.append(f"{hospital}, India")

    # Some OSM entries drop generic words like "Hospital"/"Multispeciality"
    # from the name — try a stripped-down version as a last resort.
    for word in ["Multispeciality", "Multispecialty", "Super Speciality",
                 "Superspeciality", "Speciality", "Specialty"]:
        if word.lower() in hospital.lower():
            stripped = hospital.lower().replace(word.lower(), "").strip(" ,-")
            variants.append(f"{stripped}, Karnataka, India")
            break

    # De-duplicate while preserving order
    seen = set()
    out = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def geocode_with_retries(geocode_fn, query: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return geocode_fn(query)
        except GeocoderTimedOut:
            wait = 2 * attempt
            print(f"    ⏳ timeout on '{query}', retrying in {wait}s…")
            time.sleep(wait)
        except GeocoderServiceError as e:
            wait = 3 * attempt
            print(f"    ⚠ service error ({e}) on '{query}', retrying in {wait}s…")
            time.sleep(wait)
    return None


def run_compare(df, geocode, district_col, cache, address_col=None):
    """
    Geocode every hospital fresh via Nominatim, but never touch the CSV.
    Instead, measure how far the newly-geocoded point is from the existing
    one and write a report so you can decide, hospital by hospital,
    whether the existing coordinate should be replaced.
    """
    rows = []
    total = len(df)
    for idx, row in df.iterrows():
        hospital = str(row["hospital"]).strip()
        existing_lat, existing_lon = row.get("latitude"), row.get("longitude")
        district = row.get(district_col) if district_col else None
        address = row.get(address_col) if address_col else None

        print(f"[{idx+1}/{total}] Checking: {hospital}")

        cache_key = hospital.lower().strip()
        cached = cache.get(cache_key)
        location = None
        matched_query = ""

        if cached and cached.get("status") == "geocoded":
            new_lat, new_lon = cached["latitude"], cached["longitude"]
            matched_query = cached.get("matched_query", "(cached)")
        else:
            new_lat = new_lon = None
            for query in build_query_variants(hospital, district, address):
                location = geocode_with_retries(geocode, query)
                if location and in_karnataka(location.latitude, location.longitude):
                    new_lat, new_lon = round(location.latitude, 6), round(location.longitude, 6)
                    matched_query = query
                    cache[cache_key] = {
                        "status": "geocoded", "latitude": new_lat, "longitude": new_lon,
                        "matched_query": query, "display_name": location.address,
                    }
                    break
            if new_lat is None:
                cache[cache_key] = {"status": "failed"}

        # Save cache incrementally too, so a crash mid-run doesn't lose
        # everything resolved so far (defensive, in addition to try/except
        # around the final save below).
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            try:
                save_cache(cache)
            except Exception as e:
                print(f"    ⚠️ Warning: could not save cache checkpoint ({e})")

        if new_lat is None:
            rows.append({
                "hospital": hospital, "existing_lat": existing_lat, "existing_lon": existing_lon,
                "new_lat": None, "new_lon": None, "distance_km": None,
                "matched_query": "", "status": "geocode_failed",
            })
            print("    ❌ could not geocode — flagged for manual review")
            continue

        if in_karnataka(existing_lat, existing_lon):
            dist_km = geodesic((existing_lat, existing_lon), (new_lat, new_lon)).km
            status = "mismatch" if dist_km > MISMATCH_KM_THRESHOLD else "matches_closely"
            symbol = "⚠" if status == "mismatch" else "✓"
            print(f"    {symbol} existing vs OSM: {dist_km:.2f} km apart ({status})")
        else:
            dist_km = None
            status = "existing_invalid_new_available"
            print(f"    ➕ no valid existing coordinate — new one available")

        rows.append({
            "hospital": hospital, "existing_lat": existing_lat, "existing_lon": existing_lon,
            "new_lat": new_lat, "new_lon": new_lon, "distance_km": dist_km,
            "matched_query": matched_query, "status": status,
        })

    return pd.DataFrame(rows)


def apply_accepted_updates(csv_path, report_path):
    """
    After reviewing geocode_report.csv from --compare, keep only the rows
    you're happy with (delete/edit others), then run:
        python geocode_hospitals.py --apply
    This copies new_lat/new_lon into the main CSV for every row still
    present in the report with status 'mismatch' or 'existing_invalid_new_available'.
    """
    if not Path(report_path).exists():
        print(f"❌ Report not found at {report_path}.")
        print("   Run --compare first to generate it (it must finish without")
        print("   errors — check the console output for a 'COMPARE DONE' banner).")
        sys.exit(1)

    df = pd.read_csv(csv_path, encoding="utf-8")
    report = pd.read_csv(report_path, encoding="utf-8")
    report = report[report["status"].isin(["mismatch", "existing_invalid_new_available"])]
    report = report.dropna(subset=["new_lat", "new_lon"])

    applied = 0
    for _, r in report.iterrows():
        mask = df["hospital"].str.strip().str.lower() == str(r["hospital"]).strip().lower()
        if mask.any():
            df.loc[mask, "latitude"]  = r["new_lat"]
            df.loc[mask, "longitude"] = r["new_lon"]
            applied += 1
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"✅ Applied {applied} coordinate updates to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Geocode hospital coordinates via Nominatim/OSM")
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to icu_beds.csv")
    parser.add_argument("--force", action="store_true",
                         help="Re-geocode every hospital, even ones with valid-looking coordinates, "
                              "and OVERWRITE the CSV directly")
    parser.add_argument("--compare", action="store_true",
                         help="Geocode every hospital and report how far existing coords are from "
                              "the OSM result, WITHOUT touching the CSV (recommended first step)")
    parser.add_argument("--apply", action="store_true",
                         help="Apply accepted rows from a previously-reviewed geocode_report.csv "
                              "(from --compare) into the CSV")
    parser.add_argument("--district-col", type=str, default=None,
                         help="Optional column name holding city/district, to improve query accuracy")
    parser.add_argument("--address-col", type=str, default=None,
                         help="Optional column name holding a full real address — used as the "
                              "primary geocoding query when present, ahead of district/name-only")
    args = parser.parse_args()

    if args.apply:
        apply_accepted_updates(args.csv, REPORT_PATH)
        return

    csv_path = args.csv
    if not csv_path.exists():
        print(f"❌ CSV not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path, encoding="utf-8")
    if "hospital" not in df.columns:
        print("❌ CSV has no 'hospital' column — cannot geocode.")
        sys.exit(1)
    if "latitude" not in df.columns:
        df["latitude"] = pd.NA
    if "longitude" not in df.columns:
        df["longitude"] = pd.NA

    district_col = args.district_col if args.district_col in df.columns else None
    address_col  = args.address_col if args.address_col in df.columns else None

    geolocator = Nominatim(user_agent=USER_AGENT, timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=MIN_DELAY_SECONDS)

    cache = load_cache()

    if args.compare:
        total = len(df)
        print(f"Comparing {total} hospitals against OSM (CSV will NOT be modified)…\n")
        report_df = run_compare(df, geocode, district_col, cache, address_col)

        try:
            save_cache(cache)
        except Exception as e:
            print(f"⚠️ Warning: could not save cache ({e}) — continuing anyway")

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(REPORT_PATH, index=False, encoding="utf-8")

        n_mismatch = (report_df["status"] == "mismatch").sum()
        n_close    = (report_df["status"] == "matches_closely").sum()
        n_new      = (report_df["status"] == "existing_invalid_new_available").sum()
        n_failed   = (report_df["status"] == "geocode_failed").sum()

        print("\n" + "═" * 60)
        print("COMPARE DONE — nothing was written to the CSV")
        print(f"  Matches closely (≤{MISMATCH_KM_THRESHOLD} km):  {n_close}")
        print(f"  ⚠ Mismatches (>{MISMATCH_KM_THRESHOLD} km):     {n_mismatch}")
        print(f"  ➕ No valid existing coord, new one found:  {n_new}")
        print(f"  ❌ Could not geocode at all:                {n_failed}")
        print(f"\n  Review: {REPORT_PATH}")
        print(f"  Then run: python geocode_hospitals.py --apply")
        print("  (this copies new_lat/new_lon into the CSV for 'mismatch' and")
        print("   'existing_invalid_new_available' rows — edit/delete rows in")
        print("   the report first if you don't want a particular one applied)")
        print("═" * 60)
        return

    report_rows = []
    updated = 0
    kept = 0
    failed = 0

    total = len(df)
    print(f"Processing {total} hospitals from {csv_path}…\n")

    for idx, row in df.iterrows():
        hospital = str(row["hospital"]).strip()
        existing_lat, existing_lon = row.get("latitude"), row.get("longitude")
        district = row.get(district_col) if district_col else None
        address = row.get(address_col) if address_col else None

        # 1. Skip if existing coordinates already look valid, unless --force
        if not args.force and in_karnataka(existing_lat, existing_lon):
            report_rows.append({
                "hospital": hospital, "status": "kept_existing",
                "latitude": existing_lat, "longitude": existing_lon,
                "matched_query": "", "source": "original_csv",
            })
            kept += 1
            continue

        # 2. Check cache first (avoids re-hitting the API on re-runs)
        cache_key = hospital.lower().strip()
        if not args.force and cache_key in cache and cache[cache_key].get("status") == "geocoded":
            c = cache[cache_key]
            df.at[idx, "latitude"]  = c["latitude"]
            df.at[idx, "longitude"] = c["longitude"]
            report_rows.append({
                "hospital": hospital, "status": "geocoded_from_cache",
                "latitude": c["latitude"], "longitude": c["longitude"],
                "matched_query": c.get("matched_query", ""), "source": "nominatim",
            })
            updated += 1
            continue

        # 3. Try each query variant against Nominatim until one lands in Karnataka
        print(f"[{idx+1}/{total}] Geocoding: {hospital}")
        found = False
        for query in build_query_variants(hospital, district, address):
            location = geocode_with_retries(geocode, query)
            if location and in_karnataka(location.latitude, location.longitude):
                df.at[idx, "latitude"]  = round(location.latitude, 6)
                df.at[idx, "longitude"] = round(location.longitude, 6)
                cache[cache_key] = {
                    "status": "geocoded",
                    "latitude": round(location.latitude, 6),
                    "longitude": round(location.longitude, 6),
                    "matched_query": query,
                    "display_name": location.address,
                }
                report_rows.append({
                    "hospital": hospital, "status": "geocoded",
                    "latitude": round(location.latitude, 6),
                    "longitude": round(location.longitude, 6),
                    "matched_query": query, "source": "nominatim",
                })
                print(f"    ✅ {location.latitude:.5f}, {location.longitude:.5f}  (query: \"{query}\")")
                updated += 1
                found = True
                break
            elif location:
                print(f"    ✗ result outside Karnataka for \"{query}\" — trying next variant")

        if not found:
            cache[cache_key] = {"status": "failed"}
            fallback_note = ""
            if in_karnataka(existing_lat, existing_lon) is False and pd.notna(existing_lat):
                fallback_note = "kept_original_out_of_bounds"
            report_rows.append({
                "hospital": hospital, "status": "failed_needs_manual_review",
                "latitude": existing_lat, "longitude": existing_lon,
                "matched_query": "", "source": fallback_note or "none",
            })
            print(f"    ❌ could not resolve — flagged for manual review")
            failed += 1

        # Checkpoint periodically so progress isn't lost on interruption
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            try:
                save_cache(cache)
            except Exception as e:
                print(f"    ⚠️ Warning: could not save cache checkpoint ({e})")
            df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"    💾 checkpoint saved ({idx+1}/{total})")

    # Final save
    try:
        save_cache(cache)
    except Exception as e:
        print(f"⚠️ Warning: could not save final cache ({e}) — continuing anyway")

    df.to_csv(csv_path, index=False, encoding="utf-8")

    report_df = pd.DataFrame(report_rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_PATH, index=False, encoding="utf-8")

    print("\n" + "═" * 60)
    print("DONE")
    print(f"  Kept existing (already valid):  {kept}")
    print(f"  Newly geocoded:                 {updated}")
    print(f"  Failed / needs manual review:   {failed}")
    print(f"  Updated CSV:  {csv_path}")
    print(f"  Full report:  {REPORT_PATH}")
    if failed:
        print(f"\n⚠  {failed} hospitals need manual review — see status="
              f"'failed_needs_manual_review' rows in {REPORT_PATH}.")
        print("   Common causes: hospital renamed/closed on OSM, name typo,")
        print("   or a very small clinic OSM doesn't index. Search Google")
        print("   Maps manually for those and paste coordinates in by hand.")
    print("═" * 60)


if __name__ == "__main__":
    main()
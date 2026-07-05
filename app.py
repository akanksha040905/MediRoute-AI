"""
app.py — MediRoute AI · Karnataka ICU Allocation
High-contrast dark theme. All text explicitly styled. Professional charts.
"""

import logging, warnings
from datetime import datetime

import folium
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from streamlit_folium import st_folium

from config import (
    DEFAULT_LAT, DEFAULT_LON, MAP_ZOOM,
    SEVERITY_LEVELS, BED_ALERT_THRESHOLD, CSV_COLUMNS,
    PROXIMITY_CANDIDATES,
)
from data_manager import (
    load_data, add_hospital, update_hospital,
    delete_hospital, log_allocation, load_log,
)
from model import load_or_train_model, predict_beds, train_model
from utils import enrich_with_routes, rank_hospitals, detect_location_from_ip

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="MediRoute AI — Karnataka ICU",
    page_icon="🏥", layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════
#  THEME CONFIG FILE  (.streamlit/config.toml handles base)
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg:      #07111f;
  --surf:    #0e1c2f;
  --surf2:   #152338;
  --surf3:   #1b2d46;
  --bdr:     #1f3452;
  --bdr2:    #2a4468;
  --teal:    #00d4aa;
  --teal2:   #00ffcc;
  --red:     #ff4560;
  --amber:   #ffb830;
  --green:   #00e096;
  --violet:  #9b72f5;
  --blue:    #4d9ef5;
  --txt:     #e8f2ff;
  --txt2:    #8aaac8;
  --txt3:    #4a6882;
}

/* ── Base ── */
html, body, .stApp, [class*="css"] { background: var(--bg) !important; }
.main { background: var(--bg) !important; }
.main .block-container { background: var(--bg) !important; padding: 1.6rem 2rem 4rem !important; max-width: 1500px !important; }

/* ── ALL TEXT VISIBLE ── */
p, div, li, td, th, label, small { color: var(--txt) !important; font-family: 'Inter', sans-serif !important; }
h1,h2,h3,h4,h5 { color: var(--txt) !important; font-weight: 700 !important; font-family: 'Inter', sans-serif !important; }
.stMarkdown p, .stMarkdown div { color: var(--txt) !important; }
.stCaption, .caption { color: var(--txt3) !important; font-size: 0.72rem !important; }
a { color: var(--teal) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: var(--surf) !important; border-right: 1px solid var(--bdr) !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] div, [data-testid="stSidebar"] label { color: var(--txt2) !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: var(--surf2) !important; border: 1px solid var(--bdr) !important;
  border-top: 2px solid var(--teal) !important; border-radius: 10px !important;
  padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] div,
[data-testid="stMetricLabel"] span,
[data-testid="stMetric"] [data-testid="stMetricLabel"] { color: var(--txt3) !important; font-size: 0.67rem !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.10em !important; }
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] { color: var(--txt) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 1.55rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] div,
[data-testid="stMetricDelta"] { color: var(--txt2) !important; font-size: 0.72rem !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--bdr) !important; }
[data-testid="stTabs"] button[role="tab"] {
  background: transparent !important; color: var(--txt2) !important;
  border: none !important; border-bottom: 2px solid transparent !important;
  font-size: 0.82rem !important; font-weight: 500 !important;
  padding: 10px 18px !important; font-family: 'Inter', sans-serif !important;
}
[data-testid="stTabs"] button[role="tab"]:hover { color: var(--txt) !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--teal) !important; border-bottom: 2px solid var(--teal) !important; font-weight: 700 !important; }
[data-testid="stTabs"] p, [data-testid="stTabs"] div,
[data-testid="stTabs"] label { color: var(--txt) !important; }

/* ── Buttons ── */
[data-testid="stButton"] > button {
  background: var(--surf2) !important; color: var(--txt2) !important;
  border: 1px solid var(--bdr) !important; border-radius: 8px !important;
  font-size: 0.83rem !important; font-family: 'Inter', sans-serif !important;
}
[data-testid="stButton"] > button:hover { border-color: var(--teal) !important; color: var(--teal) !important; }
[data-testid="stButton"] > button[kind="primary"] {
  background: var(--teal) !important; color: #03111e !important;
  border: none !important; font-weight: 700 !important; font-size: 0.88rem !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover { background: var(--teal2) !important; box-shadow: 0 0 24px rgba(0,212,170,0.4) !important; }

/* ── Inputs ── */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
  background: var(--surf2) !important; border: 1px solid var(--bdr) !important;
  color: var(--txt) !important; border-radius: 8px !important; font-family: 'Inter', sans-serif !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--txt3) !important; }
[data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus {
  border-color: var(--teal) !important; box-shadow: 0 0 0 2px rgba(0,212,170,0.18) !important;
}
[data-testid="stTextInput"] label, [data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label, [data-testid="stSlider"] label,
[data-testid="stCheckbox"] label, [data-testid="stCheckbox"] span,
.stSelectSlider label { color: var(--txt2) !important; font-size: 0.80rem !important; }

/* Selectbox dropdown */
[data-baseweb="select"] div, [data-baseweb="select"] span { background: var(--surf2) !important; color: var(--txt) !important; border-color: var(--bdr) !important; }
[data-baseweb="popover"] ul { background: var(--surf2) !important; }
[data-baseweb="popover"] li { color: var(--txt) !important; }

/* Select slider */
[data-testid="stSlider"] [role="slider"] { background: var(--teal) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"],
[data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] [data-testid="stTickBarMin"] { color: var(--txt2) !important; }

/* Checkbox */
[data-testid="stCheckbox"] p { color: var(--txt) !important; font-size: 0.86rem !important; }
[data-testid="stCheckbox"] input[type="checkbox"] { accent-color: var(--teal) !important; }

/* ── Expanders ── */
[data-testid="stExpander"] { background: var(--surf) !important; border: 1px solid var(--bdr) !important; border-radius: 10px !important; margin-bottom: 8px !important; }
[data-testid="stExpander"] summary { color: var(--txt) !important; font-size: 0.85rem !important; font-weight: 500 !important; padding: 14px 18px !important; }
[data-testid="stExpander"] summary:hover { color: var(--teal) !important; }
[data-testid="stExpander"] p, [data-testid="stExpander"] div { color: var(--txt) !important; }
[data-testid="stExpander"] label { color: var(--txt) !important; }
[data-testid="stExpander"] summary span:not([class*="icon"]):not([data-testid*="Icon"]) { color: var(--txt) !important; }
[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] { color: var(--txt2) !important; font-size: 1rem !important; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border: 1px solid var(--bdr) !important; border-radius: 10px !important; }
[data-testid="stDataFrame"] * { color: var(--txt) !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 8px !important; }
[data-testid="stAlert"] p, [data-testid="stAlert"] div { color: var(--txt) !important; }

/* ── Form ── */
[data-testid="stForm"] { background: var(--surf) !important; border: 1px solid var(--bdr) !important; border-radius: 12px !important; padding: 20px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--surf); }
::-webkit-scrollbar-thumb { background: var(--bdr2); border-radius: 4px; }

hr { border: none !important; border-top: 1px solid var(--bdr) !important; margin: 16px 0 !important; }

/* ═══ CUSTOM COMPONENTS ═══ */
.page-title { font-size: 1.55rem; font-weight: 800; color: var(--txt) !important; letter-spacing: -0.03em; margin-bottom: 2px; }
.page-title .acc { color: var(--teal) !important; }
.page-sub { font-size: 0.69rem; color: var(--txt3) !important; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 20px; }

/* Panel */
.panel { background: var(--surf); border: 1px solid var(--bdr); border-radius: 12px; padding: 18px 20px; margin-bottom: 12px; }
.plabel { font-size: 0.65rem; font-weight: 700; color: var(--txt3) !important; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px; }

/* Section label */
.slabel { font-size: 0.65rem; font-weight: 700; color: var(--txt3) !important; text-transform: uppercase; letter-spacing: 0.12em; margin: 14px 0 8px; }

/* Rec banner */
.rec-banner {
  background: linear-gradient(135deg, #091729 0%, #0c1e36 100%);
  border: 1px solid var(--teal); border-radius: 14px;
  padding: 22px 26px; margin-bottom: 20px; position: relative; overflow: hidden;
}
.rec-banner::after {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, var(--teal), transparent);
}
.rec-eye { font-size: 0.63rem; font-weight: 700; color: var(--teal) !important; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 8px; }
.rec-name { font-size: 1.15rem; font-weight: 800; color: var(--txt) !important; margin-bottom: 16px; letter-spacing: -0.02em; }
.rec-name .t { color: var(--teal) !important; }
.rec-stats { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; }
.rs { background: rgba(255,255,255,0.04); border: 1px solid var(--bdr); border-radius: 8px; padding: 8px 14px; min-width: 85px; }
.rs .rl { font-size: 0.61rem; font-weight: 700; color: var(--txt3) !important; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 2px; }
.rs .rv { font-size: 1.0rem; font-weight: 700; color: var(--txt) !important; font-family: 'JetBrains Mono', monospace; }
.rv.teal { color: var(--teal) !important; }
.rv.red  { color: var(--red) !important; }
.rv.amb  { color: var(--amber) !important; }

/* Phone pill */
.pill { display: inline-flex; align-items: center; gap: 6px; background: rgba(0,212,170,0.08); border: 1px solid rgba(0,212,170,0.30); color: var(--teal) !important; border-radius: 100px; padding: 5px 15px; font-size: 0.79rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

/* Severity badge */
.sev { display: inline-block; border-radius: 5px; padding: 3px 11px; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; }
.sev-Low      { background: rgba(0,224,150,0.12);  color: #00e096 !important; border: 1px solid rgba(0,224,150,0.30); }
.sev-Medium   { background: rgba(255,184,48,0.12); color: #ffb830 !important; border: 1px solid rgba(255,184,48,0.30); }
.sev-High     { background: rgba(255,69,96,0.12);  color: #ff4560 !important; border: 1px solid rgba(255,69,96,0.30); }
.sev-Critical { background: rgba(255,69,96,0.18);  color: #ff1a3d !important; border: 1px solid rgba(255,26,61,0.55); animation: pls 1.3s ease infinite; }
@keyframes pls { 0%,100%{ box-shadow: 0 0 0 0 rgba(255,26,61,0.5); } 50%{ box-shadow: 0 0 0 7px rgba(255,26,61,0); } }

/* Low bed strip */
.lowbed { background: rgba(255,69,96,0.06); border: 1px solid rgba(255,69,96,0.22); border-left: 3px solid var(--red); border-radius: 8px; padding: 10px 14px; margin-top: 10px; font-size: 0.76rem; }
.lowbed *, .lowbed b, .lowbed div { color: var(--red) !important; }

/* Delete warning */
.del { background: rgba(255,69,96,0.06); border: 1px solid rgba(255,69,96,0.22); border-left: 3px solid var(--red); border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; font-size: 0.84rem; color: var(--red) !important; }

/* Live dot */
.ldot { display: inline-block; width: 7px; height: 7px; background: var(--teal); border-radius: 50%; margin-right: 6px; box-shadow: 0 0 6px var(--teal); animation: blink 2s ease-in-out infinite; }
@keyframes blink { 0%,100%{ opacity:1; } 50%{ opacity:0.2; } }

/* Arch table */
.atbl { border-collapse: collapse; width: 100%; }
.atbl td { border-bottom: 1px solid var(--bdr); padding: 7px 0; }
.atbl .ak { color: var(--txt3) !important; font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; padding-right: 14px; white-space: nowrap; }
.atbl .av { color: var(--txt) !important; font-size: 0.78rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  MATPLOTLIB DARK THEME
# ══════════════════════════════════════════════════════
TEAL   = "#00d4aa"
RED    = "#ff4560"
AMBER  = "#ffb830"
VIOLET = "#9b72f5"
BLUE   = "#4d9ef5"
GREEN  = "#00e096"

plt.rcParams.update({
    "figure.facecolor": "#07111f",
    "axes.facecolor":   "#0e1c2f",
    "axes.edgecolor":   "#1f3452",
    "axes.labelcolor":  "#8aaac8",
    "axes.titlecolor":  "#e8f2ff",
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
    "axes.labelsize":   9,
    "axes.axisbelow":   True,
    "text.color":       "#e8f2ff",
    "xtick.color":      "#8aaac8",
    "ytick.color":      "#e8f2ff",
    "xtick.labelsize":  8.5,
    "ytick.labelsize":  8.5,
    "grid.color":       "#152338",
    "grid.linestyle":   "--",
    "grid.alpha":       0.7,
    "axes.grid":        True,
    "legend.facecolor": "#0e1c2f",
    "legend.edgecolor": "#1f3452",
    "legend.labelcolor":"#e8f2ff",
    "legend.fontsize":  8.5,
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "figure.dpi":       130,
})

# ══════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════
if "df_raw" not in st.session_state:
    st.session_state.df_raw = load_data()
if "patient_lat"    not in st.session_state: st.session_state.patient_lat    = DEFAULT_LAT
if "patient_lon"    not in st.session_state: st.session_state.patient_lon    = DEFAULT_LON
if "top3"           not in st.session_state: st.session_state.top3           = None
if "location_name"  not in st.session_state: st.session_state.location_name  = ""
if "location_set"   not in st.session_state: st.session_state.location_set   = False

# ── Reverse geocode: lat/lon → human readable place name ──────────────────
def reverse_geocode(lat: float, lon: float) -> str:
    try:
        import requests as _req
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        r   = _req.get(url, timeout=4,
                       headers={"User-Agent": "MediRouteAI/1.0"})
        data = r.json()
        addr = data.get("address", {})
        # Build a clean readable name: neighbourhood / suburb / city
        parts = []
        for key in ["suburb", "neighbourhood", "quarter", "village",
                    "town", "city", "county", "state_district"]:
            if addr.get(key):
                parts.append(addr[key])
            if len(parts) >= 2:
                break
        state = addr.get("state", "Karnataka")
        return ", ".join(parts) + f", {state}" if parts else data.get("display_name","")[:60]
    except Exception:
        return ""
if "model_bundle"  not in st.session_state:
    with st.spinner("Initialising MediRoute AI…"):
        st.session_state.model_bundle = load_or_train_model(st.session_state.df_raw)

df_raw = st.session_state.df_raw
bundle = st.session_state.model_bundle
df     = predict_beds(bundle, df_raw)

total_hosp  = len(df)
total_avail = int(df["available_beds"].sum())
total_cap   = int(df["total_beds"].sum())
crit_count  = int(df["critical_patients"].sum())
avg_occ     = (1 - total_avail / max(total_cap, 1)) * 100
low_bed     = df[df["available_beds"] <= BED_ALERT_THRESHOLD]

# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 16px;">
      <div style="font-size:1.05rem;font-weight:800;color:#00d4aa;letter-spacing:0.03em;">MediRoute AI</div>
      <div style="font-size:0.69rem;color:#4a6882;margin-top:3px;">Karnataka ICU Network</div>
    </div><hr>
    <div style="font-size:0.64rem;color:#4a6882;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;">
      <span class="ldot"></span>Live System Status
    </div>
    """, unsafe_allow_html=True)

    st.metric("Hospitals Online",   total_hosp)
    st.metric("Available ICU Beds", total_avail,
              delta=f"{avg_occ:.0f}% occupancy", delta_color="inverse")
    st.metric("Critical Patients",  crit_count)

    if not low_bed.empty:
        names   = low_bed["hospital"].tolist()
        bullets = "".join(f"<div>· {h}</div>" for h in names[:6])
        more    = f"<div style='color:#4a6882'>+{len(names)-6} more…</div>" if len(names) > 6 else ""
        st.markdown(f'<div class="lowbed"><b>⚠ {len(names)} hospitals critically low</b><div style="margin-top:5px">{bullets}{more}</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption(f"Last sync · {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄  Refresh & Retrain", use_container_width=True):
        st.session_state.df_raw = load_data()
        st.session_state.model_bundle = train_model(st.session_state.df_raw)
        st.rerun()

# ══════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚑  Find Hospital", "🏥  Registry",
    "📊  Analytics",     "📋  Allocation Log", "⚙  Model Info",
])

# ══════════════════════════════════════════════════════
#  TAB 1 — FIND HOSPITAL
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="page-title">Find <span class="acc">Best ICU</span></div>
    <div class="page-sub">Proximity-first · AI-scored · Karnataka hospitals</div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 2], gap="large")

    with left:
        # Severity
        st.markdown('<div class="panel"><div class="plabel">Emergency Severity</div>', unsafe_allow_html=True)
        severity = st.select_slider("_sev", options=SEVERITY_LEVELS, value="Medium",
                                    label_visibility="collapsed")
        cands = PROXIMITY_CANDIDATES.get(severity, 30)
        sev_desc = {
            "Low":      f"Stable patient · Scans {cands} nearest hospitals · Prioritises resources",
            "Medium":   f"Urgent · Scans {cands} nearest · Balances distance and capacity",
            "High":     f"Emergency · Scans {cands} nearest · Strong proximity bias",
            "Critical": f"Life-threatening · Scans {cands} nearest · Nearest hospital first always",
        }
        st.markdown(f"""
        <div style="margin-top:10px;display:flex;align-items:center;gap:10px;">
          <span class="sev sev-{severity}">{severity}</span>
        </div>
        <div style="font-size:0.77rem;color:#8aaac8;margin-top:8px;line-height:1.5;">{sev_desc[severity]}</div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Location — auto GPS on page load, no buttons needed
        st.markdown('<div class="panel"><div class="plabel">Patient Location</div>', unsafe_allow_html=True)

        from streamlit_js_eval import get_geolocation
        loc = get_geolocation()

        if loc and loc.get("coords"):
            g_lat = float(loc["coords"]["latitude"])
            g_lon = float(loc["coords"]["longitude"])
            # Only update if changed (avoid infinite rerun)
            if (abs(g_lat - st.session_state.patient_lat) > 0.0001 or
                abs(g_lon - st.session_state.patient_lon) > 0.0001 or
                not st.session_state.location_set):
                st.session_state.patient_lat   = g_lat
                st.session_state.patient_lon   = g_lon
                st.session_state.location_name = reverse_geocode(g_lat, g_lon)
                st.session_state.location_set  = True

        lat      = st.session_state.patient_lat
        lon      = st.session_state.patient_lon
        loc_name = st.session_state.get("location_name", "")

        if st.session_state.get("location_set") and loc_name:
            st.markdown(f"""
            <div style="background:#0a1f0f;border:1px solid #00d4aa55;
                        border-left:3px solid #00d4aa;border-radius:8px;padding:12px 16px;">
              <div style="font-size:0.62rem;color:#4a6882;text-transform:uppercase;
                          letter-spacing:0.1em;margin-bottom:4px;">Your Location (GPS)</div>
              <div style="font-size:0.92rem;font-weight:700;color:#e8f2ff;">📍 {loc_name}</div>
              <div style="font-size:0.71rem;color:#4a6882;margin-top:3px;
                          font-family:JetBrains Mono,monospace;">{lat:.5f}, {lon:.5f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#0e1c2f;border:1px solid #1f3452;border-radius:8px;
                        padding:12px 16px;text-align:center;">
              <div style="font-size:0.84rem;color:#8aaac8;margin-bottom:4px;">
                📍 Detecting your location…
              </div>
              <div style="font-size:0.72rem;color:#4a6882;">
                Allow location access when your browser asks
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Requirements
        st.markdown('<div class="panel"><div class="plabel">Special Requirements</div>', unsafe_allow_html=True)
        needs_vent = st.checkbox("Requires ventilator support")
        needs_spec = st.checkbox("Requires ICU specialist on site")
        st.markdown('</div>', unsafe_allow_html=True)

        find_btn = st.button("🚑  FIND BEST HOSPITAL", type="primary", use_container_width=True)

    with right:
        if find_btn:
            with st.spinner(f"Scanning {PROXIMITY_CANDIDATES.get(severity,30)} nearest hospitals…"):
                df_routed = enrich_with_routes(df, lat, lon, severity=severity)
                df_filt   = df_routed.copy()
                if needs_vent: df_filt = df_filt[df_filt["ventilators"] > 0]
                if needs_spec: df_filt = df_filt[df_filt["icu_specialist_count"] >= 1]
                if df_filt.empty:
                    st.warning("No hospital meets the filter — showing all nearby.")
                    df_filt = df_routed
                top3 = rank_hospitals(df_filt, severity)
                st.session_state.top3 = top3
                best = top3.iloc[0]
                log_allocation(lat, lon, severity, best["hospital"],
                               best["eta_min"], int(best["predicted_beds"]),
                               str(best.get("contact_number","N/A")))

        if st.session_state.top3 is not None:
            top3  = st.session_state.top3
            best  = top3.iloc[0]
            avail = int(best["available_beds"])
            pred  = int(best["predicted_beds"])
            eta   = best["eta_min"]
            dist  = best["distance_km"]
            conf  = best["confidence"]
            cont  = str(best.get("contact_number","N/A"))
            eta_c = "red" if eta > 30 else "amb" if eta > 15 else "teal"
            bed_c = "red" if avail <= BED_ALERT_THRESHOLD else "teal"

            st.markdown(f"""
            <div class="rec-banner">
              <div class="rec-eye">✦ Primary Recommendation</div>
              <div class="rec-name">🏥 <span class="t">{best['hospital']}</span></div>
              <div class="rec-stats">
                <div class="rs"><div class="rl">Distance</div><div class="rv">{dist:.1f} km</div></div>
                <div class="rs"><div class="rl">ETA</div><div class="rv {eta_c}">{eta:.0f} min</div></div>
                <div class="rs"><div class="rl">Available Beds</div><div class="rv {bed_c}">{avail}</div></div>
                <div class="rs"><div class="rl">Predicted Safe</div><div class="rv">{pred}</div></div>
                <div class="rs"><div class="rl">Confidence</div><div class="rv teal">{conf*100:.0f}%</div></div>
              </div>
              <span class="pill">📞 {cont}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="slabel">Top 3 Recommendations — sorted by distance from your location</div>', unsafe_allow_html=True)
            rank_labels = ["PRIMARY", "ALTERNATIVE 1", "ALTERNATIVE 2"]
            rank_colors = ["#00d4aa", "#ffb830", "#9b72f5"]
            for i, (_, row) in enumerate(top3.iterrows()):
                rc = str(row.get("contact_number","N/A"))
                bed_warn = row["available_beds"] <= BED_ALERT_THRESHOLD
                warn_html = f'<div style="margin-top:8px;padding:8px 12px;background:rgba(255,69,96,0.1);border:1px solid rgba(255,69,96,0.3);border-radius:6px;font-size:0.82rem;color:#ff4560;">Only {int(row["available_beds"])} bed(s) remaining — act immediately.</div>' if bed_warn else ""
                st.markdown(f'''
                <div style="background:#0e1c2f;border:1px solid {rank_colors[i]};border-radius:12px;padding:18px 20px;margin-bottom:12px;">
                  <div style="font-size:0.62rem;font-weight:700;color:{rank_colors[i]};text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px;">{rank_labels[i]}</div>
                  <div style="font-size:1.0rem;font-weight:700;color:#e8f2ff;margin-bottom:14px;">{row["hospital"]}</div>
                  <div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:12px;">
                    <div style="background:rgba(255,255,255,0.04);border:1px solid #1f3452;border-radius:8px;padding:8px 14px;min-width:80px;">
                      <div style="font-size:0.60rem;font-weight:700;color:#4a6882;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">Distance</div>
                      <div style="font-size:0.95rem;font-weight:700;color:#e8f2ff;font-family:JetBrains Mono,monospace;">{row["distance_km"]:.1f} km</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border:1px solid #1f3452;border-radius:8px;padding:8px 14px;min-width:80px;">
                      <div style="font-size:0.60rem;font-weight:700;color:#4a6882;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">ETA</div>
                      <div style="font-size:0.95rem;font-weight:700;color:{'#ff4560' if row['eta_min']>30 else '#ffb830' if row['eta_min']>15 else '#00d4aa'};font-family:JetBrains Mono,monospace;">{row["eta_min"]:.0f} min</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border:1px solid #1f3452;border-radius:8px;padding:8px 14px;min-width:80px;">
                      <div style="font-size:0.60rem;font-weight:700;color:#4a6882;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">Available Beds</div>
                      <div style="font-size:0.95rem;font-weight:700;color:{'#ff4560' if row['available_beds']<=BED_ALERT_THRESHOLD else '#00d4aa'};font-family:JetBrains Mono,monospace;">{int(row["available_beds"])}</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border:1px solid #1f3452;border-radius:8px;padding:8px 14px;min-width:80px;">
                      <div style="font-size:0.60rem;font-weight:700;color:#4a6882;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">Predicted Safe</div>
                      <div style="font-size:0.95rem;font-weight:700;color:#e8f2ff;font-family:JetBrains Mono,monospace;">{int(row["predicted_beds"])}</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border:1px solid #1f3452;border-radius:8px;padding:8px 14px;min-width:80px;">
                      <div style="font-size:0.60rem;font-weight:700;color:#4a6882;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">Ventilators</div>
                      <div style="font-size:0.95rem;font-weight:700;color:#e8f2ff;font-family:JetBrains Mono,monospace;">{int(row["ventilators"])}</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border:1px solid #1f3452;border-radius:8px;padding:8px 14px;min-width:80px;">
                      <div style="font-size:0.60rem;font-weight:700;color:#4a6882;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">Queue</div>
                      <div style="font-size:0.95rem;font-weight:700;color:#e8f2ff;font-family:JetBrains Mono,monospace;">{int(row["waiting_queue"])}</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border:1px solid #1f3452;border-radius:8px;padding:8px 14px;min-width:80px;">
                      <div style="font-size:0.60rem;font-weight:700;color:#4a6882;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">O2 Supply</div>
                      <div style="font-size:0.95rem;font-weight:700;color:#e8f2ff;font-family:JetBrains Mono,monospace;">{row.get("oxygen_supply_pct","N/A")}%</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.04);border:1px solid #1f3452;border-radius:8px;padding:8px 14px;min-width:80px;">
                      <div style="font-size:0.60rem;font-weight:700;color:#4a6882;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">ICU Specialists</div>
                      <div style="font-size:0.95rem;font-weight:700;color:#e8f2ff;font-family:JetBrains Mono,monospace;">{int(row.get("icu_specialist_count",0))}</div>
                    </div>
                  </div>
                  <span class="pill">📞 {rc}</span>
                  {warn_html}
                </div>
                ''', unsafe_allow_html=True)

            # ── Google Maps direct link ───────────────────────────────────
            st.markdown('<div class="slabel" style="margin-top:18px;">Open in Google Maps</div>', unsafe_allow_html=True)

            hosp_rows = [row for _, row in top3.iterrows()]

            # ── Correct Google Maps URLs ──────────────────────────────────
            # Navigate: origin + destination using the Maps Directions API format.
            # FIX: destination now always uses the hospital's exact lat,lon
            # coordinates instead of its name. Passing the name as a text
            # query lets Google Maps search-match it to a similarly named
            # place or the wrong branch/address — coordinates are always
            # exact, since that's the same lat/lon used to draw the map pin.
            import urllib.parse

            def nav_link(dest_lat, dest_lon, dest_name=""):
                dest = f"{dest_lat},{dest_lon}"
                return (f"https://www.google.com/maps/dir/?api=1"
                        f"&origin={lat},{lon}"
                        f"&destination={dest}"
                        f"&travelmode=driving")

            # All 3 hospitals route: use waypoints param
            wpts = "|".join(f"{r['latitude']},{r['longitude']}" for r in hosp_rows[1:])
            all3_url = (f"https://www.google.com/maps/dir/?api=1"
                        f"&origin={lat},{lon}"
                        f"&destination={hosp_rows[-1]['latitude']},{hosp_rows[-1]['longitude']}"
                        f"&waypoints={hosp_rows[0]['latitude']},{hosp_rows[0]['longitude']}"
                        + (f"|{hosp_rows[1]['latitude']},{hosp_rows[1]['longitude']}" if len(hosp_rows) > 1 else "")
                        + f"&travelmode=driving")

            nav1 = nav_link(hosp_rows[0]['latitude'], hosp_rows[0]['longitude'], hosp_rows[0]['hospital'])
            nav2 = nav_link(hosp_rows[1]['latitude'], hosp_rows[1]['longitude'], hosp_rows[1]['hospital']) if len(hosp_rows) > 1 else ""
            nav3 = nav_link(hosp_rows[2]['latitude'], hosp_rows[2]['longitude'], hosp_rows[2]['hospital']) if len(hosp_rows) > 2 else ""

            h2_btn = f'''<a href="{nav2}" target="_blank" style="text-decoration:none;flex:1;min-width:160px;">
              <div class="gmaps-btn gmaps-alt">🥈 {hosp_rows[1]["hospital"]}<br>
              <span>{hosp_rows[1]["distance_km"]:.1f} km · {hosp_rows[1]["eta_min"]:.0f} min · Navigate</span></div>
            </a>''' if nav2 else ""

            h3_btn = f'''<a href="{nav3}" target="_blank" style="text-decoration:none;flex:1;min-width:160px;">
              <div class="gmaps-btn gmaps-alt">🥉 {hosp_rows[2]["hospital"]}<br>
              <span>{hosp_rows[2]["distance_km"]:.1f} km · {hosp_rows[2]["eta_min"]:.0f} min · Navigate</span></div>
            </a>''' if nav3 else ""

            st.markdown(f"""
            <style>
            .gmaps-row {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px; }}
            .gmaps-btn {{
              flex:1; min-width:160px; background:#152338; border:1px solid #1f3452;
              border-radius:10px; padding:12px 16px; font-size:0.82rem; font-weight:600;
              color:#e8f2ff !important; cursor:pointer; transition:all 0.18s; line-height:1.5;
            }}
            .gmaps-btn:hover {{ border-color:#00d4aa; background:#0e1c2f; }}
            .gmaps-btn span {{ font-size:0.71rem; color:#4a6882 !important; font-weight:400; display:block; }}
            .gmaps-nav  {{ background:linear-gradient(135deg,#0d2a52,#112e5c); border:1px solid #4d9ef5 !important; }}
            .gmaps-all  {{ background:linear-gradient(135deg,#1e1040,#241450); border:1px solid #9b72f5 !important; }}
            .gmaps-alt  {{ background:linear-gradient(135deg,#0d2420,#102820); border:1px solid #00d4aa88 !important; }}
            </style>

            <div class="gmaps-row">
              <a href="{nav1}" target="_blank" style="text-decoration:none;flex:1;min-width:200px;">
                <div class="gmaps-btn gmaps-nav">
                  🧭 Navigate to Best Hospital
                  <span>{hosp_rows[0]["hospital"]} · {hosp_rows[0]["distance_km"]:.1f} km · {hosp_rows[0]["eta_min"]:.0f} min ETA · Opens Google Maps with your current location as origin</span>
                </div>
              </a>
              <a href="{all3_url}" target="_blank" style="text-decoration:none;flex:1;min-width:200px;">
                <div class="gmaps-btn gmaps-all">
                  🗺 Route Through All 3 Hospitals
                  <span>Opens Google Maps with all recommended hospitals as stops</span>
                </div>
              </a>
            </div>
            <div class="gmaps-row">
              <a href="{nav1}" target="_blank" style="text-decoration:none;flex:1;min-width:160px;">
                <div class="gmaps-btn gmaps-alt">🥇 {hosp_rows[0]["hospital"]}
                <span>{hosp_rows[0]["distance_km"]:.1f} km · {hosp_rows[0]["eta_min"]:.0f} min · {int(hosp_rows[0]["available_beds"])} beds · Tap to navigate</span></div>
              </a>
              {h2_btn}
              {h3_btn}
            </div>
            """, unsafe_allow_html=True)

            # ── Folium interactive map ────────────────────────────────────
            st.markdown('<div class="slabel" style="margin-top:6px;">Interactive Route Map — click markers for details</div>', unsafe_allow_html=True)

            # Use OpenStreetMap as base then style with CartoDB dark
            m = folium.Map(
                location=[lat, lon],
                zoom_start=12,
                tiles=None,
                prefer_canvas=True,
            )

            # Dark tiles — CartoDB dark matter
            folium.TileLayer(
                tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                attr="CartoDB",
                name="Dark",
                max_zoom=19,
            ).add_to(m)

            # Patient location — pulsing blue circle + marker
            folium.CircleMarker(
                [lat, lon], radius=18,
                color="#4d9ef5", fill=True, fill_color="#4d9ef5",
                fill_opacity=0.15, weight=2, opacity=0.6,
            ).add_to(m)
            folium.CircleMarker(
                [lat, lon], radius=8,
                color="#4d9ef5", fill=True, fill_color="#4d9ef5",
                fill_opacity=0.9, weight=2,
                tooltip="📍 Your Location",
            ).add_to(m)

            # Route colour by ETA — traffic light style
            def route_color(eta):
                if eta <= 10: return "#00d4aa"   # green — fast
                if eta <= 20: return "#ffb830"   # amber — moderate
                return "#ff4560"                  # red — slow

            rank_labels_map = ["PRIMARY", "ALT 1", "ALT 2"]
            for i, (_, row) in enumerate(top3.iterrows()):
                rc    = str(row.get("contact_number","N/A"))
                rcol  = route_color(row["eta_min"])
                avail = int(row["available_beds"])
                beds_col = "#ff4560" if avail <= BED_ALERT_THRESHOLD else "#00d4aa"

                # Rich HTML popup — Google Maps style
                # FIX: navigation link in popup now uses exact hospital
                # coordinates as destination, not the hospital name.
                popup_html = f"""
                <div style="font-family:Inter,sans-serif;min-width:240px;background:#0e1c2f;
                            color:#e8f2ff;border-radius:10px;overflow:hidden;">
                  <div style="background:{'#003d2e' if i==0 else '#1a2640'};
                              padding:10px 14px;border-bottom:1px solid #1f3452;">
                    <div style="font-size:0.62rem;font-weight:700;color:{rcol};
                                text-transform:uppercase;letter-spacing:0.1em;">
                      {rank_labels_map[i]}
                    </div>
                    <div style="font-size:0.95rem;font-weight:700;margin-top:3px;">
                      {row['hospital']}
                    </div>
                  </div>
                  <div style="padding:10px 14px;">
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">
                      <div>
                        <div style="font-size:0.60rem;color:#4a6882;text-transform:uppercase;">Distance</div>
                        <div style="font-size:0.92rem;font-weight:700;color:#e8f2ff;">{row['distance_km']:.1f} km</div>
                      </div>
                      <div>
                        <div style="font-size:0.60rem;color:#4a6882;text-transform:uppercase;">ETA</div>
                        <div style="font-size:0.92rem;font-weight:700;color:{rcol};">{row['eta_min']:.0f} min</div>
                      </div>
                      <div>
                        <div style="font-size:0.60rem;color:#4a6882;text-transform:uppercase;">Available Beds</div>
                        <div style="font-size:0.92rem;font-weight:700;color:{beds_col};">{avail}</div>
                      </div>
                      <div>
                        <div style="font-size:0.60rem;color:#4a6882;text-transform:uppercase;">Ventilators</div>
                        <div style="font-size:0.92rem;font-weight:700;color:#e8f2ff;">{int(row['ventilators'])}</div>
                      </div>
                    </div>
                    <div style="background:rgba(0,212,170,0.08);border:1px solid rgba(0,212,170,0.25);
                                border-radius:6px;padding:6px 10px;font-size:0.78rem;
                                font-weight:700;color:#00d4aa;margin-bottom:8px;">
                      📞 {rc}
                    </div>
                    <a href="https://www.google.com/maps/dir/?api=1&origin={lat},{lon}&destination={row['latitude']},{row['longitude']}&travelmode=driving"
                       target="_blank"
                       style="display:block;background:#1a3a6b;border:1px solid #4d9ef5;
                              border-radius:6px;padding:7px 10px;font-size:0.78rem;
                              font-weight:700;color:#4d9ef5;text-decoration:none;text-align:center;">
                      🧭 Navigate here in Google Maps
                    </a>
                  </div>
                </div>
                """

                # Draw route line first (behind markers)
                if row["route"]:
                    # Animated dashed route for primary, solid for others
                    folium.PolyLine(
                        row["route"],
                        color=rcol,
                        weight=6 if i == 0 else 3,
                        opacity=0.95 if i == 0 else 0.55,
                        tooltip=f"{row['hospital']} — {row['eta_min']:.0f} min ETA",
                        dash_array=None if i == 0 else "8 6",
                    ).add_to(m)

                # Hospital marker — numbered circle
                icon_html = f"""
                <div style="
                  width:32px;height:32px;
                  background:{rcol};
                  border:2px solid #07111f;
                  border-radius:50%;
                  display:flex;align-items:center;justify-content:center;
                  font-size:0.78rem;font-weight:800;color:#07111f;
                  box-shadow:0 0 12px {rcol}88;
                ">{i+1}</div>
                """
                folium.Marker(
                    [row["latitude"], row["longitude"]],
                    popup=folium.Popup(popup_html, max_width=280),
                    tooltip=f"#{i+1} {row['hospital']} — {row['distance_km']:.1f} km",
                    icon=folium.DivIcon(html=icon_html, icon_size=(32, 32), icon_anchor=(16, 16)),
                ).add_to(m)

            # All other hospitals — tiny dots (skip rows with missing coordinates)
            df_valid = df.dropna(subset=["latitude", "longitude"])
            for _, row in df_valid.iterrows():
                if row["hospital"] not in top3["hospital"].values:
                    dot_c = "#ff4560" if row["available_beds"] <= BED_ALERT_THRESHOLD else "#1f3452"
                    folium.CircleMarker(
                        [row["latitude"], row["longitude"]],
                        radius=4, color=dot_c,
                        fill=True, fill_color=dot_c,
                        fill_opacity=0.6, weight=1,
                        tooltip=f"{row['hospital']} — {int(row['available_beds'])} beds",
                    ).add_to(m)

            # Fit map bounds to show patient + all 3 hospitals
            all_lats = [lat] + top3["latitude"].tolist()
            all_lons = [lon] + top3["longitude"].tolist()
            m.fit_bounds([
                [min(all_lats) - 0.02, min(all_lons) - 0.02],
                [max(all_lats) + 0.02, max(all_lons) + 0.02],
            ])

            # Render
            map_data = st_folium(m, width="100%", height=560, returned_objects=["last_clicked"])

            # Click to set patient location
            if map_data and map_data.get("last_clicked"):
                clicked = map_data["last_clicked"]
                new_lat = clicked.get("lat")
                new_lon = clicked.get("lng")
                if new_lat and new_lon:
                    st.session_state.patient_lat = new_lat
                    st.session_state.patient_lon = new_lon
                    name = reverse_geocode(new_lat, new_lon)
                    st.session_state.location_name = name
                    loc_str = f"{name} ({new_lat:.5f}, {new_lon:.5f})" if name else f"{new_lat:.5f}, {new_lon:.5f}"
                    st.info(f"📍 Location updated to {loc_str} — click Find Best Hospital to re-search.")

            st.caption("💡 Click any numbered marker for hospital details · Click map to move patient pin · Use Google Maps buttons above for navigation")

# ══════════════════════════════════════════════════════
#  TAB 2 — REGISTRY
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="page-title">Hospital <span class="acc">Registry</span></div><div class="page-sub">Karnataka hospitals · Search · Add · Edit · Remove</div>', unsafe_allow_html=True)
    sv, sa, se, sd = st.tabs(["📋  View All","➕  Add","✏  Edit","🗑  Delete"])

    with sv:
        srch = st.text_input("Search hospitals", placeholder="Victoria, NIMHANS, Hubli…")
        disp = df[df["hospital"].str.contains(srch, case=False, na=False)] if srch else df.copy()
        cols = [c for c in ["hospital","contact_number","available_beds","predicted_beds",
                             "confidence","total_beds","critical_patients","ventilators",
                             "waiting_queue","oxygen_supply_pct","icu_specialist_count","last_updated"]
                if c in disp.columns]
        st.dataframe(disp[cols], use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(disp)} of {total_hosp} hospitals")

    with sa:
        with st.form("add_form"):
            ca, cb = st.columns(2)
            hname    = ca.text_input("Hospital Name *")
            hcontact = cb.text_input("Contact Number", placeholder="+91-080-XXXXXXXX")
            hlat     = ca.number_input("Latitude *",           value=15.31, format="%.6f")
            hlon     = cb.number_input("Longitude *",          value=75.71, format="%.6f")
            htotal   = ca.number_input("Total Beds *",         min_value=0, step=1)
            havail   = cb.number_input("Available Beds *",     min_value=0, step=1)
            havgp    = ca.number_input("Avg Daily Patients *", min_value=0, step=1)
            hcrit    = cb.number_input("Critical Patients *",  min_value=0, step=1)
            hvents   = ca.number_input("Ventilators *",        min_value=0, step=1)
            hqueue   = cb.number_input("Waiting Queue *",      min_value=0, step=1)
            hstaff   = ca.number_input("ICU Specialists",      min_value=0, step=1, value=5)
            hoxy     = cb.slider("Oxygen Supply %", 0, 100, 80)
            if st.form_submit_button("Add Hospital", type="primary"):
                nr = {"hospital":hname,"latitude":hlat,"longitude":hlon,
                      "total_beds":htotal,"available_beds":havail,"avg_daily_patients":havgp,
                      "critical_patients":hcrit,"ventilators":hvents,"waiting_queue":hqueue,
                      "icu_specialist_count":hstaff,"oxygen_supply_pct":hoxy,
                      "contact_number":hcontact or "N/A"}
                updf, errs = add_hospital(st.session_state.df_raw, nr)
                if errs:
                    for e in errs: st.error(e)
                else:
                    st.session_state.df_raw = updf
                    st.session_state.model_bundle = train_model(updf)
                    st.success(f"'{hname}' added."); st.rerun()

    with se:
        sel_h = st.selectbox("Hospital to edit", df["hospital"].tolist(), key="edit_sel")
        erow  = df[df["hospital"] == sel_h].iloc[0]
        with st.form("edit_form"):
            ea, eb = st.columns(2)
            e_con = ea.text_input("Contact Number",   value=str(erow.get("contact_number","N/A")))
            e_avl = ea.number_input("Available Beds", value=int(erow["available_beds"]),            min_value=0)
            e_que = eb.number_input("Waiting Queue",  value=int(erow["waiting_queue"]),             min_value=0)
            e_vnt = ea.number_input("Ventilators",    value=int(erow["ventilators"]),               min_value=0)
            e_oxy = eb.slider("Oxygen %", 0, 100, int(erow.get("oxygen_supply_pct", 80)))
            e_stf = ea.number_input("ICU Specialists",value=int(erow.get("icu_specialist_count",5)),min_value=0)
            e_crt = eb.number_input("Critical Patients",value=int(erow["critical_patients"]),       min_value=0)
            if st.form_submit_button("Save Changes", type="primary"):
                st.session_state.df_raw = update_hospital(
                    st.session_state.df_raw, sel_h,
                    {"available_beds":e_avl,"waiting_queue":e_que,"ventilators":e_vnt,
                     "oxygen_supply_pct":e_oxy,"icu_specialist_count":e_stf,
                     "critical_patients":e_crt,"contact_number":e_con})
                st.success("Updated."); st.rerun()

    with sd:
        del_h = st.selectbox("Hospital to remove", df["hospital"].tolist(), key="del_sel")
        st.markdown(f'<div class="del">⚠ This permanently removes <b>{del_h}</b> and retrains the model.</div>', unsafe_allow_html=True)
        if st.button("Confirm Delete", type="primary"):
            st.session_state.df_raw = delete_hospital(st.session_state.df_raw, del_h)
            st.session_state.model_bundle = train_model(st.session_state.df_raw)
            st.success(f"'{del_h}' removed."); st.rerun()

# ══════════════════════════════════════════════════════
#  TAB 3 — ANALYTICS
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="page-title">ICU <span class="acc">Analytics</span></div><div class="page-sub">Karnataka network · Live predictions · Resource overview</div>', unsafe_allow_html=True)

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Hospitals",  total_hosp)
    k2.metric("Total Capacity",   f"{total_cap:,}")
    k3.metric("Available Beds",   f"{total_avail:,}")
    k4.metric("System Occupancy", f"{avg_occ:.1f}%")
    st.markdown("<hr>", unsafe_allow_html=True)

    # Chart 1 — Strained hospitals
    st.markdown('<div class="slabel">25 Most Strained — Available vs Predicted Safe Beds</div>', unsafe_allow_html=True)
    df_s = df.sort_values("available_beds").head(25)
    bar_c = [RED if r["available_beds"] <= BED_ALERT_THRESHOLD
             else AMBER if r["available_beds"] <= 20 else TEAL
             for _, r in df_s.iterrows()]
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    y  = list(range(len(df_s)))
    b1 = ax1.barh(y, df_s["available_beds"].values, color=bar_c,  label="Available beds", zorder=3, height=0.55, edgecolor="none")
    ax1.barh(y, df_s["predicted_beds"].values,  color=BLUE, alpha=0.30, label="Predicted safe", zorder=2, height=0.55, edgecolor="none")
    ax1.set_yticks(y); ax1.set_yticklabels(df_s["hospital"].values, fontsize=8.5)
    ax1.set_xlabel("Beds"); ax1.set_title("Bed Availability — 25 Most Strained Hospitals")
    ax1.axvline(BED_ALERT_THRESHOLD, color=RED, linestyle="--", linewidth=1.2, alpha=0.7)
    for bar, val in zip(b1, df_s["available_beds"].values):
        ax1.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                 str(int(val)), va="center", fontsize=8, color="#8aaac8")
    legend_handles = [
        mpatches.Patch(color=RED,  label=f"Critical (≤{BED_ALERT_THRESHOLD} beds)"),
        mpatches.Patch(color=AMBER,label="Low (11–20 beds)"),
        mpatches.Patch(color=TEAL, label="OK (>20 beds)"),
        mpatches.Patch(color=BLUE, alpha=0.5, label="Predicted safe capacity"),
    ]
    ax1.legend(handles=legend_handles, loc="lower right"); plt.tight_layout()
    st.pyplot(fig1); plt.close(fig1)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="slabel" style="margin-top:18px;">Available Beds Distribution</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(df["available_beds"], bins=22, color=TEAL, edgecolor="#07111f", alpha=0.85, zorder=3)
        ax2.axvline(df["available_beds"].mean(), color=RED, linestyle="--", linewidth=1.8,
                    label=f"Mean = {df['available_beds'].mean():.1f}")
        ax2.axvline(BED_ALERT_THRESHOLD, color=AMBER, linestyle=":", linewidth=1.4,
                    label=f"Alert = {BED_ALERT_THRESHOLD}")
        ax2.set_xlabel("Available Beds"); ax2.set_ylabel("Hospitals")
        ax2.set_title("Available Beds Distribution"); ax2.legend(); plt.tight_layout()
        st.pyplot(fig2); plt.close(fig2)

    with c2:
        st.markdown('<div class="slabel" style="margin-top:18px;">Occupancy Rate Distribution</div>', unsafe_allow_html=True)
        occ = (df["total_beds"] - df["available_beds"]) / df["total_beds"].replace(0,1) * 100
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.hist(occ, bins=22, color=VIOLET, edgecolor="#07111f", alpha=0.85, zorder=3)
        ax3.axvline(occ.mean(), color=RED, linestyle="--", linewidth=1.8, label=f"Mean = {occ.mean():.1f}%")
        ax3.axvline(90, color=AMBER, linestyle=":", linewidth=1.4, label="90% danger line")
        ax3.set_xlabel("Occupancy (%)"); ax3.set_ylabel("Hospitals")
        ax3.set_title("Occupancy Rate Distribution"); ax3.legend(); plt.tight_layout()
        st.pyplot(fig3); plt.close(fig3)

    # Chart 4 — Ventilators & Specialists
    st.markdown('<div class="slabel" style="margin-top:18px;">Top 20 — Ventilators & ICU Specialists</div>', unsafe_allow_html=True)
    dft = df.nlargest(20, "ventilators")
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    w4 = 0.38; xp = list(range(len(dft)))
    ax4.bar([p-w4/2 for p in xp], dft["ventilators"].values,          width=w4, color=TEAL,   alpha=0.90, label="Ventilators",    zorder=3, edgecolor="none")
    ax4.bar([p+w4/2 for p in xp], dft["icu_specialist_count"].values, width=w4, color=VIOLET, alpha=0.90, label="ICU Specialists", zorder=3, edgecolor="none")
    ax4.set_xticks(xp); ax4.set_xticklabels(dft["hospital"].values, rotation=38, ha="right", fontsize=8)
    ax4.set_ylabel("Count"); ax4.set_title("Ventilator & Specialist Coverage — Top 20")
    ax4.legend(); plt.tight_layout(); st.pyplot(fig4); plt.close(fig4)

    # Chart 5 — Low oxygen
    dlo = df[df["oxygen_supply_pct"] < 85].sort_values("oxygen_supply_pct")
    if dlo.empty:
        st.success("✅ All hospitals above 85% oxygen supply.")
    else:
        st.markdown(f'<div class="slabel" style="margin-top:18px;">Hospitals Below 85% Oxygen Supply ({len(dlo)})</div>', unsafe_allow_html=True)
        oc = [RED if v < 70 else AMBER for v in dlo["oxygen_supply_pct"]]
        fig5, ax5 = plt.subplots(figsize=(12, max(3, len(dlo)*0.32)))
        ax5.barh(dlo["hospital"].values, dlo["oxygen_supply_pct"].values,
                 color=oc, zorder=3, height=0.6, edgecolor="none")
        ax5.axvline(85, color=AMBER, linestyle="--", linewidth=1.4, label="85% threshold")
        ax5.axvline(70, color=RED,   linestyle=":",  linewidth=1.2, label="70% critical")
        for i, (val, _) in enumerate(zip(dlo["oxygen_supply_pct"], dlo["hospital"])):
            ax5.text(val+0.5, i, f"{val:.1f}%", va="center", fontsize=8, color="#8aaac8")
        ax5.set_xlabel("Oxygen Supply (%)"); ax5.set_xlim(0, 105)
        ax5.set_title("Low Oxygen Supply Hospitals")
        ax5.legend(); plt.tight_layout(); st.pyplot(fig5); plt.close(fig5)

# ══════════════════════════════════════════════════════
#  TAB 4 — ALLOCATION LOG
# ══════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="page-title">Allocation <span class="acc">Log</span></div><div class="page-sub">Full audit trail of AI recommendations</div>', unsafe_allow_html=True)
    log_df = load_log()
    if log_df.empty:
        st.info("No allocations yet. Use Find Hospital to generate entries.")
    else:
        lm1,lm2,lm3,lm4 = st.columns(4)
        lm1.metric("Total Allocations",  len(log_df))
        lm2.metric("Unique Hospitals",   log_df["recommended_hospital"].nunique())
        lm3.metric("Avg ETA (min)",      f"{log_df['eta_min'].mean():.1f}")
        lm4.metric("Avg Predicted Beds", f"{log_df['predicted_beds'].mean():.1f}")
        st.markdown("<hr>", unsafe_allow_html=True)
        st.dataframe(log_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

        lc1, lc2 = st.columns(2)
        with lc1:
            freq = log_df["recommended_hospital"].value_counts().head(15)
            fig6, ax6 = plt.subplots(figsize=(6, 4))
            freq.plot(kind="bar", ax=ax6, color=TEAL, alpha=0.85, zorder=3, edgecolor="none")
            ax6.set_ylabel("Recommendations"); ax6.set_title("Most Recommended Hospitals")
            ax6.set_xticklabels(freq.index, rotation=40, ha="right", fontsize=8)
            plt.tight_layout(); st.pyplot(fig6); plt.close(fig6)

        with lc2:
            if "severity" in log_df.columns:
                sc  = log_df["severity"].value_counts()
                svc = {"Low": GREEN, "Medium": AMBER, "High": RED, "Critical": "#7a0019"}
                fig7, ax7 = plt.subplots(figsize=(5, 4))
                ax7.pie(sc.values, labels=sc.index,
                        colors=[svc.get(s,"#4a6882") for s in sc.index],
                        autopct="%1.1f%%", startangle=90,
                        wedgeprops={"edgecolor":"#07111f","linewidth":2},
                        textprops={"color":"#e8f2ff","fontsize":9})
                ax7.set_title("Cases by Severity")
                plt.tight_layout(); st.pyplot(fig7); plt.close(fig7)

        st.download_button("⬇  Download as CSV",
                           log_df.to_csv(index=False).encode("utf-8"),
                           "allocation_log.csv", "text/csv")

# ══════════════════════════════════════════════════════
#  TAB 5 — MODEL INFO
# ══════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="page-title">ML <span class="acc">Model</span> Info</div><div class="page-sub">Ensemble GBR + Random Forest · Predicts safe ICU capacity</div>', unsafe_allow_html=True)
    metrics = bundle.metrics
    if metrics and "note" not in metrics:
        mm1,mm2,mm3,mm4 = st.columns(4)
        mm1.metric("MAE (beds)",       metrics.get("MAE","N/A"))
        mm2.metric("R² Score",         metrics.get("R2","N/A"))
        mm3.metric("CV MAE (5-fold)",  metrics.get("CV_MAE","N/A"))
        mm4.metric("Training Samples", metrics.get("n_samples","N/A"))
    else:
        st.info(metrics.get("note","No metrics."))

    st.markdown("<hr>", unsafe_allow_html=True)
    ma, mb = st.columns([1, 2])

    with ma:
        st.markdown('<div class="slabel" style="margin-bottom:8px;">Architecture</div>', unsafe_allow_html=True)
        arch = [
            ("Model 1",  "Gradient Boosting (GBR)"),
            ("Model 2",  "Random Forest (RF)"),
            ("Blend",    "60% actual + 40% ML forward"),
            ("Target",   "safe_capacity"),
            ("Features", "11 incl. 5 derived"),
            ("Scaler",   "StandardScaler × 2"),
            ("CV",       "KFold 5-fold neg-MAE"),
            ("Dataset",  "Karnataka hospitals"),
            ("Ranking",  "Proximity-first, then scored"),
        ]
        rows = "".join(f'<tr><td class="ak">{k}</td><td class="av">{v}</td></tr>' for k,v in arch)
        st.markdown(f'<div class="panel" style="padding:6px 16px;"><table class="atbl">{rows}</table></div>', unsafe_allow_html=True)

        if bundle.feature_importance:
            fi = dict(sorted(bundle.feature_importance.items(), key=lambda x: x[1]))
            fig8, ax8 = plt.subplots(figsize=(5, len(fi)*0.42+1))
            b8 = ax8.barh(list(fi.keys()), list(fi.values()), color=TEAL, alpha=0.85, zorder=3, edgecolor="none")
            for bar, val in zip(b8, fi.values()):
                ax8.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                         f"{val:.3f}", va="center", fontsize=7.5, color="#8aaac8")
            ax8.set_xlabel("Importance"); ax8.set_title("Feature Importances (RF)")
            plt.tight_layout(); st.pyplot(fig8); plt.close(fig8)

    with mb:
        st.markdown('<div class="slabel" style="margin-bottom:8px;">All Hospitals — Predictions & Confidence</div>', unsafe_allow_html=True)
        cdf = df[["hospital","contact_number","available_beds","predicted_beds","confidence"]].copy()
        cdf["confidence_pct"] = (cdf["confidence"]*100).round(1).astype(str) + "%"
        st.dataframe(cdf.drop("confidence", axis=1), use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("Retrain model on current data", type="secondary"):
        with st.spinner("Retraining…"):
            st.session_state.model_bundle = train_model(st.session_state.df_raw)
        st.success("Model retrained."); st.rerun()
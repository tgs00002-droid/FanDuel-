import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

import streamlit as st
import plotly.express as px

# =========================
# CONFIG
# =========================
API_HOST = "https://api.the-odds-api.com"
SPORT_KEY = "basketball_nba"
BOOK_KEY = "fanduel"

DEFAULT_THROTTLE = 0.5

# =========================
# HELPERS
# =========================
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def american_to_prob(odds: Any) -> float:
    """Implied probability including vig."""
    if odds is None:
        return np.nan
    try:
        o = float(odds)
    except Exception:
        return np.nan

    if o > 0:
        return 100.0 / (o + 100.0)
    if o < 0:
        return (-o) / ((-o) + 100.0)
    return np.nan


def no_vig_two_way(p1: float, p2: float) -> Tuple[float, float, float]:
    """Remove vig for a two-outcome market. Returns (p1_fair, p2_fair, hold)."""
    if any(np.isnan([p1, p2])) or p1 <= 0 or p2 <= 0:
        return np.nan, np.nan, np.nan
    hold = (p1 + p2) - 1.0
    denom = p1 + p2
    return p1 / denom, p2 / denom, hold


# =========================
# HTTP CLIENT
# =========================
def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def fetch_odds(
    api_key: str,
    regions: str,
    markets: str,
    odds_format: str,
    include_links: bool,
    throttle_s: float,
) -> List[Dict[str, Any]]:
    """
    v4 odds endpoint.
    We filter to FanDuel using bookmakers=fanduel.
    """
    time.sleep(max(0.0, float(throttle_s)))

    url = f"{API_HOST}/v4/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,                 # required by API even if bookmakers used
        "markets": markets,                 # comma-separated: h2h,spreads,totals
        "oddsFormat": odds_format,          # american or decimal
        "bookmakers": BOOK_KEY,             # FanDuel only
        "dateFormat": "iso",
        "includeLinks": "true" if include_links else "false",
    }

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Odds API error {r.status_code}: {r.text[:400]}")
    return r.json()


# =========================
# PARSING
# =========================
def pick_market(bookmakers: List[Dict[str, Any]], market_key: str) -> Optional[Dict[str, Any]]:
    for b in bookmakers or []:
        if b.get("key") == BOOK_KEY:
            for m in b.get("markets", []) or []:
                if m.get("key") == market_key:
                    return m
    return None


def outcomes_to_map(market: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    if not market:
        return out
    for o in market.get("outcomes", []) or []:
        name = str(o.get("name"))
        out[name] = {"price": o.get("price"), "point": o.get("point")}
    return out


def build_board(events: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for e in events:
        home = e.get("home_team")
        away = e.get("away_team")
        start = e.get("commence_time")

        bks = e.get("bookmakers", []) or []

        h2h = outcomes_to_map(pick_market(bks, "h2h"))
        spr = outcomes_to_map(pick_market(bks, "spreads"))
        tot = outcomes_to_map(pick_market(bks, "totals"))

        # Moneyline
        home_ml = h2h.get(home, {}).get("price")
        away_ml = h2h.get(away, {}).get("price")

        # Spread
        home_spread_price = spr.get(home, {}).get("price")
        home_spread_point = spr.get(home, {}).get("point")
        away_spread_price = spr.get(away, {}).get("price")
        away_spread_point = spr.get(away, {}).get("point")

        # Totals (Over/Under)
        over_price = tot.get("Over", {}).get("price")
        over_point = tot.get("Over", {}).get("point")
        under_price = tot.get("Under", {}).get("price")
        under_point = tot.get("Under", {}).get("point")

        # Probabilities + hold (moneyline only)
        p_home = american_to_prob(home_ml)
        p_away = american_to_prob(away_ml)
        p_home_fair, p_away_fair, hold = no_vig_two_way(p_home, p_away)

        rows.append(
            {
                "Start (UTC)": start,
                "Away": away,
                "Home": home,
                "ML Away": away_ml,
                "ML Home": home_ml,
                "Spread Away": away_spread_point,
                "Spread Price Away": away_spread_price,
                "Spread Home": home_spread_point,
                "Spread Price Home": home_spread_price,
                "Total": over_point if over_point is not None else under_point,
                "Over": over_price,
                "Under": under_price,
                "Implied Away (%)": (p_away * 100) if not np.isnan(p_away) else np.nan,
                "Implied Home (%)": (p_home * 100) if not np.isnan(p_home) else np.nan,
                "No-Vig Away (%)": (p_away_fair * 100) if not np.isnan(p_away_fair) else np.nan,
                "No-Vig Home (%)": (p_home_fair * 100) if not np.isnan(p_home_fair) else np.nan,
                "ML Hold (%)": (hold * 100) if not np.isnan(hold) else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Start (UTC)"] = pd.to_datetime(df["Start (UTC)"], errors="coerce", utc=True)
    df = df.sort_values("Start (UTC)").reset_index(drop=True)
    return df


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="FanDuel NBA Odds Board", layout="wide")
st.title("FanDuel NBA Odds Board (via The Odds API)")
st.caption(f"Last refresh (UTC): {now_utc_iso()}")

# Read secret safely
api_key = None
try:
    api_key = st.secrets.get("ODDS_API_KEY", None)
except Exception:
    api_key = None
api_key = api_key or os.getenv("ODDS_API_KEY")

with st.sidebar:
    st.subheader("Settings")
    st.write("API key is read from Streamlit Secrets: `ODDS_API_KEY`.")
    regions = st.selectbox("Region", ["us", "us2", "uk", "eu", "au"], index=0)
    odds_format = st.selectbox("Odds format", ["american", "decimal"], index=0)
    markets_list = st.multiselect("Markets", ["h2h", "spreads", "totals"], default=["h2h", "spreads", "totals"])
    include_links = st.toggle("Include deep links (if available)", value=False)
    throttle = st.slider("Throttle (seconds)", 0.0, 1.5, DEFAULT_THROTTLE, 0.1)
    refresh = st.button("Refresh")

if not api_key:
    st.error("Missing ODDS_API_KEY. Add it in Streamlit Secrets and redeploy.")
    st.stop()

# Cache results to avoid hammering API and burning requests
@st.cache_data(ttl=60)
def cached_fetch(api_key: str, regions: str, markets: str, odds_format: str, include_links: bool, throttle: float):
    return fetch_odds(api_key, regions, markets, odds_format, include_links, throttle)

if refresh:
    cached_fetch.clear()

try:
    with st.spinner("Fetching FanDuel NBA odds..."):
        events = cached_fetch(
            api_key=api_key,
            regions=regions,
            markets=",".join(markets_list),
            odds_format=odds_format,
            include_links=include_links,
            throttle=throttle,
        )
except Exception as e:
    st.error("Failed to fetch odds.")
    st.code(str(e))
    st.stop()

df = build_board(events)

if df.empty:
    st.warning("No games returned. (Possible reasons: no NBA games today, region/plan limits, offseason.)")
    st.stop()

# Table
st.subheader("FanDuel Lines")
st.dataframe(df, use_container_width=True)

# Metrics
c1, c2, c3 = st.columns(3)
c1.metric("Games", len(df))
c2.metric("Avg ML Hold (%)", float(np.nanmean(df["ML Hold (%)"])) if df["ML Hold (%)"].notna().any() else 0.0)
c3.metric("Markets", ", ".join(markets_list))

# Visualization: ML hold by matchup
if df["ML Hold (%)"].notna().any():
    chart = df[["Away", "Home", "ML Hold (%)"]].copy()
    chart["Matchup"] = chart["Away"] + " @ " + chart["Home"]
    fig = px.bar(chart, x="Matchup", y="ML Hold (%)")
    st.plotly_chart(fig, use_container_width=True)

st.caption("For analytics/educational use. Betting involves risk.")

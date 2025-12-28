import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px


API_HOST = "https://api.the-odds-api.com"
SPORT_KEY = "basketball_nba"  # NBA :contentReference[oaicite:1]{index=1}
BOOK_KEY = "fanduel"          # FanDuel bookmaker key shown in v4 responses :contentReference[oaicite:2]{index=2}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def american_to_prob(odds: float) -> float:
    # implied probability (includes vig)
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return np.nan
    o = float(odds)
    if o > 0:
        return 100.0 / (o + 100.0)
    else:
        return (-o) / ((-o) + 100.0)


def no_vig_two_way(p1: float, p2: float) -> Tuple[float, float, float]:
    # normalize two implied probs to remove vig; returns (p1_fair, p2_fair, hold)
    if any(np.isnan([p1, p2])) or (p1 <= 0) or (p2 <= 0):
        return np.nan, np.nan, np.nan
    hold = (p1 + p2) - 1.0
    denom = (p1 + p2)
    return p1 / denom, p2 / denom, hold


@st.cache_data(ttl=60)  # refresh “live-ish” without hammering API
def fetch_odds(api_key: str, regions: str, markets: str, odds_format: str, include_links: bool) -> List[Dict[str, Any]]:
    """
    The Odds API v4:
    GET /v4/sports/{sport}/odds
    supports bookmakers=fanduel and markets=h2h,spreads,totals :contentReference[oaicite:3]{index=3}
    """
    url = f"{API_HOST}/v4/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,                    # still required; bookmakers takes priority if provided :contentReference[oaicite:4]{index=4}
        "markets": markets,                    # h2h,spreads,totals :contentReference[oaicite:5]{index=5}
        "oddsFormat": odds_format,             # american/decimal :contentReference[oaicite:6]{index=6}
        "bookmakers": BOOK_KEY,                # FanDuel only :contentReference[oaicite:7]{index=7}
        "dateFormat": "iso",
        "includeLinks": "true" if include_links else "false",  # deep links supported :contentReference[oaicite:8]{index=8}
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"API error {r.status_code}: {r.text[:400]}")
    return r.json()


def pick_market(bookmakers: List[Dict[str, Any]], market_key: str) -> Optional[Dict[str, Any]]:
    # We filtered to fanduel; still safe to find it
    for b in bookmakers or []:
        if b.get("key") == BOOK_KEY:
            for m in b.get("markets", []) or []:
                if m.get("key") == market_key:
                    return m
    return None


def outcomes_to_map(market: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    # returns {outcome_name: {"price":..., "point":...}}
    out = {}
    if not market:
        return out
    for o in market.get("outcomes", []) or []:
        out[str(o.get("name"))] = {"price": o.get("price"), "point": o.get("point")}
    return out


def build_board(events: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for e in events:
        home = e.get("home_team")
        away = e.get("away_team")
        start = e.get("commence_time")

        # Each event contains bookmakers array; fanduel key is "fanduel" :contentReference[oaicite:9]{index=9}
        bks = e.get("bookmakers", [])

        h2h = outcomes_to_map(pick_market(bks, "h2h"))
        spr = outcomes_to_map(pick_market(bks, "spreads"))
        tot = outcomes_to_map(pick_market(bks, "totals"))

        # Moneyline
        home_ml = h2h.get(home, {}).get("price")
        away_ml = h2h.get(away, {}).get("price")

        # Spread (price + point per side)
        home_spread_price = spr.get(home, {}).get("price")
        home_spread_point = spr.get(home, {}).get("point")
        away_spread_price = spr.get(away, {}).get("price")
        away_spread_point = spr.get(away, {}).get("point")

        # Totals (Over/Under). Outcome names often "Over"/"Under"
        over_price = tot.get("Over", {}).get("price")
        over_point = tot.get("Over", {}).get("point")
        under_price = tot.get("Under", {}).get("price")
        under_point = tot.get("Under", {}).get("point")

        # Implied probs + hold (no-vig) for ML
        p_home = american_to_prob(home_ml) if home_ml is not None else np.nan
        p_away = american_to_prob(away_ml) if away_ml is not None else np.nan
        p_home_fair, p_away_fair, hold = no_vig_two_way(p_home, p_away)

        rows.append(
            {
                "Start (UTC)": start,
                "Away": away,
                "Home": home,
                "ML Away": away_ml,
                "ML Home": home_ml,
                "Implied Away": p_away,
                "Implied Home": p_home,
                "No-Vig Away": p_away_fair,
                "No-Vig Home": p_home_fair,
                "ML Hold": hold,
                "Spread Away": away_spread_point,
                "Spread Price Away": away_spread_price,
                "Spread Home": home_spread_point,
                "Spread Price Home": home_spread_price,
                "Total": over_point if over_point is not None else under_point,
                "Over": over_price,
                "Under": under_price,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Start (UTC)"] = pd.to_datetime(df["Start (UTC)"], errors="coerce", utc=True)
    df = df.sort_values("Start (UTC)").reset_index(drop=True)
    return df


# ---------------- UI ----------------
st.set_page_config(page_title="FanDuel NBA Odds Board", layout="wide")
st.title("FanDuel NBA Odds Board (Live via The Odds API)")
st.caption(f"Updated: {now_utc_iso()} • Source: The Odds API (v4) with bookmakers=fanduel")

with st.sidebar:
    st.subheader("Settings")
    regions = st.selectbox("Region", ["us", "us2", "uk", "eu", "au"], index=0)
    odds_format = st.selectbox("Odds Format", ["american", "decimal"], index=0)
    markets = st.multiselect("Markets", ["h2h", "spreads", "totals"], default=["h2h", "spreads", "totals"])
    include_links = st.toggle("Include Links (if available)", value=False)
    refresh = st.button("Refresh now")

# API key: Streamlit secrets first, fallback to env var
api_key = st.secrets.get("ODDS_API_KEY", None) if hasattr(st, "secrets") else None
api_key = api_key or os.getenv("ODDS_API_KEY")

if not api_key:
    st.error("Missing API key. Add ODDS_API_KEY to Streamlit secrets or environment variables.")
    st.stop()

if refresh:
    fetch_odds.clear()

try:
    events = fetch_odds(
        api_key=api_key,
        regions=regions,
        markets=",".join(markets),
        odds_format=odds_format,
        include_links=include_links,
    )
except Exception as e:
    st.error("Failed to fetch odds.")
    st.code(str(e))
    st.stop()

df = build_board(events)

if df.empty:
    st.warning("No games returned (could be offseason or API plan/region limits). Try a different region.")
    st.stop()

# Format columns for display
show = df.copy()
for c in ["Implied Away", "Implied Home", "No-Vig Away", "No-Vig Home", "ML Hold"]:
    show[c] = (show[c] * 100).round(2)

st.subheader("FanDuel Lines (Moneyline • Spread • Total)")
st.dataframe(show, use_container_width=True)

st.subheader("Quick Insights")
c1, c2, c3 = st.columns(3)
c1.metric("Games", len(df))
c2.metric("Avg ML Hold (%)", float(np.nanmean(df["ML Hold"]) * 100) if df["ML Hold"].notna().any() else 0.0)
c3.metric("Markets pulled", ", ".join(markets))

# Simple visualization: Hold by game (moneyline)
if df["ML Hold"].notna().any():
    chart = df[["Start (UTC)", "Away", "Home", "ML Hold"]].copy()
    chart["Matchup"] = chart["Away"] + " @ " + chart["Home"]
    chart["ML Hold (%)"] = (chart["ML Hold"] * 100)
    fig = px.bar(chart, x="Matchup", y="ML Hold (%)")
    st.plotly_chart(fig, use_container_width=True)

st.caption("Note: Betting involves risk. This dashboard is for analytics/educational use.")

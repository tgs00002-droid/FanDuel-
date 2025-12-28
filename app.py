import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px


# =========================
# CONFIG
# =========================
API_HOST = "https://api.the-odds-api.com"
SPORT_KEY = "basketball_nba"

DEFAULT_THROTTLE = 0.4
CACHE_TTL_SECONDS = 60  # helps protect your 500 credits/month


# =========================
# UTIL
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
    """Return (p1_fair, p2_fair, hold)."""
    if any(np.isnan([p1, p2])) or p1 <= 0 or p2 <= 0:
        return np.nan, np.nan, np.nan
    hold = (p1 + p2) - 1.0
    denom = p1 + p2
    return p1 / denom, p2 / denom, hold


def fmt_american(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        x = int(x)
    except Exception:
        return str(x)
    return f"+{x}" if x > 0 else str(x)


def safe_get_secret(name: str) -> Optional[str]:
    try:
        return st.secrets.get(name)
    except Exception:
        return None


# =========================
# API
# =========================
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_odds(
    api_key: str,
    regions: str,
    markets: str,
    odds_format: str,
    include_links: bool,
    throttle_s: float,
) -> List[Dict[str, Any]]:
    """Fetch NBA odds from The Odds API v4."""
    time.sleep(max(0.0, float(throttle_s)))

    url = f"{API_HOST}/v4/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,          # "h2h,spreads,totals"
        "oddsFormat": odds_format,   # "american" or "decimal"
        "dateFormat": "iso",
        "includeLinks": "true" if include_links else "false",
    }

    r = requests.get(url, params=params, timeout=30)

    # Helpful error messages
    if r.status_code == 401:
        raise RuntimeError("401 Unauthorized: API key invalid or not being read from Secrets.")
    if r.status_code == 429:
        raise RuntimeError("429 Rate limited: reduce refresh frequency / increase cache TTL.")
    if r.status_code != 200:
        raise RuntimeError(f"API error {r.status_code}: {r.text[:500]}")

    return r.json()


# =========================
# PARSING HELPERS
# =========================
def find_book_market(book: Dict[str, Any], market_key: str) -> Optional[Dict[str, Any]]:
    for m in book.get("markets", []) or []:
        if m.get("key") == market_key:
            return m
    return None


def outcome_map(market: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    if not market:
        return out
    for o in market.get("outcomes", []) or []:
        name = str(o.get("name"))
        out[name] = {"price": o.get("price"), "point": o.get("point")}
    return out


def flatten_event(event: Dict[str, Any], selected_markets: List[str]) -> pd.DataFrame:
    """
    Return a per-bookmaker table for one event:
    columns include ML (h2h), spreads, totals.
    """
    home = event.get("home_team")
    away = event.get("away_team")
    start = event.get("commence_time")
    books = event.get("bookmakers", []) or []

    rows = []
    for b in books:
        bkey = b.get("key")
        btitle = b.get("title", bkey)

        h2h = outcome_map(find_book_market(b, "h2h")) if "h2h" in selected_markets else {}
        spr = outcome_map(find_book_market(b, "spreads")) if "spreads" in selected_markets else {}
        tot = outcome_map(find_book_market(b, "totals")) if "totals" in selected_markets else {}

        rows.append(
            {
                "Start (UTC)": start,
                "Away": away,
                "Home": home,
                "Book": btitle,
                # Moneyline
                "ML Away": h2h.get(away, {}).get("price"),
                "ML Home": h2h.get(home, {}).get("price"),
                # Spread
                "Spread Away": spr.get(away, {}).get("point"),
                "Spread Price Away": spr.get(away, {}).get("price"),
                "Spread Home": spr.get(home, {}).get("point"),
                "Spread Price Home": spr.get(home, {}).get("price"),
                # Total
                "Total": tot.get("Over", {}).get("point") or tot.get("Under", {}).get("point"),
                "Over": tot.get("Over", {}).get("price"),
                "Under": tot.get("Under", {}).get("price"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Start (UTC)"] = pd.to_datetime(df["Start (UTC)"], errors="coerce", utc=True)
        df = df.sort_values(["Start (UTC)", "Book"]).reset_index(drop=True)
    return df


def best_line_summary(per_book: pd.DataFrame) -> pd.DataFrame:
    """
    Build a best-line table per matchup:
    - Best ML for each side (highest +odds / least negative)
    - Best spreads and totals prices
    """
    if per_book.empty:
        return per_book

    out_rows = []

    for (start, away, home), g in per_book.groupby(["Start (UTC)", "Away", "Home"], dropna=False):
        # Moneyline bests
        def best_price(series):
            # Higher is better for bettor (e.g. +110 > +105 > -110 > -120)
            s = pd.to_numeric(series, errors="coerce")
            if s.dropna().empty:
                return np.nan
            return s.max()

        best_ml_away = best_price(g["ML Away"])
        best_ml_home = best_price(g["ML Home"])

        # Determine which books offer best
        def best_books(col, best_val):
            if np.isnan(best_val):
                return ""
            s = pd.to_numeric(g[col], errors="coerce")
            books = g.loc[s == best_val, "Book"].astype(str).unique().tolist()
            return ", ".join(books[:3]) + ("..." if len(books) > 3 else "")

        best_ml_away_books = best_books("ML Away", best_ml_away)
        best_ml_home_books = best_books("ML Home", best_ml_home)

        # Implied / no-vig / hold from BEST prices (approx “best market”)
        p_away = american_to_prob(best_ml_away)
        p_home = american_to_prob(best_ml_home)
        p_away_fair, p_home_fair, hold = no_vig_two_way(p_away, p_home)

        out_rows.append(
            {
                "Start (UTC)": start,
                "Matchup": f"{away} @ {home}",
                "Best ML Away": best_ml_away,
                "Best ML Away Books": best_ml_away_books,
                "Best ML Home": best_ml_home,
                "Best ML Home Books": best_ml_home_books,
                "Implied Away (%)": p_away * 100 if not np.isnan(p_away) else np.nan,
                "Implied Home (%)": p_home * 100 if not np.isnan(p_home) else np.nan,
                "No-Vig Away (%)": p_away_fair * 100 if not np.isnan(p_away_fair) else np.nan,
                "No-Vig Home (%)": p_home_fair * 100 if not np.isnan(p_home_fair) else np.nan,
                "Hold (%)": hold * 100 if not np.isnan(hold) else np.nan,
            }
        )

    out = pd.DataFrame(out_rows)
    out["Start (UTC)"] = pd.to_datetime(out["Start (UTC)"], errors="coerce", utc=True)
    out = out.sort_values("Start (UTC)").reset_index(drop=True)

    # Pretty formatting columns for display
    out["Best ML Away"] = out["Best ML Away"].apply(fmt_american)
    out["Best ML Home"] = out["Best ML Home"].apply(fmt_american)

    return out


# =========================
# UI
# =========================
st.set_page_config(page_title="NBA Odds Board", layout="wide")
st.title("NBA Live Odds Board (All Sportsbooks)")
st.caption(f"Last refresh (UTC): {now_utc_iso()} • Source: The Odds API")

# Read key safely
api_key = safe_get_secret("ODDS_API_KEY") or os.getenv("ODDS_API_KEY")

with st.sidebar:
    st.subheader("Controls")
    regions = st.selectbox("Region", ["us", "us2", "us,us2", "uk", "eu", "au"], index=2)
    odds_format = st.selectbox("Odds format", ["american", "decimal"], index=0)
    markets_list = st.multiselect("Markets", ["h2h", "spreads", "totals"], default=["h2h", "spreads", "totals"])
    include_links = st.toggle("Include links (if plan supports)", value=False)
    throttle = st.slider("Throttle (seconds)", 0.0, 1.5, DEFAULT_THROTTLE, 0.1)
    force_refresh = st.button("Force refresh")

st.write("✅ API key loaded:", bool(api_key))

if not api_key:
    st.error("Missing ODDS_API_KEY. Add it to Streamlit Secrets and redeploy.")
    st.stop()

if force_refresh:
    fetch_odds.clear()

# Fetch
try:
    events = fetch_odds(
        api_key=api_key,
        regions=regions,
        markets=",".join(markets_list),
        odds_format=odds_format,
        include_links=include_links,
        throttle_s=throttle,
    )
except Exception as e:
    st.error("Failed to fetch odds.")
    st.code(str(e))
    st.stop()

if not events:
    st.warning("0 games returned. This can happen if there are no NBA games right now (offseason/no games today).")
    st.stop()

# Flatten all events to per-book table
per_book_frames = [flatten_event(e, markets_list) for e in events]
per_book = pd.concat(per_book_frames, ignore_index=True) if per_book_frames else pd.DataFrame()

# Filters
matchups = sorted(per_book["Away"].astype(str).str.cat(per_book["Home"].astype(str), sep=" @ ").unique().tolist())
selected = st.selectbox("Select matchup", matchups, index=0) if matchups else None

if selected:
    away_sel, home_sel = selected.split(" @ ")
    view = per_book[(per_book["Away"] == away_sel) & (per_book["Home"] == home_sel)].copy()
else:
    view = per_book.copy()

# Best line summary
st.subheader("Best Line (Across All Books)")
best = best_line_summary(per_book)
st.dataframe(best, use_container_width=True)

# Per book table
st.subheader("Per-Book Odds (Selected Matchup)")
if view.empty:
    st.warning("No rows for this matchup.")
else:
    show = view.copy()
    # Pretty odds display when american
    if odds_format == "american":
        for col in ["ML Away", "ML Home", "Spread Price Away", "Spread Price Home", "Over", "Under"]:
            show[col] = show[col].apply(fmt_american)
    st.dataframe(show, use_container_width=True)

# Simple chart: hold by matchup (from best lines)
if not best.empty and best["Hold (%)"].notna().any():
    chart = best[["Matchup", "Hold (%)"]].copy()
    fig = px.bar(chart, x="Matchup", y="Hold (%)")
    st.plotly_chart(fig, use_container_width=True)

st.caption("For analytics/educational use. Betting involves risk.")

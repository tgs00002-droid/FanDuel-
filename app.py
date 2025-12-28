import os
import time
import sqlite3
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

DB_PATH = "odds_history.sqlite"
CACHE_TTL_SECONDS = 30          # frequent enough to see changes, but not too costly
DEFAULT_REFRESH_SECONDS = 60    # suggested "check for changes" interval


# =========================
# UTILS
# =========================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds")

def safe_get_secret(name: str) -> Optional[str]:
    try:
        return st.secrets.get(name)
    except Exception:
        return None

def american_to_prob(odds: Any) -> float:
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
    if any(np.isnan([p1, p2])) or p1 <= 0 or p2 <= 0:
        return np.nan, np.nan, np.nan
    hold = (p1 + p2) - 1.0
    denom = p1 + p2
    return p1 / denom, p2 / denom, hold

def fmt_american(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        x = int(float(x))
    except Exception:
        return str(x)
    return f"+{x}" if x > 0 else str(x)

def to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


# =========================
# DB
# =========================
def db_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db() -> None:
    with db_conn() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                captured_at TEXT NOT NULL,
                event_id TEXT NOT NULL,
                commence_time TEXT,
                home_team TEXT,
                away_team TEXT,
                book TEXT NOT NULL,
                market TEXT NOT NULL,
                outcome TEXT NOT NULL,
                price REAL,
                point REAL,
                PRIMARY KEY (captured_at, event_id, book, market, outcome)
            );
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_event ON odds_snapshots(event_id);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_event_market ON odds_snapshots(event_id, market, outcome);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_captured ON odds_snapshots(captured_at);")


def save_snapshot_long(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with db_conn() as con:
        con.executemany(
            """
            INSERT OR IGNORE INTO odds_snapshots
            (captured_at, event_id, commence_time, home_team, away_team, book, market, outcome, price, point)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                (
                    r["captured_at"],
                    r["event_id"],
                    r.get("commence_time"),
                    r.get("home_team"),
                    r.get("away_team"),
                    r["book"],
                    r["market"],
                    r["outcome"],
                    r.get("price"),
                    r.get("point"),
                )
                for r in rows
            ],
        )


def read_history(event_id: str, market: str, outcome: str, book: Optional[str] = None, limit: int = 5000) -> pd.DataFrame:
    q = """
    SELECT captured_at, book, price, point
    FROM odds_snapshots
    WHERE event_id = ?
      AND market = ?
      AND outcome = ?
    """
    params = [event_id, market, outcome]
    if book:
        q += " AND book = ?"
        params.append(book)
    q += " ORDER BY captured_at ASC LIMIT ?"
    params.append(limit)

    with db_conn() as con:
        df = pd.read_sql_query(q, con, params=params)
    if df.empty:
        return df
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True, errors="coerce")
    return df


def last_change_time(df_hist: pd.DataFrame) -> Optional[pd.Timestamp]:
    """
    Return the timestamp where price/point last changed (compared to previous record).
    """
    if df_hist is None or df_hist.empty:
        return None
    # define a "value" signature
    sig = df_hist[["price", "point"]].astype(float).round(6)
    changed = (sig != sig.shift(1)).any(axis=1)
    # ignore first row
    changed.iloc[0] = False
    if not changed.any():
        return None
    return df_hist.loc[changed, "captured_at"].iloc[-1]


# =========================
# API
# =========================
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_odds(api_key: str, regions: str, markets: str, odds_format: str) -> List[Dict[str, Any]]:
    url = f"{API_HOST}/v4/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": "iso",
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"API error {r.status_code}: {r.text[:500]}")
    return r.json()


def event_to_long_rows(event: Dict[str, Any], captured_at_iso: str) -> List[Dict[str, Any]]:
    """
    Turn one event into long rows for DB storage.
    We store price/point for each (book, market, outcome).
    """
    out = []
    event_id = event.get("id")
    commence_time = event.get("commence_time")
    home = event.get("home_team")
    away = event.get("away_team")

    for b in event.get("bookmakers", []) or []:
        book = b.get("title") or b.get("key") or "Unknown"
        for m in b.get("markets", []) or []:
            market = m.get("key")
            for o in m.get("outcomes", []) or []:
                outcome = str(o.get("name"))
                out.append(
                    {
                        "captured_at": captured_at_iso,
                        "event_id": event_id,
                        "commence_time": commence_time,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "market": market,
                        "outcome": outcome,
                        "price": to_float(o.get("price")),
                        "point": to_float(o.get("point")),
                    }
                )
    return out


# =========================
# BOARD LOGIC (CLEAN SUMMARY)
# =========================
def best_price(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return np.nan
    return s.max()

def best_books(df: pd.DataFrame, col: str, best_val: float) -> str:
    if np.isnan(best_val) or df.empty:
        return ""
    s = pd.to_numeric(df[col], errors="coerce")
    books = df.loc[s == best_val, "Book"].astype(str).unique().tolist()
    return ", ".join(books[:2]) + ("…" if len(books) > 2 else "")

def make_clean_board(events: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - best_lines: one row per game (clean, user-friendly)
      - per_book: per-book table for selected matchup
    """
    per_book_rows = []
    for e in events:
        event_id = e.get("id")
        home = e.get("home_team")
        away = e.get("away_team")
        start = e.get("commence_time")
        for b in e.get("bookmakers", []) or []:
            book = b.get("title") or b.get("key")
            markets = {m.get("key"): m for m in (b.get("markets") or [])}

            def get_outcome(mkey: str, oname: str) -> Tuple[Optional[float], Optional[float]]:
                m = markets.get(mkey)
                if not m:
                    return None, None
                for o in m.get("outcomes", []) or []:
                    if str(o.get("name")) == oname:
                        return o.get("price"), o.get("point")
                return None, None

            away_ml, _ = get_outcome("h2h", away)
            home_ml, _ = get_outcome("h2h", home)

            away_sp_price, away_sp_point = get_outcome("spreads", away)
            home_sp_price, home_sp_point = get_outcome("spreads", home)

            over_price, total_point = get_outcome("totals", "Over")
            under_price, _ = get_outcome("totals", "Under")

            per_book_rows.append(
                {
                    "Event ID": event_id,
                    "Start (UTC)": start,
                    "Away": away,
                    "Home": home,
                    "Book": book,
                    "ML Away": away_ml,
                    "ML Home": home_ml,
                    "Spread Away": away_sp_point,
                    "Spread Price Away": away_sp_price,
                    "Spread Home": home_sp_point,
                    "Spread Price Home": home_sp_price,
                    "Total": total_point,
                    "Over": over_price,
                    "Under": under_price,
                }
            )

    per_book = pd.DataFrame(per_book_rows)
    if per_book.empty:
        return per_book, per_book

    per_book["Start (UTC)"] = pd.to_datetime(per_book["Start (UTC)"], utc=True, errors="coerce")
    per_book = per_book.sort_values(["Start (UTC)", "Away", "Home", "Book"]).reset_index(drop=True)

    best_rows = []
    for (eid, start, away, home), g in per_book.groupby(["Event ID", "Start (UTC)", "Away", "Home"], dropna=False):
        best_away = best_price(g["ML Away"])
        best_home = best_price(g["ML Home"])
        away_books = best_books(g, "ML Away", best_away)
        home_books = best_books(g, "ML Home", best_home)

        p_away = american_to_prob(best_away)
        p_home = american_to_prob(best_home)
        p_away_fair, p_home_fair, hold = no_vig_two_way(p_away, p_home)

        best_rows.append(
            {
                "Event ID": eid,
                "Start (UTC)": start,
                "Matchup": f"{away} @ {home}",
                "Best Away ML": fmt_american(best_away),
                "Best Home ML": fmt_american(best_home),
                "Best Away Book": away_books,
                "Best Home Book": home_books,
                "Market Win % (Away)": (p_away * 100) if not np.isnan(p_away) else np.nan,
                "Market Win % (Home)": (p_home * 100) if not np.isnan(p_home) else np.nan,
                "Fair Win % (Away)": (p_away_fair * 100) if not np.isnan(p_away_fair) else np.nan,
                "Fair Win % (Home)": (p_home_fair * 100) if not np.isnan(p_home_fair) else np.nan,
                "Sportsbook Margin %": (hold * 100) if not np.isnan(hold) else np.nan,
            }
        )

    best_lines = pd.DataFrame(best_rows).sort_values("Start (UTC)").reset_index(drop=True)
    return best_lines, per_book


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="NBA Odds Board", layout="wide")
init_db()

st.title("NBA Live Odds Board")
st.caption(f"Last refresh (UTC): {utc_now_iso()}  •  Source: The Odds API")

st.markdown(
    """
**How to read this:**
- **Best Line** = most favorable odds across sportsbooks.
- **Market Win %** includes sportsbook margin (vig).
- **Fair Win %** removes vig to estimate a fair probability.
- **Sportsbook Margin %** (lower is better).
"""
)

api_key = safe_get_secret("ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
if not api_key:
    st.error("Missing ODDS_API_KEY. Add it to Streamlit → Settings → Secrets.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    regions = st.selectbox("Region", ["us,us2", "us", "us2", "uk", "eu", "au"], index=0)
    markets_list = st.multiselect("Markets", ["h2h", "spreads", "totals"], default=["h2h", "spreads", "totals"])

    with st.expander("Advanced"):
        odds_format = st.selectbox("Odds format", ["american", "decimal"], index=0)
        poll_seconds = st.slider("Auto-refresh (seconds)", 15, 180, DEFAULT_REFRESH_SECONDS, 5)
        force_refresh = st.button("Force refresh now")

# Optional auto-refresh (lightweight)
st.write(f"✅ API key loaded: **True**")
st.caption("Tip: The app records a timestamped snapshot each refresh so you can see when odds changed.")

if force_refresh:
    fetch_odds.clear()

# Fetch odds
try:
    events = fetch_odds(
        api_key=api_key,
        regions=regions,
        markets=",".join(markets_list),
        odds_format="american",
    )
except Exception as e:
    st.error("Failed to fetch odds.")
    st.code(str(e))
    st.stop()

if not events:
    st.warning("0 games returned. If it’s offseason or no games today, that’s normal.")
    st.stop()

# Save snapshot to DB
captured_at = utc_now_iso()
long_rows = []
for ev in events:
    long_rows.extend(event_to_long_rows(ev, captured_at))
save_snapshot_long(long_rows)

# Build cleaner tables
best_lines, per_book = make_clean_board(events)

tabs = st.tabs(["🏀 Best Lines (Simple)", "📈 Line Movement (When odds changed)", "📚 What these metrics mean"])

# ---------------- Tab 1: Simple board
with tabs[0]:
    st.subheader("Best Line (Simple view)")

    # Clean formatting
    show = best_lines.copy()
    for c in ["Market Win % (Away)", "Market Win % (Home)", "Fair Win % (Away)", "Fair Win % (Home)", "Sportsbook Margin %"]:
        show[c] = show[c].round(2)

    # Quick insight
    avg_margin = float(np.nanmean(best_lines["Sportsbook Margin %"])) if best_lines["Sportsbook Margin %"].notna().any() else np.nan
    if not np.isnan(avg_margin):
        st.info(f"💡 Average sportsbook margin across games: **{avg_margin:.2f}%** (lower is better).")

    st.dataframe(show, use_container_width=True)

    st.caption("If this feels too dense, use the Line Movement tab and pick ONE game + ONE market.")

# ---------------- Tab 2: Line movement
with tabs[1]:
    st.subheader("Line Movement (with timestamps)")

    matchups = best_lines["Matchup"].tolist()
    matchup = st.selectbox("Choose a matchup", matchups, index=0)

    # Find event_id
    eid = best_lines.loc[best_lines["Matchup"] == matchup, "Event ID"].iloc[0]

    colA, colB, colC = st.columns([1.2, 1.0, 1.0])

    with colA:
        market = st.selectbox("Market", ["h2h", "spreads", "totals"], index=0)
    with colB:
        # outcome choices depend on market
        # read teams from per_book
        row = per_book.loc[per_book["Event ID"] == eid].iloc[0]
        away = row["Away"]
        home = row["Home"]
        if market == "h2h":
            outcome = st.selectbox("Outcome", [away, home], index=0)
        elif market == "spreads":
            outcome = st.selectbox("Outcome", [away, home], index=0)
        else:
            outcome = st.selectbox("Outcome", ["Over", "Under"], index=0)
    with colC:
        # optional book filter
        books = sorted(per_book.loc[per_book["Event ID"] == eid, "Book"].astype(str).unique().tolist())
        book = st.selectbox("Book (optional)", ["All books"] + books, index=0)
        book_filter = None if book == "All books" else book

    hist = read_history(event_id=eid, market=market, outcome=outcome, book=book_filter)

    if hist.empty:
        st.warning("No history yet for that selection. Click Force refresh a couple times to record snapshots.")
    else:
        # If All books, show best price at each timestamp (max price)
        if book_filter is None:
            # collapse by timestamp -> best price
            h = hist.groupby("captured_at", as_index=False).agg({"price": "max", "point": "max"})
            h["book"] = "Best across books"
        else:
            h = hist.copy()

        # last change time
        lct = last_change_time(h.rename(columns={"captured_at": "captured_at"}))
        if lct is None:
            st.success("No odds changes detected yet (based on your saved snapshots).")
        else:
            st.info(f"🕒 Last changed at: **{lct.strftime('%Y-%m-%d %H:%M:%S UTC')}**")

        # chart y-axis
        y_col = "price" if market != "spreads" else "price"
        fig = px.line(h, x="captured_at", y="price", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        if market in ["spreads", "totals"]:
            # also plot point if available
            if h["point"].notna().any():
                fig2 = px.line(h, x="captured_at", y="point", markers=True)
                st.plotly_chart(fig2, use_container_width=True)

        # show raw history
        disp = h.copy()
        disp["captured_at"] = disp["captured_at"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        disp["price"] = disp["price"].apply(fmt_american)
        st.dataframe(disp, use_container_width=True)

# ---------------- Tab 3: explainers
with tabs[2]:
    st.markdown(
        """
### What these metrics mean (plain English)

**Moneyline (ML)**  
Just “who wins”. Odds show payout.

**Market Win %**  
Probability implied by the odds, **including** sportsbook margin (vig).

**Fair Win % (No-Vig)**  
We normalize both sides so they sum to 100%. This removes vig and gives a fairer probability estimate.

**Sportsbook Margin % (Hold)**  
How much the sportsbook is taking in the market.  
Lower margin usually means better pricing for bettors.

### Why timestamps matter
We save snapshots each refresh. Odds only “change” when a new snapshot differs from the previous one.
"""
    )

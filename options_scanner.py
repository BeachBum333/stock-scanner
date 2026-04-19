"""
options_scanner.py — Unusual Options Activity Scanner
Data source: yfinance (free, no API key needed)

Detects:
  - Unusual call/put volume (vol > 3x open interest)
  - Put/Call ratio (sentiment gauge)
  - IV spike vs historical volatility
  - Top strikes by volume
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Optional, List
import time


@st.cache_data(ttl=900)   # cache 15 minutes
def get_options_data(ticker: str) -> Optional[Dict]:
    """
    Fetch options chain for nearest 3 expiries and compute:
      - Unusual calls/puts (vol/OI > 3x)
      - Put/Call volume ratio
      - IV spike flag
      - Top strikes tables
    Returns None if options are not available for the ticker.
    """
    try:
        import yfinance as yf

        tk    = yf.Ticker(ticker)
        dates = tk.options

        if not dates:
            return None

        all_calls, all_puts = [], []

        for expiry in dates[:3]:
            try:
                chain = tk.option_chain(expiry)
                c = chain.calls.copy()
                p = chain.puts.copy()
                c["expiry"] = expiry
                p["expiry"] = expiry
                all_calls.append(c)
                all_puts.append(p)
            except Exception:
                continue

        if not all_calls:
            return None

        calls = pd.concat(all_calls, ignore_index=True)
        puts  = pd.concat(all_puts,  ignore_index=True)

        # ── Clean & type-cast ────────────────────────────────────
        for df in [calls, puts]:
            df["volume"]            = pd.to_numeric(df.get("volume", 0),            errors="coerce").fillna(0)
            df["openInterest"]      = pd.to_numeric(df.get("openInterest", 0),      errors="coerce").fillna(0)
            df["impliedVolatility"] = pd.to_numeric(df.get("impliedVolatility", 0), errors="coerce").fillna(0)
            df["lastPrice"]         = pd.to_numeric(df.get("lastPrice", 0),         errors="coerce").fillna(0)

        # ── Vol/OI ratio ─────────────────────────────────────────
        calls["vol_oi_ratio"] = calls["volume"] / calls["openInterest"].replace(0, 1)
        puts["vol_oi_ratio"]  = puts["volume"]  / puts["openInterest"].replace(0, 1)

        unusual_calls = (
            calls[calls["vol_oi_ratio"] >= 3]
            .sort_values("volume", ascending=False)
            .head(8)
        )
        unusual_puts = (
            puts[puts["vol_oi_ratio"] >= 3]
            .sort_values("volume", ascending=False)
            .head(8)
        )

        # ── Put/Call Ratio ────────────────────────────────────────
        total_call_vol = calls["volume"].sum()
        total_put_vol  = puts["volume"].sum()
        pcr            = total_put_vol / max(total_call_vol, 1)

        pcr_signal = (
            "Bullish (heavy calls)" if pcr < 0.7 else
            "Bearish (heavy puts)"  if pcr > 1.3 else
            "Neutral"
        )

        # ── IV Spike Detection ────────────────────────────────────
        info  = tk.fast_info
        price = getattr(info, "last_price", None) or getattr(info, "previous_close", 0)

        if price and price > 0:
            atm_calls = calls[abs(calls["strike"] - price) / price < 0.05]
            atm_puts  = puts[abs(puts["strike"]  - price) / price < 0.05]
            atm_iv    = pd.concat([atm_calls, atm_puts])["impliedVolatility"].mean()
        else:
            atm_iv = calls["impliedVolatility"].mean()

        if np.isnan(atm_iv):
            atm_iv = 0.0

        iv_spike = bool(atm_iv > 0.50)
        iv_pct   = round(atm_iv * 100, 1)

        iv_signal = (
            "Very High IV (>80%)" if atm_iv > 0.80 else
            "Elevated IV (50-80%)" if atm_iv > 0.50 else
            "Moderate IV (30-50%)" if atm_iv > 0.30 else
            "Low IV (<30%)"
        )

        # ── Top Strikes by Volume ─────────────────────────────────
        display_cols = ["expiry", "strike", "lastPrice", "volume",
                        "openInterest", "impliedVolatility", "vol_oi_ratio"]

        top_calls = calls[[c for c in display_cols if c in calls.columns]].nlargest(10, "volume")
        top_puts  = puts[[c  for c in display_cols if c in puts.columns]].nlargest(10, "volume")

        return {
            "ticker":         ticker,
            "unusual_calls":  unusual_calls,
            "unusual_puts":   unusual_puts,
            "top_calls":      top_calls,
            "top_puts":       top_puts,
            "pcr":            round(pcr, 3),
            "pcr_signal":     pcr_signal,
            "total_call_vol": int(total_call_vol),
            "total_put_vol":  int(total_put_vol),
            "total_call_oi":  int(calls["openInterest"].sum()),
            "total_put_oi":   int(puts["openInterest"].sum()),
            "atm_iv_pct":     iv_pct,
            "iv_spike":       iv_spike,
            "iv_signal":      iv_signal,
            "expiries_used":  list(dates[:3]),
        }

    except Exception as e:
        return {"error": str(e)}


def render_options_panel(ticker: str) -> None:
    """Render full options activity panel for a ticker inside Streamlit."""

    with st.spinner(f"Loading options data for {ticker}…"):
        data = get_options_data(ticker)

    if data is None:
        st.warning(f"No options available for {ticker}.")
        return
    if "error" in data:
        st.error(f"Options error: {data['error']}")
        return

    # ── Header metrics ────────────────────────────────────────────────────────
    st.markdown(f"### 📊 Options Activity — **{ticker}**")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Put/Call Ratio",  f"{data['pcr']:.2f}",     help="<0.7 bullish | >1.3 bearish")
    c2.metric("PCR Signal",      data["pcr_signal"])
    c3.metric("ATM IV",          f"{data['atm_iv_pct']}%")
    c4.metric("IV Signal",       data["iv_signal"])
    c5.metric("Total Call Vol",  f"{data['total_call_vol']:,}")

    c6, c7, c8 = st.columns(3)
    c6.metric("Total Put Vol",   f"{data['total_put_vol']:,}")
    c7.metric("Call OI",         f"{data['total_call_oi']:,}")
    c8.metric("Put OI",          f"{data['total_put_oi']:,}")

    st.divider()

    if data["iv_spike"]:
        st.warning(
            f"⚡ **IV Spike Detected** — ATM IV is {data['atm_iv_pct']}% (>50%). "
            f"Expect larger-than-normal moves. Option premiums are expensive — "
            f"consider spreads instead of naked long options."
        )

    # ── Unusual Activity ─────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    def _render_unusual(df, label):
        if df.empty:
            st.info(f"No unusual {label} activity detected.")
            return
        cols = [c for c in ["expiry","strike","lastPrice","volume",
                             "openInterest","vol_oi_ratio","impliedVolatility"]
                if c in df.columns]
        st.dataframe(
            df[cols].style.format({
                "lastPrice":         "${:.2f}",
                "vol_oi_ratio":      "{:.1f}x",
                "impliedVolatility": "{:.1%}",
                "volume":            "{:,.0f}",
                "openInterest":      "{:,.0f}",
            }),
            use_container_width=True, height=280
        )

    with col_l:
        st.markdown("#### 🟢 Unusual CALL Activity (Vol/OI ≥ 3x)")
        _render_unusual(data["unusual_calls"], "call")

    with col_r:
        st.markdown("#### 🔴 Unusual PUT Activity (Vol/OI ≥ 3x)")
        _render_unusual(data["unusual_puts"], "put")

    st.divider()

    # ── Top Strikes ──────────────────────────────────────────────────────────
    st.markdown("#### 🏆 Top Strikes by Volume")
    t1, t2 = st.tabs(["📈 Top Calls", "📉 Top Puts"])

    def _fmt(df):
        cols = [c for c in ["expiry","strike","lastPrice","volume",
                             "openInterest","impliedVolatility"] if c in df.columns]
        return df[cols].style.format({
            "lastPrice":         "${:.2f}",
            "impliedVolatility": "{:.1%}",
            "volume":            "{:,.0f}",
            "openInterest":      "{:,.0f}",
        })

    with t1:
        st.dataframe(_fmt(data["top_calls"]), use_container_width=True, height=320)
    with t2:
        st.dataframe(_fmt(data["top_puts"]),  use_container_width=True, height=320)


def scan_options_for_list(tickers: List[str], max_workers: int = 5) -> pd.DataFrame:
    """Batch scan tickers for unusual options activity. Returns summary DataFrame."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows, futures = [], {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for t in tickers:
            futures[ex.submit(get_options_data, t)] = t
            time.sleep(0.05)

        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                d = fut.result()
                if d and "error" not in d:
                    rows.append({
                        "Ticker":         ticker,
                        "PCR":            d["pcr"],
                        "PCR Signal":     d["pcr_signal"],
                        "ATM IV %":       d["atm_iv_pct"],
                        "IV Spike":       "YES" if d["iv_spike"] else "—",
                        "Unusual Calls":  len(d["unusual_calls"]),
                        "Unusual Puts":   len(d["unusual_puts"]),
                        "Call Vol":       d["total_call_vol"],
                        "Put Vol":        d["total_put_vol"],
                    })
            except Exception:
                pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Activity Score"] = (
        df["Unusual Calls"] * 2 +
        df["Unusual Puts"]  * 2 +
        (df["IV Spike"] == "YES").astype(int) * 3
    )
    return df.sort_values("Activity Score", ascending=False)

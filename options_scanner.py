"""
options_scanner.py — Unusual Options Activity Scanner

Uses yfinance (free) to detect:
  • High Volume/OI ratio  → unusual positioning
  • Call volume spike     → bullish bets
  • Put volume spike      → bearish / hedging
  • 0DTE / weekly activity (near-term expirations)
  • Put/Call ratio extremes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, date, timedelta
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────

def _safe_ratio(num, denom, default=0.0):
    try:
        return round(float(num) / float(denom), 2) if denom and denom != 0 else default
    except Exception:
        return default


def get_options_activity(ticker: str, max_expirations: int = 3) -> Dict:
    """
    Fetches options chain for the nearest `max_expirations` expiry dates.

    Returns a dict with:
      passed          bool — True if unusual activity detected
      signal          str  — "BULLISH CALLS", "BEARISH PUTS", "NEUTRAL", etc.
      pcr             float — Put/Call Ratio (volume-based)
      total_call_vol  int
      total_put_vol   int
      top_calls       DataFrame — highest volume calls
      top_puts        DataFrame — highest volume puts
      unusual_calls   DataFrame — vol/OI > 2
      unusual_puts    DataFrame — vol/OI > 2
      iv_skew         float — avg put IV - avg call IV
      expirations     list of dates scanned
    """
    import yfinance as yf

    empty = {
        "passed": False, "signal": "No Data", "pcr": 0.0,
        "total_call_vol": 0, "total_put_vol": 0,
        "top_calls": pd.DataFrame(), "top_puts": pd.DataFrame(),
        "unusual_calls": pd.DataFrame(), "unusual_puts": pd.DataFrame(),
        "iv_skew": 0.0, "expirations": [],
    }

    try:
        t    = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return empty

        # Scan nearest N expirations
        scan_exps = list(exps[:max_expirations])

        all_calls, all_puts = [], []

        for exp in scan_exps:
            try:
                chain = t.option_chain(exp)
                calls = chain.calls.copy()
                puts  = chain.puts.copy()

                calls["expiry"] = exp
                puts["expiry"]  = exp

                # Safe volume/OI ratio
                calls["vol_oi"] = calls.apply(
                    lambda r: _safe_ratio(r.get("volume", 0), r.get("openInterest", 0)), axis=1)
                puts["vol_oi"]  = puts.apply(
                    lambda r: _safe_ratio(r.get("volume", 0), r.get("openInterest", 0)), axis=1)

                # Days to expiry
                try:
                    dte = (pd.to_datetime(exp).date() - date.today()).days
                except Exception:
                    dte = 999
                calls["dte"] = dte
                puts["dte"]  = dte

                all_calls.append(calls)
                all_puts.append(puts)
            except Exception:
                continue

        if not all_calls and not all_puts:
            return empty

        calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
        puts_df  = pd.concat(all_puts,  ignore_index=True) if all_puts  else pd.DataFrame()

        # ── Totals ──────────────────────────────────────────────────────────
        total_call_vol = int(calls_df["volume"].fillna(0).sum()) if not calls_df.empty else 0
        total_put_vol  = int(puts_df["volume"].fillna(0).sum())  if not puts_df.empty  else 0
        pcr            = _safe_ratio(total_put_vol, total_call_vol, default=1.0)

        # ── Unusual activity (vol/OI > 2) ───────────────────────────────────
        unusual_cols = ["contractSymbol", "strike", "expiry", "dte",
                        "volume", "openInterest", "vol_oi",
                        "impliedVolatility", "lastPrice"]

        def _safe_cols(df, cols):
            return df[[c for c in cols if c in df.columns]]

        unusual_calls = pd.DataFrame()
        unusual_puts  = pd.DataFrame()

        if not calls_df.empty:
            uc = calls_df[calls_df["vol_oi"] > 2].sort_values("volume", ascending=False).head(10)
            unusual_calls = _safe_cols(uc, unusual_cols)

        if not puts_df.empty:
            up = puts_df[puts_df["vol_oi"] > 2].sort_values("volume", ascending=False).head(10)
            unusual_puts  = _safe_cols(up, unusual_cols)

        # ── Top volume ───────────────────────────────────────────────────────
        top_calls = _safe_cols(
            calls_df.sort_values("volume", ascending=False).head(5), unusual_cols
        ) if not calls_df.empty else pd.DataFrame()

        top_puts = _safe_cols(
            puts_df.sort_values("volume", ascending=False).head(5), unusual_cols
        ) if not puts_df.empty else pd.DataFrame()

        # ── IV Skew (put IV - call IV) ────────────────────────────────────
        try:
            avg_call_iv = calls_df["impliedVolatility"].fillna(0).mean()
            avg_put_iv  = puts_df["impliedVolatility"].fillna(0).mean()
            iv_skew     = round(float(avg_put_iv - avg_call_iv), 4)
        except Exception:
            iv_skew = 0.0

        # ── Signal determination ─────────────────────────────────────────
        unusual_call_vol = unusual_calls["volume"].sum() if not unusual_calls.empty else 0
        unusual_put_vol  = unusual_puts["volume"].sum()  if not unusual_puts.empty  else 0

        # 0DTE / weekly activity
        near_term_calls = calls_df[calls_df["dte"] <= 7]["volume"].sum() if not calls_df.empty else 0
        near_term_puts  = puts_df[puts_df["dte"] <= 7]["volume"].sum()   if not puts_df.empty  else 0

        high_call_activity = (unusual_call_vol > 0 or near_term_calls > 1000)
        high_put_activity  = (unusual_put_vol  > 0 or near_term_puts  > 1000)

        if pcr < 0.5 and high_call_activity:
            signal = "BULLISH CALLS 🟢"
            passed = True
        elif pcr > 2.0 and high_put_activity:
            signal = "BEARISH PUTS 🔴"
            passed = True
        elif high_call_activity and total_call_vol > total_put_vol * 1.5:
            signal = "CALL SWEEP 🟢"
            passed = True
        elif high_put_activity and total_put_vol > total_call_vol * 1.5:
            signal = "PUT SWEEP 🔴"
            passed = True
        elif high_call_activity or high_put_activity:
            signal = "UNUSUAL ACTIVITY ⚡"
            passed = True
        else:
            signal = "Normal"
            passed = False

        return {
            "passed":         passed,
            "signal":         signal,
            "pcr":            round(pcr, 2),
            "total_call_vol": total_call_vol,
            "total_put_vol":  total_put_vol,
            "top_calls":      top_calls,
            "top_puts":       top_puts,
            "unusual_calls":  unusual_calls,
            "unusual_puts":   unusual_puts,
            "iv_skew":        iv_skew,
            "expirations":    scan_exps,
        }

    except Exception as e:
        return {**empty, "signal": f"Error: {e}"}


def batch_options_scan(tickers: List[str],
                       min_call_vol: int = 500,
                       max_pcr: float = 3.0) -> List[Dict]:
    """
    Scan a list of tickers for unusual options activity.
    Returns list of result dicts sorted by total options volume.
    """
    results = []
    prog    = st.progress(0, text="Scanning options…")
    total   = len(tickers)

    for i, sym in enumerate(tickers):
        try:
            res = get_options_activity(sym)
            if res["passed"] and (res["total_call_vol"] + res["total_put_vol"]) >= min_call_vol:
                results.append({"ticker": sym, **res})
        except Exception:
            pass
        prog.progress((i + 1) / total, text=f"Options scan: {i+1}/{total}")

    prog.empty()

    results.sort(
        key=lambda r: r["total_call_vol"] + r["total_put_vol"],
        reverse=True
    )
    return results

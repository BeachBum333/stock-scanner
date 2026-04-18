"""
data_fetcher.py — Market data layer.
Primary: Alpaca Markets | Fallback: yfinance (free, no key needed)
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time


def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[needed + [c for c in df.columns if c not in needed]]


def _alpaca_get_bars(symbols, api_key, secret_key, period_days=400):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key, secret_key)
    end    = datetime.now()
    start  = end - timedelta(days=period_days)
    result = {}

    for i in range(0, len(symbols), 50):
        batch = symbols[i:i+50]
        try:
            req  = StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                                    start=start, end=end, adjustment="all")
            bars   = client.get_stock_bars(req)
            df_all = bars.df
            if df_all.empty:
                continue
            for sym in batch:
                try:
                    sym_df = df_all.loc[sym].copy()
                    sym_df.index = pd.DatetimeIndex(sym_df.index)
                    sym_df = _normalise_cols(sym_df)
                    if len(sym_df) >= 30:
                        result[sym] = sym_df
                except KeyError:
                    pass
        except Exception as e:
            st.warning(f"Alpaca batch error: {e}")
        time.sleep(0.1)
    return result


def _alpaca_get_snapshots(symbols, api_key, secret_key):
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockSnapshotRequest
        client = StockHistoricalDataClient(api_key, secret_key)
        snaps  = client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=symbols))
        return {sym: snap for sym, snap in snaps.items()}
    except Exception:
        return {}


def _yfinance_get_bars(symbols, period_days=400):
    import yfinance as yf
    result = {}
    period = f"{period_days}d"

    for i in range(0, len(symbols), 50):
        batch = symbols[i:i+50]
        try:
            raw = yf.download(batch, period=period, interval="1d",
                              auto_adjust=True, group_by="ticker",
                              progress=False, threads=True)
            for sym in batch:
                try:
                    df = raw.copy() if len(batch) == 1 else raw[sym].copy()
                    df = _normalise_cols(df)
                    df = df.dropna(subset=["close"])
                    if len(df) >= 30:
                        result[sym] = df
                except Exception:
                    pass
        except Exception as e:
            st.warning(f"yfinance batch error: {e}")
        time.sleep(0.05)
    return result


def fetch_stock_data(symbols, api_key="", secret_key="", period_days=400, use_alpaca=True):
    if use_alpaca and api_key and secret_key:
        try:
            data = _alpaca_get_bars(symbols, api_key, secret_key, period_days)
            if data:
                return data
            st.info("Alpaca returned no data — switching to yfinance.")
        except ImportError:
            st.warning("alpaca-py not installed. Run: pip install alpaca-py")
        except Exception as e:
            st.warning(f"Alpaca error: {e}. Falling back to yfinance.")
    return _yfinance_get_bars(symbols, period_days)


def fetch_premarket_bars(symbols, api_key, secret_key):
    if not (api_key and secret_key):
        return {}
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        client = StockHistoricalDataClient(api_key, secret_key)
        today  = datetime.now().date()
        start  = datetime.combine(today, datetime.min.time()).replace(hour=4, minute=0)
        req    = StockBarsRequest(symbol_or_symbols=symbols,
                                  timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                                  start=start, end=datetime.now())
        bars   = client.get_stock_bars(req)
        df_all = bars.df
        result = {}
        if not df_all.empty:
            for sym in symbols:
                try:
                    result[sym] = _normalise_cols(df_all.loc[sym].copy())
                except KeyError:
                    pass
        return result
    except Exception as e:
        st.warning(f"Premarket data unavailable: {e}")
        return {}


def get_current_prices(symbols, api_key="", secret_key=""):
    if api_key and secret_key:
        snaps = _alpaca_get_snapshots(symbols, api_key, secret_key)
        if snaps:
            prices = {}
            for sym, snap in snaps.items():
                try:
                    prices[sym] = snap.latest_trade.price
                except Exception:
                    try:
                        prices[sym] = snap.daily_bar.close
                    except Exception:
                        pass
            if prices:
                return prices

    import yfinance as yf
    try:
        raw = yf.download(symbols, period="2d", interval="1d",
                          auto_adjust=True, group_by="ticker",
                          progress=False, threads=True)
        prices = {}
        for sym in symbols:
            try:
                df = raw[sym] if len(symbols) > 1 else raw
                prices[sym] = float(df["Close"].dropna().iloc[-1])
            except Exception:
                pass
        return prices
    except Exception:
        return {}


def market_is_open() -> Tuple[bool, str]:
    from datetime import timezone
    import zoneinfo
    try:
        et = zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        from datetime import timedelta
        et = timezone(timedelta(hours=-4))

    now    = datetime.now(et)
    wd     = now.weekday()
    hour   = now.hour
    minute = now.minute

    if wd >= 5:
        return False, "Market Closed (Weekend)"
    if hour < 4:
        return False, "Pre-Pre-Market (before 4 AM ET)"
    if hour < 9 or (hour == 9 and minute < 30):
        return True,  "Pre-Market (4-9:30 AM ET)"
    if hour < 16:
        return True,  "Market Open (9:30 AM - 4 PM ET)"
    if hour < 20:
        return True,  "After-Hours (4-8 PM ET)"
    return False, "Market Closed"

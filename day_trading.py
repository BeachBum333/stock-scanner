"""
day_trading.py — Day Trading Strategy Implementations

Strategies:
  1. Opening Range Breakout (ORB) — first 15-min high/low
  2. VWAP Cross & Bounce
  3. Momentum Scalp Setup
  4. High-of-Day (HOD) Breakout
  5. Pre-market Gap & Go
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, date


# ── Helpers ─────────────────────────────────────────────────────────────────

def _today_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Filter intraday DataFrame to today's bars only."""
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        today = pd.Timestamp.now().date()
        mask  = df.index.normalize().date == today
        return df[mask]
    except Exception:
        return df  # fallback: return as-is


def _calc_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP from intraday bars."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()


def _calc_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, VWAP, rel_vol to an intraday bar DataFrame."""
    df = df.copy()

    # VWAP (reset each day)
    df["vwap"] = _calc_vwap(df)

    # Volume
    df["vol_avg"] = df["volume"].rolling(20, min_periods=1).mean()
    df["rel_vol"] = df["volume"] / df["vol_avg"].replace(0, np.nan)

    # RSI (14)
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema9        = df["close"].ewm(span=9,  adjust=False).mean()
    ema21       = df["close"].ewm(span=21, adjust=False).mean()
    df["macd"]  = ema9 - ema21
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()

    # MAs
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    return df


# ═══════════════════════════════════════════════════════════
#  STRATEGY 1: Opening Range Breakout (ORB)
# ═══════════════════════════════════════════════════════════

def opening_range_breakout(df_intraday: pd.DataFrame,
                           or_bars: int = 3) -> Dict:
    """
    First `or_bars` x 5-min bars (default = 15 min) define the Opening Range.
    Signal fires when price breaks above OR High (bull) or below OR Low (bear)
    on elevated volume.
    """
    empty = {"passed": False, "score": 0, "detail": {}, "pattern": "ORB",
             "grade": "N/A", "direction": "—"}

    today = _today_bars(df_intraday)
    if len(today) < or_bars + 1:
        return empty

    df_ind = _calc_intraday_indicators(today)

    # Opening range
    or_slice  = df_ind.iloc[:or_bars]
    or_high   = or_slice["high"].max()
    or_low    = or_slice["low"].min()
    or_range  = or_high - or_low

    latest    = df_ind.iloc[-1]
    price     = latest["close"]
    rel_vol   = latest["rel_vol"]
    rsi       = latest["rsi"]
    vwap      = latest["vwap"]

    bull = price > or_high
    bear = price < or_low

    if not (bull or bear):
        return {**empty, "or_high": round(or_high, 2), "or_low": round(or_low, 2)}

    direction = "BULL 🟢" if bull else "BEAR 🔴"

    if bull:
        checks = {
            "Price > OR High":        True,
            "Rel Vol > 1.5x":         bool(rel_vol > 1.5),
            "Price > VWAP":           bool(price > vwap),
            "RSI 50–80":              bool(50 <= rsi <= 80),
            "EMA9 > EMA21":           bool(latest["ema9"] > latest["ema21"]),
            "OR Range < 2% of Price": bool(or_range / price < 0.02),
        }
    else:
        checks = {
            "Price < OR Low":         True,
            "Rel Vol > 1.5x":         bool(rel_vol > 1.5),
            "Price < VWAP":           bool(price < vwap),
            "RSI 20–50":              bool(20 <= rsi <= 50),
            "EMA9 < EMA21":           bool(latest["ema9"] < latest["ema21"]),
            "OR Range < 2% of Price": bool(or_range / price < 0.02),
        }

    score  = sum(checks.values())
    passed = score >= 3
    grade  = "A" if score >= 5 else "B" if score >= 4 else "C"

    return {
        "passed":    passed,
        "score":     score,
        "total":     6,
        "detail":    checks,
        "grade":     grade,
        "pattern":   "ORB",
        "direction": direction,
        "or_high":   round(or_high, 2),
        "or_low":    round(or_low,  2),
        "vwap":      round(vwap,    2),
        "rel_vol":   round(rel_vol, 1),
        "rsi":       round(rsi,     1),
    }


# ═══════════════════════════════════════════════════════════
#  STRATEGY 2: VWAP Cross & Bounce
# ═══════════════════════════════════════════════════════════

def vwap_strategy(df_intraday: pd.DataFrame) -> Dict:
    """
    Bullish: price was below VWAP, crosses above with momentum.
    Bearish: price was above VWAP, crosses below.
    Bounce:  price pulls back to VWAP in an uptrend and holds.
    """
    empty = {"passed": False, "score": 0, "detail": {}, "pattern": "VWAP",
             "grade": "N/A", "direction": "—"}

    today = _today_bars(df_intraday)
    if len(today) < 5:
        return empty

    df_ind  = _calc_intraday_indicators(today)
    latest  = df_ind.iloc[-1]
    prev    = df_ind.iloc[-2]

    price     = latest["close"]
    vwap      = latest["vwap"]
    prev_vwap = prev["vwap"]
    rel_vol   = latest["rel_vol"]
    rsi       = latest["rsi"]

    # Cross detection
    bull_cross = prev["close"] < prev_vwap and price > vwap
    bear_cross = prev["close"] > prev_vwap and price < vwap

    # Bounce detection (within 0.3% of VWAP, holding above)
    near_vwap  = abs(price - vwap) / vwap < 0.003
    bull_bounce= near_vwap and price >= vwap and rsi > 50
    bear_bounce= near_vwap and price <= vwap and rsi < 50

    if not (bull_cross or bear_cross or bull_bounce or bear_bounce):
        return empty

    direction = "BULL 🟢" if (bull_cross or bull_bounce) else "BEAR 🔴"
    setup     = "Cross" if (bull_cross or bear_cross) else "Bounce"

    if direction == "BULL 🟢":
        checks = {
            "Bull VWAP Cross/Bounce":   bool(bull_cross or bull_bounce),
            "Rel Vol > 1.3x":           bool(rel_vol > 1.3),
            "RSI > 50":                 bool(rsi > 50),
            "Price > EMA9":             bool(price > latest["ema9"]),
            "MACD Bullish":             bool(latest["macd"] > latest["macd_sig"]),
        }
    else:
        checks = {
            "Bear VWAP Cross/Bounce":   bool(bear_cross or bear_bounce),
            "Rel Vol > 1.3x":           bool(rel_vol > 1.3),
            "RSI < 50":                 bool(rsi < 50),
            "Price < EMA9":             bool(price < latest["ema9"]),
            "MACD Bearish":             bool(latest["macd"] < latest["macd_sig"]),
        }

    score  = sum(checks.values())
    passed = score >= 3
    grade  = "A" if score == 5 else "B" if score >= 4 else "C"

    return {
        "passed":    passed,
        "score":     score,
        "total":     5,
        "detail":    checks,
        "grade":     grade,
        "pattern":   "VWAP",
        "direction": direction,
        "setup":     setup,
        "price":     round(price,   2),
        "vwap":      round(vwap,    2),
        "rel_vol":   round(rel_vol, 1),
        "rsi":       round(rsi,     1),
    }


# ═══════════════════════════════════════════════════════════
#  STRATEGY 3: Momentum Scalp Setup
# ═══════════════════════════════════════════════════════════

def momentum_scalp(df_intraday: pd.DataFrame) -> Dict:
    """
    Fast momentum: EMA crossover + RSI surge + volume spike + price > VWAP.
    Designed for 5-min chart scalp entries.
    """
    empty = {"passed": False, "score": 0, "detail": {}, "pattern": "Momentum Scalp",
             "grade": "N/A"}

    today = _today_bars(df_intraday)
    if len(today) < 10:
        return empty

    df_ind = _calc_intraday_indicators(today)
    latest = df_ind.iloc[-1]
    prev   = df_ind.iloc[-2]

    price   = latest["close"]
    vwap    = latest["vwap"]
    rel_vol = latest["rel_vol"]
    rsi     = latest["rsi"]

    # EMA crossover (9 over 21)
    ema_cross_bull = prev["ema9"] <= prev["ema21"] and latest["ema9"] > latest["ema21"]
    ema_cross_bear = prev["ema9"] >= prev["ema21"] and latest["ema9"] < latest["ema21"]

    # Strong momentum bar
    bar_range    = latest["high"] - latest["low"]
    body         = abs(latest["close"] - latest["open"])
    strong_bar   = bar_range > 0 and body / bar_range > 0.6   # body > 60% of range

    # Price momentum
    pct_chg  = (price - prev["close"]) / prev["close"]
    momentum = abs(pct_chg) >= 0.003  # 0.3%+ move this bar

    bull = ema_cross_bull or (latest["ema9"] > latest["ema21"] and price > vwap)
    bear = ema_cross_bear or (latest["ema9"] < latest["ema21"] and price < vwap)

    if not (bull or bear):
        return empty

    direction = "BULL 🟢" if bull else "BEAR 🔴"

    checks = {
        "EMA Cross / Aligned":  bool(ema_cross_bull if bull else ema_cross_bear) or bool(bull or bear),
        "Price vs VWAP":        bool(price > vwap if bull else price < vwap),
        "Rel Vol > 2x":         bool(rel_vol > 2.0),
        "RSI in Range":         bool(55 <= rsi <= 85 if bull else 15 <= rsi <= 45),
        "Strong Momentum Bar":  bool(strong_bar and momentum),
        "MACD Confirms":        bool(latest["macd"] > latest["macd_sig"] if bull
                                     else latest["macd"] < latest["macd_sig"]),
    }

    score  = sum(checks.values())
    passed = score >= 4
    grade  = "A" if score >= 6 else "B" if score >= 5 else "C" if score >= 4 else "D"

    return {
        "passed":    passed,
        "score":     score,
        "total":     6,
        "detail":    checks,
        "grade":     grade,
        "pattern":   "Momentum Scalp",
        "direction": direction,
        "rel_vol":   round(rel_vol, 1),
        "rsi":       round(rsi,     1),
        "pct_chg":   round(pct_chg * 100, 2),
        "vwap":      round(vwap,    2),
    }


# ═══════════════════════════════════════════════════════════
#  STRATEGY 4: High-of-Day (HOD) Breakout
# ═══════════════════════════════════════════════════════════

def hod_breakout(df_intraday: pd.DataFrame,
                 df_daily: Optional[pd.DataFrame] = None) -> Dict:
    """
    Detects when price breaks above the current intraday HOD (high of day)
    or the prior day's high on volume — a classic continuation breakout.
    """
    empty = {"passed": False, "score": 0, "detail": {}, "pattern": "HOD Breakout",
             "grade": "N/A"}

    today = _today_bars(df_intraday)
    if len(today) < 5:
        return empty

    df_ind    = _calc_intraday_indicators(today)
    latest    = df_ind.iloc[-1]
    price     = latest["close"]
    hod       = today["high"].max()
    rel_vol   = latest["rel_vol"]
    rsi       = latest["rsi"]
    vwap      = latest["vwap"]

    # Prior day's high (if daily data available)
    prev_high = df_daily["high"].iloc[-2] if (df_daily is not None and len(df_daily) >= 2) else None

    at_hod        = abs(price - hod) / hod < 0.002   # within 0.2% of HOD
    breaks_prev_h = bool(prev_high and price > prev_high)

    if not (at_hod or breaks_prev_h):
        return empty

    checks = {
        "At / Breaking HOD":       bool(at_hod),
        "Breaking Prev Day High":  bool(breaks_prev_h),
        "Rel Vol > 1.5x":          bool(rel_vol > 1.5),
        "Price > VWAP":            bool(price > vwap),
        "RSI 50–80":               bool(50 <= rsi <= 80),
        "EMA9 > EMA21":            bool(latest["ema9"] > latest["ema21"]),
    }

    score  = sum(checks.values())
    passed = score >= 3
    grade  = "A" if score >= 5 else "B" if score >= 4 else "C"

    return {
        "passed":     passed,
        "score":      score,
        "total":      6,
        "detail":     checks,
        "grade":      grade,
        "pattern":    "HOD Breakout",
        "direction":  "BULL 🟢",
        "hod":        round(hod,   2),
        "prev_high":  round(prev_high, 2) if prev_high else "N/A",
        "vwap":       round(vwap,  2),
        "rel_vol":    round(rel_vol, 1),
        "rsi":        round(rsi,   1),
    }


# ═══════════════════════════════════════════════════════════
#  STRATEGY 5: Pre-market Gap & Go
# ═══════════════════════════════════════════════════════════

def premarket_gap_go(df_intraday: pd.DataFrame,
                     df_daily: Optional[pd.DataFrame] = None) -> Dict:
    """
    Gap & Go: stock gapped up/down in premarket and continues in the same
    direction at open. Entry: break of first 5-min candle high (bull) or low (bear).
    """
    empty = {"passed": False, "score": 0, "detail": {}, "pattern": "Gap & Go",
             "grade": "N/A", "direction": "—"}

    if df_intraday is None or df_intraday.empty:
        return empty

    df_ind = _calc_intraday_indicators(df_intraday)

    # Gap = today's first bar open vs yesterday's close
    if df_daily is not None and len(df_daily) >= 2:
        prev_close = df_daily["close"].iloc[-2]
    else:
        prev_close = df_intraday["open"].iloc[0]   # fallback

    first_bar  = df_intraday.iloc[0]
    today_open = first_bar["open"]
    gap_pct    = (today_open - prev_close) / prev_close

    # Need at least a 2% gap
    if abs(gap_pct) < 0.02:
        return empty

    direction  = "BULL 🟢" if gap_pct > 0 else "BEAR 🔴"
    latest     = df_ind.iloc[-1]
    price      = latest["close"]
    rel_vol    = latest["rel_vol"]
    rsi        = latest["rsi"]
    vwap       = latest["vwap"]

    # Gap & Go: price continuing in direction of gap
    continuing = (price > today_open) if gap_pct > 0 else (price < today_open)

    if gap_pct > 0:
        checks = {
            "Gap Up > 2%":        bool(gap_pct >= 0.02),
            "Continuing Up":      bool(continuing),
            "Price > VWAP":       bool(price > vwap),
            "Rel Vol > 2x":       bool(rel_vol > 2.0),
            "RSI > 55":           bool(rsi > 55),
            "Gap > 5% (Strong)":  bool(abs(gap_pct) >= 0.05),
        }
    else:
        checks = {
            "Gap Down > 2%":      bool(abs(gap_pct) >= 0.02),
            "Continuing Down":    bool(continuing),
            "Price < VWAP":       bool(price < vwap),
            "Rel Vol > 2x":       bool(rel_vol > 2.0),
            "RSI < 45":           bool(rsi < 45),
            "Gap > 5% (Strong)":  bool(abs(gap_pct) >= 0.05),
        }

    score  = sum(checks.values())
    passed = score >= 3
    grade  = "A" if score >= 5 else "B" if score >= 4 else "C"

    return {
        "passed":    passed,
        "score":     score,
        "total":     6,
        "detail":    checks,
        "grade":     grade,
        "pattern":   "Gap & Go",
        "direction": direction,
        "gap_pct":   round(gap_pct * 100, 2),
        "vwap":      round(vwap,     2),
        "rel_vol":   round(rel_vol,  1),
        "rsi":       round(rsi,      1),
    }


# ═══════════════════════════════════════════════════════════
#  DAY TRADING MASTER RUNNER
# ═══════════════════════════════════════════════════════════

DAY_STRATEGY_MAP = {
    "ORB (15-min)":    opening_range_breakout,
    "VWAP Cross":      vwap_strategy,
    "Momentum Scalp":  momentum_scalp,
    "HOD Breakout":    hod_breakout,
    "Gap & Go":        premarket_gap_go,
}


def run_day_strategies(ticker: str,
                       df_intraday: pd.DataFrame,
                       df_daily: Optional[pd.DataFrame],
                       selected: list) -> Dict:
    """Run selected day-trading strategies on intraday bars."""

    if df_intraday is None or df_intraday.empty:
        return None

    row = {
        "Ticker":   ticker,
        "Price":    round(df_intraday["close"].iloc[-1], 2),
        "Signals":  [],
        "_detail":  {},
        "# Signals": 0,
    }

    for name in selected:
        fn = DAY_STRATEGY_MAP.get(name)
        if fn is None:
            continue
        try:
            if name in ("HOD Breakout", "Gap & Go"):
                res = fn(df_intraday, df_daily)
            else:
                res = fn(df_intraday)

            row["_detail"][name] = res
            if res.get("passed"):
                grade = res.get("grade", "")
                row["Signals"].append(f"{name} [{grade}]")
                row["# Signals"] += 1
        except Exception as e:
            row["_detail"][name] = {"passed": False, "error": str(e)}

    row["Signal List"] = ", ".join(row["Signals"]) if row["Signals"] else "—"
    return row

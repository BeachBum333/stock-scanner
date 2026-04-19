"""
strategies.py — All stock scanning strategy implementations.

Strategies:
  1. Minervini SEPA / Trend Template + VCP
  2. Pradeep Bonde Momentum Burst + Episodic Pivot
  3. Bullish Island Reversal
  4. Gap Up / Gap Down (Momentum)
  5. Failed Breakdown (Bear Trap)
  6. Bullish / Bearish Momentum
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════
#  INDICATOR ENGINE  — adds all technical indicators to df
# ═══════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    df["ma10"]  = close.rolling(10).mean()
    df["ma20"]  = close.rolling(20).mean()
    df["ma50"]  = close.rolling(50).mean()
    df["ma150"] = close.rolling(150).mean()
    df["ma200"] = close.rolling(200).mean()

    df["vol_avg20"] = vol.rolling(20).mean()
    df["rel_vol"]   = vol / df["vol_avg20"].replace(0, np.nan)

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12           = close.ewm(span=12, adjust=False).mean()
    ema26           = close.ewm(span=26, adjust=False).mean()
    df["macd"]      = ema12 - ema26
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    bb_mid          = close.rolling(20).mean()
    bb_std          = close.rolling(20).std()
    df["bb_upper"]  = bb_mid + 2 * bb_std
    df["bb_lower"]  = bb_mid - 2 * bb_std
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)

    df["high_52w"] = high.rolling(252).max()
    df["low_52w"]  = low.rolling(252).min()

    hl   = high - low
    hc   = (high - close.shift()).abs()
    lc   = (low  - close.shift()).abs()
    tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    df["pct_chg"] = close.pct_change()
    df["gap_pct"] = (df["open"] - close.shift()) / close.shift()

    return df


# ═══════════════════════════════════════════════════════════
#  STRATEGY 1: Minervini SEPA / Trend Template + VCP
# ═══════════════════════════════════════════════════════════

def minervini_sepa(df: pd.DataFrame, min_rs: float = 70.0) -> Dict:
    if len(df) < 252:
        return {"passed": False, "score": 0, "detail": {}, "grade": "N/A", "pattern": "Minervini SEPA"}

    latest   = df.iloc[-1]
    price    = latest["close"]
    ma50     = latest["ma50"]
    ma150    = latest["ma150"]
    ma200    = latest["ma200"]
    high_52w = latest["high_52w"]
    low_52w  = latest["low_52w"]

    ma200_1mo = df["ma200"].iloc[-22] if len(df) >= 22 else np.nan
    ma200_up  = bool(ma200 > ma200_1mo) if not np.isnan(ma200_1mo) else False

    last10_range = (df["high"].iloc[-10:] - df["low"].iloc[-10:]).mean()
    prior_range  = (df["high"].iloc[-30:-10] - df["low"].iloc[-30:-10]).mean()
    vcp_tighten  = bool(last10_range < prior_range * 0.7) if prior_range > 0 else False
    vol_dry_up   = bool(df["rel_vol"].iloc[-5:].mean() < 0.8)

    checks = {
        "Price > MA50":             bool(price > ma50),
        "Price > MA150":            bool(price > ma150),
        "Price > MA200":            bool(price > ma200),
        "MA150 > MA200":            bool(ma150 > ma200),
        "MA200 Trending Up":        ma200_up,
        "MA50 > MA150 & MA200":     bool(ma50 > ma150 and ma50 > ma200),
        ">=25% Above 52W Low":      bool(price >= low_52w  * 1.25),
        "Within 25% of 52W High":   bool(price >= high_52w * 0.75),
        "VCP Tightening":           vcp_tighten,
    }

    score  = sum(checks.values())
    passed = all([checks[k] for k in list(checks.keys())[:8]])
    grade  = "A" if score >= 9 else "B" if score >= 7 else "C" if score >= 5 else "D"

    return {
        "passed":  passed,
        "score":   score,
        "total":   9,
        "detail":  checks,
        "grade":   grade,
        "pattern": "Minervini SEPA",
        "rsi":     round(latest["rsi"], 1),
        "rel_vol": round(latest["rel_vol"], 1),
        "pct_from_high": round((price / high_52w - 1) * 100, 1),
    }


# ═══════════════════════════════════════════════════════════
#  STRATEGY 2: Pradeep Bonde Momentum Burst + Episodic Pivot
# ═══════════════════════════════════════════════════════════

def bonde_momentum(df: pd.DataFrame) -> Dict:
    if len(df) < 30:
        return {"passed": False, "score": 0, "detail": {}, "pattern": "Bonde Momentum"}

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    pct_change  = latest["pct_chg"]
    rel_vol     = latest["rel_vol"]
    vol         = latest["volume"]
    close       = latest["close"]

    bb_width_now   = latest["bb_width"]
    bb_width_20ago = df["bb_width"].iloc[-20:].mean()
    bb_squeeze     = bool(bb_width_now < bb_width_20ago * 0.75)

    low_vol_base = bool(df["rel_vol"].iloc[-6:-1].mean() < 0.8)
    above_ma50   = bool(close > latest["ma50"])
    above_ma20   = bool(close > latest["ma20"])
    episodic     = bool(pct_change >= 0.08 and rel_vol >= 3.0)
    burst_base   = bool(pct_change >= 0.04 and vol > prev["volume"] and vol > 100_000)

    checks = {
        "Up >=4% Today":    bool(pct_change >= 0.04),
        "Vol > Yesterday":  bool(vol > prev["volume"]),
        "Vol > 100K":       bool(vol > 100_000),
        "Above MA50":       above_ma50,
        "Above MA20":       above_ma20,
        "Low-Vol Base":     low_vol_base or bb_squeeze,
        "Rel Vol >=1.5x":   bool(rel_vol >= 1.5),
        "Episodic Pivot":   episodic,
    }

    score  = sum(checks.values())
    passed = burst_base
    setup  = "Episodic Pivot" if episodic else "Momentum Burst" if burst_base else "Not Triggered"
    grade  = "A" if episodic else "B" if score >= 6 else "C" if score >= 4 else "D"

    return {
        "passed":   passed,
        "score":    score,
        "total":    8,
        "detail":   checks,
        "grade":    grade,
        "pattern":  "Bonde Momentum",
        "setup":    setup,
        "pct_chg":  round(pct_change * 100, 2),
        "rel_vol":  round(rel_vol, 1),
        "rsi":      round(latest["rsi"], 1),
    }


# ═══════════════════════════════════════════════════════════
#  STRATEGY 3: Bullish Island Reversal
# ═══════════════════════════════════════════════════════════

def bullish_island_reversal(df: pd.DataFrame, lookback: int = 15) -> Dict:
    if len(df) < lookback + 5:
        return {"passed": False, "score": 0, "detail": {}, "pattern": "Island Reversal"}

    result = {"passed": False, "score": 0, "detail": {}, "pattern": "Island Reversal",
              "grade": "N/A", "gap_down_date": None, "island_days": 0}

    latest      = df.iloc[-1]
    gap_up_today = latest["gap_pct"] >= 0.01
    if not gap_up_today:
        result["detail"]["Gap Up Today"] = False
        return result

    for j in range(2, lookback):
        idx_gap = -j
        if abs(idx_gap) >= len(df):
            break

        gap_down_bar = df.iloc[idx_gap]
        prev_bar     = df.iloc[idx_gap - 1]

        if gap_down_bar["open"] >= prev_bar["close"] * 0.99:
            continue

        island_slice = df.iloc[idx_gap: -1]
        island_high  = island_slice["high"].max()
        island_low   = island_slice["low"].min()
        island_range = (island_high - island_low) / island_low if island_low > 0 else 1

        if island_range > 0.10:
            continue

        gap_fills    = latest["open"] >= prev_bar["close"] * 0.97
        high_vol     = latest["rel_vol"] >= 1.5
        closes_above = latest["close"] > island_high

        checks = {
            "Gap Down Found":      True,
            "Tight Island":        bool(island_range <= 0.08),
            "Gap Up Today":        True,
            "Gap Fills Prior":     bool(gap_fills),
            "High Volume":         bool(high_vol),
            "Closes Above Island": bool(closes_above),
        }

        score  = sum(checks.values())
        passed = checks["Gap Down Found"] and checks["Gap Up Today"] and score >= 4

        result.update({
            "passed":        passed,
            "score":         score,
            "total":         6,
            "detail":        checks,
            "grade":         "A" if score >= 6 else "B" if score >= 4 else "C",
            "gap_down_date": str(df.index[idx_gap].date()) if hasattr(df.index[idx_gap], "date") else str(df.index[idx_gap]),
            "island_days":   j - 1,
            "rel_vol":       round(latest["rel_vol"], 1),
            "gap_up_pct":    round(latest["gap_pct"] * 100, 2),
        })
        return result

    return result


# ═══════════════════════════════════════════════════════════
#  STRATEGY 4: Gap Up / Gap Down
# ═══════════════════════════════════════════════════════════

def gap_scanner(df: pd.DataFrame,
                min_gap_pct: float = 0.03,
                min_rel_vol: float = 1.5) -> Dict:
    if len(df) < 22:
        return {"passed": False, "score": 0, "detail": {}, "pattern": "Gap Scanner"}

    latest   = df.iloc[-1]
    gap_pct  = latest["gap_pct"]
    rel_vol  = latest["rel_vol"]
    close    = latest["close"]
    high_52w = latest["high_52w"]
    low_52w  = latest["low_52w"]

    direction = "UP" if gap_pct > 0 else "DOWN"
    abs_gap   = abs(gap_pct)

    if direction == "UP":
        checks = {
            f"Gap >={int(min_gap_pct*100)}%": bool(abs_gap >= min_gap_pct),
            "Rel Vol >=1.5x":                 bool(rel_vol >= min_rel_vol),
            "Above MA50":                     bool(close > latest["ma50"]),
            "Near 52W High (<=15%)":          bool(close >= high_52w * 0.85),
            "RSI 40-80":                      bool(40 <= latest["rsi"] <= 80),
            "MACD Bullish":                   bool(latest["macd"] > latest["macd_sig"]),
        }
    else:
        checks = {
            f"Gap >={int(min_gap_pct*100)}%": bool(abs_gap >= min_gap_pct),
            "Rel Vol >=1.5x":                 bool(rel_vol >= min_rel_vol),
            "Below MA50":                     bool(close < latest["ma50"]),
            "Near 52W Low (<=15%)":           bool(close <= low_52w * 1.15),
            "RSI 20-60":                      bool(20 <= latest["rsi"] <= 60),
            "MACD Bearish":                   bool(latest["macd"] < latest["macd_sig"]),
        }

    score  = sum(checks.values())
    passed = bool(abs_gap >= min_gap_pct and rel_vol >= min_rel_vol)
    grade  = "A" if score >= 5 else "B" if score >= 4 else "C" if score >= 3 else "D"

    return {
        "passed":    passed,
        "score":     score,
        "total":     6,
        "detail":    checks,
        "grade":     grade,
        "pattern":   "Gap Scanner",
        "direction": direction,
        "gap_pct":   round(gap_pct * 100, 2),
        "rel_vol":   round(rel_vol, 1),
        "rsi":       round(latest["rsi"], 1),
    }


# ═══════════════════════════════════════════════════════════
#  STRATEGY 5: Failed Breakdown (Bear Trap)
# ═══════════════════════════════════════════════════════════

def failed_breakdown(df: pd.DataFrame, support_lookback: int = 20) -> Dict:
    if len(df) < support_lookback + 2:
        return {"passed": False, "score": 0, "detail": {}, "pattern": "Failed Breakdown"}

    latest = df.iloc[-1]
    window = df.iloc[-(support_lookback + 1):-1]
    support = window["low"].min()

    today_low   = latest["low"]
    today_close = latest["close"]
    today_high  = latest["high"]
    day_range   = today_high - today_low

    broke_below = bool(today_low < support * 0.993)
    recovered   = bool(today_close > support)
    upper_close = bool(day_range > 0 and (today_close - today_low) / day_range > 0.60)
    high_vol    = bool(latest["rel_vol"] >= 1.2)

    rsi_now  = latest["rsi"]
    rsi_5ago = df["rsi"].iloc[-6]
    rsi_div  = bool(rsi_now > rsi_5ago)
    ma50_held = bool(today_close > latest["ma50"])

    checks = {
        "Broke Below Support":  broke_below,
        "Closed Back Above":    recovered,
        "Bullish Candle Close": upper_close,
        "High Volume":          high_vol,
        "RSI Divergence":       rsi_div,
        "MA50 Held":            ma50_held,
    }

    score  = sum(checks.values())
    passed = bool(broke_below and recovered and upper_close)
    grade  = "A" if score >= 5 else "B" if score >= 4 else "C" if score >= 3 else "D"

    return {
        "passed":   passed,
        "score":    score,
        "total":    6,
        "detail":   checks,
        "grade":    grade,
        "pattern":  "Failed Breakdown",
        "support":  round(support, 2),
        "low":      round(today_low, 2),
        "close":    round(today_close, 2),
        "rel_vol":  round(latest["rel_vol"], 1),
        "rsi":      round(rsi_now, 1),
    }


# ═══════════════════════════════════════════════════════════
#  STRATEGY 6: Bullish / Bearish Momentum
# ═══════════════════════════════════════════════════════════

def momentum_filter(df: pd.DataFrame) -> Dict:
    if len(df) < 200:
        return {"passed": False, "score": 0, "detail": {}, "pattern": "Momentum"}

    latest   = df.iloc[-1]
    close    = latest["close"]
    rsi      = latest["rsi"]
    macd     = latest["macd"]
    macd_sig = latest["macd_sig"]
    rel_vol  = latest["rel_vol"]
    high_52w = latest["high_52w"]
    low_52w  = latest["low_52w"]

    bull_checks = {
        "Price > MA20":         bool(close > latest["ma20"]),
        "Price > MA50":         bool(close > latest["ma50"]),
        "Price > MA200":        bool(close > latest["ma200"]),
        "MA50 > MA200":         bool(latest["ma50"] > latest["ma200"]),
        "RSI 50-80":            bool(50 <= rsi <= 80),
        "MACD > Signal":        bool(macd > macd_sig),
        "MACD > 0":             bool(macd > 0),
        "Near 52W High (<=10%)":bool(close >= high_52w * 0.90),
        "Rel Vol >=1.2x":       bool(rel_vol >= 1.2),
    }
    bull_score = sum(bull_checks.values())
    is_bullish = bull_score >= 7

    bear_checks = {
        "Price < MA20":         bool(close < latest["ma20"]),
        "Price < MA50":         bool(close < latest["ma50"]),
        "Price < MA200":        bool(close < latest["ma200"]),
        "MA50 < MA200":         bool(latest["ma50"] < latest["ma200"]),
        "RSI 20-50":            bool(20 <= rsi <= 50),
        "MACD < Signal":        bool(macd < macd_sig),
        "MACD < 0":             bool(macd < 0),
        "Near 52W Low (<=10%)": bool(close <= low_52w * 1.10),
        "Rel Vol >=1.2x":       bool(rel_vol >= 1.2),
    }
    bear_score = sum(bear_checks.values())
    is_bearish = bear_score >= 7

    direction = "BULLISH" if is_bullish else "BEARISH" if is_bearish else "NEUTRAL"
    passed    = is_bullish or is_bearish
    score     = bull_score if is_bullish else bear_score
    grade     = "A" if score >= 9 else "B" if score >= 7 else "C" if score >= 5 else "D"

    return {
        "passed":            passed,
        "score":             score,
        "total":             9,
        "detail":            bull_checks if is_bullish else bear_checks,
        "grade":             grade,
        "pattern":           "Momentum",
        "direction":         direction,
        "rsi":               round(rsi, 1),
        "rel_vol":           round(rel_vol, 1),
        "macd":              round(macd, 3),
        "pct_from_52w_high": round((close / high_52w - 1) * 100, 1),
    }


# ═══════════════════════════════════════════════════════════
#  DAY TRADING STRATEGY 1: VWAP Trend
# ═══════════════════════════════════════════════════════════

def vwap_strategy(df: pd.DataFrame) -> Dict:
    """
    VWAP computed from rolling daily bars (monthly window).
    Bullish: price > VWAP + rel vol high + RSI rising.
    Bearish: price < VWAP + rel vol high + RSI falling.
    """
    if len(df) < 22:
        return {"passed": False, "score": 0, "detail": {}, "pattern": "VWAP"}

    df = df.copy()
    # Typical price × volume cumulative over rolling 20 days
    df["tp"]   = (df["high"] + df["low"] + df["close"]) / 3
    df["tpv"]  = df["tp"] * df["volume"]
    df["vwap"] = df["tpv"].rolling(20).sum() / df["volume"].rolling(20).sum()

    latest   = df.iloc[-1]
    close    = latest["close"]
    vwap     = latest["vwap"]
    rel_vol  = latest["rel_vol"]
    rsi      = latest["rsi"]
    rsi_prev = df["rsi"].iloc[-4]

    above_vwap    = bool(close > vwap)
    pct_from_vwap = (close - vwap) / vwap

    if above_vwap:
        checks = {
            "Price Above VWAP":    True,
            "VWAP Gap > 0.5%":     bool(pct_from_vwap > 0.005),
            "Rel Vol >= 1.5x":     bool(rel_vol >= 1.5),
            "RSI Rising":          bool(rsi > rsi_prev),
            "RSI 45-80":           bool(45 <= rsi <= 80),
            "MA20 Aligned":        bool(close > latest["ma20"]),
        }
        direction = "LONG"
    else:
        checks = {
            "Price Below VWAP":    True,
            "VWAP Gap > 0.5%":     bool(abs(pct_from_vwap) > 0.005),
            "Rel Vol >= 1.5x":     bool(rel_vol >= 1.5),
            "RSI Falling":         bool(rsi < rsi_prev),
            "RSI 20-55":           bool(20 <= rsi <= 55),
            "MA20 Aligned":        bool(close < latest["ma20"]),
        }
        direction = "SHORT"

    score  = sum(checks.values())
    passed = score >= 4
    grade  = "A" if score >= 6 else "B" if score >= 4 else "C"

    return {
        "passed":    passed,
        "score":     score,
        "total":     6,
        "detail":    checks,
        "grade":     grade,
        "pattern":   "VWAP",
        "direction": direction,
        "vwap":      round(vwap, 2),
        "pct_from_vwap": round(pct_from_vwap * 100, 2),
        "rsi":       round(rsi, 1),
        "rel_vol":   round(rel_vol, 1),
    }


# ═══════════════════════════════════════════════════════════
#  DAY TRADING STRATEGY 2: Opening Range Breakout (ORB)
# ═══════════════════════════════════════════════════════════

def opening_range_breakout(df: pd.DataFrame) -> Dict:
    """
    Approximates ORB using daily bar structure.
    Bullish ORB: gap up + today closed in top 25% of range + high volume.
    Bearish ORB: gap down + today closed in bottom 25% + high volume.
    """
    if len(df) < 5:
        return {"passed": False, "score": 0, "detail": {}, "pattern": "ORB"}

    latest    = df.iloc[-1]
    day_range = latest["high"] - latest["low"]
    close_pos = (latest["close"] - latest["low"]) / day_range if day_range > 0 else 0.5
    gap_pct   = latest["gap_pct"]
    rel_vol   = latest["rel_vol"]

    bullish_orb = gap_pct > 0.005
    bearish_orb = gap_pct < -0.005

    if bullish_orb:
        checks = {
            "Gap Up Open":           bool(gap_pct > 0.005),
            "Closed Top 25% Range":  bool(close_pos >= 0.75),
            "Rel Vol >= 2x":         bool(rel_vol >= 2.0),
            "Above Yesterday High":  bool(latest["close"] > df.iloc[-2]["high"]),
            "RSI > 50":              bool(latest["rsi"] > 50),
            "MACD Bullish":          bool(latest["macd"] > latest["macd_sig"]),
        }
        direction = "BULLISH ORB"
    elif bearish_orb:
        checks = {
            "Gap Down Open":         bool(gap_pct < -0.005),
            "Closed Bot 25% Range":  bool(close_pos <= 0.25),
            "Rel Vol >= 2x":         bool(rel_vol >= 2.0),
            "Below Yesterday Low":   bool(latest["close"] < df.iloc[-2]["low"]),
            "RSI < 50":              bool(latest["rsi"] < 50),
            "MACD Bearish":          bool(latest["macd"] < latest["macd_sig"]),
        }
        direction = "BEARISH ORB"
    else:
        return {"passed": False, "score": 0, "detail": {"No significant gap": False},
                "pattern": "ORB", "grade": "D", "direction": "NONE"}

    score  = sum(checks.values())
    passed = score >= 4
    grade  = "A" if score >= 6 else "B" if score >= 4 else "C"

    return {
        "passed":    passed,
        "score":     score,
        "total":     6,
        "detail":    checks,
        "grade":     grade,
        "pattern":   "ORB",
        "direction": direction,
        "gap_pct":   round(gap_pct * 100, 2),
        "close_pos": round(close_pos * 100, 1),
        "rel_vol":   round(rel_vol, 1),
        "rsi":       round(latest["rsi"], 1),
    }


# ═══════════════════════════════════════════════════════════
#  DAY TRADING STRATEGY 3: High Relative Volume Spike
# ═══════════════════════════════════════════════════════════

def high_relative_volume(df: pd.DataFrame, threshold: float = 3.0) -> Dict:
    """
    Detects stocks with unusually high volume today (3x+ average).
    Combined with price momentum = high-probability day trade.
    """
    if len(df) < 22:
        return {"passed": False, "score": 0, "detail": {}, "pattern": "High Rel Vol"}

    latest  = df.iloc[-1]
    rel_vol = latest["rel_vol"]
    pct_chg = latest["pct_chg"]
    close   = latest["close"]

    # Price range expansion: today's range vs avg range
    today_range = latest["high"] - latest["low"]
    avg_range   = (df["high"] - df["low"]).rolling(20).mean().iloc[-1]
    range_exp   = today_range / avg_range if avg_range > 0 else 1

    checks = {
        f"Rel Vol >= {threshold}x":  bool(rel_vol >= threshold),
        "Range Expansion >= 1.5x":   bool(range_exp >= 1.5),
        "Price Moving (>1%)":        bool(abs(pct_chg) >= 0.01),
        "Above MA20":                bool(close > latest["ma20"]),
        "RSI Active (40-85)":        bool(40 <= latest["rsi"] <= 85),
        "MACD Confirming":           bool(
            (pct_chg > 0 and latest["macd"] > latest["macd_sig"]) or
            (pct_chg < 0 and latest["macd"] < latest["macd_sig"])
        ),
    }

    score     = sum(checks.values())
    passed    = bool(rel_vol >= threshold and abs(pct_chg) >= 0.01)
    grade     = "A" if score >= 6 else "B" if score >= 4 else "C"
    direction = "UP" if pct_chg > 0 else "DOWN"

    return {
        "passed":     passed,
        "score":      score,
        "total":      6,
        "detail":     checks,
        "grade":      grade,
        "pattern":    "High Rel Vol",
        "direction":  direction,
        "rel_vol":    round(rel_vol, 1),
        "range_exp":  round(range_exp, 1),
        "pct_chg":    round(pct_chg * 100, 2),
        "rsi":        round(latest["rsi"], 1),
    }


# ═══════════════════════════════════════════════════════════
#  DAY TRADING STRATEGY 4: Pre-Market Gap & Go
# ═══════════════════════════════════════════════════════════

def premarket_gap_go(df: pd.DataFrame) -> Dict:
    """
    Identifies Gap & Go setups: stock gaps and CONTINUES in gap direction
    (as opposed to fading back). Best day-trading momentum play.
    """
    if len(df) < 10:
        return {"passed": False, "score": 0, "detail": {}, "pattern": "Gap & Go"}

    latest  = df.iloc[-1]
    prev    = df.iloc[-2]
    gap_pct = latest["gap_pct"]
    abs_gap = abs(gap_pct)

    if abs_gap < 0.03:
        return {"passed": False, "score": 0,
                "detail": {"Gap >= 3% Required": False},
                "pattern": "Gap & Go", "grade": "D"}

    gap_up = gap_pct > 0

    if gap_up:
        close_pos = (latest["close"] - latest["low"]) / max(latest["high"] - latest["low"], 0.01)
        checks = {
            "Gap Up >= 3%":           bool(gap_pct >= 0.03),
            "Gap Up >= 5% (Strong)":  bool(gap_pct >= 0.05),
            "Continues Up (no fade)": bool(close_pos >= 0.5),
            "Rel Vol >= 2x":          bool(latest["rel_vol"] >= 2.0),
            "Above Prev Day High":    bool(latest["close"] > prev["high"]),
            "RSI Momentum (50-85)":   bool(50 <= latest["rsi"] <= 85),
        }
    else:
        close_pos = (latest["close"] - latest["low"]) / max(latest["high"] - latest["low"], 0.01)
        checks = {
            "Gap Down >= 3%":          bool(gap_pct <= -0.03),
            "Gap Down >= 5% (Strong)": bool(gap_pct <= -0.05),
            "Continues Down (no bounce)": bool(close_pos <= 0.5),
            "Rel Vol >= 2x":           bool(latest["rel_vol"] >= 2.0),
            "Below Prev Day Low":      bool(latest["close"] < prev["low"]),
            "RSI Weak (15-50)":        bool(15 <= latest["rsi"] <= 50),
        }

    score     = sum(checks.values())
    passed    = score >= 4
    grade     = "A" if score >= 6 else "B" if score >= 5 else "C" if score >= 4 else "D"
    direction = "GAP UP" if gap_up else "GAP DOWN"

    return {
        "passed":    passed,
        "score":     score,
        "total":     6,
        "detail":    checks,
        "grade":     grade,
        "pattern":   "Gap & Go",
        "direction": direction,
        "gap_pct":   round(gap_pct * 100, 2),
        "rel_vol":   round(latest["rel_vol"], 1),
        "rsi":       round(latest["rsi"], 1),
    }


# ═══════════════════════════════════════════════════════════
#  DAY TRADING STRATEGY 5: Intraday Momentum
# ═══════════════════════════════════════════════════════════

def intraday_momentum(df: pd.DataFrame) -> Dict:
    """
    Multi-factor intraday momentum score combining
    volume, price action, trend and indicator alignment.
    """
    if len(df) < 22:
        return {"passed": False, "score": 0, "detail": {}, "pattern": "Intraday Momentum"}

    latest  = df.iloc[-1]
    prev    = df.iloc[-2]
    close   = latest["close"]
    pct_chg = latest["pct_chg"]

    # Candle body ratio: large body = conviction
    candle_range = latest["high"] - latest["low"]
    candle_body  = abs(latest["close"] - latest["open"])
    body_ratio   = candle_body / candle_range if candle_range > 0 else 0

    # Price vs yesterday
    above_prev_close = close > prev["close"]

    # Checks
    checks = {
        "Move > 2% Today":         bool(abs(pct_chg) >= 0.02),
        "Rel Vol >= 2x":           bool(latest["rel_vol"] >= 2.0),
        "Strong Candle (>50% body)": bool(body_ratio > 0.5),
        "Trending (above MA20)":   bool(close > latest["ma20"]) if pct_chg > 0 else bool(close < latest["ma20"]),
        "MACD Aligned":            bool(latest["macd"] > latest["macd_sig"]) if pct_chg > 0
                                   else bool(latest["macd"] < latest["macd_sig"]),
        "BB Expansion":            bool(latest["bb_width"] > df["bb_width"].iloc[-10:].mean() * 1.2),
        "Above/Below Prev Close":  above_prev_close if pct_chg > 0 else not above_prev_close,
    }

    score     = sum(checks.values())
    passed    = score >= 5
    grade     = "A" if score >= 7 else "B" if score >= 5 else "C"
    direction = "BULLISH" if pct_chg > 0 else "BEARISH"

    return {
        "passed":     passed,
        "score":      score,
        "total":      7,
        "detail":     checks,
        "grade":      grade,
        "pattern":    "Intraday Momentum",
        "direction":  direction,
        "pct_chg":    round(pct_chg * 100, 2),
        "body_ratio": round(body_ratio * 100, 1),
        "rel_vol":    round(latest["rel_vol"], 1),
        "rsi":        round(latest["rsi"], 1),
    }


# ═══════════════════════════════════════════════════════════
#  MASTER RUNNER
# ═══════════════════════════════════════════════════════════

# Swing strategies
SWING_STRATEGIES = {
    "Minervini SEPA":   minervini_sepa,
    "Bonde Momentum":   bonde_momentum,
    "Island Reversal":  bullish_island_reversal,
    "Gap Up/Down":      gap_scanner,
    "Failed Breakdown": failed_breakdown,
    "Momentum Filter":  momentum_filter,
}

# Day trading strategies
DAY_STRATEGIES = {
    "VWAP Trend":         vwap_strategy,
    "Opening Range (ORB)": opening_range_breakout,
    "High Rel Volume":    high_relative_volume,
    "Gap & Go":           premarket_gap_go,
    "Intraday Momentum":  intraday_momentum,
}

STRATEGY_MAP = {**SWING_STRATEGIES, **DAY_STRATEGIES}


def run_strategies(ticker: str, df_raw: pd.DataFrame, selected: list) -> Dict:
    if df_raw is None or len(df_raw) < 30:
        return None

    df  = compute_indicators(df_raw)
    row = {
        "Ticker":  ticker,
        "Price":   round(df.iloc[-1]["close"], 2),
        "Chg%":    round(df.iloc[-1]["pct_chg"] * 100, 2),
        "Vol":     int(df.iloc[-1]["volume"]),
        "RelVol":  round(df.iloc[-1]["rel_vol"], 1),
        "RSI":     round(df.iloc[-1]["rsi"], 1),
        "Signals": [],
        "_detail": {},
    }

    signal_count = 0
    for name in selected:
        fn = STRATEGY_MAP.get(name)
        if fn is None:
            continue
        try:
            res = fn(df)
            row["_detail"][name] = res
            if res.get("passed"):
                row["Signals"].append(f"{name} [{res.get('grade','')}]")
                signal_count += 1
        except Exception as e:
            row["_detail"][name] = {"passed": False, "error": str(e)}

    row["# Signals"]  = signal_count
    row["Signal List"] = ", ".join(row["Signals"]) if row["Signals"] else "—"
    return row

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
#  MASTER RUNNER
# ═══════════════════════════════════════════════════════════

STRATEGY_MAP = {
    "Minervini SEPA":   minervini_sepa,
    "Bonde Momentum":   bonde_momentum,
    "Island Reversal":  bullish_island_reversal,
    "Gap Up/Down":      gap_scanner,
    "Failed Breakdown": failed_breakdown,
    "Momentum Filter":  momentum_filter,
}


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
